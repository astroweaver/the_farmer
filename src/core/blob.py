# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Class function to handle potentially blended sources (i.e. blobs)

Known Issues
------------
None


"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table, Column
from tractor import NCircularGaussianPSF, PixelizedPSF, Image, Tractor, FluxesPhotoCal, NullWCS, ConstantSky, GalaxyShape, Fluxes, PixPos
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev
from tractor.pointsource import PointSource
from time import time
import photutils
import sep
from matplotlib.colors import LogNorm

from .subimage import Subimage
from .utils import SimpleGalaxy
import config as conf


class Blob(Subimage):
    """TODO: docstring"""

    def __init__(self, brick, blob_id):
        """TODO: docstring"""

        blobmask = np.array(brick.blobmap == blob_id, bool)
        mask_frac = blobmask.sum() / blobmask.size
        if (mask_frac > conf.SPARSE_THRESH) & (blobmask.size > conf.SPARSE_SIZE):
            print('Blob is rejected as mask is sparse - likely an artefact issue.')
            blob = None

        self.brick_wcs = brick.wcs 
        self.mosaic_origin = brick.mosaic_origin
        self.brick_id = brick.brick_id

        # Grab blob
        self.blob_id = blob_id
        self.blobmask = blobmask
        blob_sources = np.unique(brick.segmap[blobmask])

        # Dimensions
        idx, idy = blobmask.nonzero()
        xlo, xhi = np.min(idx), np.max(idx) + 1
        ylo, yhi = np.min(idy), np.max(idy) + 1
        w = xhi - xlo
        h = yhi - ylo

        # Make cutout
        blob_comps = brick._get_subimage(xlo, ylo, w, h, buffer=conf.BLOB_BUFFER)
        # FIXME: too many return values
        self.images, self.weights, self.masks, self.psfmodels, self.bands, self.wcs, self.subvector, self.slicepix, self.slice = blob_comps

        self.masks[self.slicepix] = np.logical_not(blobmask[self.slice], dtype=bool)
        self.segmap = brick.segmap[self.slice]
        self._level = 0
        self._sublevel = 0

        # Clean
        blob_sourcemask = np.in1d(brick.catalog['sid'], blob_sources)
        self.bcatalog = brick.catalog[blob_sourcemask].copy()
        self.bcatalog['x'] -= self.subvector[1]
        self.bcatalog['y'] -= self.subvector[0]
        self.n_sources = len(self.bcatalog)

        self.mids = np.ones(self.n_sources, dtype=int)
        self.model_catalog = np.zeros(self.n_sources, dtype=object)
        self.solution_catalog = np.zeros(self.n_sources, dtype=object)
        self.solved_chisq = np.zeros(self.n_sources)
        self.tr_catalogs = np.zeros((self.n_sources, 3, 2), dtype=object)
        self.chisq = np.zeros((self.n_sources, 3, 2))
        self.position_variance = np.zeros((self.n_sources, 2))
        self.parameter_variance = np.zeros((self.n_sources, 3))
        self.forced_variance = np.zeros((self.n_sources, self.n_bands))
        self.solution_tractor = None

        self.residual_catalog = np.zeros((self.n_bands), dtype=object)
        self.residual_segmap = np.zeros_like(self.segmap)
        self.n_residual_sources = np.zeros(self.n_bands, dtype=int)

        # TODO NEED TO LOOK AT OLD SCRIPT FOR AN IDEA ABOUT WHAT COMPOSITE SPITS OUT!!!

    def stage_images(self):
        """TODO: docstring"""

        timages = np.zeros(self.n_bands, dtype=object)

        # self.subtract_background()

        # TODO: try to simplify this. particularly the zip...
        for i, (image, weight, mask, psf, band) in enumerate(zip(self.images, self.weights, self.masks, self.psfmodels, self.bands)):
            tweight = weight.copy()
            tweight[mask] = 0


            print(f'I THINK THE PSF IS {psf}')
            if psf == -99:
                psfmodel = NCircularGaussianPSF([conf.PSF_SIGMA,], [1,])
                print('WARNING - Adopting FAKE PSF model!')
            else:
                psfmodel = PixelizedPSF(psf)

            timages[i] = Image(data=image,
                            invvar=tweight,
                            psf=psfmodel,
                            wcs=NullWCS(),
                            photocal=FluxesPhotoCal(band),
                            sky=ConstantSky(0.))

        self.timages = timages

    def stage_models(self):
        # Trackers

        for i, (mid, src) in enumerate(zip(self.mids, self.bcatalog)):

            freeze_position = (self.mids >= 2).all()
            if freeze_position & (conf.FREEZE_POSITION):
                position = self.tr_catalogs[i,0,0].getPosition()
            else:
                position = PixPos(src['x'], src['y'])
            flux = Fluxes(**dict(zip(self.bands, src['flux'] * np.ones(self.n_bands))))

            shape = GalaxyShape(src['a'], src['b'] / src['a'], src['theta'])

            if mid == 1:
                self.model_catalog[i] = PointSource(position, flux)
                # self.model_catalog[i].name = 'PointSource' # HACK to get around Dustin's HACK.
            elif mid == 2:
                self.model_catalog[i] = SimpleGalaxy(position, flux)
            elif mid == 3:
                self.model_catalog[i] = ExpGalaxy(position, flux, shape)
            elif mid == 4:
                self.model_catalog[i] = DevGalaxy(position, flux, shape)
            elif mid == 5:
                self.model_catalog[i] = FixedCompositeGalaxy(
                                                position, flux,
                                                SoftenedFracDev(0.5),
                                                shape, shape)
            if freeze_position & (conf.FREEZE_POSITION):
                self.model_catalog[i].freezeParams('pos')
                if conf.VERBOSE2: print(f'Position parameter frozen at {position}')

            if conf.VERBOSE2: print(f'Instatiated a model at {position}')
            if conf.VERBOSE2: print(f'Parameters: {flux}, {shape}')

    def tractor_phot(self):

        # TODO: The meaning of the following line is not clear
        idx_models = ((1, 2), (3, 4), (5,))
        if conf.VERBOSE2: print(f'Attempting to model {self.n_sources} sources.')

        self._solved = self.solution_catalog != 0

        self._level = -1

        while not self._solved.all():
            self._level += 1
            for sublevel in np.arange(len(idx_models[self._level])):
                self._sublevel = sublevel

                self.stage = f'Morph Model ({self._level}, {self._sublevel})'

                # prepare models
                self.mids[~self._solved] = idx_models[self._level][sublevel]
                self.stage_models()

                # store
                self.tr = Tractor(self.timages, self.model_catalog)

                # optimize
                if conf.VERBOSE2: print(self.stage)
                self.status = self.optimize_tractor()

                if self.status == False:
                    return False

                # clean up
                self.tr_catalogs[:, self._level, self._sublevel] = self.tr.getCatalog()

                if (self._level == 0) & (self._sublevel == 0):
                    self.position_variance = np.array([self.variance[i][:2] for i in np.arange(self.n_sources)]) # THIS MAY JUST WORK!
                    # print(f'POSITION VAR: {self.position_variance}')

                for i, src in enumerate(self.bcatalog):
                    if self._solved[i]:
                        continue
                    totalchisq = np.sum((self.tr.getChiImage(0)[self.segmap == src['sid']])**2)
                    if conf.USE_REDUCEDCHISQ:
                        totalchisq /= (np.sum(self.segmap == src['sid']) - self.model_catalog[i].numberOfParams())
                    # PENALTIES TBD
                    # residual = self.images[0] - self.tr.getModelImage(0)
                    # if np.median(residual[self.masks[0]]) < 0:
                    #     if conf.VERBOSE2: print(f'Applying heavy penalty on source #{i+1} ({self.model_catalog[i].name})!')
                    #     totalchisq = 1E30
                    if conf.VERBOSE2: print(f'Source #{i+1} with {self.model_catalog[i].name} has chisq={totalchisq:3.3f}')
                    self.chisq[i, self._level, self._sublevel] = totalchisq

                # Move unsolved to next sublevel
                if sublevel == 0:
                    self.mids[~self._solved] += 1

            # decide
            self.decide_winners()
            self._solved = self.solution_catalog != 0

        # print('Starting final optimization')
        # Final optimization
        self.model_catalog = self.solution_catalog.copy()
        self.tr = Tractor(self.timages, self.model_catalog)

        self.stage = 'Final Optimization'
        self._level, self._sublevel = 'FO', 'FO'
        if conf.VERBOSE2: print(self.stage)
        self.status = self.optimize_tractor()
        
        if not self.status:
            return False

        self.solution_catalog = self.tr.getCatalog()
        self.solution_tractor = Tractor(self.timages, self.solution_catalog)
        self.solution_model_images = np.array([self.tr.getModelImage(i) for i in np.arange(self.n_bands)])
        self.solution_chi_images = np.array([self.tr.getChiImage(i) for i in np.arange(self.n_bands)])
        self.parameter_variance = [self.variance[i][self.n_bands:] for i in np.arange(self.n_sources)]
        # print(f'PARAMETER VAR: {self.parameter_variance}')

        for i, src in enumerate(self.bcatalog):
            totalchisq = np.sum((self.tr.getChiImage(0)[self.segmap == src['sid']])**2)
            if conf.USE_REDUCEDCHISQ:
                totalchisq /= (np.sum(self.segmap == src['sid']) - self.model_catalog[i].numberOfParams())
            self.solved_chisq[i] = totalchisq

        return True

    def decide_winners(self):
        # take the model_catalog and chisq and figure out what's what
        # Only look at unsolved models!

        # holders - or else it's pure insanity.
        chisq = self.chisq[~self._solved]
        solution_catalog = self.solution_catalog[~self._solved]
        solved_chisq = self.solved_chisq[~self._solved]
        tr_catalogs = self.tr_catalogs[~self._solved]
        mids = self.mids[~self._solved]

        if self._level == 0:
            # Which have chi2(PS) < chi2(SG)?
            chmask = (chisq[:, 0, 0] < chisq[:, 0, 1])
            if chmask.any():
                solution_catalog[chmask] = tr_catalogs[chmask, 0, 0].copy()
                solved_chisq[chmask] = chisq[chmask, 0, 0]
                mids[chmask] = 1

            # So chi2(SG) is min, try more models
            mids[~chmask] = 3

        if self._level == 1:
            # For which are they nearly equally good?
            movemask = (abs(chisq[:, 1, 0] - chisq[:, 1, 1]) < conf.EXP_DEV_THRESH)

            # Has Exp beaten SG?
            expmask = (chisq[:, 1, 0] < chisq[:, 0, 1])

            # Has Dev beaten SG?
            devmask = (chisq[:, 1, 1] < chisq[:, 0, 1])

            # Which better than SG but nearly equally good?
            nextmask = expmask & devmask & movemask

            # For which was SG better
            premask = ~expmask & ~devmask

            # If Exp beats Dev by a lot
            nexpmask = expmask & ~movemask & (chisq[:, 1, 0] < chisq[:, 1, 1])

             # If Dev beats Exp by a lot
            ndevmask = devmask & ~movemask & (chisq[:, 1, 1] < chisq[:, 1, 0])

            if nextmask.any():
                mids[nextmask] = 5

            if premask.any():
                solution_catalog[premask] = tr_catalogs[premask, 0, 1].copy()
                solved_chisq[premask] = chisq[premask, 0, 1]
                mids[premask] = 2

            if nexpmask.any():

                solution_catalog[nexpmask] = tr_catalogs[nexpmask, 1, 0].copy()
                solved_chisq[nexpmask] = chisq[nexpmask, 1, 0]
                mids[nexpmask] = 3

            if ndevmask.any():

                solution_catalog[ndevmask] = tr_catalogs[ndevmask, 1, 1].copy()
                solved_chisq[ndevmask] = chisq[ndevmask, 1, 1]
                mids[ndevmask] = 4

        if self._level == 2:
            # For which did Comp beat EXP and DEV?
            compmask = (chisq[:, 2, 0] < chisq[:, 1, 0]) &\
                       (chisq[:, 2, 0] < chisq[:, 1, 1])

            if compmask.any():
                solution_catalog[compmask] = tr_catalogs[compmask, 2, 0].copy()
                solved_chisq[compmask] = chisq[compmask, 2, 0]
                mids[compmask] = 5

            # where better as EXP or DEV
            if (~compmask).any():
                ch_exp = (chisq[:, 1, 0] < chisq[:, 1, 1]) & ~compmask

                if ch_exp.any():
                    solution_catalog[ch_exp] = tr_catalogs[ch_exp, 1, 0].copy()
                    solved_chisq[ch_exp] = chisq[ch_exp, 1, 0]
                    mids[ch_exp] = 3

                ch_dev = (chisq[:, 1, 1] < chisq[:, 1, 0]) & ~compmask

                if ch_dev.any():
                    solution_catalog[ch_dev] = tr_catalogs[ch_dev, 1, 1].copy()
                    solved_chisq[ch_dev] = chisq[ch_dev, 1, 1]
                    mids[ch_dev] = 4

        # hand back
        self.chisq[~self._solved] = chisq
        self.solution_catalog[~self._solved] = solution_catalog
        self.solved_chisq[~self._solved] = solved_chisq
        self.mids[~self._solved] = mids

    def optimize_tractor(self, tr=None):

        if tr is None:
            tr = self.tr

        tr.freezeParams('images')        

        start = time()
        if conf.VERBOSE2: print(f'Starting optimization ({conf.TRACTOR_MAXSTEPS}, {conf.TRACTOR_CONTHRESH})')


        # fig, ax = plt.subplots(ncols=2)
        # back = self.backgrounds[0]
        # mean, rms = back[0], back[1]
        # norm = LogNorm(np.max([mean + rms, 1E-5]), self.images[0].max(), clip='True')
        # img_opt = dict(cmap='Greys', norm=norm)
        # ax[0].imshow(self.images[0], **img_opt)
        # ax[1].imshow(self.tr.getModelImage(0), **img_opt)
        # for s, src in enumerate(self.solution_catalog):
        #     if src == 0:
        #         continue
        #     x, y = src.pos
        #     color = 'r'

        #     ax[0].plot([x, x], [y - 10, y - 5], c=color)
        #     ax[0].plot([x - 10, x - 5], [y, y], c=color)

        # fig.suptitle(f'L{self._level}_SL{self._sublevel}')
        # fig.savefig(os.path.join(conf.PLOT_DIR, f'{self.brick_id}_{self.blob_id}_L{self._level}_SL{self._sublevel}_DEBUG0.pdf'))
        # plt.close()


        for i in range(conf.TRACTOR_MAXSTEPS):
            # if True:
            try:
                dlnp, X, alpha, var = tr.optimize(variance=True)

                # fig, ax = plt.subplots(ncols=2)
                # back = self.backgrounds[0]
                # mean, rms = back[0], back[1]
                # norm = LogNorm(np.max([mean + rms, 1E-5]), self.images[0].max(), clip='True')
                # img_opt = dict(cmap='Greys', norm=norm)
                # ax[0].imshow(self.images[0], **img_opt)
                # ax[1].imshow(self.tr.getModelImage(0), **img_opt)
                # for s, src in enumerate(self.solution_catalog):
                #     if src == 0:
                #         continue
                #     x, y = src.pos
                #     color = 'r'

                #     ax[0].plot([x, x], [y - 10, y - 5], c=color)
                #     ax[0].plot([x - 10, x - 5], [y, y], c=color)
                # fig.suptitle(f'L{self._level}_SL{self._sublevel}')    
                # fig.savefig(os.path.join(conf.PLOT_DIR, f'{self.brick_id}_{self.blob_id}_L{self._level}_SL{self._sublevel}_DEBUG{int(i+1)}.pdf'))
                # plt.close()

                if conf.VERBOSE2: print(dlnp)
            except:
                if conf.VERBOSE: print(f'WARNING - Optimization failed on step {i} for blob #{self.blob_id}')
                return False

            if dlnp < conf.TRACTOR_CONTHRESH:
                break

        if var is None:
            if conf.VERBOSE: print(f'WARNING - VAR is NONE for blob #{self.blob_id}')
            return False

        if (self.solution_catalog != 0).all():
            # print('CHANGING TO SOLUTION CATALOG FOR PARAMETERS')
            var_catalog = self.solution_catalog
        else:
            var_catalog = self.model_catalog

        self.variance = []
        counter = 0
        for i, src in enumerate(np.arange(self.n_sources)):
            n_params = var_catalog[i].numberOfParams()
            myvar = var[counter: n_params + counter]
            # print(f'{i}) {var_catalog[i].name} has {n_params} params and {len(myvar)} variances: {myvar}')
            counter += n_params
            self.variance.append(myvar)

        if np.shape(self.tr.getChiImage(0)) != np.shape(self.segmap):
            if conf.VERBOSE: print(f'WARNING - Chimap and segmap are not the same shape for #{self.blob_id}')
            return False

        # expvar = np.sum([var_catalog[i].numberOfParams() for i in np.arange(len(var_catalog))])
        # # print(f'I have {len(var)} variance parameters for {self.n_sources} sources. I expected {expvar}.')
        # for i, mod in enumerate(var_catalog):
        #     totalchisq = np.sum((self.tr.getChiImage(0)[self.segmap == self.catalog[i]['sid']])**2)

        return True

    def aperture_phot(self, band, image_type=None, sub_background=False):
        # Allow user to enter image (i.e. image, residual, model...)

        if image_type not in ('image', 'model', 'residual'):
            raise TypeError("image_type must be 'image', 'model' or 'residual'")
            return

        idx = self._band2idx(band)

        if image_type == 'image':
            image = self.images[idx]

        elif image_type == 'model':
            image = self.solution_tractor.getModelImage(idx)

        elif image_type == 'residual':
            image = self.images[idx] - self.solution_tractor.getModelImage(idx)

        background = self.backgrounds[idx]

        if (self.weights == 1).all():
            # No weight given - kinda
            var = None
            thresh = conf.RES_THRESH * background.globalrms
            if not sub_background:
                thresh += background.globalback

        else:
            thresh = conf.RES_THRESH
            tweight = self.weights[idx].copy()
            tweight[self.masks[idx]] = 0  # Well this isn't going to go well.
            var = 1. / tweight # TODO: WRITE TO UTILS

        if sub_background:
            image -= background.back()

        cat = self.solution_catalog
        xxyy = np.vstack([src.getPosition() for src in cat]).T
        apxy = xxyy - 1.

        apertures_arcsec = np.array(conf.APER_PHOT)
        apertures = apertures_arcsec / self.pixel_scale

        apflux = np.zeros((len(cat), len(apertures)), np.float32)
        apflux_err = np.zeros((len(cat), len(apertures)), np.float32)

        H,W = image.shape
        Iap = np.flatnonzero((apxy[0,:] >= 0)   * (apxy[1,:] >= 0) *
                            (apxy[0,:] <= W-1) * (apxy[1,:] <= H-1))

        if var is None:
            imgerr = None
        else:
            imgerr = np.sqrt(var)

        for i, rad in enumerate(apertures):
            aper = photutils.CircularAperture(apxy[:,Iap], rad)
            p = photutils.aperture_photometry(image, aper, error=imgerr)
            apflux[:, i] = p.field('aperture_sum')
            if var is None:
                apflux_err[:, i] = -99 * np.ones_like(apflux[:, i])
            else:
                apflux_err[:, i] = p.field('aperture_sum_err')

        band = band.replace(' ', '_')
        if f'aperphot_{band}_{image_type}' not in self.catalog.colnames:
            self.bcatalog.add_column(Column(np.zeros(len(self.bcatalog), dtype=(float, len(apertures))), name=f'aperphot_{band}_{image_type}'))
            self.bcatalog.add_column(Column(np.zeros(len(self.bcatalog), dtype=(float, len(apertures))), name=f'aperphot_{band}_{image_type}_err'))

        for idx, src in enumerate(self.solution_catalog):
            sid = self.catalog['sid'][idx]
            row = np.argwhere(self.catalog['sid'] == sid)[0][0]
            self.bcatalog[row][f'aperphot_{band}_{image_type}'] = tuple(apflux[idx])
            self.bcatalog[row][f'aperphot_{band}_{image_type}_err'] = tuple(apflux_err[idx])

    def sextract_phot(self, band, sub_background=False):
        # SHOULD WE STACK THE RESIDUALS? (No?)
        # SHOULD WE DO THIS ON THE DETECTION IMAGE TOO? (I suppose we can already...!)
        idx = self._band2idx(band)
        residual = self.images[idx] - self.solution_tractor.getModelImage(idx)
        tweight = self.weights[idx].copy()
        tweight[self.masks[idx]] = 0 # OK this isn't going to go well.
        var = 1. / tweight # TODO: WRITE TO UTILS
        background = self.backgrounds[idx]

        if (self.weights == 1).all():
            # No weight given - kinda
            var = None
            thresh = conf.RES_THRESH * background.globalrms
            if not sub_background:
                thresh += background.globalback

        else:
            thresh = conf.RES_THRESH

        if sub_background:
            residual -= background.back()

        kwargs = dict(var=var, minarea=conf.RES_MINAREA, segmentation_map=True, deblend_nthresh=conf.RES_DEBLEND_NTHRESH, deblend_cont=conf.RES_DEBLEND_CONT)
        catalog, segmap = sep.extract(residual, thresh, **kwargs)

        if f'{band}_n_residual_sources' not in self.bcatalog.colnames:
                self.bcatalog.add_column(Column(np.zeros(len(self.bcatalog), dtype=bool), name=f'{band}_n_residual_sources'))

        if len(catalog) != 0:
            self.residual_catalog[idx] = catalog
            n_residual_sources = len(catalog)
            self.residual_segmap = segmap
            self.n_residual_sources[idx] = n_residual_sources
            if conf.VERBOSE2: print(f'SExtractor Found {n_residual_sources} in {band} residual!')


            for idx, src in enumerate(self.solution_catalog):
                sid = self.bcatalog['sid'][idx]
                row = np.argwhere(self.catalog['sid'] == sid)[0][0]
                self.bcatalog[row][f'{band}_n_residual_sources'] = True

            return catalog, segmap
        else:
            if conf.VERBOSE2: print('No objects found by SExtractor.')


    def forced_phot(self):

        # print('Starting forced photometry')
        # Update the incoming models
        for i, model in enumerate(self.model_catalog):
            model.brightness = Fluxes(**dict(zip(self.bands, model.brightness[0] * np.ones(self.n_bands))))
            model.freezeAllBut('brightness')

        # Stash in Tractor
        self.tr = Tractor(self.timages, self.model_catalog)
        self.stage = 'Forced Photometry'

        # Optimize
        status = self.optimize_tractor()

        if not status:
            return status

        # Chisq
        self.forced_variance = self.variance
        # print(f'FLUX VAR: {self.forced_variance}')
        self.solution_chisq = np.zeros((self.n_sources, self.n_bands))
        for i, src in enumerate(self.bcatalog):
            for j, band in enumerate(self.bands):
                totalchisq = np.sum((self.tr.getChiImage(j)[self.segmap == src['sid']])**2)
                if conf.USE_REDUCEDCHISQ:
                    totalchisq /= (np.sum(self.segmap == src['sid']) - self.model_catalog[i].numberOfParams())
                self.solution_chisq[i, j] = totalchisq

        self.solution_catalog = self.tr.getCatalog()
        self.solution_tractor = Tractor(self.timages, self.solution_catalog)
        self.solution_model_images = np.array([self.tr.getModelImage(i) for i in np.arange(self.n_bands)])
        self.solution_chi_images = np.array([self.tr.getChiImage(i) for i in np.arange(self.n_bands)])

        # self.rows = np.zeros(len(self.solution_catalog))
        for idx, src in enumerate(self.solution_catalog):
            sid = self.bcatalog['sid'][idx]
            # row = np.argwhere(self.brick.catalog['sid'] == sid)[0][0]
            # self.rows[idx] = row
            # print(f'STASHING {sid} IN ROW {row}')
            self.get_catalog(idx, src)

        return status

    def get_catalog(self, row, src):
        for i, band in enumerate(self.bands):
            band = band.replace(' ', '_')
            self.bcatalog[row][band] = src.getBrightness().getFlux(band)
            self.bcatalog[row][band+'_err'] = np.sqrt(self.forced_variance[row][i])
            self.bcatalog[row][band+'_chisq'] = self.solution_chisq[row, i]
        self.bcatalog[row]['x_model'] = src.pos[0] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER + 1
        self.bcatalog[row]['y_model'] = src.pos[1] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER + 1
        self.bcatalog[row]['x_model_err'] = np.sqrt(self.position_variance[row, 0])
        self.bcatalog[row]['y_model_err'] = np.sqrt(self.position_variance[row, 1])
        if self.wcs is not None:
            # print(self.brick.wcs)
            skyc = self.brick_wcs.all_pix2world(self.bcatalog[row]['x_model'], self.bcatalog[row]['y_model'], 0)
            # print('COORDINATES: ', skyc)
            self.bcatalog[row]['RA'] = skyc[0]
            self.bcatalog[row]['Dec'] = skyc[1]
        try:
            self.bcatalog[row]['solmodel'] = src.name
            skip = False
        except:
            self.bcatalog[row]['solmodel'] = 'PointSource'
            skip = True
        if not skip:
            if src.name in ('SimpleGalaxy', 'ExpGalaxy', 'DevGalaxy', 'FixedCompositeGalaxy'):
                try:
                    self.bcatalog[row]['reff'] = src.shape.re
                    self.bcatalog[row]['ab'] = src.shape.ab
                    self.bcatalog[row]['phi'] = src.shape.phi
                    self.bcatalog[row]['reff_err'] = np.sqrt(self.parameter_variance[row][0])
                    self.bcatalog[row]['ab_err'] = np.sqrt(self.parameter_variance[row][1])
                    self.bcatalog[row]['phi_err'] = np.sqrt(self.parameter_variance[row][2])

                    #print(f"REFF: {self.catalog[row]['reff']}+/-{self.catalog[row]['reff_err']}")
                    #print(f"FLUX: {self.catalog[0][band]}+/-{self.catalog[0][band+'_err']}")
                except:
                    if conf.VERBOSE: print('WARNING - model parameters not added to catalog.')
