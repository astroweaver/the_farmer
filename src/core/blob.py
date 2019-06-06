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
from tractor import NCircularGaussianPSF, PixelizedPSF, Image, Tractor, FluxesPhotoCal, NullWCS, ConstantSky, EllipseESoft, Fluxes, PixPos
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev
from tractor.pointsource import PointSource
from time import time
import photutils
import sep
from matplotlib.colors import LogNorm

from .subimage import Subimage
from .utils import SimpleGalaxy, plot_detblob, plot_fblob
import config as conf


class Blob(Subimage):
    """TODO: docstring"""

    def __init__(self, brick, blob_id):
        """TODO: docstring"""

        blobmask = np.array(brick.blobmap == blob_id, bool)
        mask_frac = blobmask.sum() / blobmask.size
        if (mask_frac > conf.SPARSE_THRESH) & (blobmask.size > conf.SPARSE_SIZE):
            if conf.VERBOSE: print('Blob is rejected as mask is sparse - likely an artefact issue.')
            blob = None

        self.brick_wcs = brick.wcs.copy()
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

        self.blob_center = (xlo + w/2., ylo + h/2.)

        # Make cutout
        blob_comps = brick._get_subimage(xlo, ylo, w, h, buffer=conf.BLOB_BUFFER)
        # FIXME: too many return values
        self.images, self.weights, self.masks, self.psfmodels, self.bands, self.wcs, self.subvector, self.slicepix, self.slice = blob_comps

        self.masks[self.slicepix] = np.logical_not(blobmask[self.slice], dtype=bool)
        self.segmap = brick.segmap[self.slice]
        self._level = 0
        self._sublevel = 0

        # Clean
        
        blob_sourcemask = np.in1d(brick.catalog['source_id'], blob_sources)
        self.bcatalog = brick.catalog[blob_sourcemask].copy() # working copy
        print(self.bcatalog['x', 'y'])
        self.bcatalog['x'] -= self.subvector[1]
        self.bcatalog['y'] -= self.subvector[0]
        print(self.bcatalog['x', 'y'])
        print(self.subvector)
        self.n_sources = len(self.bcatalog)

        self.mids = np.ones(self.n_sources, dtype=int)
        self.model_catalog = np.zeros(self.n_sources, dtype=object)
        self.solution_catalog = np.zeros(self.n_sources, dtype=object)
        self.solved_chisq = np.zeros(self.n_sources)
        self.solved_bic = np.zeros(self.n_sources)
        self.solution_chisq = np.zeros(self.n_sources)
        self.tr_catalogs = np.zeros((self.n_sources, 3, 2), dtype=object)
        self.chisq = np.zeros((self.n_sources, 3, 2))
        self.bic = np.zeros((self.n_sources, 3, 2))
        # self.position_variance = np.zeros((self.n_sources, 2))
        # self.parameter_variance = np.zeros((self.n_sources, 3))
        # self.forced_variance = np.zeros((self.n_sources, self.n_bands))
        self.solution_tractor = None

        self.residual_catalog = np.zeros((self.n_bands), dtype=object)
        self.residual_segmap = np.zeros_like(self.segmap)
        self.n_residual_sources = np.zeros(self.n_bands, dtype=int)

        del brick

    def stage_images(self):
        """TODO: docstring"""

        if conf.VERBOSE2: 
            print()
            print('blob.stage_images :: Staging images...')

        timages = np.zeros(self.n_bands, dtype=object)

        if conf.SUBTRACT_BACKGROUND:
            self.subtract_background(flat=conf.USE_FLAT)
            if conf.VERBOSE2: print(f'blob.stage_images :: Subtracted background (flat={conf.USE_FLAT})')

        # TODO: try to simplify this. particularly the zip...
        for i, (image, weight, mask, psf, band) in enumerate(zip(self.images, self.weights, self.masks, self.psfmodels, self.bands)):
            if conf.VERBOSE2: print(f'blob.stage_images :: Staging image for {band}')
            tweight = weight.copy()
            tweight[mask] = 0

            if band in conf.CONSTANT_PSF:
                psfmodel = psf
                if conf.VERBOSE2: print(f'blob.stage_images :: Adopting constant PSF.')
            else:
                psfmodel = psf.at(self.blob_center[1], self.blob_center[0], NativeScale=True) # init at blob center, may need to swap!
                if conf.VERBOSE2: print(f'blob.stage_images :: Adopting varying PSF constant at ({self.blob_center})')
            # except:
            #     # psfmodel = NCircularGaussianPSF([conf.PSF_SIGMA,], [1,])
            #     raise ValueError(f'WARNING - No PSF model found for {band}!')

            timages[i] = Image(data=image,
                            invvar=tweight,
                            psf=psfmodel,
                            wcs=NullWCS(),
                            photocal=FluxesPhotoCal(band),
                            sky=ConstantSky(0.))

        self.timages = timages

    def stage_models(self):
        # Trackers
        if conf.VERBOSE2: print()
        if conf.VERBOSE2: print(f'blob.stage_models :: Loading models for blob #{self.blob_id}')

        for i, (mid, src) in enumerate(zip(self.mids, self.bcatalog)):

            freeze_position = (self.mids != 1).any()
            # print(f'DEBUG: {self.mids}')
            # print(f'DEBUG: Freeze position? {freeze_position}')
            if conf.FORCE_POSITION:
                position = PixPos(src['x'], src['y'])
                freeze_position = True
            elif freeze_position & (conf.FREEZE_POSITION):
                position = self.tr_catalogs[i,0,0].getPosition()
            else:
                position = PixPos(src['x'], src['y'])
            flux = Fluxes(**dict(zip(self.bands, np.ones(self.n_bands))))

            #shape = GalaxyShape(src['a'], src['b'] / src['a'], src['theta'])
            shape = EllipseESoft.fromRAbPhi(src['a'], src['b'] / src['a'], -0.5*src['theta'])
            #shape = EllipseESoft.fromRAbPhi(1, 1, 0)

            if mid == 1:
                self.model_catalog[i] = PointSource(position, flux)
                self.model_catalog[i].name = 'PointSource' # HACK to get around Dustin's HACK.
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

            if conf.VERBOSE2: print(f'Source #{src["source_id"]}: {self.model_catalog[i].name} model at {position}')
            if conf.VERBOSE2: print(f'               {flux}') 
            if mid not in (1,2):
                if conf.VERBOSE2: print(f'               {shape}')

    def tractor_phot(self):

        # TODO: The meaning of the following line is not clear
        idx_models = ((1, 2), (3, 4), (5,))
        if conf.VERBOSE2: 
            print()
            print(f'blob.tractor_phot :: Attempting to model {self.n_sources} sources.')

        self._solved = self.solution_catalog != 0

        self._level = -1

        if conf.PLOT:
            fig, ax = plot_detblob(self)

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

                if conf.PLOT:
                    plot_detblob(self, fig, ax, level=self._level, sublevel=self._sublevel, init=True)

                # optimize
                if conf.VERBOSE2: print(self.stage)
                self.status = self.optimize_tractor()

                if self.status == False:
                    return False

                # clean up
                self.tr_catalogs[:, self._level, self._sublevel] = self.tr.getCatalog()

                if (self._level == 0) & (self._sublevel == 0):
                    #self.position_variance = np.array([self.variance[i][:2] for i in np.arange(self.n_sources)]) # THIS MAY JUST WORK!
                    self.position_variance = self.variance
                    # print(f'POSITION VAR: {self.position_variance}')

                for i, src in enumerate(self.bcatalog):
                    if self._solved[i]:
                        continue
                    totalchisq = np.sum((self.tr.getChiImage(0)[self.segmap == src['source_id']])**2)
                    m_param = self.model_catalog[i].numberOfParams()
                    n_data = np.sum(self.segmap == src['source_id'])
                    self.bic[i, self._level, self._sublevel] = totalchisq + np.log(n_data) * m_param
                    if conf.USE_REDUCEDCHISQ:
                        totalchisq /= (n_data - m_param)
                    # PENALTIES TBD
                    # residual = self.images[0] - self.tr.getModelImage(0)
                    # if np.median(residual[self.masks[0]]) < 0:
                    #     if conf.VERBOSE2: print(f'Applying heavy penalty on source #{i+1} ({self.model_catalog[i].name})!')
                    #     totalchisq = 1E30
                    if conf.VERBOSE2: print(f'Source #{src["source_id"]} with {self.model_catalog[i].name} has chisq={totalchisq:3.3f} | bic={self.bic[i, self._level, self._sublevel]:3.3f}')
                    if conf.VERBOSE2: print(f'               Fluxes: {self.bands[0]}={self.model_catalog[i].getBrightness().getFlux(self.bands[0]):3.3f}') 
                    if self.model_catalog[i].name not in ('PointSource', 'SimpleGalaxy'):
                        if self.model_catalog[i].name == 'FixedCompositeGalaxy':
                            if conf.VERBOSE2: print(f'               {self.model_catalog[i].shapeExp}')
                            if conf.VERBOSE2: print(f'               {self.model_catalog[i].shapeDev}')
                        else:
                            if conf.VERBOSE2: print(f'               {self.model_catalog[i].shape}')
                    
                    self.chisq[i, self._level, self._sublevel] = totalchisq

                # Move unsolved to next sublevel
                if sublevel == 0:
                    self.mids[~self._solved] += 1

                if conf.PLOT:
                    plot_detblob(self, fig, ax, level=self._level, sublevel=self._sublevel)

            # decide
            self.decide_winners()
            self._solved = self.solution_catalog != 0

        # print('Starting final optimization')
        # Final optimization
        self.model_catalog = self.solution_catalog.copy()
        if conf.VERBOSE2: 
            print()
            print(f'Starting final optimization for blob #{self.blob_id}')
        for i, (mid, src) in enumerate(zip(self.mids, self.bcatalog)):
            if conf.VERBOSE2: print(f'Source #{src["source_id"]}: {self.model_catalog[i].name} model at {self.model_catalog[i].pos}')
            if conf.VERBOSE2: print(f'               Fluxes: {self.bands[0]}={self.model_catalog[i].getBrightness().getFlux(self.bands[0]):3.3f}') 
            if self.model_catalog[i].name not in ('PointSource', 'SimpleGalaxy'):
                if self.model_catalog[i].name == 'FixedCompositeGalaxy':
                    if conf.VERBOSE2: print(f'               {self.model_catalog[i].shapeExp}')
                    if conf.VERBOSE2: print(f'               {self.model_catalog[i].shapeDev}')
                else:
                    if conf.VERBOSE2: print(f'               {self.model_catalog[i].shape}')
            if conf.FREEZE_POSITION:
                self.model_catalog[i].freezeParams('pos')
                if conf.VERBOSE2: print(f'Position parameter frozen at {self.model_catalog[i].pos}')
        self.tr = Tractor(self.timages, self.model_catalog)

        if conf.VERBOSE2: print()
        self.stage = 'Final Optimization'
        # self._level, self._sublevel = 'FO', 'FO'
        if conf.VERBOSE2: print(self.stage)
        self.status = self.optimize_tractor()
        
        if not self.status:
            return False
        if conf.VERBOSE2: print(f'Optimization converged in {self.n_converge+1} steps.')

        self.solution_catalog = self.tr.getCatalog()
        self.solution_tractor = Tractor(self.timages, self.solution_catalog)
        self.solution_model_images = np.array([self.tr.getModelImage(i) for i in np.arange(self.n_bands)])
        self.solution_chi_images = np.array([self.tr.getChiImage(i) for i in np.arange(self.n_bands)])
        self.parameter_variance = self.variance
        # print(f'PARAMETER VAR: {self.parameter_variance}')

        if conf.VERBOSE2: 
            print()
            print(f'Resulting model parameters for blob #{self.blob_id}')
        self.solution_chisq = np.zeros((self.n_sources, self.n_bands))
        self.solution_bic = np.zeros((self.n_sources, self.n_bands))
        for i, src in enumerate(self.bcatalog):
            for j, band in enumerate(self.bands):
                totalchisq = np.sum((self.tr.getChiImage(j)[self.segmap == src['source_id']])**2)
                m_param = self.model_catalog[i].numberOfParams()
                n_data = np.sum(self.segmap == src['source_id'])
                self.solution_bic[i, j] = totalchisq + np.log(n_data) * m_param
                if conf.USE_REDUCEDCHISQ:
                    totalchisq /= (n_data - m_param)
                # PENALTIES TBD
                # residual = self.images[0] - self.tr.getModelImage(0)
                # if np.median(residual[self.masks[0]]) < 0:
                #     if conf.VERBOSE2: print(f'Applying heavy penalty on source #{i+1} ({self.model_catalog[i].name})!')
                #     totalchisq = 1E30
                if conf.VERBOSE2: print(f'Source #{src["source_id"]} ({band}) with {self.model_catalog[i].name} has chisq={totalchisq:3.3f} | bic={self.solution_bic[i, j]:3.3f}')

                self.solution_chisq[i, j] = totalchisq

            if conf.VERBOSE2: print(f'Source #{src["source_id"]}: {self.solution_catalog[i].name} model at {self.solution_catalog[i].pos}')
            if conf.VERBOSE2: print(f'               Fluxes: {self.bands[0]}={self.solution_catalog[i].getBrightness().getFlux(self.bands[0])}') 
            if self.solution_catalog[i].name not in ('PointSource', 'SimpleGalaxy'):
                if self.solution_catalog[i].name == 'FixedCompositeGalaxy':
                    if conf.VERBOSE2: print(f'               {self.solution_catalog[i].shapeExp}')
                    if conf.VERBOSE2: print(f'               {self.solution_catalog[i].shapeDev}')
                else:
                    if conf.VERBOSE2: print(f'               {self.solution_catalog[i].shape}')

        if conf.PLOT:
            plot_detblob(self, fig, ax, level=self._level, sublevel=self._sublevel, final_opt=True)
            plt.close()

        # self.rows = np.zeros(len(self.solution_catalog))
        for idx, src in enumerate(self.solution_catalog):
            sid = self.bcatalog['source_id'][idx]
            # row = np.argwhere(self.brick.catalog['source_id'] == sid)[0][0]
            # self.rows[idx] = row
            # print(f'STASHING {sid} IN ROW {row}')
            self.get_catalog(idx, src)

        return self.status

    def decide_winners(self, use_bic=True):

        if use_bic:
            print()
            print('Using BIC to select best-fit model')
            self.decide_winners_bic()
        else:
            print()
            print('Using chisq to select best-fit model')
            raise RuntimeError('BUGGY - NOT WORKING NOW.')
            self.decide_winners_chisq()

    def decide_winners_bic(self):

        # holders - or else it's pure insanity.
        bic= self.bic[~self._solved]
        solution_catalog = self.solution_catalog[~self._solved]
        solved_bic = self.solved_bic[~self._solved]
        tr_catalogs = self.tr_catalogs[~self._solved]
        mids = self.mids[~self._solved]

        if self._level == 0:
            # Which have chi2(PS) < chi2(SG)?
            chmask = (bic[:, 0, 0] - bic[:, 0, 1] < conf.PS_SG_THRESH)
            if chmask.any():
                solution_catalog[chmask] = tr_catalogs[chmask, 0, 0].copy()
                solved_bic[chmask] = bic[chmask, 0, 0]
                mids[chmask] = 1

            # So chi2(SG) is min, try more models
            mids[~chmask] = 3

        if self._level == 1:
            # For which are they nearly equally good?
            movemask = (abs(bic[:, 1, 0] - bic[:, 1, 1]) < conf.EXP_DEV_THRESH)

            # Has Exp beaten SG?
            expmask = (bic[:, 1, 0] < bic[:, 0, 1])

            # Has Dev beaten SG?
            devmask = (bic[:, 1, 1] < bic[:, 0, 1])

            # Which better than SG but nearly equally good?
            nextmask = expmask & devmask & movemask

            # For which was SG better
            premask = ~expmask & ~devmask

            # If Exp beats Dev by a lot
            nexpmask = expmask & ~movemask & (bic[:, 1, 0]  <  bic[:, 1, 1])

             # If Dev beats Exp by a lot
            ndevmask = devmask & ~movemask & (bic[:, 1, 1] < bic[:, 1, 0])

            if nextmask.any():
                mids[nextmask] = 5

            if premask.any():
                solution_catalog[premask] = tr_catalogs[premask, 0, 1].copy()
                solved_bic[premask] = bic[premask, 0, 1]
                mids[premask] = 2

            if nexpmask.any():

                solution_catalog[nexpmask] = tr_catalogs[nexpmask, 1, 0].copy()
                solved_bic[nexpmask] = bic[nexpmask, 1, 0]
                mids[nexpmask] = 3

            if ndevmask.any():

                solution_catalog[ndevmask] = tr_catalogs[ndevmask, 1, 1].copy()
                solved_bic[ndevmask] = bic[ndevmask, 1, 1]
                mids[ndevmask] = 4

        if self._level == 2:
            # For which did Comp beat EXP and DEV?
            compmask = (bic[:, 2, 0] < bic[:, 1, 0]) &\
                       (bic[:, 2, 0] < bic[:, 1, 1])

            if compmask.any():
                solution_catalog[compmask] = tr_catalogs[compmask, 2, 0].copy()
                solved_bic[compmask] = bic[compmask, 2, 0]
                mids[compmask] = 5

            # where better as EXP or DEV
            compmask_or = (bic[:, 2, 0] > bic[:, 1, 0]) |\
                          (bic[:, 2, 0] > bic[:, 1, 1])
            if compmask_or.any():
                ch_exp = (bic[:, 1, 0] <= bic[:, 1, 1]) & compmask_or  # using '=' to break any ties, give to exp...

                if ch_exp.any():
                    solution_catalog[ch_exp] = tr_catalogs[ch_exp, 1, 0].copy()
                    solved_bic[ch_exp] = bic[ch_exp, 1, 0]
                    mids[ch_exp] = 3

                ch_dev = (bic[:, 1, 1] < bic[:, 1, 0]) & compmask_or

                if ch_dev.any():
                    solution_catalog[ch_dev] = tr_catalogs[ch_dev, 1, 1].copy()
                    solved_bic[ch_dev] = bic[ch_dev, 1, 1]
                    mids[ch_dev] = 4

        # hand back
        self.bic[~self._solved] = bic
        self.solution_catalog[~self._solved] = solution_catalog
        self.solved_bic[~self._solved] = solved_bic
        self.mids[~self._solved] = mids

    def decide_winners_chisq(self):

        # take the model_catalog and chisq and figure out what's what
        # Only look at unsolved models!
        
        if conf.USE_REDUCEDCHISQ:
            chisq_exp = 1.0
        else:
            chisq_exp = 0.0

        # holders - or else it's pure insanity.
        chisq = self.chisq[~self._solved]
        solution_catalog = self.solution_catalog[~self._solved]
        solved_chisq = self.solved_chisq[~self._solved]
        tr_catalogs = self.tr_catalogs[~self._solved]
        mids = self.mids[~self._solved]

        if self._level == 0:
            # Which have chi2(PS) < chi2(SG)?
            chmask = ((abs(chisq_exp - chisq[:, 0, 0]) - abs(chisq_exp - chisq[:, 0, 1])) < conf.PS_SG_THRESH)
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
            expmask = (abs(chisq_exp - chisq[:, 1, 0]) < abs(chisq_exp - chisq[:, 0, 1]))

            # Has Dev beaten SG?
            devmask = (abs(chisq_exp - chisq[:, 1, 1]) < abs(chisq_exp - chisq[:, 0, 1]))

            # Which better than SG but nearly equally good?
            nextmask = expmask & devmask & movemask

            # For which was SG better
            premask = ~expmask & ~devmask

            # If Exp beats Dev by a lot
            nexpmask = expmask & ~movemask & (abs(chisq_exp - chisq[:, 1, 0])  <  abs(chisq_exp - chisq[:, 1, 1]))

             # If Dev beats Exp by a lot
            ndevmask = devmask & ~movemask & (abs(chisq_exp - chisq[:, 1, 1]) < abs(chisq_exp - chisq[:, 1, 0]))

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
            compmask = (abs(chisq_exp - chisq[:, 2, 0]) < abs(chisq_exp - chisq[:, 1, 0])) &\
                       (abs(chisq_exp - chisq[:, 2, 0]) < abs(chisq_exp - chisq[:, 1, 1]))

            if compmask.any():
                solution_catalog[compmask] = tr_catalogs[compmask, 2, 0].copy()
                solved_chisq[compmask] = chisq[compmask, 2, 0]
                mids[compmask] = 5

            # where better as EXP or DEV
            if (~compmask).any():
                ch_exp = (abs(chisq_exp - chisq[:, 1, 0]) < abs(chisq_exp - chisq[:, 1, 1])) & ~compmask

                if ch_exp.any():
                    solution_catalog[ch_exp] = tr_catalogs[ch_exp, 1, 0].copy()
                    solved_chisq[ch_exp] = chisq[ch_exp, 1, 0]
                    mids[ch_exp] = 3

                ch_dev = (abs(chisq_exp - chisq[:, 1, 1]) < abs(chisq_exp - chisq[:, 1, 0])) & ~compmask

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

        
        if conf.VERBOSE2: 
            print()
            print(f'blob.optimize_tractor :: Starting optimization ({conf.TRACTOR_MAXSTEPS}, {conf.TRACTOR_CONTHRESH})')

        self.n_converge = 0
        dlnp_init = 'NaN'
        tstart = time()
        for i in range(conf.TRACTOR_MAXSTEPS):
            try:
                dlnp, X, alpha, var = tr.optimize(shared_params=False, variance=True)
                if i == 0:
                    dlnp_init = dlnp
            except:
                if conf.VERBOSE: print(f'WARNING - Optimization failed on step {i} for blob #{self.blob_id}')
                return False

            if dlnp < conf.TRACTOR_CONTHRESH:
                if conf.VERBOSE: print(f'blob.optimize_tractor :: Converged in {i} steps ({dlnp_init:2.2f} --> {dlnp:2.2f}) ({time() - tstart:3.3f}s)')
                self.n_converge = i
                break
                

        if var is None:
            if conf.VERBOSE: print(f'WARNING - VAR is NONE for blob #{self.blob_id}')
            return False

        cat = tr.getCatalog()
        var_catalog = cat.copy()
        var_catalog.setParams(var)
        # if (cat != 0).all():
        #     var_catalog = self.solution_catalog.copy()
        #     var_catalog = var_catalog.setParams(var)
        # else:
        #     var_catalog = self.model_catalog.copy()

        self.variance = var_catalog
        # counter = 0
        # for i, src in enumerate(np.arange(self.n_sources)):
        #     n_params = var_catalog[i].numberOfParams()
        #     myvar = var[counter: n_params + counter]
        #     # print(f'{i}) {var_catalog[i].name} has {n_params} params and {len(myvar)} variances: {myvar}')
        #     counter += n_params
        #     self.variance.append(myvar)

        if np.shape(self.tr.getChiImage(0)) != np.shape(self.segmap):
            if conf.VERBOSE: print(f'WARNING - Chimap and segmap are not the same shape for #{self.blob_id}')
            return False

        # expvar = np.sum([var_catalog[i].numberOfParams() for i in np.arange(len(var_catalog))])
        # # print(f'I have {len(var)} variance parameters for {self.n_sources} sources. I expected {expvar}.')
        # for i, mod in enumerate(var_catalog):
        #     totalchisq = np.sum((self.tr.getChiImage(0)[self.segmap == self.catalog[i]['source_id']])**2)

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
            image -= self.background_images[idx]

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
        if f'FLUX_APER_{band}_{image_type}' not in self.bcatalog.colnames:
            self.bcatalog.add_column(Column(length=len(self.bcatalog), dtype=float, shape=len(apertures), name=f'FLUX_APER_{band}_{image_type}'))
            self.bcatalog.add_column(Column(length=len(self.bcatalog), dtype=float, shape=len(apertures), name=f'FLUX_APER_{band}_{image_type}_err'))

        for idx, src in enumerate(self.solution_catalog):
            sid = self.bcatalog['source_id'][idx]
            row = np.argwhere(self.bcatalog['source_id'] == sid)[0][0]
            self.bcatalog[row][f'FLUX_APER_{band}_{image_type}'] = tuple(apflux[idx])
            self.bcatalog[row][f'FLUX_APER_{band}_{image_type}_err'] = tuple(apflux_err[idx])

    def sextract_phot(self, band, sub_background=False):
        # SHOULD WE STACK THE RESIDUALS? (No?)
        # SHOULD WE DO THIS ON THE MODELING IMAGE TOO? (I suppose we can already...!)
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
                sid = self.bcatalog['source_id'][idx]
                row = np.argwhere(self.catalog['source_id'] == sid)[0][0]
                self.bcatalog[row][f'{band}_n_residual_sources'] = True

            return catalog, segmap
        else:
            if conf.VERBOSE2: print('No objects found by SExtractor.')

    def forced_phot(self):

        # print('Starting forced photometry')
        # Update the incoming models
        if conf.VERBOSE2: print()
        if conf.VERBOSE2: print('Reviewing sources to be modelled.')
        for i, model in enumerate(self.model_catalog):
            # if model == -99:
            #     if conf.VERBOSE: print(f"FAILED -- Source #{self.bcatalog[i]['source_id']} does not have a valid model!")
            #     return False

            self.model_catalog[i].brightness = Fluxes(**dict(zip(self.bands, model.brightness[0] * np.ones(self.n_bands))))
            self.model_catalog[i].freezeAllBut('brightness')

            if conf.VERBOSE2: print(f"Source #{self.bcatalog[i]['source_id']}: {self.model_catalog[i].name} model at {self.model_catalog[i].pos}")
            if conf.VERBOSE2: print(f'               {self.model_catalog[i].brightness}')
            if self.bcatalog[i]['SOLMODEL'] == 'FixedCompositeGalaxy':
                if conf.VERBOSE2: print(f'               {self.model_catalog[i].shapeExp}')
                if conf.VERBOSE2: print(f'               {self.model_catalog[i].shapeDev}')
            elif self.bcatalog[i]['SOLMODEL'] in ('ExpGalaxy', 'DevGalaxy'):
                if conf.VERBOSE2: print(f'               {self.model_catalog[i].shape}')


        # Stash in Tractor
        self.tr = Tractor(self.timages, self.model_catalog)
        self.stage = 'Forced Photometry'

        if conf.PLOT:
            axlist = [plot_fblob(self, band=band) for band in self.bands]

        # Optimize
        status = self.optimize_tractor()

        if not status:
            return status

        # Chisq
        self.forced_variance = self.variance
        self.solution_chisq = np.zeros((self.n_sources, self.n_bands))
        self.solution_bic = np.zeros((self.n_sources, self.n_bands))
        self.solution_catalog = self.tr.getCatalog()
        self.solution_tractor = Tractor(self.timages, self.solution_catalog)
        self.solution_model_images = np.array([self.tr.getModelImage(i) for i in np.arange(self.n_bands)])
        self.solution_chi_images = np.array([self.tr.getChiImage(i) for i in np.arange(self.n_bands)])

        if conf.VERBOSE2: print(f'Resulting model parameters for blob #{self.blob_id}')
        for i, src in enumerate(self.bcatalog):
            if conf.VERBOSE2: print(f'Source #{src["source_id"]}: {self.solution_catalog[i].name} model at {self.solution_catalog[i].pos}')
            if self.solution_catalog[i].name not in ('PointSource', 'SimpleGalaxy'):
                if self.solution_catalog[i].name == 'FixedCompositeGalaxy':
                    if conf.VERBOSE2: print(f'               {self.solution_catalog[i].shapeExp}')
                    if conf.VERBOSE2: print(f'               {self.solution_catalog[i].shapeDev}')
                else:
                    if conf.VERBOSE2: print(f'               {self.solution_catalog[i].shape}')
            for j, band in enumerate(self.bands):
                totalchisq = np.sum((self.tr.getChiImage(j)[self.segmap == src['source_id']])**2)
                m_param = self.model_catalog[i].numberOfParams()
                n_data = np.sum(self.segmap == src['source_id'])
                self.solution_bic[i, j] = totalchisq + np.log(n_data) * m_param
                if conf.USE_REDUCEDCHISQ:
                    totalchisq /= (n_data - m_param)
                self.solution_chisq[i, j] = totalchisq
                # PENALTIES TBD
                # residual = self.images[0] - self.tr.getModelImage(0)
                # if np.median(residual[self.masks[0]]) < 0:
                #     if conf.VERBOSE2: print(f'Applying heavy penalty on source #{i+1} ({self.model_catalog[i].name})!')
                #     totalchisq = 1E30
                if conf.VERBOSE2: print(f'Source #{src["source_id"]} with {self.model_catalog[i].name} has chisq={totalchisq:3.3f}')
                if conf.VERBOSE2: print(f'               Fluxes: {self.bands[j]}={self.solution_catalog[i].getBrightness().getFlux(self.bands[j]):3.3f}') 
                if conf.VERBOSE2: print(f'               Chisq:  {self.bands[j]}={totalchisq:3.3f}')
                if conf.VERBOSE2: print(f'               BIC:  {self.bands[j]}={self.solution_bic[i, j]:3.3f}')


        # self.rows = np.zeros(len(self.solution_catalog))
        for idx, src in enumerate(self.solution_catalog):
            self.get_catalog(idx, src, multiband_only=True)


        if conf.PLOT:
            [plot_fblob(self, band, axlist[idx][0], axlist[idx][1], final_opt=True) for idx, band in enumerate(self.bands)]


        return status

    def get_catalog(self, row, src, multiband_only=False):

        if conf.VERBOSE:
            sid = self.bcatalog['source_id'][row]
            print()
            print(f'blob.get_catalog :: Writing output entires for #{sid}')

        # Add band fluxes, flux errors
        for i, band in enumerate(self.bands):
            valid_source = True # until decided False
            band = band.replace(' ', '_')
            if band == conf.MODELING_NICKNAME:
                zpt = conf.MODELING_ZPT
                flux_var = self.parameter_variance
                if self.bcatalog[row]['MAGERR_'+band]:
                    valid_source = False
            else:
                zpt = conf.MULTIBAND_ZPT[self._band2idx(band)]
                flux_var = self.forced_variance

            self.bcatalog[row]['MAG_'+band] = -2.5 * np.log10(src.getBrightness().getFlux(band)) + zpt
            self.bcatalog[row]['MAGERR_'+band] = 1.09 * np.sqrt(flux_var[row].brightness.getParams()[i]) / src.getBrightness().getFlux(band)
            self.bcatalog[row]['FLUX_'+band] = src.getBrightness().getFlux(band)
            self.bcatalog[row]['FLUXERR_'+band] = np.sqrt(flux_var[row].brightness.getParams()[i])
            self.bcatalog[row]['CHISQ_'+band] = self.solution_chisq[row, i]
            self.bcatalog[row]['BIC_'+band] = self.solution_bic[row, i]

            # Just do the positions again - more straightforward to do it here than in interface.py
            self.bcatalog[row]['X_MODEL'] = src.pos[0] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER + 1
            self.bcatalog[row]['Y_MODEL'] = src.pos[1] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER + 1

            if conf.VERBOSE2:
                print()
                mag, magerr = self.bcatalog[row]['MAG_'+band], self.bcatalog[row]['MAGERR_'+band]
                flux, fluxerr = self.bcatalog[row]['FLUX_'+band], self.bcatalog[row]['FLUXERR_'+band]
                chisq, bic = self.bcatalog[row]['CHISQ_'+band], self.bcatalog[row]['BIC_'+band]
                print(f'     MAG_{band} = {mag:3.3f} +/- {magerr:3.3f}')
                print(f'     FLUX_{band} = {flux:3.3f} +/- {fluxerr:3.3f}')
                print(f'     CHISQ_{band} = {chisq:3.3f} | BIC_{band} = {bic:3.3f}')

        if not multiband_only:
            # Position information
            self.bcatalog[row]['x'] = self.bcatalog[row]['x'] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER + 1
            self.bcatalog[row]['y'] = self.bcatalog[row]['y'] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER + 1
            self.bcatalog[row]['X_MODEL'] = src.pos[0] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER + 1
            self.bcatalog[row]['Y_MODEL'] = src.pos[1] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER + 1
            if conf.VERBOSE2:
                print()
                print(f"     xy = {self.bcatalog[row]['x']}, {self.bcatalog[row]['y']}")
                print(f"     src.pos = {src.pos[0]}, {src.pos[1]}")
                print(f"     subvector = {self.subvector[1]}, {self.subvector[0]}")
                print(f"     mosaic_origin = {self.mosaic_origin[1]}, {self.mosaic_origin[0]}")
                print(f"     brick_buffer = {conf.BRICK_BUFFER}")
                print(f"     XY_MODEL = {self.bcatalog[row]['X_MODEL']}, {self.bcatalog[row]['Y_MODEL']}")
            self.bcatalog[row]['XERR_MODEL'] = np.sqrt(self.position_variance[row].pos.getParams()[0])
            self.bcatalog[row]['YERR_MODEL'] = np.sqrt(self.position_variance[row].pos.getParams()[1])
            if self.wcs is not None:
                skyc = self.brick_wcs.all_pix2world(self.bcatalog[row]['X_MODEL'] - self.mosaic_origin[1], self.bcatalog[row]['Y_MODEL'] - self.mosaic_origin[0], 0)
                self.bcatalog[row]['RA'] = skyc[0]
                self.bcatalog[row]['DEC'] = skyc[1]

            # Model Parameters
            self.bcatalog[row]['SOLMODEL'] = src.name
            self.bcatalog[row]['VALID_SOURCE'] = valid_source
            self.bcatalog[row]['N_CONVERGE'] = self.n_converge

            if src.name in ('SimpleGalaxy', 'ExpGalaxy', 'DevGalaxy'):
                self.bcatalog[row]['REFF'] = src.shape.re
                self.bcatalog[row]['REFF_ERR'] = np.sqrt(self.parameter_variance[row].shape.getParams()[0])
                self.bcatalog[row]['AB'] = (src.shape.e + 1) / (1 - src.shape.e)
                if src.shape.e >= 1:
                    self.bcatalog[row]['VALID_SOURCE'] = False
                self.bcatalog[row]['AB_ERR'] = np.sqrt(self.parameter_variance[row].shape.getParams()[1])
                self.bcatalog[row]['THETA'] = np.rad2deg(src.shape.theta)
                self.bcatalog[row]['THETA_ERR'] = np.sqrt(self.parameter_variance[row].shape.getParams()[2])

            elif src.name == 'FixedCompositeGalaxy':
                self.bcatalog[row]['FRACDEV'] = src.fracDev.getValue()
                self.bcatalog[row]['EXP_REFF'] = src.shapeExp.re
                self.bcatalog[row]['EXP_REFF_ERR'] = np.sqrt(self.parameter_variance[row].shapeExp.getParams()[0])
                self.bcatalog[row]['EXP_AB'] = (src.shapeExp.e + 1) / (1 - src.shapeExp.e)
                self.bcatalog[row]['EXP_AB_ERR'] = np.sqrt(self.parameter_variance[row].shapeExp.getParams()[1])
                self.bcatalog[row]['EXP_THETA'] = np.rad2deg(src.shapeExp.theta)
                self.bcatalog[row]['EXP_THETA_ERR'] = np.sqrt(self.parameter_variance[row].shapeExp.getParams()[2])
                self.bcatalog[row]['DEV_REFF'] = src.shapeDev.re
                self.bcatalog[row]['DEV_REFF_ERR'] = np.sqrt(self.parameter_variance[row].shapeDev.getParams()[0])
                self.bcatalog[row]['DEV_AB'] = (src.shapeDev.e + 1) / (1 - src.shapeDev.e)
                self.bcatalog[row]['DEV_AB_ERR'] = np.sqrt(self.parameter_variance[row].shapeDev.getParams()[1])
                self.bcatalog[row]['DEV_THETA'] = np.rad2deg(src.shapeDev.theta)
                self.bcatalog[row]['DEV_THETA_ERR'] = np.sqrt(self.parameter_variance[row].shapeDev.getParams()[2])
                # self.bcatalog[row]['reff_err'] = np.sqrt(self.parameter_variance[row][0])
                # self.bcatalog[row]['ab_err'] = np.sqrt(self.parameter_variance[row][1])
                # self.bcatalog[row]['phi_err'] = np.sqrt(self.parameter_variance[row][2])
