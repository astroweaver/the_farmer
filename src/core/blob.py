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
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import zoom
from scipy import stats

from tractor import NCircularGaussianPSF, PixelizedPSF, PixelizedPsfEx, Image, Tractor, FluxesPhotoCal, NullWCS, ConstantSky, EllipseE, EllipseESoft, Fluxes, PixPos
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev
from tractor.pointsource import PointSource
from tractor.psf import HybridPixelizedPSF
import time
import photutils
import sep
from matplotlib.colors import LogNorm
import pathos

from .subimage import Subimage
from .utils import SimpleGalaxy, create_circular_mask
from .visualization import plot_detblob, plot_fblob, plot_psf, plot_modprofile, plot_mask, plot_xsection, plot_srcprofile, plot_iterblob
import config as conf

import logging

class Blob(Subimage):
    """TODO: docstring"""

    def __init__(self, brick, blob_id):
        """TODO: docstring"""

        self.logger = logging.getLogger(f'farmer.blob.{blob_id}')
        self.rejected = False
        # fh = logging.FileHandler(f'farmer_B{blob_id}.log')
        # fh.setLevel(logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL))
        # formatter = logging.Formatter('[%(asctime)s] %(name)s :: %(levelname)s - %(message)s', '%H:%M:%S')
        # fh.setFormatter(formatter)
       
        # self.logger = pathos.logger(level=logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL), handler=fh)

        blobmask = np.array(brick.blobmap == blob_id, bool)
        mask_frac = blobmask.sum() / blobmask.size
        if (mask_frac > conf.SPARSE_THRESH) & (blobmask.size > conf.SPARSE_SIZE):
            self.logger.warning('Blob is rejected as mask is sparse - likely an artefact issue.')
            self.rejected = True

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

        
        self._is_itemblob = False

        # Make cutout
        blob_comps = brick._get_subimage(xlo, ylo, w, h, buffer=conf.BLOB_BUFFER)
        # FIXME: too many return values
        self.images, self.weights, self.masks, self.psfmodels, self.bands, self.wcs, self.subvector, self.slicepix, self.slice = blob_comps

        self.masks[self.slicepix] = np.logical_not(blobmask[self.slice], dtype=bool)
        self.segmap = brick.segmap[self.slice]
        self._level = 0
        self._sublevel = 0

        # coordinates
        self.blob_center = (xlo + h/2., ylo + w/2.)
        center_x = self.blob_center[1] - self.subvector[0] + self.mosaic_origin[1] - self.mosaic_origin[0] + 1
        center_y = self.blob_center[0] - self.subvector[1] + self.mosaic_origin[0] - self.mosaic_origin[1] + 1
        ra, dec = self.wcs.all_pix2world(center_x, center_y, 0)
        self.blob_coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)

        # Clean
        blob_sourcemask = np.in1d(brick.catalog['source_id'], blob_sources)
        self.bcatalog = brick.catalog[blob_sourcemask].copy() # working copy
        if brick.catalog['VALID_SOURCE_MODELING'].any(): # Then modeling completed (???) and we are good to check this stuff.
            valid_arr = self.bcatalog['VALID_SOURCE_MODELING']
            if len(valid_arr) > 1:
                if (valid_arr == False).all():
                    self.logger.warning('Blob is rejected as no sources are valid!')
                    self.rejected = True
            else:
                if valid_arr == False:
                    self.logger.warning('Blob is rejected as no sources are valid!')
                    self.rejected = True

        # print(self.bcatalog['x', 'y'])
        self.bcatalog['x'] -= self.subvector[1]
        self.bcatalog['y'] -= self.subvector[0]

        # print(self.bcatalog['x', 'y'])
        # print(self.subvector)
        # print(self.mosaic_origin)
        self.n_sources = len(self.bcatalog)

        self.shared_params = brick.shared_params
        self.multiband_model = brick.shared_params # HACK

        self.mids = np.ones(self.n_sources, dtype=int)
        self.model_catalog = np.zeros(self.n_sources, dtype=object)
        self.solution_catalog = np.zeros(self.n_sources, dtype=object)
        self.solved_chisq = np.zeros(self.n_sources)
        self.solved_bic = np.zeros(self.n_sources)
        self.solution_chisq = np.zeros(self.n_sources)
        self.tr_catalogs = np.zeros((self.n_sources, 3, 2), dtype=object)
        self.chisq = np.zeros((self.n_sources, 3, 2))
        self.rchisq = np.zeros((self.n_sources, 3, 2))
        self.bic = np.zeros((self.n_sources, 3, 2))
        self.noise = np.zeros((self.n_sources, self.n_bands))
        # self.norm = np.zeros((self.n_sources, self.n_bands))
        self.chi_mu = np.zeros((self.n_sources, self.n_bands))
        self.chi_sig = np.zeros((self.n_sources, self.n_bands))
        self.k2 = np.zeros((self.n_sources, self.n_bands))
        # self.position_variance = np.zeros((self.n_sources, 2))
        # self.parameter_variance = np.zeros((self.n_sources, 3))
        # self.forced_variance = np.zeros((self.n_sources, self.n_bands))
        self.solution_tractor = None
        self.psfimg = {}

        self.residual_catalog = np.zeros((self.n_bands), dtype=object)
        self.residual_segmap = np.zeros_like(self.segmap)
        self.n_residual_sources = np.zeros(self.n_bands, dtype=int)

        self.minsep = dict.fromkeys(conf.PRFMAP_PSF)

        del brick

    def stage_images(self):
        """ Collect image information (img, wgt, mask, psf, wcs) to build a Tractor Image for the blob"""

        self.logger.debug('Staging images...')

        timages = np.zeros(self.n_bands, dtype=object)

        if conf.SUBTRACT_BACKGROUND:
            self.subtract_background(flat=conf.USE_FLAT)
            self.logger.debug(f'Subtracted background (flat={conf.USE_FLAT})')

        # TODO: try to simplify this. particularly the zip...
        for i, (image, weight, mask, psf, band) in enumerate(zip(self.images, self.weights, self.masks, self.psfmodels, self.bands)):
            self.logger.debug(f'Staging image for {band}')
            band_strip = band
            if (band_strip != conf.MODELING_NICKNAME) & band_strip.startswith(conf.MODELING_NICKNAME):
                band_strip = band[len(conf.MODELING_NICKNAME)+1:]
                self.logger.debug(f'Striped name for image: {band_strip}')
            tweight = weight.copy()
            bcmask = None
            if conf.APPLY_SEGMASK:
                tweight[mask] = 0

            remove_background_psf = False
            if band_strip in conf.RMBACK_PSF:
                remove_background_psf = True

            psfplotband = band

            if (band_strip in conf.CONSTANT_PSF) & (psf is not None):
                psfmodel = psf.constantPsfAt(conf.MOSAIC_WIDTH/2., conf.MOSAIC_HEIGHT/2.) # if not spatially varying psfex model, this won't matter.
                pw, ph = np.shape(psfmodel.img)
                if remove_background_psf & (not conf.FORCE_GAUSSIAN_PSF):
                    self.logger.debug('Removing PSF background.')
                    cmask = create_circular_mask(pw, ph, radius=conf.PSF_MASKRAD / conf.PIXEL_SCALE)
                    bcmask = ~cmask.astype(bool) & (psfmodel.img > 0)
                    psfmodel.img -= np.nanmax(psfmodel.img[bcmask])
                    # psfmodel.img[np.isnan(psfmodel.img)] = 0
                    # psfmodel.img -= np.nanmax(psfmodel.img[bcmask])
                    psfmodel.img[(psfmodel.img < 0) | np.isnan(psfmodel.img)] = 0

                if conf.PSF_RADIUS > 0:
                    self.logger.debug(f'Clipping PSF ({conf.PSF_RADIUS}px radius)')
                    psfmodel.img = psfmodel.img[int(pw/2.-conf.PSF_RADIUS):int(pw/2+conf.PSF_RADIUS), int(ph/2.-conf.PSF_RADIUS):int(ph/2+conf.PSF_RADIUS)]
                    self.logger.debug(f'New shape: {np.shape(psfmodel.img)}')

                if conf.NORMALIZE_PSF & (not conf.FORCE_GAUSSIAN_PSF):
                    norm = psfmodel.img.sum()
                    self.logger.debug(f'Normalizing PSF (sum = {norm:4.4f})')
                    psfmodel.img /= norm # HACK -- force normalization to 1
                self.logger.debug('Adopting constant PSF.')

                if conf.USE_MOG_PSF:
                    self.logger.debug('Making a Gaussian Mixture PSF')
                    psfmodel = HybridPixelizedPSF(pix=psfmodel, N=10).gauss
            
            elif (band_strip in conf.PSFGRID) & (psf is not None):
                self.logger.debug('Adopting a GRIDPSF from file.')
                # find nearest prf to blob center
                psftab_coords, psftab_fname = self.psfmodels[i]

                minsep_idx, minsep, __ = self.blob_coords.match_to_catalog_sky(psftab_coords)
                psf_fname = psftab_fname[minsep_idx]
                # print(self.blob_coords[minsep_idx])
                self.logger.debug(f'Nearest PSF sample: {psf_fname} ({minsep[0].to(u.arcsec).value:2.2f}")')

                if minsep > conf.PSFGRID_MAXSEP*u.arcsec:
                    self.logger.error(f'Separation ({minsep.to(u.arcsec)}) exceeds maximum {conf.PSFGRID_MAXSEP}!')
                    return False

                self.minsep[band_strip] = minsep # record it, and add it to the output catalog!

                # open id file
                path_psffile = os.path.join(conf.PSFGRID_OUT_DIR, f'{band_strip}_OUT/{psf_fname}.psf')
                if not os.path.exists(path_psffile):
                    self.logger.error(f'PSF file has not been found! ({path_psffile}')
                    return False
                self.logger.debug(f'Adopting GRID PSF: {psf_fname}')
                
                # # Do I need to resample?
                # if  (conf.PRFMAP_PIXEL_SCALE_ORIG > 0) & (conf.PRFMAP_PIXEL_SCALE_ORIG is not None):
                    
                #     factor = conf.PRFMAP_PIXEL_SCALE_ORIG / conf.PIXEL_SCALE
                #     self.logger.debug(f'Resampling PRF with zoom factor: {factor:2.2f}')
                #     img = zoom(img, factor)
                #     if np.shape(img)[0]%2 == 0:
                #         shape_factor = np.shape(img)[0] / (np.shape(img)[0] + 1)
                #         img = zoom(img, shape_factor)
                #     self.logger.debug(f'Final PRF size: {np.shape(img)}')
                # blob_centerx = self.blob_center[0] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER + 1
                # blob_centery = self.blob_center[1] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER + 1
                # psfmodel = psf.) # init at blob center, may need to swap!
                psfmodel = PixelizedPsfEx(fn=path_psffile)
                pw, ph = np.shape(psfmodel.img)

                psfplotband = psf_fname

                if remove_background_psf & (not conf.FORCE_GAUSSIAN_PSF):
                    
                    cmask = create_circular_mask(pw, ph, radius=conf.PSF_MASKRAD / conf.PIXEL_SCALE)
                    bcmask = ~cmask.astype(bool) & (psfmodel.img > 0)
                    psf_bkg = np.nanmax(psfmodel.img[bcmask])
                    psfmodel.img -= psf_bkg
                    self.logger.debug(f'Removing PSF background. {psf_bkg:e}')
                    # psfmodel.img[np.isnan(psfmodel.img)] = 0
                    # psfmodel.img -= np.nanmax(psfmodel.img[bcmask])
                    psfmodel.img[(psfmodel.img < 0) | np.isnan(psfmodel.img)] = 0

                if conf.PSF_RADIUS > 0:
                    self.logger.debug(f'Clipping PRF ({conf.PSF_RADIUS}px radius)')
                    psfmodel.img = psfmodel.img[int(pw/2.-conf.PSF_RADIUS):int(pw/2+conf.PSF_RADIUS), int(ph/2.-conf.PSF_RADIUS):int(ph/2+conf.PSF_RADIUS)]
                    self.logger.debug(f'New shape: {np.shape(psfmodel.img)}')

                if conf.NORMALIZE_PSF & (not conf.FORCE_GAUSSIAN_PSF):
                    norm = psfmodel.img.sum()
                    self.logger.debug(f'Normalizing PSF (sum = {norm:4.4f})')
                    psfmodel.img /= norm # HACK -- force normalization to 1       
                    self.logger.debug(f'PSF has been normalized. (sum = {psfmodel.img.sum():4.4f})') 

            elif (band_strip in conf.PRFMAP_PSF) & (psf is not None):
                self.logger.debug('Adopting a PRF from file.')
                # find nearest prf to blob center
                prftab_coords, prftab_idx = self.psfmodels[i]

                if conf.USE_BLOB_IDGRID:
                    prf_idx = self.blob_id
                else:
                    minsep_idx, minsep, __ = self.blob_coords.match_to_catalog_sky(prftab_coords)
                    prf_idx = prftab_idx[minsep_idx]
                    self.logger.debug(f'Nearest PRF sample: {prf_idx} ({minsep[0].to(u.arcsec).value:2.2f}")')

                    if minsep > conf.PRFMAP_MAXSEP*u.arcsec:
                        self.logger.error(f'Separation ({minsep.to(u.arcsec)}) exceeds maximum {conf.PRFMAP_MAXSEP}!')
                        return False

                    self.minsep[band] = minsep # record it, and add it to the output catalog!

                # open id file
                pad_prf_idx = ((6 - len(str(prf_idx))) * "0") + str(prf_idx)
                path_prffile = os.path.join(conf.PRFMAP_DIR[band_strip], f'{conf.PRFMAP_FILENAME}{pad_prf_idx}.fits')
                if not os.path.exists(path_prffile):
                    self.logger.error(f'PRF file has not been found! ({path_prffile}')
                    return False
                hdul = fits.open(path_prffile)
                from scipy.ndimage.interpolation import rotate
                # img = rotate(hdul[0].data, 270)
                img = hdul[0].data
                # img = 1E-31 * np.ones_like(img)
                # img[50:-50, 50:-50] = hdul[0].data[50:-50, 50:-50]
                assert(img.shape[0] == img.shape[1]) # am I square!?
                self.logger.debug(f'PRF size: {np.shape(img)}')
                
                # Do I need to resample?
                if  (conf.PRFMAP_PIXEL_SCALE_ORIG > 0) & (conf.PRFMAP_PIXEL_SCALE_ORIG is not None):
                    
                    factor = conf.PRFMAP_PIXEL_SCALE_ORIG / conf.PIXEL_SCALE
                    self.logger.debug(f'Resampling PRF with zoom factor: {factor:2.2f}')
                    img = zoom(img, factor)
                    if np.shape(img)[0]%2 == 0:
                        shape_factor = np.shape(img)[0] / (np.shape(img)[0] + 1)
                        img = zoom(img, shape_factor)
                    self.logger.debug(f'Final PRF size: {np.shape(img)}')

                psfmodel = PixelizedPSF(img)
                pw, ph = np.shape(psfmodel.img)

                if (conf.PRFMAP_MASKRAD > 0) & (not conf.FORCE_GAUSSIAN_PSF):
                    self.logger.debug('Clipping outskirts of PRF.')
                    cmask = create_circular_mask(pw, ph, radius=conf.PRFMAP_MASKRAD / conf.PIXEL_SCALE)
                    bcmask = ~cmask.astype(bool) & (psfmodel.img > 0)
                    psfmodel.img[bcmask] = 0
                    # psfmodel.img -= np.nanmax(psfmodel.img[bcmask])
                    psfmodel.img[(psfmodel.img < 0) | np.isnan(psfmodel.img)] = 0

                if conf.PSF_RADIUS > 0:
                    self.logger.debug(f'Clipping PRF ({conf.PSF_RADIUS}px radius)')
                    psfmodel.img = psfmodel.img[int(pw/2.-conf.PSF_RADIUS):int(pw/2+conf.PSF_RADIUS), int(ph/2.-conf.PSF_RADIUS):int(ph/2+conf.PSF_RADIUS)]
                    self.logger.debug(f'New shape: {np.shape(psfmodel.img)}')

                if conf.NORMALIZE_PSF & (not conf.FORCE_GAUSSIAN_PSF):
                    norm = psfmodel.img.sum()
                    self.logger.debug(f'Normalizing PRF (sum = {norm:4.4f})')
                    psfmodel.img /= norm # HACK -- force normalization to 1       
                    self.logger.debug(f'PRF has been normalized. (sum = {psfmodel.img.sum():4.4f})')         

            elif (psf is not None):
                blob_centerx = self.blob_center[0] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER + 1
                blob_centery = self.blob_center[1] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER + 1
                psfmodel = psf.constantPsfAt(blob_centerx, blob_centery) # init at blob center, may need to swap!
                if remove_background_psf & (not conf.FORCE_GAUSSIAN_PSF):
                    pw, ph = np.shape(psfmodel.img)
                    cmask = create_circular_mask(pw, ph, radius=conf.PSF_MASKRAD / conf.PIXEL_SCALE)
                    bcmask = ~cmask.astype(bool) & (psfmodel.img > 0)
                    psfmodel.img -= np.nanmax(psfmodel.img[bcmask])
                    # psfmodel.img[np.isnan(psfmodel.img)] = 0
                    # psfmodel.img -= np.nanmax(psfmodel.img[bcmask])
                    psfmodel.img[(psfmodel.img < 0) | np.isnan(psfmodel.img)] = 0
                if conf.PSF_RADIUS > 0:
                    self.logger.debug(f'Clipping PRF ({conf.PSF_RADIUS}px radius)')
                    psfmodel.img = psfmodel.img[int(pw/2.-conf.PSF_RADIUS):int(pw/2+conf.PSF_RADIUS), int(ph/2.-conf.PSF_RADIUS):int(ph/2+conf.PSF_RADIUS)]
                    self.logger.debug(f'New shape: {np.shape(psfmodel.img)}')
                    
                if conf.NORMALIZE_PSF & (not conf.FORCE_GAUSSIAN_PSF):
                    norm = psfmodel.img.sum()
                    self.logger.debug(f'Normalizing PSF (sum = {norm:4.4f})')
                    psfmodel.img /= norm # HACK -- force normalization to 1
                self.logger.debug(f'Adopting varying PSF constant at ({blob_centerx}, {blob_centery}).')
            

            elif (psf is None):
                if conf.USE_GAUSSIAN_PSF:
                    psfmodel = NCircularGaussianPSF([conf.PSF_SIGMA / conf.PIXEL_SCALE], [1,])
                    self.logger.debug(f'Adopting {conf.PSF_SIGMA}" Gaussian PSF.')
                else:
                    raise ValueError(f'WARNING - No PSF model found for {band}!')

            if (psf is not None) & (~conf.USE_MOG_PSF):
                try:
                    psfimg = psfmodel.img
                except:
                    psfimg = psfmodel.getImage(0, 0)
            else:
                psfimg = psfmodel.getPointSourcePatch(0, 0).getImage()

            self.psfimg[band] = psfimg
            
            if (conf.PLOT > 1):
                plot_psf(psfimg, psfplotband, show_gaussian=False)

            psfmodel.img = psfmodel.img.astype('float32') # This may be redundant, but it's super important!
                    
            self.logger.debug('Making image...')
            timages[i] = Image(data=image,
                            invvar=tweight,
                            psf=psfmodel,
                            wcs=NullWCS(),
                            photocal=FluxesPhotoCal(band),
                            sky=ConstantSky(0))
            # modelminval = 0.1 * np.nanmedian(1/np.sqrt(tweight[tweight>0]))
            # timages[i].modelMinval = 1. #modelminval
            # print(f'Setting minval to {modelminval} in img pixel units.')

        self.timages = timages
        return True

    def stage_models(self):
        """ Build the Tractor Model catalog for the blob """
        # Trackers
        self.logger.debug(f'Loading models for blob #{self.blob_id}')

        for i, (mid, src) in enumerate(zip(self.mids, self.bcatalog)):

            self.logger.debug(f'Source #{src["source_id"]}')
            self.logger.debug(f"               x, y: {src['x']:3.3f}, {src['y']:3.3f}")
            self.logger.debug(f"               flux: {src['flux']:3.3f}")
            self.logger.debug(f"               cflux: {src['cflux']:3.3f}")
            self.logger.debug(f"               a, b: {src['a']:3.3f}, {src['b']:3.3f}") 
            self.logger.debug(f"               theta: {src['theta']:3.3f}")

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
            
            if conf.USE_SEP_INITIAL_FLUX:
                flux = Fluxes(**dict(zip(self.bands, src['flux'] * np.ones(len(self.bands)))))

            else:
                try:
                    qflux = np.zeros(len(self.bands))
                    src_seg = self.segmap==src['source_id']
                    for j, (img, psf) in enumerate(zip(self.images, self.psfmodels)):
                        max_img = np.nanmax(img * src_seg)
                        max_psf = np.nanmax(psf.img)
                        qflux[j] = max_img / max_psf
                    flux = Fluxes(**dict(zip(self.bands, qflux)))
                except:
                    self.logger.warning('Failed to estimate inital flux a priori. Falling back on SEP...')
                    flux = Fluxes(**dict(zip(self.bands, src['flux'] * np.ones(len(self.bands)))))
                

            #shape = GalaxyShape(src['a'], src['b'] / src['a'], src['theta'])
            pa = 90 + np.rad2deg(src['theta'])
            shape = EllipseESoft.fromRAbPhi(src['a'], src['b'] / src['a'], pa)
            # shape = EllipseESoft.fromRAbPhi(3.0, 0.6, 38)

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
                self.logger.debug(f'Position parameter frozen at {position}')
            elif conf.USE_POSITION_PRIOR:
                self.logger.info(f'Setting position prior. X = {src["x"]:2.2f}+/-{conf.POSITION_PRIOR_SIG}; Y = {src["y"]:2.2f}+/-{conf.POSITION_PRIOR_SIG}')
                self.model_catalog[i].pos.addGaussianPrior('x', src['x'], conf.POSITION_PRIOR_SIG)
                self.model_catalog[i].pos.addGaussianPrior('y', src['y'], conf.POSITION_PRIOR_SIG)

            self.logger.debug(f'Source #{src["source_id"]}: {self.model_catalog[i].name} model at {position}')
            self.logger.debug(f'               {flux}') 
            if mid not in (1,2):
                self.logger.debug(f'               {shape}')

    def optimize_tractor(self, tr=None):
        """ Iterate and optimize given a Tractor Image and Model catalog. Determines uncertainties. """

        if tr is None:
            tr = self.tr

        # if conf.USE_CERES:
        #     from tractor.ceres_optimizer import CeresOptimizer
        #     tr.optimizer = CeresOptimizer()

        tr.freezeParams('images')  

        if conf.USE_CERES:
            raise RuntimeError('CERES NOT IMPLEMENTED!')
            # self.logger.debug(f'Starting ceres optimization ({conf.TRACTOR_MAXSTEPS}, {conf.TRACTOR_CONTHRESH})') 

            # self.n_converge = 0
            # tstart = time.time()
            # R = tr.optimize_forced_photometry(
            #     minsb=0.1,
            #     shared_params=False, wantims=False, fitstats=True, variance=True,
            #     use_ceres=True, BW=8,BH=8)
            # self.logger.info(f'Blob #{self.blob_id} converged ({time.time() - tstart:3.3f}s)')

        else:

            self.logger.debug(f'Starting lsqr optimization ({conf.TRACTOR_MAXSTEPS}, {conf.TRACTOR_CONTHRESH}) (shared_params={self.shared_params})') 

            self.n_converge = 0
            dlnp_init = 'NaN'
            tstart = time.time()

            # grab inital positions
            # cat = tr.getCatalog()
            # x_orig = np.zeros(self.n_sources)
            # y_orig = np.zeros(self.n_sources)
            # for i, src in enumerate(cat):
            #     x_orig[i] = src.pos[0]
            #     y_orig[i] = src.pos[1]


            if conf.PLOT > 2:
                # [print(m.getBrightness()) for m in tr.getCatalog()]
                plot_iterblob(self, tr, iteration=0, bands=self.bands)

            for i in range(conf.TRACTOR_MAXSTEPS):

                # for mod in self.tr.getCatalog():
                #     print(mod)
                #     print(mod.pos.getGaussianPriors())
                #     print(mod.getThawedParams())

                # if i == 0:
                #     pos = PixPos(13, 30)
                #     pos.addGaussianPrior('x', 13, 2)
                #     pos.addGaussianPrior('y', 30, 2)
                #     tr.addSource(PointSource(pos, Fluxes(**dict(zip(self.bands, 0.1 * np.ones(len(self.bands)))))))

                if conf.TRY_OPTIMIZATION:
                    try:
                        # print()
                        # print(i)
                        # [print(m.getPosition()) for m in tr.getCatalog()]
                        # [print(m.getBrightness().getFlux(self.bands[0])) for m in tr.getCatalog()]
                        # [print(m.getThawedParams()) for m in tr.getCatalog()]
                        # [print(m.getFrozenParams()) for m in tr.getCatalog()]
                        # print('...')

                        # dlnp, X, alpha, var = tr.optimize(shared_params=self.shared_params, damp=conf.DAMPING, variance=True, priors=conf.USE_POSITION_PRIOR)
                        dlnp, X, alpha, var = tr.optimize(shared_params=self.shared_params, damp=conf.DAMPING, 
                                                    variance=True, priors=conf.USE_POSITION_PRIOR)

                        self.logger.debug(f'    {i+1}) dlnp = {dlnp}')
                        # print(dlnp)
                        # print(X)
                        # print(alpha)
                        # print(var)
                        # [print(m.getShape()) for m in tr.getCatalog()]
                        # [print(m.getBrightness().getFlux(self.bands[0])) for m in tr.getCatalog()]
                        # [print(m.getThawedParams()) for m in tr.getCatalog()]
                        # [print(m.getFrozenParams()) for m in tr.getCatalog()]
                        # print()
                        


                        if i == 0:
                            dlnp_init = dlnp
                    except:
                        self.logger.warning(f'WARNING - Optimization failed on step {i} for blob #{self.blob_id}')
                        return False
                else:
                    dlnp, X, alpha, var = tr.optimize(shared_params=self.shared_params, damp=conf.DAMPING, 
                                                    variance=True, priors=conf.USE_POSITION_PRIOR)
                    self.logger.debug(f'    {i+1}) dlnp = {dlnp}')
                    if i == 0:
                        dlnp_init = dlnp

                # try:  # HACK -- this sometimes fails!!!
                #     cat = tr.getCatalog()
                #     for k, src in enumerate(cat):
                #         sid = self.bcatalog['source_id'][k]
                #         for j, band in enumerate(self.bands):
                #             p = np.sum(cat[k].getUnitFluxModelPatches(tr.getImage(j))[0].patch)
                #             # m = np.max(cat[k].getUnitFluxModelPatches(tr.getImage(j))[0].patch)
                #             # self.logger.debug(f'Max of PSF convolved patch for {sid} in {band}: {m:4.4f}')
                #             # self.logger.debug(f'Max of PSF: {np.max(tr.getImage(j).getPsf().img):4.4f}')
                #             # self.logger.debug(f'Max of Model: {np.max(tr.getModelImage(j)):4.4f}')
                #             # self.logger.debug(f'Sum of PSF convolved patch for {sid} in {band}: {p:4.4f}')
                #             # self.logger.debug(f'Sum of PSF: {np.sum(tr.getImage(j).getPsf().img):4.4f}')
                #             self.norm[k, j] = p
                #             if 1. - p > conf.NORMALIZATION_THRESH:
                #                 self.logger.critical(f'The final model for {sid} in {band} is NOT normalized within threshold ({conf.NORMALIZATION_THRESH})')
                                

                # except:
                #     return False

                if conf.PLOT > 2:
                    plot_iterblob(self, tr, iteration=i+1, bands=self.bands)

                if dlnp < conf.TRACTOR_CONTHRESH:
                    self.logger.info(f'Blob #{self.blob_id} converged in {i+1} steps ({dlnp_init:2.2f} --> {dlnp:2.2f}) ({time.time() - tstart:3.3f}s)')
                    self.n_converge = i
                    break

                if conf.CORRAL_SOURCES:
                    # check if any sources have left their segment
                    cat = tr.getCatalog()[:len(self.bcatalog)]
                    trip = False
                    for idx, src in enumerate(cat):
                        sid = self.bcatalog['source_id'][idx]
                        xp, yp = src.pos[0], src.pos[1]
                        xp0, yp0 = self.bcatalog['x'][idx], self.bcatalog['y'][idx]
                        srcseg = self.segmap == sid
                        maxy, maxx = np.shape(self.segmap)

                        if (xp > maxx) | (xp < 0) | (yp < 0) | (yp > maxy):
                            self.logger.warning(f'Source {sid} has escaped the blob!')
                            trip = True

                            # gpriors = src.getLogPrior()
                            # print('Log(Prior):', gpriors)
                            src.pos.setParams([xp0, yp0])
                            # self.logger.info(f'Setting position prior. X = {xp0:2.2f}+/-{POSITION_PRIOR_SIG}; Y = {yp0:2.2f}+/-{1.}')
                            # src.pos.addGaussianPrior('x', xp0, 1.0)
                            # src.pos.addGaussianPrior('y', yp0, 1.0)


                        # elif ~srcseg[int(yp), int(xp)]:

                        #     # fig, ax = plt.subplots()
                        #     # ax.imshow(srcseg)
                        #     # ax.scatter(xp, yp)
                        #     # plt.savefig(os.path.join(conf.PLOT_DIR,f'tr_{i}_{idx}.pdf'))
                        #     # self.logger.debug('MAKING CORRAL IMAGE!')

                        #     # gpriors = src.getLogPrior()
                        #     # print('Log(Prior):', gpriors)

                        #     self.logger.warning(f'Source {sid} has escaped its segment!')
                        #     src.pos.setParams([xp0, yp0])
                        #     # self.logger.info(f'Setting position prior. X = {xp0:2.2f}+/-{1.}; Y = {yp0:2.2f}+/-{1.}')
                        #     # src.pos.addGaussianPrior('x', xp0, 1.0)
                        #     # src.pos.addGaussianPrior('y', yp0, 1.0)

                        #     # gpriors = src.getLogPrior()
                        #     # print('Log(Prior):', gpriors)

                    if trip:
                        tr.setCatalog(cat)

                
        if var is None:
            self.logger.warning(f'Variance was not output for blob #{self.blob_id}')
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
            self.logger.warning('Chimap and segmap are not the same shape for #{self.blob_id}')
            return False

        # expvar = np.sum([var_catalog[i].numberOfParams() for i in np.arange(len(var_catalog))])
        # # print(f'I have {len(var)} variance parameters for {self.n_sources} sources. I expected {expvar}.')
        # for i, mod in enumerate(var_catalog):
        #     totalchisq = np.sum((self.tr.getChiImage(0)[self.segmap == self.catalog[i]['source_id']])**2)

        return True

    def tractor_phot(self):
        """ Determines the best-fit model """

        # TODO: The meaning of the following line is not clear
        idx_models = ((1, 2), (3, 4), (5,))
        self.logger.debug(f'Attempting to model {self.n_sources} sources.')

        self._solved = self.solution_catalog != 0

        self._level = -1

        if conf.PLOT > 1:
            fig = np.ones(len(self.bands), dtype=object)
            ax = np.ones(len(self.bands), dtype=object)
            for i, band in enumerate(self.bands):
                fig[i], ax[i] = plot_detblob(self, band=band)

        while not self._solved.all():
            self._level += 1

            if self._level > 2:
                self.logger.critical(f'SOURCE LEFT UNSOLVED IN BLOB {self.blob_id}')
                raise RuntimeError(f'SOURCE LEFT UNSOLVED IN BLOB {self.blob_id}')

            for sublevel in np.arange(len(idx_models[self._level])):
                self._sublevel = sublevel

                self.stage = f'Morph Model ({self._level}, {self._sublevel})'

                # prepare models
                self.mids[~self._solved] = idx_models[self._level][sublevel]
                self.stage_models()

                # store
                self.tr = Tractor(self.timages, self.model_catalog)

                if conf.PLOT > 1:
                    for figi, axi, band in zip(fig, ax, self.bands):
                        plot_detblob(self, figi, axi, band=band, level=self._level, sublevel=self._sublevel, init=True)

                # optimize
                self.logger.debug(self.stage)
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
                    if self.multiband_model:
                        totalchisq = 0
                        opttop = 0
                        optbot = 0
                        for k in np.arange(self.n_bands):
                            try:
                                wgt = self.psfmodels[k].fwhm**-1
                            except:
                                midx = int(np.shape(self.psfmodels[k].img)[0]/2)
                                fwhm = 2.355 * np.std(self.psfmodels[k].img[midx, :])
                                wgt = fwhm**-1

                            totalchisq += np.sum((self.tr.getChiImage(k)[self.segmap == src['source_id']])**2)
                            chi2 = np.sum((self.tr.getChiImage(k)[self.segmap == src['source_id']])**2)
                            nparam = self.model_catalog[i].numberOfParams() - (len(self.bands) - 1)
                            ndof = (np.sum(self.segmap == src['source_id']) - nparam)
                            if ndof < 1:
                                ndof = 1
                            rchi2 = chi2 / ndof
                            opttop += rchi2* wgt
                            optbot += wgt
                    else:
                        totalchisq = np.sum((self.tr.getChiImage(i)[self.segmap == src['source_id']])**2)
                    m_param = self.model_catalog[i].numberOfParams()
                    n_data = np.sum(self.segmap == src['source_id']) * self.n_bands # 1, or else multimodel!
                    self.chisq[i, self._level, self._sublevel] = totalchisq
                    ndof = (n_data - m_param)
                    if ndof < 1:
                        ndof = 1
                    if self.multiband_model:
                        self.rchisq[i, self._level, self._sublevel] = opttop/optbot
                    else:
                        self.rchisq[i, self._level, self._sublevel] = totalchisq / ndof
                    self.bic[i, self._level, self._sublevel] = self.rchisq[i, self._level, self._sublevel] + np.log(n_data) * m_param
 
                    self.logger.debug(f'Source #{src["source_id"]} with {self.model_catalog[i].name} has rchisq={self.rchisq[i, self._level, self._sublevel]:3.3f} | bic={self.bic[i, self._level, self._sublevel]:3.3f}')
                    self.logger.debug(f'     with {len(self.bands)} bands, {m_param} parameters, {n_data} points in total --> NDOF = {ndof}')
                    for k, band in enumerate(self.bands):
                        self.logger.debug(f'               Positions: {self.bands[k]}={self.model_catalog[i].getPosition()}') 
                        self.logger.debug(f'               Fluxes: {self.bands[k]}={self.model_catalog[i].getBrightness().getFlux(self.bands[k]):3.3f}+/-{np.sqrt(self.variance[i].brightness.getParams()[k]):3.3f}') 

                    if self.model_catalog[i].name not in ('PointSource', 'SimpleGalaxy'):
                        if self.model_catalog[i].name == 'FixedCompositeGalaxy':
                            self.logger.debug(f'               {self.model_catalog[i].shapeExp}')
                            self.logger.debug(f'               {self.model_catalog[i].shapeDev}')
                        else:
                            self.logger.debug(f'               {self.model_catalog[i].shape}')
                    

                # Move unsolved to next sublevel
                if sublevel == 0:
                    self.mids[~self._solved] += 1
                if conf.PLOT > 1:
                    for figi, axi, band in zip(fig, ax, self.bands):
                        plot_detblob(self, figi, axi, band=band, level=self._level, sublevel=self._sublevel)

            # decide
            self.decide_winners()
            self._solved = self.solution_catalog != 0

        # print('Starting final optimization')
        # Final optimization
        self.model_catalog = self.solution_catalog.copy()
        self.logger.debug(f'Starting final optimization for blob #{self.blob_id}')
        for i, (mid, src) in enumerate(zip(self.mids, self.bcatalog)):
            self.logger.debug(f'Source #{src["source_id"]}: {self.model_catalog[i].name} model at {self.model_catalog[i].pos}')
            for k, band in enumerate(self.bands):
                    self.logger.debug(f'               Fluxes: {self.bands[k]}={self.model_catalog[i].getBrightness().getFlux(self.bands[k]):3.3f}+/-{np.sqrt(self.variance[i].brightness.getParams()[k]):3.3f}')
            if self.model_catalog[i].name not in ('PointSource', 'SimpleGalaxy'):
                if self.model_catalog[i].name == 'FixedCompositeGalaxy':
                    self.logger.debug(f'               {self.model_catalog[i].shapeExp}')
                    self.logger.debug(f'               {self.model_catalog[i].shapeDev}')
                else:
                    self.logger.debug(f'               {self.model_catalog[i].shape}')
            if conf.FREEZE_FINAL_POSITION:
                self.model_catalog[i].freezeParams('pos')
                self.logger.debug(f'Position parameter frozen at {self.model_catalog[i].pos}')
        self.tr = Tractor(self.timages, self.model_catalog)

        self.pre_solution_catalog = self.tr.getCatalog()
        self.pre_solution_tractor = Tractor(self.timages, self.pre_solution_catalog)
        self.pre_solution_model_images = np.array([self.tr.getModelImage(i) for i in np.arange(self.n_bands)])
        self.pre_solution_chi_images = np.array([self.tr.getChiImage(i) for i in np.arange(self.n_bands)])

        self.stage = 'Final Optimization'
        # self._level, self._sublevel = 'FO', 'FO'
        self.logger.debug(self.stage)
        self.status = self.optimize_tractor()
        
        if not self.status:
            return False
        self.logger.debug(f'Optimization converged in {self.n_converge+1} steps.')

        self.solution_catalog = self.tr.getCatalog()
        self.solution_tractor = Tractor(self.timages, self.solution_catalog)
        self.solution_model_images = np.array([self.tr.getModelImage(i) for i in np.arange(self.n_bands)])
        self.solution_chi_images = np.array([self.tr.getChiImage(i) for i in np.arange(self.n_bands)])
        self.parameter_variance = self.variance
        # print(f'PARAMETER VAR: {self.parameter_variance}')

        self.logger.debug(f'Resulting model parameters for blob #{self.blob_id}')
        self.solution_chisq = np.zeros((self.n_sources, self.n_bands))
        self.solution_bic = np.zeros((self.n_sources, self.n_bands))
        for i, src in enumerate(self.bcatalog):
            for j, band in enumerate(self.bands):
                totalchisq = np.sum((self.tr.getChiImage(j)[self.segmap == src['source_id']])**2)
                m_param = self.model_catalog[i].numberOfParams()  # is this bugged?!
                n_data = np.sum(self.segmap == src['source_id'])
                ndof = (n_data - m_param)
                if ndof < 1:
                    ndof = 1
                self.solution_chisq[i, j] = totalchisq / ndof
                self.solution_bic[i, j] = self.solution_chisq[i, j] + np.log(n_data) * m_param
                self.logger.debug(f'Source #{src["source_id"]} ({band}) with {self.model_catalog[i].name} has rchisq={self.solution_chisq[i, j]:3.3f} | bic={self.solution_bic[i, j]:3.3f}')

                # signal-to-noise
                self.noise[i, j] = np.median(self.background_rms_images[j][self.segmap == src['source_id']])

                sid = src['source_id']
                mod = self.solution_model_images[j]
                chi = self.solution_chi_images[j]
                res = self.images[j] - mod
                res_seg = res[self.segmap==sid].flatten()
                if len(res_seg) < 8:
                    self.k2[i,j] = -99
                else:
                    try:
                        self.k2[i,j], __ = stats.normaltest(res_seg)
                    except:
                        self.k2[i,j] = -99
                        self.logger.warning('Normality test FAILED. Setting to -99')
                chi_seg = chi[self.segmap==sid].flatten()
                self.chi_sig[i,j] = np.std(chi_seg)
                self.chi_mu[i,j] = np.mean(chi_seg)

            self.logger.debug(f'Source #{src["source_id"]}: {self.solution_catalog[i].name} model at {self.solution_catalog[i].pos}')
            self.logger.debug(f'               Fluxes: {self.bands[0]}={self.solution_catalog[i].getBrightness().getFlux(self.bands[0])}') 
            if self.solution_catalog[i].name not in ('PointSource', 'SimpleGalaxy'):
                if self.solution_catalog[i].name == 'FixedCompositeGalaxy':
                    self.logger.debug(f'               {self.solution_catalog[i].shapeExp}')
                    self.logger.debug(f'               {self.solution_catalog[i].shapeDev}')
                else:
                    self.logger.debug(f'               {self.solution_catalog[i].shape}')

        # self.rows = np.zeros(len(self.solution_catalog))
        for idx, src in enumerate(self.solution_catalog):
            # row = np.argwhere(self.brick.catalog['source_id'] == sid)[0][0]
            # self.rows[idx] = row
            # print(f'STASHING {sid} IN ROW {row}')
            self.get_catalog(idx, src, multiband_model=self.multiband_model)

        if conf.PLOT > 1:
            for figi, axi, band in zip(fig, ax, self.bands):
                plot_detblob(self, figi, axi, band=band, level=self._level, sublevel=self._sublevel, final_opt=True)

                # for k, src in enumerate(self.solution_catalog):
                #     sid = self.bcatalog['source_id'][k]
                #     plot_xsection(self, band, src, sid)

        if conf.PLOT > 0:
            for k, src in enumerate(self.solution_catalog):
                sid = self.bcatalog['source_id'][k]
                plot_srcprofile(self, src, sid, self.bands)

        return self.status

    def forced_phot(self):
        """ Forces the best-fit models """

        # print('Starting forced photometry')
        # Update the incoming models
        self.logger.debug('Reviewing sources to be modelled.')
        for i, model in enumerate(self.model_catalog):
            # if model == -99:
            #     if conf.VERBOSE: print(f"FAILED -- Source #{self.bcatalog[i]['source_id']} does not have a valid model!")
            #     return False

            ############ self.model_catalog[i].brightness = Fluxes(**dict(zip(self.bands, model.brightness[0] * np.ones(self.n_bands))))
            self.model_catalog[i].freezeAllBut('brightness')
            # # self.model_catalog[i].thawParams('sky')
            if not conf.FREEZE_FORCED_POSITION:
                self.logger.debug('Thawing position...')
                self.model_catalog[i].thawParams('pos')
            if (not conf.FREEZE_FORCED_SHAPE) & (model.name not in ('PointSource', 'SimpleGalaxy', 'FixedCompositeGalaxy')):
                self.model_catalog[i].thawParams('shape')
            elif (not conf.FREEZE_FORCED_SHAPE) & (model.name == 'FixedCompositeGalaxy'):
                self.model_catalog[i].thawParams('shapeExp')
                self.model_catalog[i].thawParams('shapeDev')

            best_band = f"{self.bcatalog[i]['BEST_MODEL_BAND']}"


            self.logger.debug(f"Source #{self.bcatalog[i]['source_id']}: {self.model_catalog[i].name} model at {self.model_catalog[i].pos}")
            self.logger.debug(f'               {self.model_catalog[i].brightness}')
            if self.bcatalog[i][f'SOLMODEL_{best_band}'] == 'FixedCompositeGalaxy':
                self.logger.debug(f'               {self.model_catalog[i].shapeExp}')
                self.logger.debug(f'               {self.model_catalog[i].shapeDev}')
            elif self.bcatalog[i][f'SOLMODEL_{best_band}'] in ('ExpGalaxy', 'DevGalaxy'):
                self.logger.debug(f'               {self.model_catalog[i].shape}')


        # Stash in Tractor
        self.tr = Tractor(self.timages, self.model_catalog)
        self.stage = 'Forced Photometry'

        # if conf.PLOT >1:
        #     axlist = [plot_fblob(self, band=band) for band in self.bands]


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

        self.logger.info(f'Resulting model parameters for blob #{self.blob_id}')
        for i, src in enumerate(self.bcatalog):
            self.logger.info(f'Source #{src["source_id"]}: {self.solution_catalog[i].name} model at {self.solution_catalog[i].pos}')
            # if self.solution_catalog[i].name not in ('PointSource', 'SimpleGalaxy'):
            #     if self.solution_catalog[i].name == 'FixedCompositeGalaxy': 
            #         shapeExp, shapeExp_err = self.solution_catalog[i].shapeExp, self.forced_variance[i].shapeExp.getParams('shapeExp')
            #         shapeDev, shapeDev_err = self.solution_catalog[i].shapeDev, self.forced_variance[i].shapeDev.getParams('shapeDev')
            #         self.logger.info(f'    ShapeExp -- Reff:      {shapeExp.re:3.3f} +/- {shapeExp_err.re:3.3f}')
            #         self.logger.info(f'    ShapeExp -- ee1:       {shapeExp.ee1:3.3f} +/- {shapeExp_err.ee1:3.3f}')
            #         self.logger.info(f'    ShapeExp -- ee2:       {shapeExp.ee2:3.3f} +/- {shapeExp_err.ee2:3.3f}')
            #         self.logger.info(f'    ShapeDev -- Reff:      {shapeDev.re:3.3f} +/- {shapeDev_err.re:3.3f}')
            #         self.logger.info(f'    ShapeDev -- ee1:       {shapeDev.ee1:3.3f} +/- {shapeDev_err.ee1:3.3f}')
            #         self.logger.info(f'    ShapeDev -- ee2:       {shapeDev.ee2:3.3f} +/- {shapeDev_err.ee2:3.3f}')
            #     else:
            #         shape, shape_err = self.solution_catalog[i].shape, self.forced_variance[i].shape
            #         self.logger.info(f'    Shape -- Reff:      {shape.re:3.3f} +/- {shape_err.re:3.3f}')
            #         self.logger.info(f'    Shape -- ee1:       {shape.ee1:3.3f} +/- {shape_err.ee1:3.3f}')
            #         self.logger.info(f'    Shape -- ee2:       {shape.ee2:3.3f} +/- {shape_err.ee2:3.3f}')
            for j, band in enumerate(self.bands):
                totalchisq = np.sum((self.tr.getChiImage(j)[self.segmap == src['source_id']])**2)
                m_param = self.model_catalog[i].numberOfParams() / self.n_bands
                n_data = np.sum(self.segmap == src['source_id'])
                self.solution_bic[i, j] = totalchisq + np.log(n_data) * m_param
                self.solution_chisq[i, j] = totalchisq / (n_data - m_param)
                # flux = self.solution_catalog[i].getBrightness().getFlux(self.bands[j])
                # fluxerr = np.sqrt(self.forced_variance[i].brightness.getParams()[j])
                # self.logger.info(f'Source #{src["source_id"]} in {self.bands[j]}')
                # self.logger.info(f'    Flux({self.bands[j]}):  {flux:3.3f} +/- {fluxerr:3.3f}') 
                # self.logger.info(f'    Chisq({self.bands[j]}): {totalchisq:3.3f}')
                # self.logger.info(f'    BIC({self.bands[j]}):   {self.solution_bic[i, j]:3.3f}')

                # signal-to-noise
                self.noise[i, j] = np.median(self.background_rms_images[j][self.segmap == src['source_id']])

                sid = src['source_id']
                mod = self.solution_model_images[j]
                chi = self.solution_chi_images[j]
                res = self.images[j] - mod
                res_seg = res[self.segmap==sid].flatten()
                if len(res_seg) < 8:
                    self.k2[i,j] = -99
                else:
                    try:
                        self.k2[i,j], __ = stats.normaltest(res_seg)
                    except:
                        self.k2[i,j] = -99
                        self.logger.warning('Normality test FAILED. Setting to -99')
                chi_seg = chi[self.segmap==sid].flatten()
                self.chi_sig[i,j] = np.std(chi_seg)
                self.chi_mu[i,j] = np.mean(chi_seg)

                if conf.PLOT > 2:
                    for k, ssrc in enumerate(self.solution_catalog):
                        sid = self.bcatalog['source_id'][k]
                        plot_xsection(self, band, ssrc, sid)

        # self.rows = np.zeros(len(self.solution_catalog))
        for idx, src in enumerate(self.solution_catalog):
            self.get_catalog(idx, src, multiband_only=True)


            if conf.PLOT > 0:
                sid = self.bcatalog['source_id'][idx]
                plot_srcprofile(self, src, sid, self.bands)

        return status

    def decide_winners(self, use_bic=conf.USE_BIC):
        """ Traffic cop to direct BIC or CHISQ trees """
        if use_bic:
            self.logger.debug('Using BIC to select best-fit model')
            self.decide_winners_bic()
        else:
            self.logger.debug('Using chisq/N to select best-fit model')
            self.decide_winners_chisq()

    def decide_winners_bic(self):
        """ Decision tree in BIC. EXPERIMENTAL. """

        # holders - or else it's pure insanity.
        sid = self.bcatalog['source_id'][~self._solved]
        bic= self.bic[~self._solved]
        rchisq = self.rchisq[~self._solved]
        solution_catalog = self.solution_catalog[~self._solved]
        solved_bic = self.solved_bic[~self._solved]
        tr_catalogs = self.tr_catalogs[~self._solved]
        mids = self.mids[~self._solved]

        if self._level == 0:
            # Which have chi2(PS) < chi2(SG) and both are chi2 > 3?
            for i, blob_id in enumerate(sid):
                self.logger.debug(f'Source #{blob_id} BIC -- PS({bic[i, 0, 0]:3.3f}) vs. SG({bic[i, 0, 1]:3.3f}) with thresh of {conf.PS_SG_THRESH:3.3f}')
            chmask = (bic[:, 0, 0] - bic[:, 0, 1] < conf.PS_SG_THRESH)
            chmask[(rchisq[:, 0, 0] > conf.CHISQ_FORCE_EXP_DEV) & (rchisq[:, 0, 1] > conf.CHISQ_FORCE_EXP_DEV)] = False # For these, keep trying! Essentially a back-door for high a/b sources.
            if chmask.any():
                solution_catalog[chmask] = tr_catalogs[chmask, 0, 0].copy()
                solved_bic[chmask] = bic[chmask, 0, 0]
                mids[chmask] = 1

            # So chi2(SG) is min, try more models
            mids[~chmask] = 3

        if self._level == 1:
            for i, blob_id in enumerate(sid):
                self.logger.debug(f'Source #{blob_id} BIC -- EXP({bic[i, 1, 0]:3.3f}) vs. DEV({bic[i, 1, 1]:3.3f}) with thresh of {conf.EXP_DEV_THRESH:3.3f}')
            # For which are they nearly equally good?
            movemask = (abs(bic[:, 1, 0] - bic[:, 1, 1]) < conf.EXP_DEV_THRESH)
            # movemask = np.ones_like(bic[:,1,0], dtype=bool)

            # Has Exp beaten SG?
            expmask = (bic[:, 1, 0] < bic[:, 0, 1])

            # Has Dev beaten SG?
            devmask = (bic[:, 1, 1] < bic[:, 0, 1])

            # Which better than SG but nearly equally good?
            nextmask = expmask & devmask & movemask

            # For which was SG better
            premask_sg = ~expmask & ~devmask & (bic[:, 0, 1] < bic[:, 0, 0])

            # For which was PS better
            premask_ps = ~expmask & ~devmask & (bic[:, 0, 0] < bic[:, 0, 1])

            # If Exp beats Dev by a lot
            nexpmask = expmask & ~movemask & (bic[:, 1, 0]  <  bic[:, 1, 1])

             # If Dev beats Exp by a lot
            ndevmask = devmask & ~movemask & (bic[:, 1, 1] < bic[:, 1, 0])

            if nextmask.any():
                mids[nextmask] = 5

            if premask_ps.any():
                solution_catalog[premask_ps] = tr_catalogs[premask_ps, 0, 0].copy()
                solved_bic[premask_ps] = bic[premask_ps, 0, 0]
                mids[premask_ps] = 1

            if premask_sg.any():
                solution_catalog[premask_sg] = tr_catalogs[premask_sg, 0, 1].copy()
                solved_bic[premask_sg] = bic[premask_sg, 0, 1]
                mids[premask_sg] = 2

            if nexpmask.any():

                solution_catalog[nexpmask] = tr_catalogs[nexpmask, 1, 0].copy()
                solved_bic[nexpmask] = bic[nexpmask, 1, 0]
                mids[nexpmask] = 3

            if ndevmask.any():

                solution_catalog[ndevmask] = tr_catalogs[ndevmask, 1, 1].copy()
                solved_bic[ndevmask] = bic[ndevmask, 1, 1]
                mids[ndevmask] = 4

        if self._level == 2:
            for i, blob_id in enumerate(sid):
                self.logger.debug(f'Source #{blob_id} BIC -- COMP({bic[i, 2, 0]:3.3f})')
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
        """ Decision tree in CHISQ. STABLE. """

        # take the model_catalog and chisq and figure out what's what
        # Only look at unsolved models!
        
        # if conf.USE_REDUCEDCHISQ:
        #     chisq_exp = 1.0
        # else:
        chisq_exp = 0.0

        # holders - or else it's pure insanity.
        sid = self.bcatalog['source_id'][~self._solved]
        chisq = self.rchisq[~self._solved]
        solution_catalog = self.solution_catalog[~self._solved]
        solved_chisq = self.solved_chisq[~self._solved]
        tr_catalogs = self.tr_catalogs[~self._solved]
        mids = self.mids[~self._solved]

        if self._level == 0:
            # Which have chi2(PS) < chi2(SG)?
            for i, blob_id in enumerate(sid):
                self.logger.debug(f'Source #{blob_id} Chi2 -- PS({chisq[i, 0, 0]:3.3f}) vs. SG({chisq[i, 0, 1]:3.3f}) with thresh of {conf.PS_SG_THRESH:3.3f}')
            chmask = ((abs(chisq_exp - chisq[:, 0, 0]) - abs(chisq_exp - chisq[:, 0, 1])) < conf.PS_SG_THRESH)
            chmask[(chisq[:, 0, 0] > conf.CHISQ_FORCE_EXP_DEV) & (chisq[:, 0, 1] > conf.CHISQ_FORCE_EXP_DEV)] = False # For these, keep trying! Essentially a back-door for high a/b sources.
            if chmask.any():
                solution_catalog[chmask] = tr_catalogs[chmask, 0, 0].copy()
                solved_chisq[chmask] = chisq[chmask, 0, 0]
                mids[chmask] = 1

            # So chi2(SG) is min, try more models
            mids[~chmask] = 3

        if self._level == 1:
            for i, blob_id in enumerate(sid):
                self.logger.debug(f'Source #{blob_id} Chi2 -- EXP({chisq[i, 1, 0]:3.3f}) vs. DEV({chisq[i, 1, 1]:3.3f}) with thresh of {conf.EXP_DEV_THRESH:3.3f}')
            # For which are they nearly equally good?
            movemask = (abs(chisq[:, 1, 0] - chisq[:, 1, 1]) < conf.EXP_DEV_THRESH)

            # Has Exp beaten SG?
            expmask = (abs(chisq_exp - chisq[:, 1, 0]) < abs(chisq_exp - chisq[:, 0, 1]))

            # Has Dev beaten SG?
            devmask = (abs(chisq_exp - chisq[:, 1, 1]) < abs(chisq_exp - chisq[:, 0, 1]))

            # Which better than SG but nearly equally good?
            nextmask = expmask & devmask & movemask

            # What if they are both bad?
            badmask = (chisq[:, 1, 0] > conf.CHISQ_FORCE_COMP) & (chisq[:, 1, 1] > conf.CHISQ_FORCE_COMP)

            nextmask |= badmask

            # For which was SG best? Then go back!
            premask_sg = ~expmask & ~devmask & (chisq[:, 0, 1] < chisq[:, 0, 0])

            # For which was PS best? The go back!
            premask_ps = (chisq[:, 0, 0] < chisq[:, 0, 1]) & (chisq[:, 0, 0] < chisq[:, 1, 0]) & (chisq[:, 0, 0] < chisq[:, 1, 1])

            # If Exp beats Dev by a lot
            nexpmask = expmask & ~movemask & (abs(chisq_exp - chisq[:, 1, 0])  <  abs(chisq_exp - chisq[:, 1, 1])) & ~badmask

             # If Dev beats Exp by a lot
            ndevmask = devmask & ~movemask & (abs(chisq_exp - chisq[:, 1, 1]) < abs(chisq_exp - chisq[:, 1, 0])) & ~badmask

            # Check solved first
            if nexpmask.any():
                solution_catalog[nexpmask] = tr_catalogs[nexpmask, 1, 0].copy()
                solved_chisq[nexpmask] = chisq[nexpmask, 1, 0]
                mids[nexpmask] = 3

            if ndevmask.any():
                solution_catalog[ndevmask] = tr_catalogs[ndevmask, 1, 1].copy()
                solved_chisq[ndevmask] = chisq[ndevmask, 1, 1]
                mids[ndevmask] = 4

            # Then which ones might advance
            if nextmask.any():
                mids[nextmask] = 5
            

            # Then which are going back (which can overrwrite)
            if premask_ps.any():
                solution_catalog[premask_ps] = tr_catalogs[premask_ps, 0, 0].copy()
                solved_chisq[premask_ps] = chisq[premask_ps, 0, 0]
                mids[premask_ps] = 1

            if premask_sg.any():
                solution_catalog[premask_sg] = tr_catalogs[premask_sg, 0, 1].copy()
                solved_chisq[premask_sg] = chisq[premask_sg, 0, 1]
                mids[premask_sg] = 2

            



        if self._level == 2:
            for i, blob_id in enumerate(sid):
                self.logger.debug(f'Source #{blob_id} Chi2 -- COMP({chisq[i, 2, 0]:3.3f})')
            # For which did Comp beat EXP and DEV?
            compmask = (abs(chisq_exp - chisq[:, 2, 0]) < abs(chisq_exp - chisq[:, 1, 0])) &\
                       (abs(chisq_exp - chisq[:, 2, 0]) < abs(chisq_exp - chisq[:, 1, 1]))

            if compmask.any():
                solution_catalog[compmask] = tr_catalogs[compmask, 2, 0].copy()
                solved_chisq[compmask] = chisq[compmask, 2, 0]
                mids[compmask] = 5

            # where better as EXP or DEV
            compmask_or = (chisq[:, 2, 0] > chisq[:, 1, 0]) | (chisq[:, 2, 0] > chisq[:, 1, 1])
            if compmask_or.any():
                ch_exp = (abs(chisq_exp - chisq[:, 1, 0]) <= abs(chisq_exp - chisq[:, 1, 1])) & compmask_or

                if ch_exp.any():
                    solution_catalog[ch_exp] = tr_catalogs[ch_exp, 1, 0].copy()
                    solved_chisq[ch_exp] = chisq[ch_exp, 1, 0]
                    mids[ch_exp] = 3

                ch_dev = (abs(chisq_exp - chisq[:, 1, 1]) < abs(chisq_exp - chisq[:, 1, 0])) & compmask_or

                if ch_dev.any():
                    solution_catalog[ch_dev] = tr_catalogs[ch_dev, 1, 1].copy()
                    solved_chisq[ch_dev] = chisq[ch_dev, 1, 1]
                    mids[ch_dev] = 4

        # hand back
        self.chisq[~self._solved] = chisq
        self.solution_catalog[~self._solved] = solution_catalog
        self.solved_chisq[~self._solved] = solved_chisq
        self.mids[~self._solved] = mids

    def aperture_phot(self, band=None, image_type=None, sub_background=False):
        """ Provides post-processing aperture photometry support """
        # Allow user to enter image (i.e. image, residual, model...)

        tstart = time.time()
        if band is None:
            self.logger.info(f'Performing aperture photometry on {conf.MODELING_NICKNAME} {image_type}...')
        else:
            self.logger.info(f'Performing aperture photometry on {band} {image_type}...')

        if image_type not in ('image', 'model', 'isomodel', 'residual'):
            raise TypeError("image_type must be 'image', 'model', 'isomodel', or 'residual'")

        if band is None:
            idx = 0
        else:
            idx = np.argwhere(self.bands == band)[0][0]
        use_iso = False

        if image_type == 'image':
            image = self.images[idx] 

        elif image_type == 'model':
            image = self.solution_tractor.getModelImage(idx)
        
        elif image_type == 'isomodel':
            use_iso = True

        elif image_type == 'residual':
            image = (self.images[idx] - self.solution_tractor.getModelImage(idx))
        
        if conf.APER_APPLY_SEGMASK & (not use_iso):
            image *= self.masks[idx]

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

        if sub_background & (not use_iso):
            image -= self.background_images[idx]

        cat = self.solution_catalog
        xxyy = np.vstack([src.getPosition() for src in cat]).T
        apxy = xxyy - 1.

        apertures_arcsec = np.array(conf.APER_PHOT)
        apertures = apertures_arcsec / self.pixel_scale / 2. # diameter in arcsec -> radius in pixels

        apflux = np.zeros((len(cat), len(apertures)), np.float32)
        apflux_err = np.zeros((len(cat), len(apertures)), np.float32)
        apmag = np.zeros((len(cat), len(apertures)), np.float32)
        apmag_err = np.zeros((len(cat), len(apertures)), np.float32)

        H,W = self.images[0].shape
        Iap = np.flatnonzero((apxy[0,:] >= 0)   * (apxy[1,:] >= 0) *
                            (apxy[0,:] <= W-1) * (apxy[1,:] <= H-1))

        if band is None:
            zpt = conf.MODELING_ZPT
        else:   
            zpt = conf.MULTIBAND_ZPT[self._band2idx(band)]

        if var is None:
            imgerr = None
        else:
            imgerr = np.sqrt(var)

        for i, rad in enumerate(apertures):
            if not use_iso: # Run with all models in image
                aper = photutils.CircularAperture(apxy[:,Iap], rad)
                self.logger.debug(f'Measuring {apertures_arcsec[i]:2.2f}" aperture flux on {len(cat)} sources.')
                p = photutils.aperture_photometry(image, aper, error=imgerr)
                # aper.plot()
                apflux[Iap, i] = p.field('aperture_sum') * 10**(-0.4 * (zpt - 23.9))
                if var is None:
                    apflux_err[Iap, i] = -99 * np.ones_like(apflux[Iap, i])
                else:
                    apflux_err[Iap, i] = p.field('aperture_sum_err') * 10**(-0.4 * (zpt - 23.9))
            for j, src in enumerate(cat):
                if use_iso: # Run with only one model in image
                    aper = photutils.CircularAperture(apxy[:,j], rad)
                    image = self.solution_tractor.getModelImage(idx, srcs=[src,])
                    if conf.APER_APPLY_SEGMASK:
                        image *= self.masks[self.band2idx(band)]
                    if sub_background:
                        image -= self.background_images[idx]
                    self.logger.debug(f'Measuring {apertures_arcsec[i]:2.2f}" aperture flux on 1 source of {len(cat)}.')
                    p = photutils.aperture_photometry(image, aper, error=imgerr)
                    # aper.plot()
                    apflux[j, i] = p.field('aperture_sum') * 10**(-0.4 * (zpt - 23.9))
                    if var is None:
                        apflux_err[j, i] = -99 * np.ones_like(apflux[j, i])
                    else:
                        apflux_err[j, i] = p.field('aperture_sum_err') * 10**(-0.4 * (zpt - 23.9))

                apmag[j, i] = - 2.5 * np.log10( apflux[j,i] ) + 23.9
                apmag_err[j, i] = 1.09 * apflux_err[j, i] / apflux[j, i]
                self.logger.debug(f'        Flux({j}, {band}, {apertures_arcsec[i]:2.2f}") = {apflux[j,i]:3.3f}/-{apflux_err[j,i]:3.3f}')
                self.logger.debug(f'        Mag({j}, {band}, {apertures_arcsec[i]:2.2f}") = {apmag[j, i]:3.3f}/-{apmag_err[j, i]:3.3f}')

        # if image_type == 'image':
            
        #     plt.figure()
        #     col = 'royalblue'
        # if image_type == 'model':
        #     col = 'orange'
        # if image_type == 'residual':
        #     col = 'brown'
        
        # plt.ion()
        # area = np.pi * apertures_arcsec**2
        # plt.plot(apertures_arcsec, apflux[0]/area, color=col, label=image_type)
        # plt.axhline(0, color='k', ls='dotted')
        # plt.axhline(self.backgrounds[0][0])
        # #plt.axhline(self.bcatalog['FLUX_'+band], color='red')
        # plt.legend()
        # plt.savefig(os.path.join(conf.PLOT_DIR, f'{band}_{self.blob_id}_NORM_{conf.NORMALIZE_PSF}_GAUSS_{conf.FORCE_GAUSSIAN_PSF}_SEGMASK_{conf.APPLY_SEGMASK}_SURFBRI.pdf'))

        if band is None:
            band = 'MODELING'
        band = band.replace(' ', '_')
        if f'FLUX_APER_{band}_{image_type}' not in self.bcatalog.colnames:
            self.bcatalog.add_column(Column(length=len(self.bcatalog), dtype=float, shape=len(apertures), name=f'FLUX_APER_{band}_{image_type}'))
            self.bcatalog.add_column(Column(length=len(self.bcatalog), dtype=float, shape=len(apertures), name=f'FLUX_APER_{band}_{image_type}_err'))
            self.bcatalog.add_column(Column(length=len(self.bcatalog), dtype=float, shape=len(apertures), name=f'MAG_APER_{band}_{image_type}'))
            self.bcatalog.add_column(Column(length=len(self.bcatalog), dtype=float, shape=len(apertures), name=f'MAG_APER_{band}_{image_type}_err'))
            # self.bcatalog.add_column(Column(length=len(self.bcatalog), dtype=float, name=f'MAG_TOTAL_{band}_{image_type}'))
            # self.bcatalog.add_column(Column(length=len(self.bcatalog), dtype=float, name=f'MAG_TOTAL_{band}_{image_type}_err'))

        for idx, src in enumerate(self.solution_catalog):
            sid = self.bcatalog['source_id'][idx]
            row = np.argwhere(self.bcatalog['source_id'] == sid)[0][0]
            self.bcatalog[row][f'FLUX_APER_{band}_{image_type}'] = tuple(apflux[idx])
            self.bcatalog[row][f'FLUX_APER_{band}_{image_type}_err'] = tuple(apflux_err[idx])
            self.bcatalog[row][f'MAG_APER_{band}_{image_type}'] = tuple(apmag[idx])
            self.bcatalog[row][f'MAG_APER_{band}_{image_type}_err'] = tuple(apmag_err[idx])
            # self.bcatalog[row][f'MAG_TOTAL_{band}_{image_type}'] = apmag[idx, -1]
            # self.bcatalog[row][f'MAG_TOTAL_{band}_{image_type}_err'] = apmag_err[idx, -1]

        self.logger.info(f'Aperture photometry complete ({time.time() - tstart:3.3f}s)')

    def sextract_phot(self, band=None, sub_background=False):
        """ Run Sextractor on the residuals and flag any sources with detections in the parent blob """
        # SHOULD WE STACK THE RESIDUALS? (No?)
        # SHOULD WE DO THIS ON THE MODELING IMAGE TOO? (I suppose we can already...!)
        if band is None:
            idx = 0
        else:
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

        if band is None:
            band = 'MODELING'
        if f'{band}_n_residual_sources' not in self.bcatalog.colnames:
                self.bcatalog.add_column(Column(np.zeros(len(self.bcatalog), dtype=bool), name=f'{band}_n_residual_sources'))

        if len(catalog) != 0:
            self.residual_catalog[idx] = catalog
            n_residual_sources = len(catalog)
            self.residual_segmap = segmap
            self.n_residual_sources[idx] = n_residual_sources
            self.logger.debug(f'SExtractor Found {n_residual_sources} in {band} residual!')


            for idx, src in enumerate(self.solution_catalog):
                sid = self.bcatalog['source_id'][idx]
                row = np.argwhere(self.catalog['source_id'] == sid)[0][0]
                self.bcatalog[row][f'{band}_n_residual_sources'] = True

            return catalog, segmap
        else:
            self.logger.debug('No objects found by SExtractor.')

    def get_catalog(self, row, src, multiband_only=False, multiband_model=False):
        """ Turn photometry into a catalog. Add flags. """

        sid = self.bcatalog['source_id'][row]
        self.logger.debug(f'blob.get_catalog :: Writing output entires for #{sid}')

        # Add band fluxes, flux errors
        for i, band in enumerate(self.bands):
            valid_source = True # until decided False
            band = band.replace(' ', '_')
            if band == conf.MODELING_NICKNAME:
                zpt = conf.MODELING_ZPT
                param_var = self.parameter_variance
            elif band.startswith(conf.MODELING_NICKNAME):
                band_name = band[len(conf.MODELING_NICKNAME)+1:]
                zpt = conf.MULTIBAND_ZPT[self._band2idx(band_name)]
                param_var = self.parameter_variance
            else:
                zpt = conf.MULTIBAND_ZPT[self._band2idx(band)]
                param_var = self.forced_variance

            self.bcatalog[row]['MAG_'+band] = -2.5 * np.log10(src.getBrightness().getFlux(band)) + zpt
            self.bcatalog[row]['MAGERR_'+band] = 1.09 * np.sqrt(param_var[row].brightness.getParams()[i]) / src.getBrightness().getFlux(band)
            self.bcatalog[row]['RAWFLUX_'+band] = src.getBrightness().getFlux(band)
            self.bcatalog[row]['RAWFLUXERR_'+band] = np.sqrt(param_var[row].brightness.getParams()[i])
            self.bcatalog[row]['FLUX_'+band] = src.getBrightness().getFlux(band) * 10**(-0.4 * (zpt - 23.9))  # Force fluxes to be in uJy!
            self.bcatalog[row]['FLUXERR_'+band] = np.sqrt(param_var[row].brightness.getParams()[i]) * 10**(-0.4 * (zpt - 23.9))
            self.bcatalog[row]['CHISQ_'+band] = self.solution_chisq[row, i]
            self.bcatalog[row]['BIC_'+band] = self.solution_bic[row, i]
            # self.bcatalog[row]['N_CONVERGE_'+band] = self.n_converge
            self.bcatalog[row]['SNR_'+band] = self.bcatalog[row]['RAWFLUX_'+band] / self.bcatalog[row]['npix'] / self.noise[row, i]
            # self.bcatalog[row]['NORM_'+band] = self.norm[row, i]
            self.bcatalog[row]['CHI_MU_'+band] = self.chi_mu[row, i]
            self.bcatalog[row]['CHI_SIG_'+band] = self.chi_sig[row, i]
            self.bcatalog[row]['CHI_K2_'+band] = self.k2[row, i]
            # self.bcatalog[row]['VALID_SOURCE_'+band] = valid_source

            if not conf.FREEZE_FORCED_POSITION:
                self.bcatalog[row][f'X_MODEL_{band}'] = src.pos[0] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER
                self.bcatalog[row][f'Y_MODEL_{band}'] = src.pos[1] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER
                self.bcatalog[row][f'XERR_MODEL_{band}'] = np.sqrt(param_var[row].pos.getParams()[0])
                self.bcatalog[row][f'YERR_MODEL_{band}'] = np.sqrt(param_var[row].pos.getParams()[1])
                if self.wcs is not None:
                    skyc = self.brick_wcs.all_pix2world(self.bcatalog[row][f'X_MODEL_{band}'] - self.mosaic_origin[0] + conf.BRICK_BUFFER, self.bcatalog[row][f'Y_MODEL_{band}'] - self.mosaic_origin[1] + conf.BRICK_BUFFER, 0)
                    self.bcatalog[row][f'RA_{band}'] = skyc[0]
                    self.bcatalog[row][f'DEC_{band}'] = skyc[1]

            if not conf.FREEZE_FORCED_SHAPE:
                # Model Parameters
                self.bcatalog[row][f'VALID_SOURCE_{band}'] = valid_source


                if src.name in ('ExpGalaxy', 'DevGalaxy'):
                    self.bcatalog[row][f'REFF_{band}'] = src.shape.logre
                    self.bcatalog[row][f'REFF_ERR_{band}'] = np.sqrt(param_var[row].shape.getParams()[0])
                    self.bcatalog[row][f'EE1_{band}'] = src.shape.ee1
                    self.bcatalog[row][f'EE2_{band}'] = src.shape.ee2
                    if (src.shape.e >= 1) | (src.shape.e <= -1):
                        # self.bcatalog[row][f'VALID_SOURCE'] = False
                        self.bcatalog[row][f'AB_{band}'] = -99.0
                        self.logger.warning(f'Source has invalid ellipticity! (e = {src.shape.e:3.3f})')
                    else:
                        self.bcatalog[row][f'AB_{band}'] = (src.shape.e + 1) / (1 - src.shape.e)
                    self.bcatalog[row][f'AB_ERR_{band}'] = np.sqrt(param_var[row].shape.getParams()[1])
                    self.bcatalog[row][f'THETA_{band}'] = np.rad2deg(src.shape.theta)
                    self.bcatalog[row][f'THETA_ERR_{band}'] = np.sqrt(param_var[row].shape.getParams()[2])

                    self.logger.info(f"    Reff:               {self.bcatalog[row][f'REFF_{band}']:3.3f} +/- {self.bcatalog[row][f'REFF_ERR_{band}']:3.3f}")
                    self.logger.info(f"    a/b:                {self.bcatalog[row][f'AB_{band}']:3.3f} +/- {self.bcatalog[row][f'AB_ERR_{band}']:3.3f}")
                    self.logger.info(f"    pa:                 {self.bcatalog[row][f'THETA_{band}']:3.3f} +/- {self.bcatalog[row][f'THETA_ERR_{band}']:3.3f}")

                elif src.name == 'FixedCompositeGalaxy':
                    self.bcatalog[row][f'FRACDEV_{band}'] = src.fracDev.getValue()
                    self.bcatalog[row][f'EXP_REFF_{band}'] = src.shapeExp.logre
                    self.bcatalog[row][f'EXP_REFF_ERR_{band}'] = np.sqrt(param_var[row].shapeExp.getParams()[0])
                    if (src.shapeExp.e >= 1) | (src.shapeExp.e <= -1):
                        # self.bcatalog[row][f'VALID_SOURCE'] = False
                        self.bcatalog[row][f'EXP_AB_{band}'] = -99.0
                        self.logger.warning(f'Source has invalid ellipticity! (e = {src.shapeExp.e:3.3f})')
                    else:
                        self.bcatalog[row][f'EXP_AB_{band}'] = (src.shapeExp.e + 1) / (1 - src.shapeExp.e)
                    self.bcatalog[row][f'EXP_AB_ERR_{band}'] = np.sqrt(param_var[row].shapeExp.getParams()[1])
                    self.bcatalog[row][f'EXP_THETA_{band}'] = np.rad2deg(src.shapeExp.theta)
                    self.bcatalog[row][f'EXP_THETA_ERR_{band}'] = np.sqrt(param_var[row].shapeExp.getParams()[2])
                    self.bcatalog[row][f'DEV_REFF_{band}'] = src.shapeDev.logre
                    self.bcatalog[row][f'DEV_REFF_ERR_{band}'] = np.sqrt(param_var[row].shapeDev.getParams()[0])
                    if (src.shapeDev.e >= 1) | (src.shapeDev.e <= -1):
                        # self.bcatalog[row][f'VALID_SOURCE'] = False
                        self.bcatalog[row][f'DEV_AB_{band}'] = -99.0
                        self.logger.warning(f'Source has invalid ellipticity! (e = {src.shapeDev.e:3.3f})')
                    else:
                        self.bcatalog[row][f'DEV_AB_{band}'] = (src.shapeDev.e + 1) / (1 - src.shapeDev.e)
                    self.bcatalog[row][f'DEV_AB_ERR_{band}'] = np.sqrt(param_var[row].shapeDev.getParams()[1])
                    self.bcatalog[row][f'DEV_THETA_{band}'] = np.rad2deg(src.shapeDev.theta)
                    self.bcatalog[row][f'DEV_THETA_ERR_{band}'] = np.sqrt(param_var[row].shapeDev.getParams()[2])
                    # self.bcatalog[row][f'reff_err'] = np.sqrt(self.parameter_variance[row][0])
                    # self.bcatalog[row][f'ab_err'] = np.sqrt(self.parameter_variance[row][1])
                    # self.bcatalog[row][f'phi_err'] = np.sqrt(self.parameter_variance[row][2])

                    self.bcatalog[row][f'EXP_EE1_{band}'] = src.shapeExp.ee1
                    self.bcatalog[row][f'EXP_EE2_{band}'] = src.shapeExp.ee2
                    self.bcatalog[row][f'DEV_EE1_{band}'] = src.shapeDev.ee1
                    self.bcatalog[row][f'DEV_EE2_{band}'] = src.shapeDev.ee2


                    if (src.shapeExp.e >= 1) | (src.shapeExp.e <= -1):
                        # self.bcatalog[row]['VALID_SOURCE'] = False
                        self.logger.warning(f'Source has invalid ellipticity! (e = {src.shapeExp.e:3.3f})')


                    if (src.shapeDev.e >= 1) | (src.shapeDev.e <= -1):
                        # self.bcatalog[row]['VALID_SOURCE'] = False
                        self.logger.warning(f'Source has invalid ellipticity! (e = {src.shapeDev.e:3.3f})')


                    self.logger.info(f"    Reff(Exp):          {self.bcatalog[row][f'EXP_REFF_{band}']:3.3f} +/- {self.bcatalog[row][f'EXP_REFF_ERR_{band}']:3.3f}")
                    self.logger.info(f"    a/b (Exp):          {self.bcatalog[row][f'EXP_AB_{band}']:3.3f} +/- {self.bcatalog[row][f'EXP_AB_ERR_{band}']:3.3f}")
                    self.logger.info(f"    pa  (Exp):          {self.bcatalog[row][f'EXP_THETA_{band}']:3.3f} +/- {self.bcatalog[row][f'EXP_THETA_ERR_{band}']:3.3f}")
                    self.logger.info(f"    Reff(Dev):          {self.bcatalog[row][f'DEV_REFF_{band}']:3.3f} +/- {self.bcatalog[row][f'DEV_REFF_ERR_{band}']:3.3f}")
                    self.logger.info(f"    a/b (Dev):          {self.bcatalog[row][f'DEV_AB_{band}']:3.3f} +/- {self.bcatalog[row][f'DEV_AB_ERR_{band}']:3.3f}")
                    self.logger.info(f"    pa  (Dev):          {self.bcatalog[row][f'DEV_THETA_{band}']:3.3f} +/- {self.bcatalog[row][f'DEV_THETA_ERR_{band}']:3.3f}")

                elif src.name not in ('PointSource', 'SimpleGalaxy'): # last resort
                    self.logger.warning(f"Source does not have a valid solution model!")
                    valid_source = False
                    self.bcatalog[row]['VALID_SOURCE'] = valid_source


            pos = 0000
            mag, magerr = self.bcatalog[row]['MAG_'+band], self.bcatalog[row]['MAGERR_'+band]
            flux, fluxerr = self.bcatalog[row]['FLUX_'+band], self.bcatalog[row]['FLUXERR_'+band]
            rawflux, rawfluxerr = self.bcatalog[row]['RAWFLUX_'+band], self.bcatalog[row]['RAWFLUXERR_'+band]
            chisq, bic = self.bcatalog[row]['CHISQ_'+band], self.bcatalog[row]['BIC_'+band]
            self.logger.info(f'    Position({band}):     {pos}')
            self.logger.info(f'    Raw Flux({band}):     {rawflux:3.3f} +/- {rawfluxerr:3.3f}')
            self.logger.info(f'    Flux({band}):         {flux:3.3f} +/- {fluxerr:3.3f} uJy')               
            self.logger.info(f'    Mag({band}):          {mag:3.3f} +/- {magerr:3.3f} AB')
            self.logger.info(f'    Chi2({band}):         {chisq:3.3f}')
            self.logger.info(f'    BIC({band}):          {bic:3.3f}')
            self.logger.info(f'    Res. Chi({band}):     {self.chi_mu[row,i]:3.3f}+/-{self.chi_sig[row,i]:3.3f}')
            self.logger.info(f'    DAgostino K2({band}): {self.k2[row,i]:3.3f}')
            self.logger.info(f'    Zpt({band}):          {zpt:3.3f} AB')
            

            
            

        # # Just do the positions again - more straightforward to do it here than in interface.py
        # # Why do we need this!? Should we not be adding in extra X/Y if the force_position is turned off?
        # self.bcatalog[row]['x'] = self.bcatalog[row]['x'] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER + 1
        # self.bcatalog[row]['y'] = self.bcatalog[row]['y'] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER + 1
        self.bcatalog[row][f'X_MODEL'] = src.pos[0] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER
        self.bcatalog[row][f'Y_MODEL'] = src.pos[1] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER
        if self.wcs is not None:
            skyc = self.brick_wcs.all_pix2world(self.bcatalog[row][f'X_MODEL'] - self.mosaic_origin[0] + conf.BRICK_BUFFER, self.bcatalog[row][f'Y_MODEL'] - self.mosaic_origin[1] + conf.BRICK_BUFFER, 0)
            self.bcatalog[row][f'RA'] = skyc[0]
            self.bcatalog[row][f'DEC'] = skyc[1]
            self.logger.info(f"    Model position:      {src.pos[0]:6.6f}, {src.pos[1]:6.6f}")
            self.logger.info(f"    Sky Model RA, Dec:   {skyc[0]:6.6f} deg, {skyc[1]:6.6f} deg")

        # Is the source located outwidth the blob?
        xmax, ymax = np.shape(self.images[0])
        if (src.pos[0] < 0) | (src.pos[0] > ymax) | (src.pos[1] < 0) | (src.pos[1] > xmax):
            self.logger.warning('Source is located outwith blob limits!')
            valid_source = False

        if (not multiband_only):
            # Position information
            if multiband_model:
                mod_band = conf.MODELING_NICKNAME
            else:
                mod_band = self.bands[0]
            self.bcatalog[row]['x'] = self.bcatalog[row]['x'] + self.subvector[1] #+ self.mosaic_origin[1] - conf.BRICK_BUFFER
            self.bcatalog[row]['y'] = self.bcatalog[row]['y'] + self.subvector[0] #+ self.mosaic_origin[0] - conf.BRICK_BUFFER
            self.bcatalog[row][f'X_MODEL_{mod_band}'] = src.pos[0] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER
            self.bcatalog[row][f'Y_MODEL_{mod_band}'] = src.pos[1] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER
            self.logger.info(f"    Detection Position: {self.bcatalog[row]['x']:3.3f}, {self.bcatalog[row]['y']:3.3f}")
            self.logger.info(f"    Model Position:     {self.bcatalog[row][f'X_MODEL_{mod_band}']:3.3f}, {self.bcatalog[row][f'Y_MODEL_{mod_band}']:3.3f}")
            self.logger.debug(f"   Blob Origin:        {self.subvector[1]:3.3f}, {self.subvector[0]:3.3f}")
            self.logger.debug(f"   Mosaic Origin:      {self.mosaic_origin[1]:3.3f}, {self.mosaic_origin[0]:3.3f}")
            self.logger.debug(f"   Brick Buffer:       {conf.BRICK_BUFFER:3.3f}")
            self.bcatalog[row][f'XERR_MODEL_{mod_band}'] = np.sqrt(self.position_variance[row].pos.getParams()[0])
            self.bcatalog[row][f'YERR_MODEL_{mod_band}'] = np.sqrt(self.position_variance[row].pos.getParams()[1])
            if self.brick_wcs is not None:
                skyc = self.brick_wcs.all_pix2world(self.bcatalog[row][f'X_MODEL_{mod_band}'] - self.mosaic_origin[0] + conf.BRICK_BUFFER, self.bcatalog[row][f'Y_MODEL_{mod_band}'] - self.mosaic_origin[1] + conf.BRICK_BUFFER, 0)
                self.bcatalog[row][f'RA_{mod_band}'] = skyc[0]
                self.bcatalog[row][f'DEC_{mod_band}'] = skyc[1]
                self.logger.info(f"    Sky Model RA, Dec:   {skyc[0]:6.6f} deg, {skyc[1]:6.6f} deg")

            # Model Parameters
            self.bcatalog[row][f'SOLMODEL_{mod_band}'] = src.name
            self.bcatalog[row][f'VALID_SOURCE_{mod_band}'] = valid_source
            self.bcatalog[row]['N_BLOB'] = self.n_sources

            if src.name in ('ExpGalaxy', 'DevGalaxy'):
                self.bcatalog[row][f'REFF_{mod_band}'] = src.shape.logre
                self.bcatalog[row][f'REFF_ERR_{mod_band}'] = np.sqrt(self.parameter_variance[row].shape.getParams()[0])
                self.bcatalog[row][f'EE1_{mod_band}'] = src.shape.ee1
                self.bcatalog[row][f'EE2_{mod_band}'] = src.shape.ee2
                if (src.shape.e >= 1) | (src.shape.e <= -1):
                    # self.bcatalog[row][f'VALID_SOURCE'] = False
                    self.bcatalog[row][f'AB_{mod_band}'] = -99.0
                    self.logger.warning(f'Source has invalid ellipticity! (e = {src.shape.e:3.3f})')
                else:
                    self.bcatalog[row][f'AB_{mod_band}'] = (src.shape.e + 1) / (1 - src.shape.e)
                self.bcatalog[row][f'AB_ERR_{mod_band}'] = np.sqrt(self.parameter_variance[row].shape.getParams()[1])
                self.bcatalog[row][f'THETA_{mod_band}'] = np.rad2deg(src.shape.theta)
                self.bcatalog[row][f'THETA_ERR_{mod_band}'] = np.sqrt(self.parameter_variance[row].shape.getParams()[2])

                self.logger.info(f"    Reff:               {self.bcatalog[row][f'REFF_{mod_band}']:3.3f} +/- {self.bcatalog[row][f'REFF_ERR_{mod_band}']:3.3f}")
                self.logger.info(f"    a/b:                {self.bcatalog[row][f'AB_{mod_band}']:3.3f} +/- {self.bcatalog[row][f'AB_ERR_{mod_band}']:3.3f}")
                self.logger.info(f"    pa:                 {self.bcatalog[row][f'THETA_{mod_band}']:3.3f} +/- {self.bcatalog[row][f'THETA_ERR_{mod_band}']:3.3f}")

            elif src.name == 'FixedCompositeGalaxy':
                self.bcatalog[row][f'FRACDEV_{mod_band}'] = src.fracDev.getValue()
                self.bcatalog[row][f'EXP_REFF_{mod_band}'] = src.shapeExp.logre
                self.bcatalog[row][f'EXP_REFF_ERR_{mod_band}'] = np.sqrt(self.parameter_variance[row].shapeExp.getParams()[0])
                if (src.shapeExp.e >= 1) | (src.shapeExp.e <= -1):
                    # self.bcatalog[row][f'VALID_SOURCE'] = False
                    self.bcatalog[row][f'EXP_AB_{mod_band}'] = -99.0
                    self.logger.warning(f'Source has invalid ellipticity! (e = {src.shapeExp.e:3.3f})')
                else:
                    self.bcatalog[row][f'EXP_AB_{mod_band}'] = (src.shapeExp.e + 1) / (1 - src.shapeExp.e)
                self.bcatalog[row][f'EXP_AB_ERR_{mod_band}'] = np.sqrt(self.parameter_variance[row].shapeExp.getParams()[1])
                self.bcatalog[row][f'EXP_THETA_{mod_band}'] = np.rad2deg(src.shapeExp.theta)
                self.bcatalog[row][f'EXP_THETA_ERR_{mod_band}'] = np.sqrt(self.parameter_variance[row].shapeExp.getParams()[2])
                self.bcatalog[row][f'DEV_REFF_{mod_band}'] = src.shapeDev.logre
                self.bcatalog[row][f'DEV_REFF_ERR_{mod_band}'] = np.sqrt(self.parameter_variance[row].shapeDev.getParams()[0])
                if (src.shapeDev.e >= 1) | (src.shapeDev.e <= -1):
                    # self.bcatalog[row][f'VALID_SOURCE'] = False
                    self.bcatalog[row][f'DEV_AB_{mod_band}'] = -99.0
                    self.logger.warning(f'Source has invalid ellipticity! (e = {src.shapeDev.e:3.3f})')
                else:
                    self.bcatalog[row][f'DEV_AB_{mod_band}'] = (src.shapeDev.e + 1) / (1 - src.shapeDev.e)
                self.bcatalog[row][f'DEV_AB_ERR_{mod_band}'] = np.sqrt(self.parameter_variance[row].shapeDev.getParams()[1])
                self.bcatalog[row][f'DEV_THETA_{mod_band}'] = np.rad2deg(src.shapeDev.theta)
                self.bcatalog[row][f'DEV_THETA_ERR_{mod_band}'] = np.sqrt(self.parameter_variance[row].shapeDev.getParams()[2])
                # self.bcatalog[row][f'reff_err'] = np.sqrt(self.parameter_variance[row][0])
                # self.bcatalog[row][f'ab_err'] = np.sqrt(self.parameter_variance[row][1])
                # self.bcatalog[row][f'phi_err'] = np.sqrt(self.parameter_variance[row][2])

                self.bcatalog[row][f'EXP_EE1_{mod_band}'] = src.shapeExp.ee1
                self.bcatalog[row][f'EXP_EE2_{mod_band}'] = src.shapeExp.ee2
                self.bcatalog[row][f'DEV_EE1_{mod_band}'] = src.shapeDev.ee1
                self.bcatalog[row][f'DEV_EE2_{mod_band}'] = src.shapeDev.ee2


                if (src.shapeExp.e >= 1) | (src.shapeExp.e <= -1):
                    # self.bcatalog[row]['VALID_SOURCE'] = False
                    self.logger.warning(f'Source has invalid ellipticity! (e = {src.shapeExp.e:3.3f})')


                if (src.shapeDev.e >= 1) | (src.shapeDev.e <= -1):
                    # self.bcatalog[row]['VALID_SOURCE'] = False
                    self.logger.warning(f'Source has invalid ellipticity! (e = {src.shapeDev.e:3.3f})')


                self.logger.info(f"    Reff(Exp):          {self.bcatalog[row][f'EXP_REFF_{mod_band}']:3.3f} +/- {self.bcatalog[row][f'EXP_REFF_ERR_{mod_band}']:3.3f}")
                self.logger.info(f"    a/b (Exp):          {self.bcatalog[row][f'EXP_AB_{mod_band}']:3.3f} +/- {self.bcatalog[row][f'EXP_AB_ERR_{mod_band}']:3.3f}")
                self.logger.info(f"    pa  (Exp):          {self.bcatalog[row][f'EXP_THETA_{mod_band}']:3.3f} +/- {self.bcatalog[row][f'EXP_THETA_ERR_{mod_band}']:3.3f}")
                self.logger.info(f"    Reff(Dev):          {self.bcatalog[row][f'DEV_REFF_{mod_band}']:3.3f} +/- {self.bcatalog[row][f'DEV_REFF_ERR_{mod_band}']:3.3f}")
                self.logger.info(f"    a/b (Dev):          {self.bcatalog[row][f'DEV_AB_{mod_band}']:3.3f} +/- {self.bcatalog[row][f'DEV_AB_ERR_{mod_band}']:3.3f}")
                self.logger.info(f"    pa  (Dev):          {self.bcatalog[row][f'DEV_THETA_{mod_band}']:3.3f} +/- {self.bcatalog[row][f'DEV_THETA_ERR_{mod_band}']:3.3f}")

            elif src.name not in ('PointSource', 'SimpleGalaxy'): # last resort
                self.logger.warning(f"Source does not have a valid solution model!")
                valid_source = False
                self.bcatalog[row]['VALID_SOURCE'] = valid_source
