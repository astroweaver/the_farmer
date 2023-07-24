# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Class to handle mosaic subimages (i.e. bricks)

Known Issues
------------
None


"""

import os
import sys
import numpy as np

from astropy.table import Column
from astropy.io import fits
from scipy.ndimage import label, binary_dilation, binary_erosion, binary_fill_holes
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import zoom
from scipy import stats

from tractor import NCircularGaussianPSF, PixelizedPSF, PixelizedPsfEx, Image, Tractor, FluxesPhotoCal, NullWCS, ConstantSky, EllipseE, EllipseESoft, Fluxes, PixPos
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev
from tractor.pointsource import PointSource
from tractor.psf import HybridPixelizedPSF

from .utils import create_circular_mask, SimpleGalaxy
from .visualization import plot_blobmap, plot_detblob, plot_fblob
from .subimage import Subimage
from .blob import Blob
import config as conf

import logging

class Brick(Subimage):
    """TODO: docstring"""

    def __init__(self,
                 images,
                 weights=None,
                 masks=None,
                 psfmodels=None,
                 wcs=None,
                 bands=None,
                 buffer=conf.BRICK_BUFFER,
                 brick_id=-99,
                 ):
        """TODO: docstring"""

        self.logger = logging.getLogger('farmer.brick')

        self.wcs = wcs
        self.images = images
        self.weights = weights

        self.masks = masks
        self.psfmodels = psfmodels
        self.bands = np.array(bands)

        self.is_modeling = False

        self.generate_backgrounds()


        super().__init__()

        self._buffer = buffer
        self.brick_id = brick_id

        self.segmap = None
        self.blobmap = None
        self.shared_params = False

        self._buff_left = self._buffer
        self._buff_right = self.dims[0] - self._buffer
        self._buff_bottom = self._buffer
        self._buff_top = self.dims[1] - self._buffer

        x0 = int(((brick_id - 1) * conf.BRICK_WIDTH) % conf.MOSAIC_WIDTH)
        y0 = int(((brick_id - 1) * conf.BRICK_HEIGHT) / conf.MOSAIC_HEIGHT) * conf.BRICK_HEIGHT
        self.mosaic_origin = np.array([x0, y0])

        self.model_images = None
        self.residual_images = None
        self.nopsf_images = None
        self.auxhdu_path = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_AUXILLARY_MAPS.fits')

    @property
    def buffer(self):
        return self._buffer

    def cleanup(self):
        """TODO: docstring"""

        self.logger.info('Cleaning catalog...')

        self.clean_segmap()

        if conf.POST_DETECTION_MASK:
            self.clean_segmap_withmask()

        self.clean_catalog()

        self.dilate()

        self.relabel()

        self.clean_blobmap()

        self.add_ids()

        self.logger.info('Finished cleaning catalog.')

    def new_segmap(self):
        """Obtain a new segmap with only pixels within the existing segmap with SNR >= target"""
        self.logger.info(f'Eroding the segmap to a new target SNR = {conf.SEGMAP_SNR}')
        segmask_init = self.segmap.copy()
        segmap = self.segmap.copy()
        segmask_init[self.segmap > 0] = 1
        segmask = segmask_init*0

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(ncols=2, figsize=(20,10))
        ax[0].imshow(segmask_init)
        ax[1].imshow(segmap)
        ax[0].scatter(self.catalog['x'], self.catalog['y'], s=1)
        plt.savefig(conf.PLOT_DIR + '/preerosion.pdf')
        plt.close('all')

        self.logger.info(f'Pre-erosion fraction: {np.sum(segmask_init)/segmask_init.size:3.3f}')
        snrmap = self.images[0] * np.sqrt(self.weights[0])
        segmask[(segmask_init == 1) & (snrmap >= conf.SEGMAP_SNR)] = 1

        fig, ax = plt.subplots(ncols=2, figsize=(20,10))
        ax[0].imshow(segmask)
        ax[1].imshow(segmap)
        ax[0].scatter(self.catalog['x'], self.catalog['y'], s=1)
        plt.savefig(conf.PLOT_DIR + '/inter1erosion.pdf')
        plt.close('all')

        # Try just cleaning it with an erosion
        radius = 2
        struct2 = create_circular_mask(2*abs(radius), 2*abs(radius), radius=abs(radius))
        segmask = binary_erosion(segmask, structure=struct2)

        fig, ax = plt.subplots(ncols=2, figsize=(20,10))
        ax[0].imshow(segmask)
        ax[1].imshow(segmap)
        ax[0].scatter(self.catalog['x'], self.catalog['y'], s=1)
        plt.savefig(conf.PLOT_DIR + '/inter2erosion.pdf')
        plt.close('all')

        # But save the small guys!
        uniq_id = np.unique(self.segmap)[1:]
        keep = np.array([np.sum(self.segmap==i) <= conf.SEGMAP_MINAREA for i in uniq_id])
        # add in all the small ones that could get totally removed
        for idx in uniq_id[keep]:
            segmask[self.segmap==idx] = 1

        segmap[segmask==0] == 0

        self.logger.info(f'Post-erosion fraction: {np.sum(segmask)/segmask.size:3.3f}')

        fig, ax = plt.subplots(ncols=2, figsize=(20,10))
        ax[0].imshow(segmask)
        ax[1].imshow(segmap)
        ax[0].scatter(self.catalog['x'], self.catalog['y'], s=1)
        plt.savefig(conf.PLOT_DIR + '/posterosion.pdf')
        plt.close('all')

        # make sure every source sits in a segment -- this was not meant to remove any!
        for idx, (x, y) in enumerate(self.catalog['x', 'y']):
            if segmap[int(y), int(x)] != idx+1:
                raise RuntimeError(f'Source #{idx+1} does not have a segment!') 


        self.segmask = segmask

        # apply to segmap
        self.segmap = segmap

    def clean_segmap(self):
        """TODO: docstring"""
        self.logger.info('Removing sources in the border region')
        coords = np.array([self.catalog['x'], self.catalog['y']]).T
        self._allowed_sources = (coords[:,0] > self._buff_left) & (coords[:,0] < self._buff_right )\
                        & (coords[:,1] > self._buff_bottom) & (coords[:,1] < self._buff_top)
        
        idx = np.where(~self._allowed_sources)[0]
        for i in idx:
            self.segmap[self.segmap == i+1] = 0

    def clean_segmap_withmask(self):
        """TODO: docstring"""
        self.logger.info('Removing sources in the mask')
        coords = np.array([self.catalog['x'], self.catalog['y']]).T
        # self._allowed_sources = ...$
        for i, coord in enumerate(coords):
            iy, ix = np.round(coord, 0).astype(int)
            if self.masks[0][ix, iy]:
                self._allowed_sources[i] = False

        idx = np.where(~self._allowed_sources)[0]
        for i in idx:
            self.segmap[self.segmap == i+1] = 0

    def clean_catalog(self):
        """TODO: docstring"""
        sid_col = np.arange(1, self.n_sources+1, dtype=int)
        self.catalog.add_column(Column(sid_col.astype(int), name='source_id'), 0)
        self.catalog = self.catalog[self._allowed_sources]
        self.n_sources = len(self.catalog)
        n_cleaned = np.sum(~self._allowed_sources)
        pc = n_cleaned / len(self._allowed_sources)
        self.logger.info(f'Cleaned out {n_cleaned} sources ({100*pc:3.3f}%)')

    def add_columns(self, modeling=True, multiband_model=False, modbrick_name=conf.MODELING_NICKNAME):
        """TODO: docstring"""
        filler = np.zeros(len(self.catalog))
        filler5 = np.zeros((len(self.catalog), 5))
        boolfiller = np.zeros(len(self.catalog), dtype=bool)
        if 'N_BLOB' not in self.catalog.colnames:
            self.catalog.add_column(Column(-99*np.ones(len(self.catalog), dtype=int), name='N_BLOB'))
        if 'BEST_MODEL_BAND' not in self.catalog.colnames:
            self.catalog.add_column(Column(np.zeros(len(self.catalog), dtype='S20'), name='BEST_MODEL_BAND'))
        if 'X_MODEL' not in self.catalog.colnames:
            self.catalog.add_column(Column(-99*np.ones(len(self.catalog), dtype=float), name='X_MODEL'))
        if 'Y_MODEL' not in self.catalog.colnames:
            self.catalog.add_column(Column(-99*np.ones(len(self.catalog), dtype=float), name='Y_MODEL'))
        if 'RA' not in self.catalog.colnames:
            self.catalog.add_column(Column(-99*np.ones(len(self.catalog), dtype=float), name='RA'))
        if 'DEC' not in self.catalog.colnames:
            self.catalog.add_column(Column(-99*np.ones(len(self.catalog), dtype=float), name='DEC'))
        if 'VALID_SOURCE' not in self.catalog.colnames:
            self.catalog.add_column(Column(np.zeros(len(self.catalog), dtype=bool), name='VALID_SOURCE'))
        for colname in self.bands:
            colname = colname.replace(' ', '_')
            self.logger.debug(f'Adding columns to catalog for {colname}')
            try:
                self.catalog.add_column(Column(filler, name=f'MAG_{colname}'))
                self.catalog.add_column(Column(filler, name=f'MAGERR_{colname}'))
                self.catalog.add_column(Column(filler, name=f'RAWFLUX_{colname}'))
                self.catalog.add_column(Column(filler, name=f'RAWFLUXERR_{colname}'))
                self.catalog.add_column(Column(filler, name=f'FLUX_{colname}'))
                self.catalog.add_column(Column(filler, name=f'FLUXERR_{colname}'))
                self.catalog.add_column(Column(filler, name=f'RAWFLUXCORE_{colname}'))
                self.catalog.add_column(Column(filler, name=f'RAWFLUXCOREERR_{colname}'))
                self.catalog.add_column(Column(filler, name=f'CHISQ_{colname}'))
                self.catalog.add_column(Column(filler, name=f'BIC_{colname}'))
                self.catalog.add_column(Column(filler, name=f'SNR_{colname}'))
                self.catalog.add_column(Column(filler, name=f'NORM_{colname}'))
                self.catalog.add_column(Column(filler, name=f'N_CONVERGE_{colname}'))
                self.catalog.add_column(Column(filler, name=f'CHI_MU_{colname}'))
                self.catalog.add_column(Column(filler, name=f'CHI_SIG_{colname}'))
                self.catalog.add_column(Column(filler, name=f'CHI_K2_{colname}'))
                self.catalog.add_column(Column(filler5, name=f'CHI_PERCENT_{colname}'))
                self.catalog.add_column(Column(filler, name=f'SEG_RAWFLUX_{colname}'))
                self.catalog.add_column(Column(filler, name=f'CHISQ_NOMODEL_{colname}'))
                self.catalog.add_column(Column(boolfiller, name=f'VALID_SOURCE_{colname}'))
                self.catalog.add_column(Column(filler, name=f'RAWDIRECTFLUXERR_{colname}'))
                self.catalog.add_column(Column(filler, name=f'DIRECTFLUXERR_{colname}'))
                # self.catalog.add_column(Column(np.zeros(len(self.catalog), dtype='S20'), name=f'SOLMODEL_{colname}'))
                if modeling | (~modeling & (not conf.FREEZE_FORCED_POSITION)):
                    self.catalog.add_column(Column(filler, name=f'X_MODEL_{colname}'))
                    self.catalog.add_column(Column(filler, name=f'Y_MODEL_{colname}'))
                    self.catalog.add_column(Column(filler, name=f'XERR_MODEL_{colname}'))
                    self.catalog.add_column(Column(filler, name=f'YERR_MODEL_{colname}'))
                    self.catalog.add_column(Column(filler, name=f'RA_{colname}'))
                    self.catalog.add_column(Column(filler, name=f'DEC_{colname}'))

            except:
                self.logger.debug(f'Columns already exist for {colname}')
            if (modeling & (not multiband_model)) | (not conf.FREEZE_FORCED_SHAPE):
                if modeling & (colname != conf.MODELING_NICKNAME):
                    self.logger.debug(f'Adding model columns to catalog. (Using MODELING band)')
                    self.catalog.add_column(Column(boolfiller, name=f'VALID_SOURCE_{conf.MODELING_NICKNAME}'))

                    for colname_fill in [f'{colext}_{colname}' for colext in ('X_MODEL', 'Y_MODEL', 'XERR_MODEL', 'YERR_MODEL', 'RA', 'DEC')]:
                        try:
                            self.catalog.add_column(Column(filler, name=colname_fill))
                        except:
                            pass
                try:
                    self.catalog.add_column(Column(np.zeros(len(self.catalog), dtype='S20'), name=f'SOLMODEL_{colname}'))
                except:
                    pass  # This is a bit silly, but it gets around the issue of an input catalog already having these columns...
                
                for colname_fill in [f'{colext}_{colname}' for colext in ('REFF', 'REFF_ERR', 'EE1', 'EE2', 'AB', 'AB_ERR', 'THETA', 'THETA_ERR',
                            'N', 'N_ERR',
                            'FRACDEV', 'EXP_REFF', 'EXP_REFF_ERR', 'EXP_EE1', 'EXP_EE2', 'EXP_AB', 'EXP_AB_ERR', 'EXP_THETA', 'EXP_THETA_ERR', 
                            'DEV_REFF', 'DEV_REFF_ERR', 'DEV_EE1', 'DEV_EE2', 'DEV_AB', 'DEV_AB_ERR', 'DEV_THETA', 'DEV_THETA_ERR' )]:
                    try:
                        self.catalog.add_column(Column(-99. * (1.+filler), name=colname_fill))
                    except:
                        pass

        if multiband_model:
            colname = modbrick_name
            try:
                self.logger.debug(f'Adding model columns to catalog. (Using MULTIBAND band(s))')
                for colname_fill in [f'{colext}_{colname}' for colext in ('X_MODEL', 'Y_MODEL', 'XERR_MODEL', 'YERR_MODEL', 'RA', 'DEC')]:
                    self.catalog.add_column(Column(filler, name=colname_fill))
                self.catalog.add_column(Column(np.zeros(len(self.catalog), dtype='S20'), name=f'SOLMODEL_{colname}'))
                self.catalog.add_column(Column(boolfiller, name=f'VALID_SOURCE_{conf.MODELING_NICKNAME}'))

                # self.catalog.add_column(Column(np.zeros(len(self.catalog), dtype=bool), name=f'VALID_SOURCE_{colname}'))
                
                for colname_fill in [f'{colext}_{colname}' for colext in ('REFF', 'REFF_ERR', 'EE1', 'EE2', 'AB', 'AB_ERR', 'THETA', 'THETA_ERR',
                            'N', 'N_ERR',
                            'FRACDEV', 'EXP_REFF', 'EXP_REFF_ERR', 'EXP_EE1', 'EXP_EE2', 'EXP_AB', 'EXP_AB_ERR', 'EXP_THETA', 'EXP_THETA_ERR', 
                            'DEV_REFF', 'DEV_REFF_ERR', 'DEV_EE1', 'DEV_EE2', 'DEV_AB', 'DEV_AB_ERR', 'DEV_THETA', 'DEV_THETA_ERR' )]:
                    self.catalog.add_column(Column(-99. * (1.+filler), name=colname_fill))
            except:
                self.logger.debug(f'Model columns already exist')

    def dilate(self, radius=conf.DILATION_RADIUS, fill_holes=True):
        """TODO: docstring"""
        # Make binary
        segmask = self.segmap.copy()  # I don't see how we can get around this.
        segmask[segmask != 0] = 1
        segmask_init = segmask.copy()

        if radius == 0:
            self.logger.info('No dilation applied.')
            # exit early!
            if fill_holes:
                segmask = binary_fill_holes(segmask).astype(int)
            self.segmask = segmask
            return segmask
        else:
            # Dilate
            struct2 = create_circular_mask(2*abs(radius), 2*abs(radius), radius=abs(radius))
            if radius > 0:
                self.logger.info(f'Dilating segments with radius of {radius}px')
                segmask = binary_dilation(segmask, structure=struct2).astype(int)
            else:
                self.logger.info(f'Eroding segments with radius of {radius}px')
                segmask = binary_erosion(segmask, structure=struct2).astype(int)

                uniq_sm = np.unique(self.segmap)[1:]
                keep = np.array([np.sum(self.segmap==i) < 50 for i in uniq_sm])
                self.logger.info(f'Found {np.sum(keep)} segments smaller than {conf.SEGMAP_MINAREA} px2 to re-instate.')
                for i in uniq_sm[keep]:
                    segmask[self.segmap == i] = 1

            if fill_holes:
                segmask = binary_fill_holes(segmask).astype(int)

            self.segmask = segmask
            return segmask

    def relabel(self):
        self.blobmap, self.n_blobs = label(self.segmask)
        self.logger.info(f'Found {self.n_blobs} blobs')

        return self.blobmap, self.n_blobs

    def clean_blobmap(self):
        """Remove regions without sources"""

        if conf.DILATION_RADIUS >= 1:
            return

        x, y = np.array(self.catalog['x']).astype(int), np.array(self.catalog['y']).astype(int)

        # Check that no sources were cut out!
        self.logger.info('Checking that all sources have a segment')
        sxy = []
        trip = False
        keep = np.ones(len(self.catalog), dtype=bool)
        for i, sid in enumerate(self.catalog['source_id']):
            px, py = x[i], y[i]
            segval = self.segmap[py, px]
            if segval != sid:
                sxy.append((px, py))
                self.logger.warning(f'Source {sid} is missing a segment, sits on {segval} at ({px}, {py}).')
                trip = True
                keep[i] = False
        
        self.logger.info('Cleaning blobmap')
        n_rm = 0
        pix_rm = 0
        uniq_bm = np.unique(self.blobmap)
        
        for i in uniq_bm:
            if i == 0:
                continue
            mask = self.blobmap==i
            n_px = np.sum(mask)

            if (mask[y, x] == 1).any(): # is there a centroid in the blob?
                continue
            else:
                self.blobmap[mask] = 0
                sids = np.unique(self.segmap[mask])
                sids = sids[sids!=0]
                self.logger.debug(f'   Removed blob #{i} with size {np.sum(mask)} ({sids})')
                n_rm += 1
                pix_rm +=1
                
                # if np.sum(mask) > 50:
                #     import matplotlib.pyplot as plt
                #     plt.figure(figsize=(20,20))
                #     plt.imshow(mask)
                #     plt.scatter(x, y, s=1)
                #     plt.savefig(conf.PLOT_DIR + f'/checkplot_{i}')
                #     plt.close('all')
            
        self.logger.info(f'Cleaned out {n_rm}/{len(uniq_bm)} blobs ({100*pix_rm/mask.size:2.3f}% by area)')
        
        self.segmap[self.blobmap == 0] = 0 # blobs are smaller than segs, so just cut away segs...
        self.segmask[self.blobmap == 0] = 0

        # Check that no sources were cut out!
        sxy = []
        trip = False
        keep = np.ones(len(self.catalog), dtype=bool)
        for i, sid in enumerate(self.catalog['source_id']):
            px, py = x[i], y[i]
            segval = self.segmap[py, px]
            if segval != sid:
                sxy.append((px, py))
                self.logger.warning(f'Source {sid} is missing a segment, sits on {segval} at ({px}, {py}).')
                trip = True
                keep[i] = False

        self.catalog = self.catalog[keep]
        self.n_sources = np.sum(keep)

        if trip:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(40, 40))
            plt.imshow(self.segmask)
            sxy = np.array(sxy)
            plt.scatter(sxy.T[0], sxy.T[1], s=1)
            plt.savefig(conf.PLOT_DIR + '/checkmissing.pdf')
            # raise RuntimeError('Will not continue!')

        self.relabel() # clean it up again.

    def add_ids(self):
        """TODO: docstring. rename sid and bid throughout"""
        brick_col = float(self.brick_id) * np.ones(self.n_sources, dtype=int)
        self.catalog.add_column(Column(brick_col.astype(int), name='brick_id'), 1)

        blob_col = np.array([np.unique(self.blobmap[self.segmap == sid])[0] for sid in self.catalog['source_id']])
        self.catalog.add_column(Column(blob_col.astype(int), name='blob_id'), 1)

        nblob_col = -99*np.ones(len(self.catalog))
        for bid in np.unique(self.catalog['blob_id']):
            nbid = np.sum(self.catalog['blob_id'] == bid)
            nblob_col[self.catalog['blob_id'] == bid] = nbid

        self.catalog.add_column(Column(nblob_col.astype(int), name='N_BLOB'), 1)

    def make_blob(self, blob_id):

        if blob_id < 1:
            raise ValueError('Blob id must be greater than 0.')

        blob = Blob(self, blob_id)

        return blob

    def run_weights(self, is_detection=False):
        
        use_rms_weights = conf.USE_RMS_WEIGHTS
        scale_weights = conf.SCALE_WEIGHTS
        median_weights = conf.MEDIAN_WEIGHTS
        for i, band in enumerate(self.bands):
            if band.startswith(conf.MODELING_NICKNAME):
                band = band[len(conf.MODELING_NICKNAME)+1:]
            if (band not in use_rms_weights) & (band not in scale_weights) & (band not in median_weights):
                continue
            self.logger.info(f'Performing any weight corrections for {band}')
            rms = self.background_rms_images[i]
            if (conf.USE_MASKED_SEP_RMS | conf.USE_MASKED_DIRECT_RMS) & (not is_detection):
                self.logger.info('Computing the masked RMS maps')
                if not hasattr(self, 'segmask'):
                    raise ValueError('Brick is missing a segment mask!')
                self.subtract_background(idx=i, use_masked=conf.USE_MASKED_SEP_RMS, apply=False, use_direct_median=conf.USE_MASKED_DIRECT_RMS) # generates advanced RMS even if its not applied!
                # So now the background_rms_images are updated
                if conf.USE_MASKED_DIRECT_RMS:
                    self.logger.info('Setting the RMS to the direct masked measurement!')
                    rms = self.masked_std[i]
            if band in use_rms_weights:
                self.logger.info('Converting the RMS image to use as a weight!')
                ok_weights = rms > 0
                self.weights[i] = np.zeros_like(rms)
                self.weights[i][ok_weights] = 1./(rms[ok_weights]**2)
            elif band in median_weights:
                self.logger.info('Using the RMS image to scale the weight!')
                wgt = self.weights[i]
                median_wrms = np.median(1/rms**2)
                median_wgt = np.median(wgt[wgt>0])
                self.logger.debug(f' Median rms weight: {median_wrms:3.6f}, and in RMS: {1./np.sqrt(median_wrms):3.6f}')
                self.logger.debug(f' Median input weight: {median_wgt:3.6f}, and in RMS: {1./np.sqrt(median_wgt):3.6f}')
                self.logger.info(f' Will flatten non-zero weights to {median_wgt:6.6f}.')
                self.weights[i] = median_wgt
            elif band in scale_weights:
                self.logger.info('Using the RMS image to scale the weight!')
                wgt = self.weights[i]
                median_wrms = np.median(1/rms**2)
                median_wgt = np.median(wgt[wgt>0])
                self.logger.debug(f' Median rms weight: {median_wrms:3.6f}, and in RMS: {1./np.sqrt(median_wrms):3.6f}')
                self.logger.debug(f' Median input weight: {median_wgt:3.6f}, and in RMS: {1./np.sqrt(median_wgt):3.6f}')
                self.logger.info(f' Will apply a factor of {median_wrms/median_wgt:6.6f} to scale input weights.')
                self.weights[i] *= median_wrms / median_wgt

    def run_background(self):
        # Just stash this here. 
        for i, band in enumerate(self.bands):
            idx = self._band2idx(band, bands=self.bands)
            if band.startswith(conf.MODELING_NICKNAME):
                band = band[len(conf.MODELING_NICKNAME)+1:]
            

            if band in conf.SUBTRACT_BACKGROUND:
                self.subtract_background(idx=idx, flat=conf.USE_FLAT, use_masked=(conf.SUBTRACT_BACKGROUND_WITH_MASK|conf.SUBTRACT_BACKGROUND_WITH_DIRECT_MEDIAN), use_direct_median=conf.SUBTRACT_BACKGROUND_WITH_DIRECT_MEDIAN)
                self.logger.debug(f'Subtracted background (flat={conf.USE_FLAT}, masked={conf.SUBTRACT_BACKGROUND_WITH_MASK}, used_direct_median={conf.SUBTRACT_BACKGROUND_WITH_DIRECT_MEDIAN})')

            elif band in conf.MANUAL_BACKGROUND.keys():
                self.images[idx] -= conf.MANUAL_BACKGROUND[band]
                self.logger.debug(f'Subtracted background manually ({conf.MANUAL_BACKGROUND[band]})')
            
            else:
                self.logger.warning(f'No background is subtracted for {band}!')

    def make_model_image(self, catalog, include_chi=True, include_nopsf=False, save=True, use_band_position=False, use_band_shape=False, modeling=False):

        self.model_images = np.zeros(shape=(self.n_bands, np.shape(self.images[0])[0], np.shape(self.images[0])[1]))
        self.chisq_images = np.zeros(shape=(self.n_bands, np.shape(self.images[0])[0], np.shape(self.images[0])[1]))

        sbands = np.zeros_like(self.bands)
        for b, band in enumerate(self.bands):
            if band.startswith(conf.MODELING_NICKNAME):
                sbands[b] = band[len(conf.MODELING_NICKNAME)+1:]
            else:
                sbands[b] = band
        

        self.bands = np.array(self.bands) # The bands attribute is being modfified somewhere to make it BACK into a list. Why? Dunno. Just don't change this.

        # Figure out which bands are to be run with which setup.
        # The output should be an attribute containing all the model images, in the right order for self.bands
        self.logger.info(f'Making Model images for {self.bands}')
        if np.in1d(sbands, conf.BANDS).any() & ~np.in1d(sbands, conf.PRFMAP_PSF).any() & ~np.in1d(sbands, conf.PSFGRID).any():
            self.logger.info('Making Models for PSF images')
            self.make_model_image_psf(catalog, include_chi=include_chi, include_nopsf=include_nopsf, save=save, use_band_position=use_band_position, use_band_shape=use_band_shape, modeling=modeling)
        if np.in1d(sbands, conf.PSFGRID).any():
            self.logger.info('Making Models for GRID PSF images')
            self.make_model_image_gridpsf(catalog, include_chi=include_chi, include_nopsf=include_nopsf, save=save, use_band_position=use_band_position, use_band_shape=use_band_shape, modeling=modeling)
        if np.in1d(sbands, conf.PRFMAP_PSF).any():
            self.logger.info('Making Models for PRF images')
            self.make_model_image_prfmap(catalog, include_chi=include_chi, include_nopsf=include_nopsf, save=save, use_band_position=use_band_position, use_band_shape=use_band_shape, modeling=modeling)


    def make_model_image_prfmap(self, catalog, include_chi=True, include_nopsf=False, save=True, use_band_position=False, use_band_shape=False, modeling=False):

        # Which indices to use?
        idx = []
        for i, b in enumerate(self.bands):
            if modeling:
                b = b[len(conf.MODELING_NICKNAME)+1:]
            if b in conf.PRFMAP_PSF:
                idx.append(i)
        # # Make Images

        self.logger.info(f'Making model images for {self.bands[idx]}')
        self.model_mask = np.zeros(len(catalog), dtype=bool)

        # loop over blobs
        for i, bid in enumerate(np.unique(catalog['blob_id'])):
            blob = Blob(self, bid)
            
            self.logger.info(f'Incoporating blob {blob.blob_id}')

            timages = np.zeros(shape=len(self.bands[idx]), dtype=object)

            # Loop over bands:
            prf_notfound = 0
            nband_added = 0
            for j, band in enumerate(self.bands[idx]):

                remove_background_psf = False
                if band in conf.RMBACK_PSF:
                    remove_background_psf = True

                ### construct PRFs
                self.logger.debug('Adopting a PRF from file.')
                # find nearest prf to blob center
                prftab_coords, prftab_idx = self.psfmodels[j]

                if conf.USE_BLOB_IDGRID:
                    prf_idx = bid
                else:
                    minsep_idx, minsep, __ = blob.blob_coords.match_to_catalog_sky(prftab_coords)
                    prf_idx = prftab_idx[minsep_idx]
                    self.logger.debug(f'Nearest PRF sample: {prf_idx} ({minsep[0].to(u.arcsec).value:2.2f}")')

                    if minsep > conf.PRFMAP_MAXSEP*u.arcsec:
                        self.logger.error(f'Separation ({minsep.to(u.arcsec)}) exceeds maximum {conf.PRFMAP_MAXSEP}!')
                        continue


                # open id file
                pad_prf_idx = ((6 - len(str(prf_idx))) * "0") + str(prf_idx)
                path_prffile = os.path.join(conf.PRFMAP_DIR[band], f'{conf.PRFMAP_FILENAME}{pad_prf_idx}.fits')
                if not os.path.exists(path_prffile):
                    self.logger.error(f'PRF file has not been found! ({path_prffile}')
                    prf_notfound += 1
                    if prf_notfound == len(self.bands[idx]):
                        self.logger.error(f'No PRF files found for any bands in selection!')
                        break
                    else:
                        continue

                hdul = fits.open(path_prffile)
                from scipy.ndimage.interpolation import rotate
                img = hdul[0].data
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
                    psfmodel.img[(psfmodel.img < 0) | np.isnan(psfmodel.img)] = 0

                if conf.PSF_RADIUS > 0:
                    psf_rad_pix = int(conf.PSF_RADIUS / conf.PIXEL_SCALE)
                    self.logger.debug(f'Clipping PRF ({psf_rad_pix}px radius)')
                    psfmodel.img = psfmodel.img[int(pw/2.-psf_rad_pix):int(pw/2+psf_rad_pix), int(ph/2.-psf_rad_pix):int(ph/2+psf_rad_pix)]
                    self.logger.debug(f'New shape: {np.shape(psfmodel.img)}')

                if conf.NORMALIZE_PSF & (not conf.FORCE_GAUSSIAN_PSF):
                    norm = psfmodel.img.sum()
                    self.logger.debug(f'Normalizing PRF (sum = {norm:4.4f})')
                    psfmodel.img /= norm # HACK -- force normalization to 1       
                    self.logger.debug(f'PRF has been normalized. (sum = {psfmodel.img.sum():4.4f})') 


                psfmodel.img = psfmodel.img.astype('float32') # This may be redundant, but it's super important!

                # make Image
                timages[j] = Image(data=np.zeros_like(self.images[0]),
                            invvar=np.ones_like(self.images[0]),
                            psf=psfmodel,
                            wcs=NullWCS(),
                            photocal=FluxesPhotoCal(band),
                            sky=ConstantSky(0.),
                            name=band)

                nband_added += 1

            if nband_added != len(self.bands[idx]):
                self.logger.error(f'Not all bands added for blob #{bid}! Skipping!')
                continue

            self.timages = timages

            # Construct models
            self.model_catalog = np.ones(len(blob.bcatalog), dtype=object)
            
            bad_blob = True
            oksource = np.zeros(blob.n_sources, dtype=bool)
            for m, src in enumerate(blob.bcatalog):

                mm_idx = np.argwhere(catalog['source_id'] == src['source_id'])[0]
                
                if (blob.bcatalog['BEST_MODEL_BAND'] == '').all():
                    self.logger.debug('No best models chosen yet.')
                    best_band = self.bands[0]
                    if not src[f'VALID_SOURCE_{self.bands[0]}']:
                        self.logger.debug('Source does not have a valid model.')
                        continue
                else:
                    best_band = src['BEST_MODEL_BAND']

                    if not src[f'VALID_SOURCE_{best_band}']:
                        self.logger.debug('Source does not have a valid model.')
                        continue

                if modeling:
                    raw_fluxes = np.array([src[f'RAWFLUX_{conf.MODELING_NICKNAME}_{band}'] for band in self.bands[idx]])
                else:
                    raw_fluxes = np.array([src[f'RAWFLUX_{band}'] for band in self.bands[idx]])

                rejected = False
                if conf.RESIDUAL_CHISQ_REJECTION is not None:
                    for j, band in enumerate(self.bands[idx]):
                        if modeling:
                            chisq_band = src[f'CHISQ_MODELING_{band}']
                        else:
                            chisq_band = src[f'CHISQ_{band}']
                        if chisq_band > conf.RESIDUAL_CHISQ_REJECTION:
                            raw_fluxes[j] = 0.0
                            self.logger.debug(f'Source has too large chisq in {band}. ({chisq_band:3.3f}) > {conf.RESIDUAL_CHISQ_REJECTION})')
                    if (raw_fluxes < 0.0).all():
                        self.logger.debug('Source has too large chisq in all bands. Rejecting!')
                        continue
                if conf.RESIDUAL_NEGFLUX_REJECTION:
                    if (raw_fluxes < 0.0).all():
                        self.logger.debug('Source has negative flux in all bands. Rejecting!')
                        continue
                    elif (raw_fluxes < 0.0).any():
                        raw_fluxes[raw_fluxes < 0.0] = 0.0
                        self.logger.debug('Source has negative flux in some bands.')

                if (conf.RESIDUAL_AB_REJECTION is not None) & (src[f'SOLMODEL_{conf.MODELING_NICKNAME}'] not in ('PointSource', 'SimpleGalaxy')):  # HACK -- does NOT apply to unfixed shapes!
                
                    if src[f'SOLMODEL_{conf.MODELING_NICKNAME}'] in ('ExpGalaxy', 'DevGalaxy'):
                        ab = np.array([src[f'AB_{conf.MODELING_NICKNAME}'] for band in self.bands[idx]])
                        if (ab > conf.RESIDUAL_AB_REJECTION).all() | (ab <= 0).all():

                            self.logger.debug('Source has exessive a/b in all bands. Rejecting!')
                        elif (ab > conf.RESIDUAL_AB_REJECTION).any() | (ab <= 0).any():
                            raw_fluxes[ab > conf.RESIDUAL_AB_REJECTION] = 0.0
                            raw_fluxes[ab <= 0] = 0.0
                            self.logger.debug('Source has exessive a/b  in some bands.')
                    else:
                        ab_exp = np.array([src[f'EXP_AB_{conf.MODELING_NICKNAME}'] for band in self.bands[idx]])
                        if (ab_exp > conf.RESIDUAL_AB_REJECTION).all() | (ab_exp <= 0).all():

                            self.logger.debug('Source has exessive exp a/b in all bands. Rejecting!')
                        elif (ab_exp > conf.RESIDUAL_AB_REJECTION).any() | (ab_exp <= 0).any():
                            raw_fluxes[ab_exp > conf.RESIDUAL_AB_REJECTION] = 0.0
                            raw_fluxes[ab_exp <= 0] = 0.0
                            self.logger.debug('Source has exessive exp a/b  in some bands.')

                        ab_dev = np.array([src[f'DEV_AB_{conf.MODELING_NICKNAME}'] for band in self.bands[idx]])
                        if (ab_dev > conf.RESIDUAL_AB_REJECTION).all() | (ab_dev <= 0).all():

                            self.logger.debug('Source has exessive dev a/b in all bands. Rejecting!')
                        elif (ab_dev > conf.RESIDUAL_AB_REJECTION).any() | (ab_dev <= 0).any():
                            raw_fluxes[ab_dev > conf.RESIDUAL_AB_REJECTION] = 0.0
                            raw_fluxes[ab_dev <= 0] = 0.0
                            self.logger.debug('Source has exessive dev a/b  in some bands.')

                if use_band_position:
                    bx_model = src[f'X_MODEL_{band}'] - self.mosaic_origin[1] + conf.BRICK_BUFFER 
                    by_model = src[f'Y_MODEL_{band}'] - self.mosaic_origin[0] + conf.BRICK_BUFFER
                else:
                    bx_model = src[f'X_MODEL_{best_band}'] - self.mosaic_origin[1] + conf.BRICK_BUFFER 
                    by_model = src[f'Y_MODEL_{best_band}'] - self.mosaic_origin[0] + conf.BRICK_BUFFER

                position = PixPos(bx_model, by_model)
                flux = Fluxes(**dict(zip(self.bands[idx], raw_fluxes)), order=self.bands[idx]) # IMAGES ARE IN NATIVE ZPT, USE RAWFLUXES!

                if src[f'SOLMODEL_{best_band}'] == "PointSource":
                    self.model_catalog[m] = PointSource(position, flux)
                    self.model_catalog[m].name = 'PointSource' # HACK to get around Dustin's HACK.
                elif src[f'SOLMODEL_{best_band}'] == "SimpleGalaxy":
                    self.model_catalog[m] = SimpleGalaxy(position, flux)
                elif src[f'SOLMODEL_{best_band}'] == "ExpGalaxy":
                    if use_band_shape:
                        shape = EllipseESoft(src[f'REFF_{band}'], src[f'EE1_{band}'], src[f'EE2_{band}'])
                    else:
                        shape = EllipseESoft(src[f'REFF_{best_band}'], src[f'EE1_{best_band}'], src[f'EE2_{best_band}'])
                    self.model_catalog[m] = ExpGalaxy(position, flux, shape)
                elif src[f'SOLMODEL_{best_band}'] == "DevGalaxy":
                    if use_band_shape:
                        shape = EllipseESoft(src[f'REFF_{band}'], src[f'EE1_{band}'], src[f'EE2_{band}'])
                    else:
                        shape = EllipseESoft(src[f'REFF_{best_band}'], src[f'EE1_{best_band}'], src[f'EE2_{best_band}'])                
                    self.model_catalog[m] = DevGalaxy(position, flux, shape)
                elif src[f'SOLMODEL_{best_band}'] == "FixedCompositeGalaxy":
                    if use_band_shape:
                        shape_exp = EllipseESoft(src[f'EXP_REFF_{band}'], src[f'EXP_EE1_{band}'], src[f'EXP_EE2_{band}'])
                        shape_dev = EllipseESoft(src[f'DEV_REFF_{band}'], src[f'DEV_EE1_{band}'], src[f'DEV_EE2_{band}'])
                    else:
                        shape_exp = EllipseESoft(src[f'EXP_REFF_{best_band}'], src[f'EXP_EE1_{best_band}'], src[f'EXP_EE2_{best_band}'])
                        shape_dev = EllipseESoft(src[f'DEV_REFF_{best_band}'], src[f'DEV_EE1_{best_band}'], src[f'DEV_EE2_{best_band}'])
                    
                    self.model_catalog[m] = FixedCompositeGalaxy(
                                                    position, flux,
                                                    SoftenedFracDev(src[f'FRACDEV_{best_band}']),
                                                    shape_exp, shape_dev)
                else:
                    self.logger.warning(f'Source #{src["source_id"]}: has no solution model at {position}')
                    continue

                self.logger.debug(f'Source #{src["source_id"]}: {self.model_catalog[m].name} model at {position}')
                self.logger.debug(f'               {flux}') 

                bad_blob = False # made it though!
                self.model_mask[mm_idx] = True
                oksource[m] = True

            # Clean
            if bad_blob:        
                self.logger.warning(f'No valid models to make model image!')
                continue

            # Tractorize
            tr = Tractor(self.timages, self.model_catalog[oksource])
            self.tr = tr

            # Add to existing array! -- this is the trick!
            self.logger.info(f'Computing model image for blob {blob.blob_id}')
            self.model_images[idx] += np.array([tr.getModelImage(k) for k in np.arange(len(self.bands))])
            if include_chi:
                self.logger.info(f'Computing chi image for blob {blob.blob_id}')
                self.chisq_images[idx] += np.array([tr.getChiImage(k) for k in np.arange(len(self.bands))])

        mtotal = len(self.model_mask)
        nmasked = np.sum(~self.model_mask)
        msrc = np.sum(self.model_mask)
        if not self.model_mask.any():        
            raise RuntimeError(f'No valid models to make model image! (of {mtotal}, {nmasked} masked)')
        self.logger.info(f'Made model image with {msrc}/{mtotal} sources. ({nmasked} are masked)')

        if save:
            if os.path.exists(self.auxhdu_path):
                self.logger.info(f'Saving image(s) to existing file, {self.auxhdu_path}')
                # Save to file
                hdul = fits.open(self.auxhdu_path, mode='update')
                for i, band in zip(idx, conf.PRFMAP_PSF):
                    hdu_img = fits.ImageHDU(data=self.images[i], name=f'{band}_IMAGE', header=self.wcs.to_header())
                    hdul.append(hdu_img)
                    hdu_mod = fits.ImageHDU(data=self.model_images[i], name=f'{band}_MODEL', header=self.wcs.to_header())
                    hdul.append(hdu_mod)
                    if include_chi:
                        hdu_chi = fits.ImageHDU(data=self.chisq_images[i], name=f'{band}_CHI', header=self.wcs.to_header())
                        hdul.append(hdu_chi)
                    if include_nopsf:
                        hdu_nopsf = fits.ImageHDU(data=self.nopsf_images[i], name=f'{band}_NOPSF', header=self.wcs.to_header())
                        hdul.append(hdu_nopsf)

                # make mask array
                self.residual_mask = np.ones_like(self.masks[i], dtype=bool)
                self.residual_mask[conf.BRICK_BUFFER:-conf.BRICK_BUFFER, conf.BRICK_BUFFER:-conf.BRICK_BUFFER] = False
                self.residual_mask[self.masks[i]] = True
                self.residual_mask[self.segmap != 0] = False
                for src in catalog[~self.model_mask]:
                    sid = src['source_id']
                    self.residual_mask[self.segmap == sid] = True

                self.residual_mask = self.residual_mask.astype(int)
                hdu_mask = fits.ImageHDU(data=self.residual_mask, name=f'{band}_MASK', header=self.wcs.to_header())
                hdul.append(hdu_mask)
                hdul.flush()

            else:
                self.logger.info(f'Saving image(s) to new file, {self.auxhdu_path}')
                # Save to file
                hdul = fits.HDUList()
                for i, band in zip(idx, conf.PRFMAP_PSF):
                    hdu_img = fits.ImageHDU(data=self.images[i], name=f'{band}_IMAGE', header=self.wcs.to_header())
                    hdul.append(hdu_img)
                    hdu_mod = fits.ImageHDU(data=self.model_images[i], name=f'{band}_MODEL', header=self.wcs.to_header())
                    hdul.append(hdu_mod)
                    if include_chi:
                        hdu_chi = fits.ImageHDU(data=self.chisq_images[i], name=f'{band}_CHI', header=self.wcs.to_header())
                        hdul.append(hdu_chi)
                    if include_nopsf:
                        hdu_nopsf = fits.ImageHDU(data=self.nopsf_images[i], name=f'{band}_NOPSF', header=self.wcs.to_header())
                        hdul.append(hdu_nopsf)

                # make mask array
                self.residual_mask = np.ones_like(self.masks[i], dtype=bool)
                self.residual_mask[conf.BRICK_BUFFER:-conf.BRICK_BUFFER, conf.BRICK_BUFFER:-conf.BRICK_BUFFER] = False
                self.residual_mask[self.masks[i]] = True
                self.residual_mask[self.segmap != 0] = False
                for src in catalog[~self.model_mask]:
                    sid = src['source_id']
                    self.residual_mask[self.segmap == sid] = True

                self.residual_mask = self.residual_mask.astype(int)
                hdu_mask = fits.ImageHDU(data=self.residual_mask, name=f'{band}_MASK', header=self.wcs.to_header())
                hdul.append(hdu_mask)

                hdul.writeto(self.auxhdu_path, overwrite=True)

    def make_model_image_gridpsf(self, catalog, include_chi=True, include_nopsf=False, save=True, use_band_position=False, use_band_shape=False, modeling=False):

        # Which indices to use?
        idx = []
        for i, b in enumerate(self.bands):
            if modeling:
                b = b[len(conf.MODELING_NICKNAME)+1:]
            if b in conf.PSFGRID:
                idx.append(i)
        # # Make Images
        self.logger.info(f'Making model images for {self.bands[idx]}')
        self.model_mask = np.zeros(len(catalog), dtype=bool)

        # loop over blobs
        for i, bid in enumerate(np.unique(catalog['blob_id'])):
            blob = Blob(self, bid)
            
            self.logger.info(f'Incoporating blob {blob.blob_id}')

            timages = np.zeros(shape=len(self.bands[idx]), dtype=object)

            # Loop over bands:
            prf_notfound = 0
            nband_added = 0
            for j, band in enumerate(self.bands[idx]):

                remove_background_psf = False
                if band in conf.RMBACK_PSF:
                    remove_background_psf = True

                ### construct PSFs
                self.logger.debug('Adopting a GRIDPSF from file.')
                # find nearest prf to blob center
                psftab_coords, psftab_fname = self.psfmodels[j]

                minsep_idx, minsep, __ = blob.blob_coords.match_to_catalog_sky(psftab_coords)
                psf_fname = psftab_fname[minsep_idx]
                self.logger.debug(f'Nearest PSF sample: {psf_fname} ({minsep[0].to(u.arcsec).value:2.2f}")')

                if minsep > conf.PSFGRID_MAXSEP*u.arcsec:
                    self.logger.error(f'Separation ({minsep.to(u.arcsec)}) exceeds maximum {conf.PSFGRID_MAXSEP}!')
                    return False

                blob.minsep[band] = minsep # record it, and add it to the output catalog!

                # open id file
                path_psffile = os.path.join(conf.PSFGRID_OUT_DIR, f'{band}_OUT/{psf_fname}.psf')
                if not os.path.exists(path_psffile):
                    self.logger.error(f'PSF file has not been found! ({path_psffile}')
                    return False
                self.logger.debug(f'Adopting GRID PSF: {psf_fname}')
                
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
                    psf_rad_pix = int(conf.PSF_RADIUS / conf.PIXEL_SCALE)
                    self.logger.debug(f'Clipping PSF ({psf_rad_pix}px radius)')
                    psfmodel.img = psfmodel.img[int(pw/2.-psf_rad_pix):int(pw/2+psf_rad_pix), int(ph/2.-psf_rad_pix):int(ph/2+psf_rad_pix)]
                    self.logger.debug(f'New shape: {np.shape(psfmodel.img)}')

                if conf.NORMALIZE_PSF & (not conf.FORCE_GAUSSIAN_PSF):
                    norm = psfmodel.img.sum()
                    self.logger.debug(f'Normalizing PSF (sum = {norm:4.4f})')
                    psfmodel.img /= norm # HACK -- force normalization to 1       
                    self.logger.debug(f'PSF has been normalized. (sum = {psfmodel.img.sum():4.4f})') 


                psfmodel.img = psfmodel.img.astype('float32') # This may be redundant, but it's super important!

                # make Image
                timages[j] = Image(data=np.zeros_like(self.images[0]),
                            invvar=np.ones_like(self.images[0]),
                            psf=psfmodel,
                            wcs=NullWCS(),
                            photocal=FluxesPhotoCal(band),
                            sky=ConstantSky(0.),
                            name=band)

                nband_added += 1

            if nband_added != len(self.bands[idx]):
                self.logger.error(f'Not all bands added for blob #{bid}! Skipping!')
                continue

            self.timages = timages

            # Construct models
            self.model_catalog = np.ones(len(blob.bcatalog), dtype=object)
            
            bad_blob = True
            oksource = np.zeros(blob.n_sources, dtype=bool)
            for m, src in enumerate(blob.bcatalog):

                mm_idx = np.argwhere(catalog['source_id'] == src['source_id'])[0]
                
                if (blob.bcatalog['BEST_MODEL_BAND'] == '').all():
                    self.logger.debug('No best models chosen yet.')
                    best_band = self.bands[0]
                    if not src[f'VALID_SOURCE_{self.bands[0]}']:
                        self.logger.debug('Source does not have a valid model.')
                        continue
                else:
                    best_band = src['BEST_MODEL_BAND']

                    if not src[f'VALID_SOURCE_{best_band}']:
                        self.logger.debug('Source does not have a valid model.')
                        continue

                if modeling:
                    raw_fluxes = np.array([src[f'RAWFLUX_{conf.MODELING_NICKNAME}_{band}'] for band in self.bands[idx]])
                else:
                    raw_fluxes = np.array([src[f'RAWFLUX_{band}'] for band in self.bands[idx]])

                rejected = False
            
                if conf.RESIDUAL_CHISQ_REJECTION is not None:
                    for j, band in enumerate(self.bands[idx]):
                        if modeling:
                            chisq_band = src[f'CHISQ_MODELING_{band}']
                        else:
                            chisq_band = src[f'CHISQ_{band}']
                        if chisq_band > conf.RESIDUAL_CHISQ_REJECTION:
                            raw_fluxes[j] = 0.0
                            self.logger.debug(f'Source has too large chisq in {band}. ({chisq_band:3.3f}) > {conf.RESIDUAL_CHISQ_REJECTION})')
                    if (raw_fluxes < 0.0).all():
                        self.logger.debug('Source has too large chisq in all bands. Rejecting!')
                        continue

                if conf.RESIDUAL_NEGFLUX_REJECTION:
                    if (raw_fluxes < 0.0).all():
                        self.logger.debug('Source has negative flux in all bands. Rejecting!')
                        continue
                    elif (raw_fluxes < 0.0).any():
                        raw_fluxes[raw_fluxes < 0.0] = 0.0
                        self.logger.debug('Source has negative flux in some bands.')

                if (conf.RESIDUAL_AB_REJECTION is not None) & (src[f'SOLMODEL_{conf.MODELING_NICKNAME}'] not in ('PointSource', 'SimpleGalaxy')):  # HACK -- does NOT apply to unfixed shapes!
                
                    if src[f'SOLMODEL_{conf.MODELING_NICKNAME}'] in ('ExpGalaxy', 'DevGalaxy'):
                        ab = np.array([src[f'AB_{conf.MODELING_NICKNAME}'] for band in self.bands[idx]])
                        if (ab > conf.RESIDUAL_AB_REJECTION).all() | (ab <= 0).all():

                            self.logger.debug('Source has exessive a/b in all bands. Rejecting!')
                        elif (ab > conf.RESIDUAL_AB_REJECTION).any() | (ab <= 0).any():
                            raw_fluxes[ab > conf.RESIDUAL_AB_REJECTION] = 0.0
                            raw_fluxes[ab <= 0] = 0.0
                            self.logger.debug('Source has exessive a/b  in some bands.')
                    else:
                        ab_exp = np.array([src[f'EXP_AB_{conf.MODELING_NICKNAME}'] for band in self.bands[idx]])
                        if (ab_exp > conf.RESIDUAL_AB_REJECTION).all() | (ab_exp <= 0).all():

                            self.logger.debug('Source has exessive exp a/b in all bands. Rejecting!')
                        elif (ab_exp > conf.RESIDUAL_AB_REJECTION).any() | (ab_exp <= 0).any():
                            raw_fluxes[ab_exp > conf.RESIDUAL_AB_REJECTION] = 0.0
                            raw_fluxes[ab_exp <= 0] = 0.0
                            self.logger.debug('Source has exessive exp a/b  in some bands.')

                        ab_dev = np.array([src[f'DEV_AB_{conf.MODELING_NICKNAME}'] for band in self.bands[idx]])
                        if (ab_dev > conf.RESIDUAL_AB_REJECTION).all() | (ab_dev <= 0).all():

                            self.logger.debug('Source has exessive dev a/b in all bands. Rejecting!')
                        elif (ab_dev > conf.RESIDUAL_AB_REJECTION).any() | (ab_dev <= 0).any():
                            raw_fluxes[ab_dev > conf.RESIDUAL_AB_REJECTION] = 0.0
                            raw_fluxes[ab_dev <= 0] = 0.0
                            self.logger.debug('Source has exessive dev a/b  in some bands.')

                if use_band_position:
                    bx_model = src[f'X_MODEL_{band}'] - self.mosaic_origin[1] + conf.BRICK_BUFFER 
                    by_model = src[f'Y_MODEL_{band}'] - self.mosaic_origin[0] + conf.BRICK_BUFFER
                else:
                    bx_model = src[f'X_MODEL_{best_band}'] - self.mosaic_origin[1] + conf.BRICK_BUFFER 
                    by_model = src[f'Y_MODEL_{best_band}'] - self.mosaic_origin[0] + conf.BRICK_BUFFER

                position = PixPos(bx_model, by_model)
                flux = Fluxes(**dict(zip(self.bands[idx], raw_fluxes)), order=self.bands[idx]) # IMAGES ARE IN NATIVE ZPT, USE RAWFLUXES!

                if src[f'SOLMODEL_{best_band}'] == "PointSource":
                    self.model_catalog[m] = PointSource(position, flux)
                    self.model_catalog[m].name = 'PointSource' # HACK to get around Dustin's HACK.
                elif src[f'SOLMODEL_{best_band}'] == "SimpleGalaxy":
                    self.model_catalog[m] = SimpleGalaxy(position, flux)
                elif src[f'SOLMODEL_{best_band}'] == "ExpGalaxy":
                    if use_band_shape:
                        shape = EllipseESoft(src[f'REFF_{band}'], src[f'EE1_{band}'], src[f'EE2_{band}'])
                    else:
                        shape = EllipseESoft(src[f'REFF_{best_band}'], src[f'EE1_{best_band}'], src[f'EE2_{best_band}'])
                    self.model_catalog[m] = ExpGalaxy(position, flux, shape)
                elif src[f'SOLMODEL_{best_band}'] == "DevGalaxy":
                    if use_band_shape:
                        shape = EllipseESoft(src[f'REFF_{band}'], src[f'EE1_{band}'], src[f'EE2_{band}'])
                    else:
                        shape = EllipseESoft(src[f'REFF_{best_band}'], src[f'EE1_{best_band}'], src[f'EE2_{best_band}'])                
                    self.model_catalog[m] = DevGalaxy(position, flux, shape)
                elif src[f'SOLMODEL_{best_band}'] == "FixedCompositeGalaxy":
                    if use_band_shape:
                        shape_exp = EllipseESoft(src[f'EXP_REFF_{band}'], src[f'EXP_EE1_{band}'], src[f'EXP_EE2_{band}'])
                        shape_dev = EllipseESoft(src[f'DEV_REFF_{band}'], src[f'DEV_EE1_{band}'], src[f'DEV_EE2_{band}'])
                    else:
                        shape_exp = EllipseESoft(src[f'EXP_REFF_{best_band}'], src[f'EXP_EE1_{best_band}'], src[f'EXP_EE2_{best_band}'])
                        shape_dev = EllipseESoft(src[f'DEV_REFF_{best_band}'], src[f'DEV_EE1_{best_band}'], src[f'DEV_EE2_{best_band}'])
                    
                    self.model_catalog[m] = FixedCompositeGalaxy(
                                                    position, flux,
                                                    SoftenedFracDev(src[f'FRACDEV_{best_band}']),
                                                    shape_exp, shape_dev)
                else:
                    self.logger.warning(f'Source #{src["source_id"]}: has no solution model at {position}')
                    continue

                self.logger.debug(f'Source #{src["source_id"]}: {self.model_catalog[m].name} model at {position}')
                self.logger.debug(f'               {flux}') 

                bad_blob = False # made it though!
                self.model_mask[mm_idx] = True
                oksource[m] = True

            # Clean
            if bad_blob:        
                self.logger.warning(f'No valid models to make model image!')
                continue

            # Tractorize
            tr = Tractor(self.timages, self.model_catalog[oksource])
            self.tr = tr

            # Add to existing array! -- this is the trick!
            self.logger.info(f'Computing model image for blob {blob.blob_id}')
            self.model_images[idx] += np.array([tr.getModelImage(k) for k in np.arange(len(self.bands))])
            if include_chi:
                self.logger.info(f'Computing chi image for blob {blob.blob_id}')
                self.chisq_images[idx] += np.array([tr.getChiImage(k) for k in np.arange(len(self.bands))])

        mtotal = len(self.model_mask)
        nmasked = np.sum(~self.model_mask)
        msrc = np.sum(self.model_mask)
        if not self.model_mask.any():        
            raise RuntimeError(f'No valid models to make model image! (of {mtotal}, {nmasked} masked)')
        self.logger.info(f'Made model image with {msrc}/{mtotal} sources. ({nmasked} are masked)')

        if save:
            if os.path.exists(self.auxhdu_path):
                self.logger.info(f'Saving image(s) to existing file, {self.auxhdu_path}')
                # Save to file
                hdul = fits.open(self.auxhdu_path, mode='update')
                for i, band in zip(idx, conf.PRFMAP_PSF):
                    hdu_img = fits.ImageHDU(data=self.images[i], name=f'{band}_IMAGE', header=self.wcs.to_header())
                    hdul.append(hdu_img)
                    hdu_mod = fits.ImageHDU(data=self.model_images[i], name=f'{band}_MODEL', header=self.wcs.to_header())
                    hdul.append(hdu_mod)
                    if include_chi:
                        hdu_chi = fits.ImageHDU(data=self.chisq_images[i], name=f'{band}_CHI', header=self.wcs.to_header())
                        hdul.append(hdu_chi)
                    if include_nopsf:
                        hdu_nopsf = fits.ImageHDU(data=self.nopsf_images[i], name=f'{band}_NOPSF', header=self.wcs.to_header())
                        hdul.append(hdu_nopsf)

                # make mask array
                self.residual_mask = np.ones_like(self.masks[i], dtype=bool)
                self.residual_mask[conf.BRICK_BUFFER:-conf.BRICK_BUFFER, conf.BRICK_BUFFER:-conf.BRICK_BUFFER] = False
                self.residual_mask[self.masks[i]] = True
                self.residual_mask[self.segmap != 0] = False
                for src in catalog[~self.model_mask]:
                    sid = src['source_id']
                    self.residual_mask[self.segmap == sid] = True

                self.residual_mask = self.residual_mask.astype(int)
                hdu_mask = fits.ImageHDU(data=self.residual_mask, name=f'{band}_MASK', header=self.wcs.to_header())
                hdul.append(hdu_mask)
                hdul.flush()

            else:
                self.logger.info(f'Saving image(s) to new file, {self.auxhdu_path}')
                # Save to file
                hdul = fits.HDUList()
                for i, band in zip(idx, conf.PRFMAP_PSF):
                    hdu_img = fits.ImageHDU(data=self.images[i], name=f'{band}_IMAGE', header=self.wcs.to_header())
                    hdul.append(hdu_img)
                    hdu_mod = fits.ImageHDU(data=self.model_images[i], name=f'{band}_MODEL', header=self.wcs.to_header())
                    hdul.append(hdu_mod)
                    if include_chi:
                        hdu_chi = fits.ImageHDU(data=self.chisq_images[i], name=f'{band}_CHI', header=self.wcs.to_header())
                        hdul.append(hdu_chi)
                    if include_nopsf:
                        hdu_nopsf = fits.ImageHDU(data=self.nopsf_images[i], name=f'{band}_NOPSF', header=self.wcs.to_header())
                        hdul.append(hdu_nopsf)

                # make mask array
                self.residual_mask = np.ones_like(self.masks[i], dtype=bool)
                self.residual_mask[conf.BRICK_BUFFER:-conf.BRICK_BUFFER, conf.BRICK_BUFFER:-conf.BRICK_BUFFER] = False
                self.residual_mask[self.masks[i]] = True
                self.residual_mask[self.segmap != 0] = False
                for src in catalog[~self.model_mask]:
                    sid = src['source_id']
                    self.residual_mask[self.segmap == sid] = True

                self.residual_mask = self.residual_mask.astype(int)
                hdu_mask = fits.ImageHDU(data=self.residual_mask, name=f'{band}_MASK', header=self.wcs.to_header())
                hdul.append(hdu_mask)

                hdul.writeto(self.auxhdu_path, overwrite=True)

    def make_model_image_psf(self, catalog, include_chi=True, include_nopsf=False, save=True, use_band_position=False, use_band_shape=False, modeling=False):

        # Which indices to use?
        idx = []
        for i, b in enumerate(self.bands):
            if modeling:
                b = b[len(conf.MODELING_NICKNAME)+1:]
            if b not in conf.PRFMAP_PSF:
                idx.append(i)


        # Make Images
        self.logger.info(f'Making a model image for {self.bands[idx]}')

        timages = np.zeros(self.n_bands, dtype=object)

        for i, (psf, band) in enumerate(zip(self.psfmodels[idx], self.bands[idx])):

            band_orig = band
            if band.startswith(conf.MODELING_NICKNAME):
                band = band[len(conf.MODELING_NICKNAME)+1:]

            remove_background_psf = False
            if band in conf.RMBACK_PSF:
                remove_background_psf = True

            if (band in conf.CONSTANT_PSF) & (psf is not None):
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
                    psf_rad_pix = int(conf.PSF_RADIUS / conf.PIXEL_SCALE)
                    self.logger.debug(f'Clipping PSF ({psf_rad_pix}px radius)')
                    psfmodel.img = psfmodel.img[int(pw/2.-psf_rad_pix):int(pw/2+psf_rad_pix), int(ph/2.-psf_rad_pix):int(ph/2+psf_rad_pix)]
                    self.logger.debug(f'New shape: {np.shape(psfmodel.img)}')

                if conf.NORMALIZE_PSF & (not conf.FORCE_GAUSSIAN_PSF):
                    norm = psfmodel.img.sum()
                    self.logger.debug(f'Normalizing PSF (sum = {norm:4.4f})')
                    psfmodel.img /= norm # HACK -- force normalization to 1
                self.logger.debug('Adopting constant PSF.')

                if conf.USE_MOG_PSF:
                    self.logger.debug('Making a Gaussian Mixture PSF')
                    psfmodel = HybridPixelizedPSF(pix=psfmodel, N=10).gauss

            elif (psf is not None):
                raise RuntimeError('Position dependent PSFs in brick-scale model images is not supported!')

            elif (psf is None):
                if conf.USE_GAUSSIAN_PSF:
                    psfmodel = NCircularGaussianPSF([conf.PSF_SIGMA / conf.PIXEL_SCALE], [1,])
                    self.logger.debug(f'Adopting {conf.PSF_SIGMA}" Gaussian PSF.')
                else:
                    raise ValueError(f'WARNING - No PSF model found for {band}!')

            timages[i] = Image(data=np.zeros_like(self.images[0]),
                            invvar=np.ones_like(self.images[0]),
                            psf=psfmodel,
                            wcs=NullWCS(),
                            photocal=FluxesPhotoCal(band_orig),
                            sky=ConstantSky(0.),
                            name=band)

        self.timages = timages

        # Make models
        self.model_catalog = np.zeros(len(catalog), dtype=object)
        self.model_mask = np.zeros(len(catalog), dtype=bool)
        nb = []
        for ix in np.arange(len(self.bands)):
            if ~self.bands[ix].startswith(conf.MODELING_NICKNAME) & (ix in idx) & modeling:
                nb.append(f'{conf.MODELING_NICKNAME}_{self.bands[ix]}')
            else:
                nb.append(self.bands[ix])
        self.mbands = np.array(nb)

        for i, src in enumerate(catalog):
            
            best_band = conf.MODELING_NICKNAME
            raw_fluxes = np.array([src[f'RAWFLUX_{band}'] for band in self.mbands[idx]])

            
            rejected = False
            if conf.RESIDUAL_CHISQ_REJECTION is not None:
                for j, band in enumerate(self.mbands[idx]):
                    chisq_band = src[f'CHISQ_{band}']
                    # print(chisq_band)
                    if chisq_band > conf.RESIDUAL_CHISQ_REJECTION:
                        raw_fluxes[j] = 0.0
                        self.logger.debug(f'Source has too large chisq in {band}. ({chisq_band:3.3f}) > {conf.RESIDUAL_CHISQ_REJECTION})')
                if (raw_fluxes <= 0.0).all():

                    self.logger.debug('Source has too large chisq in all bands. Rejecting!')
                    rejected=True
                
                elif (raw_fluxes < 0.0).any():
                    raw_fluxes[raw_fluxes < 0.0] = 0.0
                    self.logger.debug('Source has too large in some bands.')

            if conf.RESIDUAL_NEGFLUX_REJECTION:
                if (raw_fluxes <= 0.0).all():

                    self.logger.debug('Source has negative flux in all bands. Rejecting!')
                    rejected=True
                elif (raw_fluxes < 0.0).any():
                    raw_fluxes[raw_fluxes < 0.0] = 0.0
                    self.logger.debug('Source has negative flux in some bands.')

            if (conf.RESIDUAL_AB_REJECTION is not None) & (src[f'SOLMODEL_{best_band}'] not in ('PointSource', 'SimpleGalaxy')):  # HACK -- does NOT apply to unfixed shapes!
                
                if src[f'SOLMODEL_{best_band}'] in ('ExpGalaxy', 'DevGalaxy'):
                    ab = np.array([src[f'AB_{best_band}'] for band in self.mbands[idx]])
                    if (ab > conf.RESIDUAL_AB_REJECTION).all() | (ab <= 0).all():

                        self.logger.debug('Source has exessive a/b in all bands. Rejecting!')
                        rejected=True
                    elif (ab > conf.RESIDUAL_AB_REJECTION).any() | (ab <= 0).any():
                        raw_fluxes[ab > conf.RESIDUAL_AB_REJECTION] = 0.0
                        raw_fluxes[ab <= 0] = 0.0
                        self.logger.debug('Source has exessive a/b in some bands.')
                else:
                    ab_exp = np.array([src[f'EXP_AB_{best_band}'] for band in self.mbands[idx]])
                    if (ab_exp > conf.RESIDUAL_AB_REJECTION).all() | (ab_exp <= 0).all():

                        self.logger.debug('Source has exessive exp a/b in all bands. Rejecting!')
                        rejected=True
                    elif (ab_exp > conf.RESIDUAL_AB_REJECTION).any() | (ab_exp <= 0).any():
                        raw_fluxes[ab_exp > conf.RESIDUAL_AB_REJECTION] = 0.0
                        raw_fluxes[ab_exp <= 0] = 0.0
                        self.logger.debug('Source has exessive exp a/b in some bands.')

                    ab_dev = np.array([src[f'DEV_AB_{best_band}'] for band in self.mbands[idx]])
                    if (ab_dev > conf.RESIDUAL_AB_REJECTION).all() | (ab_dev <= 0).all():

                        self.logger.debug('Source has exessive dev a/b in all bands. Rejecting!')
                        rejected=True
                    elif (ab_dev > conf.RESIDUAL_AB_REJECTION).any() | (ab_dev <= 0).any():
                        raw_fluxes[ab_dev > conf.RESIDUAL_AB_REJECTION] = 0.0
                        raw_fluxes[ab_dev <= 0] = 0.0
                        self.logger.debug('Source has exessive dev a/b in some bands.')
                        # dont specify rejection here -- if there is only one band then the elif wont trigger anyways.

            if use_band_position:
                bx_model = src[f'X_MODEL_{band}'] - self.mosaic_origin[1] + conf.BRICK_BUFFER 
                by_model = src[f'Y_MODEL_{band}'] - self.mosaic_origin[0] + conf.BRICK_BUFFER
            else:
                bx_model = src[f'X_MODEL_{best_band}'] - self.mosaic_origin[1] + conf.BRICK_BUFFER 
                by_model = src[f'Y_MODEL_{best_band}'] - self.mosaic_origin[0] + conf.BRICK_BUFFER

            position = PixPos(bx_model, by_model)
            flux = Fluxes(**dict(zip(self.bands[idx], raw_fluxes)), order=self.bands[idx]) # IMAGES ARE IN NATIVE ZPT, USE RAWFLUXES!

            if src[f'SOLMODEL_{best_band}'] == "PointSource":
                self.model_catalog[i] = PointSource(position, flux)
                self.model_catalog[i].name = 'PointSource' # HACK to get around Dustin's HACK.
            elif src[f'SOLMODEL_{best_band}'] == "SimpleGalaxy":
                self.model_catalog[i] = SimpleGalaxy(position, flux)
            elif src[f'SOLMODEL_{best_band}'] == "ExpGalaxy":
                if use_band_shape:
                    shape = EllipseESoft(src[f'REFF_{band}'], src[f'EE1_{band}'], src[f'EE2_{band}'])
                else:
                    shape = EllipseESoft(src[f'REFF_{best_band}'], src[f'EE1_{best_band}'], src[f'EE2_{best_band}'])
                self.model_catalog[i] = ExpGalaxy(position, flux, shape)
            elif src[f'SOLMODEL_{best_band}'] == "DevGalaxy":
                if use_band_shape:
                    shape = EllipseESoft(src[f'REFF_{band}'], src[f'EE1_{band}'], src[f'EE2_{band}'])
                else:
                    shape = EllipseESoft(src[f'REFF_{best_band}'], src[f'EE1_{best_band}'], src[f'EE2_{best_band}'])                
                self.model_catalog[i] = DevGalaxy(position, flux, shape)
            elif src[f'SOLMODEL_{best_band}'] == "FixedCompositeGalaxy":
                if use_band_shape:
                    shape_exp = EllipseESoft(src[f'EXP_REFF_{band}'], src[f'EXP_EE1_{band}'], src[f'EXP_EE2_{band}'])
                    shape_dev = EllipseESoft(src[f'DEV_REFF_{band}'], src[f'DEV_EE1_{band}'], src[f'DEV_EE2_{band}'])
                else:
                    shape_exp = EllipseESoft(src[f'EXP_REFF_{best_band}'], src[f'EXP_EE1_{best_band}'], src[f'EXP_EE2_{best_band}'])
                    shape_dev = EllipseESoft(src[f'DEV_REFF_{best_band}'], src[f'DEV_EE1_{best_band}'], src[f'DEV_EE2_{best_band}'])
                
                self.model_catalog[i] = FixedCompositeGalaxy(
                                                position, flux,
                                                SoftenedFracDev(src[f'FRACDEV_{best_band}']),
                                                shape_exp, shape_dev)
            else:
                self.logger.warning(f'Source #{src["source_id"]}: has no solution model at {position}')
                rejected = True
                
            if not rejected:
                self.model_mask[i] = True  
                self.logger.debug(f'Source #{src["source_id"]}: {self.model_catalog[i].name} model at {position}')
                self.logger.debug(f'               {flux}') 

            

        # Clean
        mtotal = len(self.model_catalog)
        nmasked = np.sum(~self.model_mask)
        msrc = np.sum(self.model_mask)
        if not self.model_mask.any():        
            raise RuntimeError(f'No valid models to make model image! (of {mtotal}, {nmasked} masked)')
        self.logger.info(f'Making model image with {msrc}/{mtotal} sources. ({nmasked} are masked)')

        # Tractorize


        self.logger.info(f'Computing model image...')
        if conf.MODEL_APPLY_SEGMAP:
            tr = Tractor(self.timages, self.model_catalog)
            self.tr = tr
            for i, mod in enumerate(self.model_catalog):
                trs = Tractor(self.timages, [mod,])
                sid = catalog['source_id'][i]
                npix = catalog['npix'][i]
                for k in np.arange(len(idx)):
                    ming = trs.getModelImage(k)
                    assert(np.sum(self.segmap==sid) == npix)

                    ming[self.segmap != sid] = 0
                    self.model_images[k] += ming
        else:
            self.model_catalog = self.model_catalog[self.model_mask]
            tr = Tractor(self.timages, self.model_catalog)
            self.tr = tr
            self.model_images[idx] = [tr.getModelImage(k) for k in np.arange(len(idx))]

        if include_chi:
            self.logger.info(f'Computing chi image...')
            if conf.MODEL_APPLY_SEGMAP:
                for i, mod in enumerate(self.model_catalog):
                    trs = Tractor(self.timages, [mod,])
                    sid = catalog[i]['source_id']
                    npix = catalog['npix'][i]
                    for k in np.arange(len(idx)):
                        ming = trs.getChiImage(k)
                        assert(np.sum(self.segmap==sid) == npix)
                        ming[self.segmap != sid] = 0
                        self.chisq_images[k] += ming
            else:
                self.chisq_images[idx] = [tr.getChiImage(k) for k in np.arange(len(idx))]


        # If no psf:
        if include_nopsf:
            self.logger.warning('INCLUSION OF NO PSF IS CURRENTLY DISABLED.')
            include_nopsf = False

        if save:
            if os.path.exists(self.auxhdu_path):
                self.logger.info(f'Saving image(s) to existing file, {self.auxhdu_path}')
                # Save to file
                hdul = fits.open(self.auxhdu_path, mode='update')
                for i, band in enumerate(self.mbands[idx]):
                    hdu_img = fits.ImageHDU(data=self.images[i], name=f'{band}_IMAGE', header=self.wcs.to_header())
                    hdul.append(hdu_img)
                    hdu_mod = fits.ImageHDU(data=self.model_images[i], name=f'{band}_MODEL', header=self.wcs.to_header())
                    hdul.append(hdu_mod)
                    if include_chi:
                        hdu_chi = fits.ImageHDU(data=self.chisq_images[i], name=f'{band}_CHI', header=self.wcs.to_header())
                        hdul.append(hdu_chi)
                    if include_nopsf:
                        hdu_nopsf = fits.ImageHDU(data=self.nopsf_images[i], name=f'{band}_NOPSF', header=self.wcs.to_header())
                        hdul.append(hdu_nopsf)

                # make mask array
                self.residual_mask = np.ones_like(self.masks[i], dtype=bool)
                self.residual_mask[conf.BRICK_BUFFER:-conf.BRICK_BUFFER, conf.BRICK_BUFFER:-conf.BRICK_BUFFER] = False
                self.residual_mask[self.masks[i]] = True
                self.residual_mask[self.segmap != 0] = False
                for src in catalog[~self.model_mask]:
                    sid = src['source_id']
                    self.residual_mask[self.segmap == sid] = True

                self.residual_mask = self.residual_mask.astype(int)
                hdu_mask = fits.ImageHDU(data=self.residual_mask, name=f'{band}_MASK', header=self.wcs.to_header())
                hdul.append(hdu_mask)
                hdul.flush()

            else:
                self.logger.info(f'Saving image(s) to new file, {self.auxhdu_path}')
                # Save to file
                hdul = fits.HDUList()
                for i, band in enumerate(self.mbands[idx]):
                    hdu_img = fits.ImageHDU(data=self.images[i], name=f'{band}_IMAGE', header=self.wcs.to_header())
                    hdul.append(hdu_img)
                    hdu_mod = fits.ImageHDU(data=self.model_images[i], name=f'{band}_MODEL', header=self.wcs.to_header())
                    hdul.append(hdu_mod)
                    if include_chi:
                        hdu_chi = fits.ImageHDU(data=self.chisq_images[i], name=f'{band}_CHI', header=self.wcs.to_header())
                        hdul.append(hdu_chi)
                    if include_nopsf:
                        hdu_nopsf = fits.ImageHDU(data=self.nopsf_images[i], name=f'{band}_NOPSF', header=self.wcs.to_header())
                        hdul.append(hdu_nopsf)

                # make mask array
                self.residual_mask = np.ones_like(self.masks[i], dtype=bool)
                self.residual_mask[conf.BRICK_BUFFER:-conf.BRICK_BUFFER, conf.BRICK_BUFFER:-conf.BRICK_BUFFER] = False
                self.residual_mask[self.masks[i]] = True
                self.residual_mask[self.segmap != 0] = False
                for src in catalog[~self.model_mask]:
                    sid = src['source_id']
                    self.residual_mask[self.segmap == sid] = True

                self.residual_mask = self.residual_mask.astype(int)
                hdu_mask = fits.ImageHDU(data=self.residual_mask, name=f'{band}_MASK', header=self.wcs.to_header())
                hdul.append(hdu_mask)
                hdul.writeto(self.auxhdu_path, overwrite=True)

    def make_residual_image(self, catalog=None, band=None, include_chi=True, include_nopsf=False, save=True, use_band_position=False, use_band_shape=False, modeling=False):
        # Make model image or load it in
        self.logger.info(f'Making residual image')

        if include_nopsf:
            self.logger.warning('INCLUSION OF NO PSF IS CURRENTLY DISABLED.')
            include_nopsf = False

        if self.model_images is None:
            if catalog is None:
                raise RuntimeError('ERROR - I need a catalog to make the model image first!') # save here is false, since we want to save later on.
            self.make_model_image(catalog, include_chi=include_chi, include_nopsf=include_nopsf, save=False, use_band_position=use_band_position, use_band_shape=use_band_shape, modeling=modeling)
        
        # Background
        self.subtract_background(flat=conf.USE_FLAT)
        self.logger.debug(f'Subtracted background (flat={conf.USE_FLAT})')

        # Subtract
        self.logger.info('Constructing residual image...')
        self.residual_images = self.images - self.model_images

        

        # Save to file
        if save:

            if os.path.exists(self.auxhdu_path):
                self.logger.info(f'Saving image(s) to existing file, {self.auxhdu_path}')
                hdul = fits.open(self.auxhdu_path, mode='update')
                for i, band in enumerate(self.bands):
                    hdu_img = fits.ImageHDU(data=self.images[i], name=f'{band}_IMAGE', header=self.wcs.to_header())
                    hdul.append(hdu_img)
                    hdu_mod = fits.ImageHDU(data=self.model_images[i], name=f'{band}_MODEL', header=self.wcs.to_header())
                    hdul.append(hdu_mod)
                    if include_chi:
                        hdu_chi = fits.ImageHDU(data=self.chisq_images[i], name=f'{band}_CHI', header=self.wcs.to_header())
                        hdul.append(hdu_chi)
                    if include_nopsf:
                        hdu_nopsf = fits.ImageHDU(data=self.nopsf_images[i], name=f'{band}_NOPSF', header=self.wcs.to_header())
                        hdul.append(hdu_nopsf)

                    hdu_residual = fits.ImageHDU(data=self.residual_images[i], name=f'{band}_RESIDUAL', header=self.wcs.to_header())
                    hdul.append(hdu_residual)

                    # make mask array
                    self.residual_mask = np.ones_like(self.masks[i], dtype=bool)
                    self.residual_mask[conf.BRICK_BUFFER:-conf.BRICK_BUFFER, conf.BRICK_BUFFER:-conf.BRICK_BUFFER] = False
                    self.residual_mask[self.masks[i]] = True
                    self.residual_mask[self.segmap != 0] = False
                    for src in catalog[~self.model_mask]:
                        sid = src['source_id']
                        self.residual_mask[self.segmap == sid] = True

                    self.residual_mask = self.residual_mask.astype(int)
                    hdu_mask = fits.ImageHDU(data=self.residual_mask, name=f'{band}_MASK', header=self.wcs.to_header())
                    hdul.append(hdu_mask)
                # Save
                hdul.flush()

            else:

                self.logger.info(f'brick.make_residual_image :: Saving image(s) to new file, s{self.auxhdu_path}')
                hdul = fits.HDUList()
                for i, band in enumerate(self.bands):
                    hdu_img = fits.ImageHDU(data=self.images[i], name=f'{band}_IMAGE', header=self.wcs.to_header())
                    hdul.append(hdu_img)
                    hdu_mod = fits.ImageHDU(data=self.model_images[i], name=f'{band}_MODEL', header=self.wcs.to_header())
                    hdul.append(hdu_mod)
                    if include_chi:
                        hdu_chi = fits.ImageHDU(data=self.chisq_images[i], name=f'{band}_CHI', header=self.wcs.to_header())
                        hdul.append(hdu_chi)
                    if include_nopsf:
                        hdu_nopsf = fits.ImageHDU(data=self.nopsf_images[i], name=f'{band}_NOPSF', header=self.wcs.to_header())
                        hdul.append(hdu_nopsf)

                    hdu_residual = fits.ImageHDU(data=self.residual_images[i], name=f'{band}_RESIDUAL', header=self.wcs.to_header())
                    hdul.append(hdu_residual)
                
                    # make mask array
                    self.residual_mask = np.ones_like(self.masks[i], dtype=bool)
                    self.residual_mask[conf.BRICK_BUFFER:-conf.BRICK_BUFFER, conf.BRICK_BUFFER:-conf.BRICK_BUFFER] = False
                    self.residual_mask[self.masks[i]] = True
                    self.residual_mask[self.segmap != 0] = False
                    for src in catalog[~self.model_mask]:
                        sid = src['source_id']
                        self.residual_mask[self.segmap == sid] = True

                    self.residual_mask = self.residual_mask.astype(int)
                    hdu_mask = fits.ImageHDU(data=self.residual_mask, name=f'{band}_MASK', header=self.wcs.to_header())
                    hdul.append(hdu_mask)

                # Save
                hdul.writeto(self.auxhdu_path, overwrite=True)

    def estimate_effective_area(self, catalog, band, modeling=False):
        self.logger.info(f'Calculating the effective area for {band}')
        inner_area_pix = (conf.BRICK_WIDTH) * (conf.BRICK_HEIGHT)
        if len(self.masks > 0):  # bit of an assumption, but OK
            mask = self.masks[0]
        else:
            # make mask array
            if band == conf.MODELING_NICKNAME:
                idx = 0
            elif band.startswith(conf.MODELING_NICKNAME):
                band_name = band[len(conf.MODELING_NICKNAME)+1:]
                idx = self._band2idx(band_name)
            else:
                idx = self._band2idx(band)

            mask = self.masks[idx]
        total_area_pix = np.size(mask)
        residual_mask = np.zeros_like(mask, dtype=bool) # everything masked
        residual_mask[conf.BRICK_BUFFER:-conf.BRICK_BUFFER, conf.BRICK_BUFFER:-conf.BRICK_BUFFER] = True  # inner region ok
        residual_mask[mask] = False # mask is not ok
        residual_mask[self.segmap != 0] = False # assume everything failed
        for src in catalog:
            self.logger.debug(f"Evaluating source #{src['source_id']}")

            if modeling:
                raw_fluxes = src[f'FLUX_{conf.MODELING_NICKNAME}_{band}']
                try:
                    solmodel = src[f'SOLMODEL_{conf.MODELING_NICKNAME}_{band}']
                except:
                    solmodel = src[f'SOLMODEL_{conf.MODELING_NICKNAME}']
            else:
                raw_fluxes = src[f'FLUX_{band}']
                solmodel = src[f'SOLMODEL_{conf.MODELING_NICKNAME}']

            if conf.RESIDUAL_CHISQ_REJECTION is not None:
                if modeling:
                    chisq_band = src[f'CHISQ_MODELING_{band}']
                else:
                    chisq_band = src[f'CHISQ_{band}']
                if chisq_band > conf.RESIDUAL_CHISQ_REJECTION:
                    raw_fluxes = 0.0
                    self.logger.debug(f'Source has too large chisq. ({chisq_band:3.3f}) > {conf.RESIDUAL_CHISQ_REJECTION})') 
                    continue

            if conf.RESIDUAL_NEGFLUX_REJECTION:
                if raw_fluxes <= 0.0:
                    self.logger.debug('Source has negative flux. Rejecting!')
                    continue

            if (conf.RESIDUAL_AB_REJECTION is not None) & (src[f'SOLMODEL_{conf.MODELING_NICKNAME}'] not in ('PointSource', 'SimpleGalaxy')):  # HACK -- does NOT apply to unfixed shapes!
                
                if src[f'SOLMODEL_{conf.MODELING_NICKNAME}'] in ('ExpGalaxy', 'DevGalaxy'):
                    ab = src[f'AB_{conf.MODELING_NICKNAME}']
                    if (ab > conf.RESIDUAL_AB_REJECTION) | (ab <= 0):
                        self.logger.debug('Source has exessive a/b. Rejecting!')
                        continue
                else:
                    ab_exp = src[f'EXP_AB_{conf.MODELING_NICKNAME}']
                    if (ab_exp > conf.RESIDUAL_AB_REJECTION) | (ab_exp <= 0):
                        self.logger.debug('Source has exessive exp a/b. Rejecting!')
                        continue

                    ab_dev = src[f'DEV_AB_{conf.MODELING_NICKNAME}']
                    if (ab_dev > conf.RESIDUAL_AB_REJECTION) | (ab_dev <= 0):
                        self.logger.debug('Source has exessive dev a/b. Rejecting!')
                        continue

            sid = src['source_id']
            residual_mask[self.segmap == sid] = True

        # HACK -- this is a slight estimationf in the case that a source bleeds into the buffer, but it is EXACT for the good area!
        good_area_pix = np.sum(residual_mask)
        bad_area_pix = inner_area_pix - good_area_pix
        self.logger.info(f'Total effective area for brick #{self.brick_id}: {good_area_pix*(conf.PIXEL_SCALE/3600)**2:4.4f} deg2 ({good_area_pix/inner_area_pix*100:3.3f}%)')
        self.logger.debug(f'Total masked area for brick #{self.brick_id}: {bad_area_pix*(conf.PIXEL_SCALE/3600)**2:4.4f} deg2 ({bad_area_pix/inner_area_pix*100:3.3f}%)')
        

        return good_area_pix, inner_area_pix
