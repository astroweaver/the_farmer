import config as conf
from .utils import clean_catalog, get_fwhm, map_discontinuous, SimpleGalaxy, read_wcs, cumulative, set_priors
from .utils import recursively_save_dict_contents_to_group, recursively_load_dict_contents_from_group, dcoord_to_offset, get_params
from .utils import get_detection_kernel

import logging
import os
import sep
import numpy as np
import time
import h5py
import copy
import sys
import gc

from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
from  matplotlib.colors import LogNorm, SymLogNorm, Normalize, ListedColormap, BoundaryNorm
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.table import Table, Column
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from matplotlib.patches import Rectangle, Circle, Ellipse
from tractor import RaDecPos
from tqdm import tqdm
from scipy import stats, ndimage
import matplotlib.backends.backend_pdf


from tractor import PixelizedPSF, PixelizedPsfEx, Image, Tractor, FluxesPhotoCal, ConstantSky, EllipseESoft, Fluxes, Catalog
from tractor.sersic import SersicIndex, SersicGalaxy
from tractor.sercore import SersicCoreGalaxy
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev
from tractor.pointsource import PointSource
if conf.USE_CERES:
    try:
        from tractor.ceres_optimizer import CeresOptimizer as optimizer
    except ImportError:
        logging.warning('Ceres not found! Falling back to ConstrainedOptimizer. Install Ceres for faster optimization!')
        from tractor.constrained_optimizer import ConstrainedOptimizer as optimizer
else:
    from tractor.constrained_optimizer import ConstrainedOptimizer as optimizer


class BaseImage():
    """
    Base class for images: mosaics, bricks, and groups
    Useful for estimating image properties + detecting sources
    """

    def __init__(self):
        self.data = {}
        self.headers = {}
        self.segments = {}
        self.catalogs = {}
        self.bands = []
        self.model_catalog = OrderedDict()
        self.model_tracker = OrderedDict()
        self.model_priors = None
        self.phot_priors = None

        # Load the logger
        self.logger = logging.getLogger(f'farmer.image')


    def add_tracker(self, init_stage=0):
        catalog = self.catalogs[self.catalog_band][self.catalog_imgtype]
        if not hasattr(self, 'stage'):
            self.stage = init_stage
            self.solved = np.zeros(len(catalog), dtype=bool)
        elif self.stage is None:
            self.stage = init_stage
            self.solved = np.zeros(len(catalog), dtype=bool)

        if self.stage == 0:
            self.logger.debug(f'Tracking for Stage {self.stage}, {np.sum(self.solved)}/{len(self.solved)} models solved.')

        if self.stage == init_stage:
            self.model_tracker[self.type] = {}
        self.model_tracker[self.type][self.stage] = {}

        for src in catalog:
            source_id = src['id']
            if source_id not in self.model_catalog:
                self.model_catalog[source_id] = PointSource(None, None)  # this is a bit of a DT-dependent move...
                self.model_tracker[source_id] = {}
                self.source_ids = np.array(catalog['id'])
            self.model_tracker[source_id][self.stage] = {}

    def reset_models(self):
        self.engine = None
        self.stage = None
        self.model_tracker = OrderedDict()
        self.model_catalog = OrderedDict()

    def cleanup_after_detection(self, keep_segmap=True, keep_groupmap=False):
        """Remove non-essential data after source detection to save memory.
        
        Deletes temporary arrays like background estimates and RMS that are
        not needed after detection completes. Useful for low-memory workflows.
        
        Args:
            keep_segmap: If True, keep segmentation map (needed for groups)
            keep_groupmap: If True, keep group map (rarely needed)
        """
        if not hasattr(self, 'data'):
            return
        
        self.logger.debug('Cleaning up post-detection data...')
        
        # Remove temporary background/RMS data
        temp_keys = ['background', 'rms', 'back', 'rmsimg', 'backimg']
        for band in self.data:
            for key in temp_keys:
                if key in self.data[band]:
                    del self.data[band][key]
                    self.logger.debug(f'  Deleted {band}/{key}')
        
        # Optionally remove segmentation maps
        if not keep_segmap:
            for band in self.data:
                if 'segmap' in self.data[band]:
                    del self.data[band]['segmap']
                    self.logger.debug(f'  Deleted {band}/segmap')
        
        if not keep_groupmap:
            for band in self.data:
                if 'groupmap' in self.data[band]:
                    del self.data[band]['groupmap']
                    self.logger.debug(f'  Deleted {band}/groupmap')
        
        self.logger.debug('Cleanup complete.')

    def cleanup_after_modeling(self, keep_models=True, clear_tracker=True):
        """Remove non-essential data after modeling to save memory.
        
        Clears model tracking information and computed residuals/chi images
        if they won't be written to disk. Keeps the final model catalog.
        
        Args:
            keep_models: If True, keep model_catalog (needed for writing)
            clear_tracker: If True, delete detailed model_tracker (analysis only)
        """
        if not hasattr(self, 'model_tracker'):
            return
        
        self.logger.debug('Cleaning up post-modeling data...')
        
        # Clear model tracker (detailed convergence history)
        if clear_tracker:
            self.model_tracker = OrderedDict()
            self.logger.debug('  Deleted model_tracker')
        
        # Remove computed residuals/chi images (only needed if writing)
        for band in self.data:
            for key in ['residual', 'chi']:
                if key in self.data[band]:
                    del self.data[band][key]
                    self.logger.debug(f'  Deleted {band}/{key}')
        
        self.logger.debug('Cleanup complete.')

    def cleanup_headers(self, keep_wcs_only=True):
        """Remove FITS headers from memory to save space.
        
        Headers are only needed for WCS information during image operations.
        This method keeps WCS information while removing bulky header objects.
        
        Args:
            keep_wcs_only: If True, keep only WCS not full headers
        """
        if not hasattr(self, 'headers'):
            return
        
        self.logger.debug('Cleaning up FITS headers...')
        
        if keep_wcs_only:
            # Extract WCS if not already stored
            if not hasattr(self, 'wcs'):
                self.wcs = {}
                
            for band in self.headers:
                for imgtype in self.headers[band]:
                    if band not in self.wcs:
                        try:
                            self.wcs[band] = WCS(self.headers[band][imgtype])
                            self.logger.debug(f'  Extracted WCS from {band}/{imgtype}')
                        except (KeyError, ValueError, TypeError) as e:
                            self.logger.debug(f'  Failed to extract WCS from {band}/{imgtype}: {e}')
            
            # Clear headers
            self.headers = {}
            self.logger.debug('  Deleted all headers (WCS preserved)')
        else:
            self.headers = {}
    
   
    def get_image(self, imgtype=None, band=None):

        if self.type == 'mosaic':
            image = self.data[imgtype]
        elif (imgtype in ('segmap', 'groupmap')) & (band != 'detection'):
            image = self.data[band][imgtype]
        else:
            image = self.data[band][imgtype].data

        # tsum, mean, med, std = np.nansum(image), np.nanmean(image), np.nanmedian(image), np.nanstd(image)
        self.logger.debug(f'Getting {imgtype} image for {band}')# ( {tsum:2.2f} / {mean:2.2f} / {med:2.2f} / {std:2.2f} )')
        return image

    def get_psfmodel(self, band, coord=None):
        # If you run models on a brick/mosaic **or reconstruct** one, I'll always grab the one nearest the center
        # If you run models on a group, I'll always grab the one nearest to the center of the group
        if band == 'detection':
            self.logger.warning('Detection bands do not have PSFs!')
            return
        if self.type == 'mosaic':
            psfcoords, psflist = self.data['psfcoords'], self.data['psflist']
        else:
            psfcoords, psflist = self.data[band]['psfcoords'], self.data[band]['psflist']

        if np.any(psfcoords == 'none'): # single psf!
            if coord is not None:
                self.logger.debug(f'{band} has only a single PSF! Coordinates ignored.')
            psf_path = psflist
            # Convert bytes to string if needed (HDF5 compatibility)
            try:
                psf_path = psf_path.decode('utf-8')
            except (AttributeError, UnicodeDecodeError):
                pass  # Already a string or not decodeable
            self.logger.debug(f'Found a constant PSF for {band}.')
        elif np.size(psfcoords) == 1:
            self.logger.debug(f'{band} has only a single PSF! Coordinates ignored.')
            psf_path = psflist[0]
            # Convert bytes to string if needed (HDF5 compatibility)
            try:
                psf_path = psf_path.decode('utf-8')
            except (AttributeError, UnicodeDecodeError):
                pass  # Already a string or not decodeable
        else:
            if coord is None:
                if self.type != 'group':
                    self.logger.debug(f'{band} has mutliple PSFs but no coordinates supplied. Picking the nearest.')
                coord = self.position
            
            # find nearest to coord
            psf_idx, d2d, __ = coord.match_to_catalog_sky(psfcoords, 1)
            d2d = d2d[0]
            self.logger.debug(f'Found the nearest PSF for {band} {d2d.to(u.arcmin)} away.')
            psf_path = psflist[psf_idx]
            # Convert bytes to string if needed (HDF5 compatibility)
            try:
                psf_path = psf_path.decode('utf-8')
            except (AttributeError, UnicodeDecodeError):
                pass  # Already a string or not decodeable
            # Reformat PSF filename to use zero-padded numbers (e.g., gp000123.fits)
            try:
                num = int(psf_path.split('gp')[-1].split('.fits')[0])
                psf_path = psf_path.replace(str(num), f'{num:06}')  
            except (ValueError, IndexError):
                pass  # Filename doesn't match expected pattern

        # Load PSF model from file
        if psf_path.endswith('.psf'):
            # Try loading as PsfEx format first
            try:
                psfmodel = PixelizedPsfEx(fn=psf_path)
                self.logger.debug(f'PSF model for {band} identified as PixelizedPsfEx.')

            except (ValueError, RuntimeError) as e:
                # Fall back to generic pixelized PSF format
                self.logger.debug(f'PsfEx loading failed ({e}), trying PixelizedPSF...')
                img = fits.getdata(psf_path)
                img[(img<1e-31) | np.isnan(img)] = 1e-31  # Replace bad pixels
                img = img.astype('float32')
                psfmodel = PixelizedPSF(img)
                self.logger.debug(f'PSF model for {band} identified as PixelizedPSF.')
            
        elif psf_path.endswith('.fits'):
            img = fits.getdata(psf_path)
            img[(img<1e-31) | np.isnan(img)] = 1e-31
            img = img.astype('float32')
            psfmodel = PixelizedPSF(img)
            self.logger.debug(f'PSF model for {band} identified as PixelizedPSF.')

        if conf.RENORM_PSF is not None:
            psfmodel.img *= conf.RENORM_PSF / np.nansum(psfmodel.img)
            self.logger.warning(f'PSF model has been renormalized to {conf.RENORM_PSF}. This WILL affect photometry!')

        return psfmodel

    def set_image(self, image, imgtype=None, band=None):
        if self.type == 'mosaic':
            self.data[imgtype] = image

        else:
            if imgtype in self.data[band]:
                self.data[band][imgtype].data = image
            else:
                self.data[band][imgtype] = Cutout2D(np.zeros_like(self.data[band]['science'].data), self.position, self.buffsize, wcs=self.wcs[band], mode='partial')
                self.data[band][imgtype].data = image

        self.logger.debug(f'Setting {imgtype} image for {band} (sum = {np.nansum(image):2.5f})')


    def set_property(self, value, property, band=None):
        if self.type == 'mosaic':
            self.properties[property] = value

        else:
            self.properties[band][property] = value

    def get_property(self, property, band=None):
        if self.type == 'mosaic':
            return self.properties[property]
        else:
            return self.properties[band][property]

    def get_wcs(self, band=None, imgtype='science'):

        if self.type == 'mosaic':
            return self.wcs

        else:
            return self.data[band][imgtype].wcs


    def estimate_background(self, image=None, band=None, imgtype='science'):
        
        if image is None:
            image = self.get_image(imgtype, band)
        if image.dtype.byteorder == '>':
                image = image.astype(image.dtype.newbyteorder())
        self.logger.debug(f'Estimating background...')
        background = sep.Background(image, 
                                bw = conf.BACK_BW, bh = conf.BACK_BH,
                                fw = conf.BACK_FW, fh = conf.BACK_FH)

        self.set_image(background.back(), imgtype='background', band=band)
        self.set_image(background.rms(), imgtype='rms', band=band)
        self.set_property(background.globalrms, 'rms', band)
        self.set_property(background.globalback, 'background', band)

        return background


    def _extract(self, band='detection', imgtype='science', wgttype='weight', masktype='mask', background=None):
        var = None
        mask = None
        image = self.get_image(imgtype, band) # these are cutouts, remember.

        if conf.USE_DETECTION_WEIGHT:
            try:
                wgt = self.data[band][wgttype].data
                var = np.where(wgt>0, 1/np.sqrt(wgt), 0)
            except (KeyError, AttributeError):
                raise RuntimeError(f'Weight image "{wgttype}" not found for band {band}!')
        if conf.USE_DETECTION_MASK:
            try:
                mask = self.data[band][masktype].data
            except (KeyError, AttributeError):
                raise RuntimeError(f'Mask image "{masktype}" not found for band {band}!')

        # Deal with background
        if background is None:
            background = 0
        elif ~np.isscalar(background):
            assert np.shape(background)==np.shape(image), f'Background {np.shape(background)} does not have the same shape as image {np.shape(image)}!'

        # Grab the convolution filter
        convfilt = None
        if conf.FILTER_KERNEL is not None:
            convfilt = get_detection_kernel(conf.FILTER_KERNEL)

        # Do the detection
        self.logger.debug(f'Detection will be performed with thresh = {conf.THRESH}')
        kwargs = dict(var=var, mask=mask, minarea=conf.MINAREA, filter_kernel=convfilt, 
                filter_type=conf.FILTER_TYPE, segmentation_map=True, 
                clean = conf.CLEAN, clean_param = conf.CLEAN_PARAM,
                deblend_nthresh=conf.DEBLEND_NTHRESH, deblend_cont=conf.DEBLEND_CONT)
        sep.set_extract_pixstack(conf.PIXSTACK_SIZE)
        tstart = time.time()
        catalog, segmap = sep.extract(image-background, conf.THRESH, **kwargs)

        if len(catalog) == 0:
            self.logger.error('No objects found! Check overlap of mosaic with this brick. May be OK. Exiting...')
            sys.exit()

        self.logger.info(f'Detection found {len(catalog)} sources. ({time.time()-tstart:2.2}s)')
        catalog = Table(catalog)

        # Apply mask now?
        if conf.APPLY_DETECTION_MASK & (mask is not None):
            catalog, segmap = clean_catalog(catalog, mask, segmap)
        elif conf.APPLY_DETECTION_MASK & (mask is None):
            raise RuntimeError('Cannot apply detection mask when there is no mask supplied!')

        return catalog, segmap

    def estimate_properties(self, band=None, imgtype='science'):
        self.logger.debug(f'Estimating properties for {band}...')
        # Try to retrieve cached statistics, compute if not available
        try:
            mean = self.get_property('mean', band)
            median = self.get_property('median', band)
            rms = self.get_property('clipped_rms', band)
            return mean, median, rms
        except (KeyError, AttributeError):
            # Properties not yet computed, calculate now
            image = self.get_image(imgtype, band)
            if self.type == 'mosaic':
                cleanimage = image[image!=0]
                mean = np.nanmean(cleanimage)
                median = np.nanmedian(cleanimage)
                rms = np.nanstd(cleanimage)
                del cleanimage
            else:
                mean, median, rms = sigma_clipped_stats(image[image!=0])
            self.logger.debug(f'Estimated stats of \"{imgtype}\" image (@3 sig)')
            self.logger.debug(f'    Mean:   {mean:2.3f}')
            self.logger.debug(f'    Median: {median:2.3f}')
            self.logger.debug(f'    RMS:    {rms:2.3f}')

            self.set_property(mean, 'mean', band)
            self.set_property(median, 'median', band)
            self.set_property(rms, 'clipped_rms', band)
            return mean, median, rms

    def generate_weight(self, band=None, imgtype='science', overwrite=False):
        """Generate inverse variance weight map from image RMS.
        """
        # Try to use cached RMS, compute if not available
        try:
            rms = self.get_property('clipped_rms', band)
        except (KeyError, AttributeError):
            __, __, rms = self.estimate_properties(band, imgtype)
        image = self.get_image(imgtype, band)
        if ('weight' in self.data[band].keys()) & ~overwrite:
            raise RuntimeError('Cannot overwrite exiting weight (overwrite=False)')
        weight = np.ones_like(image) * np.where(rms>0, 1/rms**2, 0)
        self.set_image(weight, 'weight', band)

        return weight

    def generate_mask(self, band=None, imgtype='weight', overwrite=False):
        """Uses zero weight portions to make an effective mask
        """

        image = self.get_image(imgtype, band)
        if ('weight' in self.data[band].keys()) & ~overwrite:
            raise RuntimeError('Cannot overwrite exiting weight (overwrite=False)')
        mask = image == 0
        self.set_image(mask, 'mask', band)

        return mask

    def get_background(self, band=None):
        if self.get_property('backtype', band=band) == 'flat':
            return self.get_property('background', band=band)
        elif self.get_property('backtype', band=band) == 'variable':
            return self.get_image(band=band, imgtype='background')

    def stage_images(self, bands=None, data_imgtype='science'):
        if bands is None:
            bands = self.get_bands()
        elif np.isscalar(bands):
            bands = [bands,]
        if 'detection' in bands:
            bands.remove('detection')

        self.images = OrderedDict()

        self.logger.debug(f'Staging images for The Tractor... (image --> {data_imgtype})')
        for band in bands:
            psfmodel = self.get_psfmodel(band=band)

            data = self.get_image(band=band, imgtype=data_imgtype)
            data[np.isnan(data)] = 0
            weight = self.get_image(band=band, imgtype='weight').copy()
            masked = self.get_image(band=band, imgtype='mask').copy()
            if self.type == 'group':
                # Mask pixels outside this group to prevent contamination
                try:
                    x, y = self.get_image(band=band, imgtype='groupmap')[self.group_id]
                    filler = np.ones(masked.shape, dtype=bool)
                    filler[x, y] = False
                    masked[filler] = 1
                    del filler
                except (KeyError, IndexError) as e:
                    self.logger.debug(f'Failed to mask group {self.group_id} in band {band} ({e}). Continuing...')
            weight[np.isnan(data) | (masked==1) | np.isnan(masked)] = 0

            # ensure that there are no nans in data or weight
            data[np.isnan(data) | ~np.isfinite(data)] = 0
            weight[np.isnan(weight) | ~np.isfinite(weight)] = 0

            if np.sum(weight) == 0:
                self.logger.debug(f'All weight pixels in {band} are zero! Skipping this band.')
                continue

            self.images[band] = Image(
                data=data,
                invvar=weight,
                psf=psfmodel,
                wcs=read_wcs(self.get_wcs(band=band, imgtype=data_imgtype)),
                photocal=FluxesPhotoCal(band),
                sky=ConstantSky(0)
            )
            self.logger.debug(f'  ✓ {band}')

    def update_models(self, bands=None, data_imgtype='science', existing_catalog=None):

        if bands is None:
            bands = self.get_bands()
        elif np.isscalar(bands):
            bands = [bands,]

        if 'detection' in bands:
            bands.remove('detection')

        # Trackers
        self.logger.debug(f'Updating models for group #{self.group_id}')

        if existing_catalog is None:
            existing_catalog = self.existing_model_catalog

        for src in self.catalogs[self.catalog_band][self.catalog_imgtype]:

            source_id = src['id']
            model_type = existing_catalog[source_id].name
            self.logger.debug(f'Source #{source_id} ({model_type})')

            # add new bands using average (zpt-corr) fluxes as guesses
            existing_bands = np.array(existing_catalog[source_id].brightness.getParamNames())
            existing_fluxes = np.array(existing_catalog[source_id].brightness.getParams())
            existing_zpt = np.array([conf.BANDS[band]['zeropoint'] for band in existing_bands])

            fluxes = OrderedDict()
            filler = {}
            for band in bands:
                if band not in existing_bands:
                    zpt = conf.BANDS[band]['zeropoint']
                    conv_fluxes = existing_fluxes * 10**(-0.4 * (existing_zpt - zpt))
                    fluxes[band] = np.mean(conv_fluxes)
                    filler[band] = 0
                else:
                    fluxes[band] = existing_fluxes[existing_bands==band][0]
                    filler[band] = 0
            self.model_catalog[source_id] = copy.deepcopy(existing_catalog[source_id])
            self.model_catalog[source_id].brightness =  Fluxes(**fluxes, order=list(fluxes.keys()))
            self.model_catalog[source_id].variance.brightness =  Fluxes(**filler, order=list(filler.keys()))

            # update priors
            self.model_catalog[source_id] = set_priors(self.model_catalog[source_id], self.phot_priors)
    
    def stage_models(self, bands=conf.MODEL_BANDS, data_imgtype='science'):
        """ Build the Tractor Model catalog for the group """

        if bands is None:
            bands = self.get_bands()
        elif np.isscalar(bands):
            bands = [bands,]

        if 'detection' in bands:
            bands.remove('detection')

        # Trackers
        if hasattr(self, 'group_id'):
            self.logger.debug(f'Staging models for group #{self.group_id}')
        else:
            self.logger.debug(f'Staging models for brick #{self.brick_id}')

        for src in self.catalogs[self.catalog_band][self.catalog_imgtype]:

            source_id = src['id']
            if source_id not in self.model_catalog:
                self.logger.debug(f'Source #{source_id} is not in the model catalog! Skipping...')
                continue
                
            # inital position
            position = RaDecPos(src['ra'], src['dec'])

            # initial fluxes
            # Only use bands that were actually staged (e.g., skip bands with all zero weight pixels)
            staged_bands = [band for band in bands if band in self.images]
            
            # If no bands were staged, skip this source
            if len(staged_bands) == 0:
                self.logger.debug(f'No bands staged for source #{source_id}. Skipping model creation.')
                continue
            qflux = np.zeros(len(staged_bands))
            for j, band in enumerate(staged_bands):
                src_seg = self.data[band]['segmap'][source_id]
                try:
                    qflux[j] = np.nansum(self.images[band].data[src_seg[0], src_seg[1]])
                except Exception as e:
                    self.logger.debug(f'Failed to sum flux for source #{source_id} in band {band}: {e}')
                    qflux[j] = 0
            flux = Fluxes(**dict(zip(staged_bands, qflux)), order=staged_bands)

            # initial shapes pa = 90. * u.deg + np.rad2deg(shape.theta) * u.deg
            pa = 90 - np.rad2deg(src['theta'])
            self.logger.debug(f'Source #{source_id}: qflux[0]={qflux[0]}, axis_ratio={src["b"]/src["a"]}, pa={pa}, theta={src["theta"]}')
            # Use the first available band for pixel scale (or detection if available)
            if 'detection' in self.pixel_scales:
                shape_band = 'detection'
            elif len(staged_bands) > 0:
                shape_band = staged_bands[0]
            else:
                self.logger.debug(f'No bands available for shape calculation for source #{source_id}.')
                continue
            pixscl = self.pixel_scales[shape_band][0].to(u.arcsec).value
            guess_radius = np.sqrt(src['a']*src['b']) * pixscl
            shape = EllipseESoft.fromRAbPhi(guess_radius, src['b'] / src['a'], pa)
            nre = SersicIndex(1) # Just a guess for the seric index
            fluxcore = Fluxes(**dict(zip(staged_bands, np.zeros(len(staged_bands)))), order=staged_bands) # Just a simple init condition

            # set shape bounds for sanity
            shape.lowers = [-5, -np.inf, -np.inf]
            shape.uppers = [1, np.inf, np.inf]

            # assign model
            if isinstance(self.model_catalog[source_id], PointSource):
                model = PointSource(position, flux)
                model.name = 'PointSource'
            elif isinstance(self.model_catalog[source_id], SimpleGalaxy):
                model = SimpleGalaxy(position, flux)
            elif isinstance(self.model_catalog[source_id], ExpGalaxy):
                model = ExpGalaxy(position, flux, shape)
            elif isinstance(self.model_catalog[source_id], DevGalaxy):
                model = DevGalaxy(position, flux, shape)
            elif isinstance(self.model_catalog[source_id], FixedCompositeGalaxy):
                model = FixedCompositeGalaxy(
                                                position, flux,
                                                SoftenedFracDev(0.5),
                                                shape, shape)
            elif isinstance(self.model_catalog[source_id], SersicGalaxy):
                model = SersicGalaxy(position, flux, shape, nre)
            elif isinstance(self.model_catalog[source_id], SersicCoreGalaxy):
                model = SersicCoreGalaxy(position, flux, shape, nre, fluxcore)

            self.logger.debug(f'Source #{source_id}: {model.name} model at {position}')
            self.logger.debug(f'               {flux}') 
            if hasattr(model, 'fluxCore'):
                self.logger.debug(f'               {fluxcore}')
            if hasattr(model, 'shape'):
                self.logger.debug(f'               {shape}')

            model = set_priors(model, self.model_priors)
            model.variance = model.copy()
            model.statistics = {}
            self.model_catalog[source_id] = model

    def optimize(self):
        self.engine.optimizer = optimizer()

        # cat = self.engine.getCatalog()
        self.logger.debug('Running engine...')
        tstart = time.time()

        # Suppress Ceres C++ optimizer output at file descriptor level (once, not per step)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stdout = os.dup(1)  # Save stdout fd
        saved_stderr = os.dup(2)  # Save stderr fd
        os.dup2(devnull_fd, 1)   # Redirect stdout
        os.dup2(devnull_fd, 2)   # Redirect stderr

        try:
            for i in range(conf.MAX_STEPS):
                # Run one optimization step
                try:
                    dlnp, X, alpha, var = self.engine.optimize(variance=True, damping=conf.DAMPING)
                except (RuntimeError, ValueError, np.linalg.LinAlgError, IndexError) as e:
                    self.logger.debug(f'Optimization failed on step {i+1}: {e}')
                    if not conf.IGNORE_FAILURES:
                        raise RuntimeError(f'Optimization failed on step {i+1}: {e}')
                    return False

                if conf.PLOT > 4:
                    self.build_all_images(set_engine=False, reconstruct=False, bands=self.engine.bands)
                    self.plot_image(tag=f's{self.stage}_n{i}', band=self.engine.bands,
                                     show_catalog=True, imgtype=('science', 'model', 'residual'))

                self.logger.debug(f'   step: {i+1} dlnp: {dlnp:2.5f}')
                if dlnp < conf.DLNP_CRIT:
                    break

            self.variance = var

            if dlnp < conf.DLNP_CRIT:
                self.logger.debug(f'Fit converged in {i+1} steps ({time.time()-tstart:2.2f}s)')
            else:
                self.logger.warning(f'Fit did not converge in {i+1} steps ({time.time()-tstart:2.2f}s)')
            for source_id in self.model_tracker:
                self.model_tracker[source_id][self.stage]['nstep'] = i+1   # TODO should check this against MAX_STEP in a binary flag output...

            # return i, dlnp, X, alpha, var
            return True
        finally:
            # Restore stdout/stderr
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(devnull_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)

    def store_models(self):
        self.logger.debug(f'Storing models...')

        cat = self.engine.getCatalog()
        cat_variance = copy.deepcopy(cat)
        cat_variance.setParams(self.variance)

        for i, source_id in enumerate(self.source_ids):
            model, variance = cat[i], cat_variance[i]
            # To store stuff like chi2, bic, residual stats, group chi2
            model.variance = variance
            self.model_tracker[source_id][self.stage]['model'] = model

            self.logger.debug(f'Source #{source_id}: {model.name} model at {model.pos}')
            self.logger.debug(f'               {model.brightness}') 
            if hasattr(model, 'fluxCore'):
                self.logger.debug(f'               {model.fluxcore}')
            if hasattr(model, 'shape'):
                self.logger.debug(f'               {model.shape}')

            if self.solved.all() | (self.stage == 11):
                low_idx = 0
                if self.stage == 11:
                    low_idx = 10
                self.model_catalog[source_id] = model
                self.model_catalog[source_id].group_id = self.group_id
                self.model_catalog[source_id].statistics = self.model_tracker[source_id][self.stage]
                for stat in self.model_tracker[source_id][low_idx]:
                    if stat in self.bands:
                        for substat in self.model_tracker[source_id][low_idx][stat]:
                            if substat.endswith('chisq'):
                                self.model_catalog[source_id].statistics[stat][f'{substat}_nomodel'] = \
                                                self.model_tracker[source_id][low_idx][stat][substat]
                    elif substat.endswith('chisq'):
                        self.model_catalog[source_id].statistics[f'{stat}_nomodel'] = \
                                                    self.model_tracker[source_id][low_idx][stat]

    def stage_engine(self, bands=conf.MODEL_BANDS):
        self.add_tracker()
        self.stage_images(bands=bands)
        
        # Check if any images were successfully staged
        if len(self.images) == 0:
            self.logger.error('No images were successfully staged. All bands may have zero weight pixels.')
            return False
            
        self.stage_models(bands=bands)
        self.engine = Tractor(list(self.images.values()), list(self.model_catalog.values()))
        self.engine.bands = list(self.images.keys())
        self.engine.freezeParam('images')

    def force_models(self, bands=None):

        if bands is None: bands = self.bands
        tstart = time.time()
        self.logger.debug('Measuring photometry...')

        self.existing_model_catalog = copy.deepcopy(self.model_catalog)
        if len(self.existing_model_catalog) == 0:
            self.logger.warning('No existing models to force!')
            return False
        self.engine = None
        self.stage = None

        self.model_priors = conf.PHOT_PRIORS
        self.reset_models()

        self.add_tracker(init_stage=10)
        self.stage_images(bands=bands)
        
        # Check if any images were successfully staged
        if len(self.images) == 0:
            self.logger.error('No images were successfully staged. All bands may have zero weight pixels.')
            return False
            
        self.stage_models(bands=bands)
        self.measure_stats(bands=bands, stage=self.stage)
        if conf.PLOT > 2:
            self.plot_image(tag=f's{self.stage}', band=bands, show_catalog=True, imgtype=('science', 'model', 'residual', 'chi'))

        self.stage = 11
        self.add_tracker()
        # Only update models with bands that have actual image data
        optimized_bands = list(self.images.keys())
        self.update_models(bands=optimized_bands)
        self.engine = Tractor(list(self.images.values()), list(self.model_catalog.values()))
        self.engine.bands = optimized_bands
        self.engine.freezeParam('images')

        status = self.optimize()
        if not status:
            return False
        
        # Add back any missing bands that were requested but had zero weight
        missing_bands = [b for b in bands if b not in optimized_bands and b != 'detection']
        if missing_bands:
            self.logger.debug(f'Adding back {len(missing_bands)} zero-weight bands to models: {missing_bands}')
            
            # First, expand self.variance to include zero variance for missing bands
            # Get the current catalog to determine parameter structure
            cat = self.engine.getCatalog()
            n_models = len(cat)
            
            # Count parameters per model for optimized bands
            n_params_per_model_optimized = len(cat[0].getParams()) if n_models > 0 else 0
            
            # Each missing band adds one parameter per model (flux)
            n_params_per_model_missing = len(missing_bands)
            
            # Expand variance array with zeros for missing band parameters
            variance_expanded = []
            for i in range(n_models):
                # Get variance for this model's optimized parameters
                start_idx = i * n_params_per_model_optimized
                end_idx = start_idx + n_params_per_model_optimized
                model_var = self.variance[start_idx:end_idx]
                
                # Add zero variance for missing band fluxes
                missing_var = [0.0] * n_params_per_model_missing
                
                # Combine: optimized params variance + missing params variance
                variance_expanded.extend(list(model_var) + missing_var)
            
            self.variance = np.array(variance_expanded)
            
            # Now add missing bands to the models
            for source_id in self.model_catalog:
                model = self.model_catalog[source_id]
                # Get existing bands and fluxes
                existing_bands = list(model.getBrightness().getParamNames())
                existing_fluxes = dict(zip(existing_bands, model.getBrightness().getParams()))
                existing_var_fluxes = dict(zip(existing_bands, model.variance.getBrightness().getParams()))
                
                # Add missing bands with zero flux and variance
                for band in missing_bands:
                    existing_fluxes[band] = 0.0
                    existing_var_fluxes[band] = 0.0
                
                # Rebuild brightness in the order: optimized bands first, then missing bands
                all_bands = optimized_bands + missing_bands
                model.brightness = Fluxes(**{b: existing_fluxes[b] for b in all_bands}, order=all_bands)
                model.variance.brightness = Fluxes(**{b: existing_var_fluxes[b] for b in all_bands}, order=all_bands)
            
            # Add placeholder statistics for missing bands so model_tracker doesn't break
            for band in missing_bands:
                if self.type == 'group':
                    self.model_tracker[self.type][self.stage][band] = {
                        'chisq': 0.0, 'rchisq': 0.0, 'rchisqmodel': np.nan,
                        'ndata': 0, 'nparam': 0, 'ndof': 0, 'nres': 0,
                        'chi_k2': np.nan
                    }
                    for pc in (5, 16, 50, 84, 95):
                        self.model_tracker[self.type][self.stage][band][f'chi_pc{pc:02d}'] = np.nan
                
                for source_id in self.source_ids:
                    self.model_tracker[source_id][self.stage][band] = {
                        'chisq': 0.0, 'rchisq': 0.0, 'rchisqmodel': np.nan,
                        'ndata': 0, 'nparam': 0, 'ndof': 0, 'nres': 0,
                        'chi_k2': np.nan, 'flag': True
                    }
                    for pc in (5, 16, 50, 84, 95):
                        self.model_tracker[source_id][self.stage][band][f'chi_pc{pc}'] = np.nan
        
        self.measure_stats(bands=optimized_bands, stage=self.stage) 
        self.store_models()
        if conf.PLOT > 2:
                self.plot_image(band=bands, imgtype=('science', 'model', 'residual'))     
        if conf.PLOT > 0:
                self.plot_summary(bands=bands, source_id='group', tag='PHOT')

        self.logger.info(f'Photometry completed ({time.time()-tstart:2.2f}s)')


    def determine_models(self, bands=conf.MODEL_BANDS):

        self.logger.debug('Determining best-choice models...')

        # clean up
        self.model_priors = conf.MODEL_PRIORS
        self.reset_models()

        # stage 0
        self.add_tracker()
        self.stage_images(bands=bands)
        
        # Check if any images were successfully staged
        if len(self.images) == 0:
            self.logger.error('No images were successfully staged. All bands may have zero weight pixels.')
            return False
            
        # self.stage_models(bands=bands)
        # self.measure_stats(bands=bands, stage=self.stage)
        if conf.PLOT > 2:
            self.plot_image(tag=f's{self.stage}', band=bands, show_catalog=True, imgtype=('science', 'model', 'residual', 'chi'))

        tstart = time.time()
        while not self.solved.all():
            self.stage += 1
            self.add_tracker()
            self.stage_models(bands=bands)
        
            self.engine = Tractor(list(self.images.values()), list(self.model_catalog.values()))
            self.engine.bands = list(self.images.keys())
            self.engine.freezeParam('images')

            if conf.PLOT > 3:
                self.build_all_images(reconstruct=False, set_engine=False)
                self.plot_image(tag=f's{self.stage}_init', band=bands, show_catalog=True, imgtype=('science', 'model', 'residual', 'chi'))

            status = self.optimize()
            if not status:
                return False
            self.measure_stats(bands=bands, stage=self.stage)
            self.store_models()
            if conf.PLOT > 2:
                self.plot_image(tag=f's{self.stage}', band=bands, show_catalog=True, imgtype=('science', 'model', 'residual', 'chi'))
            self.decision_tree()

        # run final time
        self.stage += 1
        self.add_tracker()
        self.stage_models(bands=bands)
    
        self.engine = Tractor(list(self.images.values()), list(self.model_catalog.values()))
        self.engine.bands = list(self.images.keys())
        self.engine.freezeParam('images')

        status = self.optimize()
        if not status:
            return False

        self.measure_stats(bands=bands, stage=self.stage) 
        self.store_models()
        if conf.PLOT > 2:
                self.plot_image(band=bands, imgtype=('science', 'model', 'residual'))     
        if conf.PLOT > 1:
                self.plot_summary(bands=bands, source_id='group', tag='MODEL')

        self.logger.info(f'Modelling completed ({time.time()-tstart:2.2f}s)')
        return True

    def decision_tree(self):

        self.logger.debug('Running Decision Tree...')
        for i, source_id in enumerate(self.source_ids):

            if not self.solved[self.source_ids == source_id]:       
                # After stage 1, reset to SimpleGalaxies
                if self.stage == 1:
                    self.model_catalog[source_id] = SimpleGalaxy(None, None)
                    self.logger.debug(f' Source #{source_id} ... trying SimpleGalaxy')

                # After stage 2, compare PS(1) to SG(2)
                if self.stage == 2:
                    ps_chi2 = self.model_tracker[source_id][1]['total']['rchisq']
                    sg_chi2 = self.model_tracker[source_id][2]['total']['rchisq']

                    if (ps_chi2 > conf.SUFFICIENT_THRESH) & (sg_chi2 > conf.SUFFICIENT_THRESH): # neither win
                        self.model_catalog[source_id] = ExpGalaxy(None, None, None)
                        self.logger.debug(f' Source #{source_id} ... trying ExpGalaxy')
                        continue

                    delta_chi2 = ps_chi2 - (sg_chi2 + conf.SIMPLEGALAXY_PENALTY)
                    if (delta_chi2 <= 0) & (ps_chi2 <= conf.SUFFICIENT_THRESH): # pointsource wins
                        self.model_catalog[source_id] = PointSource(None, None)
                        self.solved[i] = True
                        self.logger.debug(f' Source #{source_id} ... solved as PointSource')
                        continue

                    # elif (sg_chi2 <= conf.SUFFICIENT_THRESH): # simplegalaxy wins outright
                    #     self.model_catalog[source_id] = SimpleGalaxy(None, None, None)
                    #     self.solved[i] = True
                    #     self.logger.debug(f' Source #{source_id} ... solved as SimpleGalaxy')
                    #     continue

                    else: # Just continue
                        self.model_catalog[source_id] = ExpGalaxy(None, None, None)
                        self.logger.debug(f' Source #{source_id} ... trying ExpGalaxy')
                        continue

                # After stage 3, try DevGalaxy
                if self.stage == 3:
                    self.model_catalog[source_id] = DevGalaxy(None, None, None)
                    self.logger.debug(f' Source #{source_id} ... trying DevGalaxy')
                    continue

                # After stage 4, compare EXP(3) to DEV(4)
                if self.stage == 4:
                    ps_chi2 = self.model_tracker[source_id][1]['total']['rchisq']
                    sg_chi2 = self.model_tracker[source_id][2]['total']['rchisq']
                    exp_chi2 = self.model_tracker[source_id][3]['total']['rchisq']
                    dev_chi2 = self.model_tracker[source_id][4]['total']['rchisq']

                    if (exp_chi2 > conf.SUFFICIENT_THRESH) & (dev_chi2 > conf.SUFFICIENT_THRESH): # neither win
                        self.model_catalog[source_id] = FixedCompositeGalaxy(None, None, None, None, None)
                        self.logger.debug(f' Source #{source_id} ... trying CompostieGalaxy')
                        continue

                    elif (sg_chi2 <= exp_chi2) & (sg_chi2 <= dev_chi2) & (sg_chi2 <= conf.SUFFICIENT_THRESH): # sg is better, go back
                        self.model_catalog[source_id] = SimpleGalaxy(None, None, None)
                        self.solved[i] = True
                        self.logger.debug(f' Source #{source_id} ... solved as SimpleGalaxy')
                        continue

                    elif abs(exp_chi2 - dev_chi2) < conf.EXP_DEV_SIMILAR_THRESH: # if exp and dev are similar
                        self.model_catalog[source_id] = FixedCompositeGalaxy(None, None, None, None, None)
                        self.logger.debug(f' Source #{source_id} ... trying CompostieGalaxy')
                        continue

                    elif (exp_chi2 < dev_chi2) & (exp_chi2 <= conf.SUFFICIENT_THRESH): # exp is just bettter and wins
                        self.model_catalog[source_id] = ExpGalaxy(None, None, None)
                        self.solved[i] = True
                        self.logger.debug(f' Source #{source_id} ... solved as ExpGalaxy')
                        continue

                    elif (dev_chi2 < exp_chi2) & (dev_chi2 <= conf.SUFFICIENT_THRESH): # dev is just better and wins
                        self.model_catalog[source_id] = DevGalaxy(None, None, None)
                        self.solved[i] = True
                        self.logger.debug(f' Source #{source_id} ... solved as DevGalaxy')
                        continue

                # After stage 5, check exp and dev
                if self.stage == 5:
                    exp_chi2 = self.model_tracker[source_id][3]['total']['rchisq']
                    dev_chi2 = self.model_tracker[source_id][4]['total']['rchisq']
                    comp_chi2 = self.model_tracker[source_id][5]['total']['rchisq']

                    if (exp_chi2 <= comp_chi2) & (exp_chi2 <= conf.SUFFICIENT_THRESH): # exp wins
                        self.model_catalog[source_id] = ExpGalaxy(None, None, None)
                        self.solved[i] = True
                        self.logger.debug(f' Source #{source_id} ... solved as ExpGalaxy')
                        continue

                    elif (dev_chi2 <= comp_chi2) & (dev_chi2 <= conf.SUFFICIENT_THRESH): # dev wins
                        self.model_catalog[source_id] = DevGalaxy(None, None, None)
                        self.solved[i] = True
                        self.logger.debug(f' Source #{source_id} ... solved as DevGalaxy')
                        continue

                    elif (comp_chi2 <= conf.SUFFICIENT_THRESH): # comp wins
                        self.model_catalog[source_id] = FixedCompositeGalaxy(None, None, None, None, None)
                        self.solved[i] = True
                        self.logger.debug(f' Source #{source_id} ... solved as CompositeGalaxy')
                        continue

                    else: # wow that thing is crap. Pick the lowest chi2
                        ps_chi2 = self.model_tracker[source_id][1]['total']['rchisq']
                        sg_chi2 = self.model_tracker[source_id][2]['total']['rchisq']
                        chi2 = np.array([ps_chi2, sg_chi2, exp_chi2, dev_chi2, comp_chi2])

                        if np.argmin(chi2) == 0:
                            self.model_catalog[source_id] = PointSource(None, None)
                            self.solved[i] = True
                            self.logger.debug(f' Source #{source_id} ... solved as PointSource')
                            continue

                        if np.argmin(chi2) == 1:
                            self.model_catalog[source_id] = SimpleGalaxy(None, None)
                            self.solved[i] = True
                            self.logger.debug(f' Source #{source_id} ... solved as SimpleGalaxy')
                            continue

                        if np.argmin(chi2) == 2:
                            self.model_catalog[source_id] = ExpGalaxy(None, None, None)
                            self.solved[i] = True
                            self.logger.debug(f' Source #{source_id} ... solved as ExpGalaxy')
                            continue

                        if np.argmin(chi2) == 3:
                            self.model_catalog[source_id] = DevGalaxy(None, None, None)
                            self.solved[i] = True
                            self.logger.debug(f' Source #{source_id} ... solved as DevGalaxy')
                            continue

                        if np.argmin(chi2) == 4:
                            self.model_catalog[source_id] = FixedCompositeGalaxy(None, None, None, None, None)
                            self.solved[i] = True
                            self.logger.debug(f' Source #{source_id} ... solved as CompositeGalaxy')
                            continue

                        self.logger.warning(f' Source #{source_id} did not meet Chi2 requirements! ({np.argmin(chi2):2.2f} > {conf.SUFFICIENT_THRESH}:2.2f)')

    def measure_stats(self, bands=None, stage=None):
        if bands is None:
            bands = self.engine.bands
        elif np.isscalar(bands):
            bands = [bands,]

        # Use current stage if not explicitly provided
        if stage is None:
            try:
                stage = self.stage
            except AttributeError:
                pass  # No stage set yet

        self.build_all_images(bands=bands, reconstruct=False)

        q_pc = (5, 16, 50, 84, 95)

        self.logger.debug('Measuing statistics...')
        # group Chi2, <Chi>, sig(Chi)
        if self.type == 'group':
            self.logger.debug(f'{self.type} #{self.group_id}')
            ntotal_pix = 0
            ntotalres_elem = 0
            totchi = []
            rchi2_model_top = []
            rchi2_model_bot = []
            for band in bands:
                # Skip bands that were not staged (e.g., all weight pixels are zero)
                if band not in self.images:
                    self.logger.debug(f'Band {band} not in staged images. Skipping.')
                    continue
                    
                groupmap = self.get_image('groupmap', band)[self.group_id]
                segmap = self.get_image('segmap', band)
                chi = self.get_image('chi', band=band)[groupmap[0], groupmap[1]].flatten()
                totchi += list(chi)
                chi2, chi_pc = np.sum(chi**2), np.nanpercentile(chi, q=q_pc)
                if np.isscalar(chi_pc):
                    chi_pc = np.nan * np.ones(5)
                area = 0
                for source_id in self.catalogs[self.catalog_band][self.catalog_imgtype]['id']:
                    if source_id not in segmap:
                        self.logger.debug(f'Source {source_id} missing segmap for band {band}. Skipping.')
                        continue
                    data = self.images[band].data.copy()
                    segmask = np.zeros(shape=data.shape, dtype=bool)
                    segmask[segmap[source_id][0], segmap[source_id][1]] = True
                    data[(self.images[band].invvar <= 0) | ~segmask] = 0
                    area += get_fwhm(data)**2
                nres_elem = area / (get_fwhm(self.images[band].psf.img))**2
                ndata = np.sum(self.images[band].invvar[groupmap[0], groupmap[1]] > 0) # number of pixels
                try:
                    nparam = self.engine.getCatalog().numberOfParams() - np.sum(np.array(bands)!=band)  
                    model_bands = self.engine.bands
                    src_model = self.engine.getModelImage(model_bands == band)
                    chi_model = self.engine.getChiImage(model_bands == band)
                    rchi2_model = np.sum(chi_model**2 * src_model) / np.sum(src_model)
                    rchi2_model_top.append(np.sum(chi_model**2 * src_model))
                    rchi2_model_bot.append(np.sum(src_model))
                except (AttributeError, ValueError, ZeroDivisionError) as e:
                    self.logger.debug(f'Chi image calculation failed for {band}: {e}')
                    nparam = 0
                    rchi2_model = np.nan
                    rchi2_model_top.append(np.nan)
                    rchi2_model_bot.append(np.nan)
                ndof = np.max([1, ndata - nparam])
                rchi2 = chi2 / ndof
                
                self.logger.debug(f'   {band}: chi2/N = {rchi2:2.2f} ({rchi2_model:2.2f})')
                self.logger.debug(f'   {band}: N(data) = {ndata} ({nres_elem})')
                self.logger.debug(f'   {band}: N(param) = {nparam}')
                self.logger.debug(f'   {band}: N(DOF) = {ndof}')
                self.logger.debug(f'   {band}: Med(chi) = {chi_pc[2]:2.2f}')
                self.logger.debug(f'   {band}: Width(chi) = {chi_pc[3]-chi_pc[1]:2.2f}')
                
                ntotal_pix += ndata 
                ntotalres_elem += nres_elem
                if stage is not None:
                    self.model_tracker[self.type][stage][band] = {}
                    self.model_tracker[self.type][stage][band]['chisq'] = chi2
                    self.model_tracker[self.type][stage][band]['rchisq'] = chi2 /ndof
                    self.model_tracker[self.type][stage][band]['rchisqmodel'] = rchi2_model
                    for pc, chi_npc in zip(q_pc, chi_pc):
                        self.model_tracker[self.type][stage][band][f'chi_pc{pc:02d}'] = chi_npc
                    if len(chi) >= 8:
                        self.model_tracker[self.type][stage][band]['chi_k2'] = stats.normaltest(chi)[0]
                    else:
                        self.model_tracker[self.type][stage][band]['chi_k2'] = np.nan
                    self.model_tracker[self.type][stage][band]['ndata'] = ndata
                    self.model_tracker[self.type][stage][band]['nparam'] = nparam
                    self.model_tracker[self.type][stage][band]['ndof'] = ndof
                    self.model_tracker[self.type][stage][band]['nres'] = nres_elem
            try:
                nparam = self.engine.getCatalog().numberOfParams()
            except (AttributeError, TypeError) as e:
                self.logger.debug(f'Failed to get number of parameters: {e}')
                nparam = 0
            ndof = np.max([len(bands), ntotal_pix - nparam])
            # Only include bands that were staged and have model_tracker entries
            tracked_bands = [band for band in bands if band in self.model_tracker[self.type][stage]]
            chi2 = np.sum(np.array([self.model_tracker[self.type][stage][band]['chisq'] for band in tracked_bands]))
            self.model_tracker[self.type][stage]['total'] = {}
            self.model_tracker[self.type][stage]['total']['chisq'] = chi2
            tot_rchi2_model = np.sum(rchi2_model_top) / np.sum(rchi2_model_bot)
            self.model_tracker[self.type][stage]['total']['rchisqmodel'] = tot_rchi2_model
            self.model_tracker[self.type][stage]['total']['rchisq'] = chi2 / ndof
            chi_pc = np.nanpercentile(totchi, q=q_pc)
            for pc, chi_npc in zip(q_pc, chi_pc):
                self.model_tracker[self.type][stage]['total'][f'chi_pc{pc}'] = chi_npc
            if len(totchi) >= 8:
                self.model_tracker[self.type][stage]['total']['chi_k2'] = stats.normaltest(totchi)[0]
            else:
                self.model_tracker[self.type][stage]['total']['chi_k2'] = np.nan
            self.model_tracker[self.type][stage]['total']['ndata'] = ntotal_pix
            self.model_tracker[self.type][stage]['total']['nparam'] = nparam
            self.model_tracker[self.type][stage]['total']['ndof'] = ndof
            self.model_tracker[self.type][stage]['total']['nres'] = ntotalres_elem

            self.logger.debug(f'   Total: chi2/N = {chi2/ndof:2.2f} ({tot_rchi2_model:2.2f})')
            self.logger.debug(f'   Total: N(data) = {ntotal_pix} ({ntotalres_elem})')
            self.logger.debug(f'   Total: N(param) = {nparam}')
            self.logger.debug(f'   Total: N(DOF) = {ndof}')
            self.logger.debug(f'   Total: Med(chi) = {chi_pc[2]:2.2f}')
            self.logger.debug(f'   Total: Width(chi) = {chi_pc[3]-chi_pc[1]:2.2f}')

        for i, src in enumerate(self.catalogs[self.catalog_band][self.catalog_imgtype]):
            source_id = src['id']
            model = self.model_catalog[source_id]
            try:
                modelname = model.name
            except AttributeError:
                modelname = 'PointSource'
            if self.stage == 0:
                modelname = 'None'
            issolved = ''
            if self.solved[i]:
                issolved = ' - SOLVED'
            self.logger.debug(f'Source #{source_id} ({modelname}{issolved})')
            ntotal_pix = 0
            ntotalres_elem = 0
            totchi = []
            rchi2_model_top = []
            rchi2_model_bot = []
            for band in bands:
                # Skip bands that were not staged (e.g., all weight pixels are zero)
                if band not in self.images:
                    self.logger.debug(f'Band {band} not in staged images. Skipping.')
                    continue
                    
                segmap = self.get_image('segmap', band)
                if source_id not in segmap:
                    self.logger.debug(f'Source {source_id} missing segmap for band {band}. Skipping.')
                    continue
                chi = self.get_image('chi', band=band)[segmap[source_id][0], segmap[source_id][1]].flatten()
                totchi += list(chi)
                chi2, chi_pc = np.nansum(chi**2), np.nanpercentile(chi, q=q_pc)
                if np.isscalar(chi_pc):
                    chi_pc = np.nan * np.ones(len(q_pc))
                
                data = self.images[band].data.copy()
                segmask = np.zeros(shape=data.shape, dtype=bool)
                segmask[segmap[source_id][0], segmap[source_id][1]] = True
                data[(self.images[band].invvar <= 0) | ~segmask] = 0
                ndata = len(segmap[source_id][0]) # number of pixels
                nres_elem = (get_fwhm(data) / get_fwhm(self.images[band].psf.img))**2
                sci = self.get_image('science', band)
                wht = self.get_image('weight', band)
                mask = self.get_image('mask', band)
                flag = np.sum((sci[segmask] == 0) | (np.isnan(sci[segmask])) | (wht[segmask] <= 0) | (mask[segmask] == 1)  ) > 0
                try:
                    nparam = self.model_catalog[source_id].numberOfParams() - np.nansum(np.array(bands)!=band).astype(np.int32)
                    tr = Tractor([self.images[band],], Catalog(*[model,]))
                    src_model = tr.getModelImage(0)
                    chi_model = tr.getChiImage(0)
                    rchi2_model = np.nansum(chi_model**2 * src_model) / np.nansum(src_model)
                    rchi2_model_top.append(np.nansum(chi_model**2 * src_model))
                    rchi2_model_bot.append(np.nansum(src_model))
                except (AttributeError, ValueError, ZeroDivisionError) as e:
                    self.logger.debug(f'Chi image calculation failed for source {source_id} in {band}: {e}')
                    nparam = 0
                    rchi2_model = np.nan
                    rchi2_model_top.append(np.nan)
                    rchi2_model_bot.append(np.nan)
                ndof = np.max([1, ndata - nparam]).astype(np.int32)
                rchi2 = chi2 / ndof
                
                self.logger.debug(f'   {band}: chi2/N = {rchi2:2.2f} ({rchi2_model:2.2f})')
                self.logger.debug(f'   {band}: N(data) = {ndata} ({nres_elem})')
                self.logger.debug(f'   {band}: N(param) = {nparam}')
                self.logger.debug(f'   {band}: N(DOF) = {ndof}')
                self.logger.debug(f'   {band}: Med(chi) = {chi_pc[2]:2.2f}')
                self.logger.debug(f'   {band}: Width(chi) = {chi_pc[3]-chi_pc[1]:2.2f}')
                self.logger.debug(f'   {band}: Flagged? {flag}')

                ntotal_pix += ndata
                ntotalres_elem += nres_elem
                if stage is not None:
                    self.model_tracker[source_id][stage][band] = {}
                    self.model_tracker[source_id][stage][band]['rchisq'] = chi2 / ndof
                    self.model_tracker[source_id][stage][band]['rchisqmodel'] = rchi2_model
                    self.model_tracker[source_id][stage][band]['chisq'] = chi2
                    for pc, chi_npc in zip(q_pc, chi_pc):
                        self.model_tracker[source_id][stage][band][f'chi_pc{pc}'] = chi_npc
                    if len(chi) >= 8:
                        self.model_tracker[source_id][stage][band]['chi_k2'] = stats.normaltest(chi)[0]
                    else:
                        self.model_tracker[source_id][stage][band]['chi_k2'] = np.nan
                    self.model_tracker[source_id][stage][band]['ndata'] = ndata
                    self.model_tracker[source_id][stage][band]['nparam'] = nparam
                    self.model_tracker[source_id][stage][band]['ndof'] = ndof
                    self.model_tracker[source_id][stage][band]['nres'] = nres_elem
                    self.model_tracker[source_id][stage][band]['flag'] = flag
            try:
                nparam = self.model_catalog[source_id].numberOfParams()
            except (AttributeError, TypeError, KeyError) as e:
                self.logger.debug(f'Failed to get parameters for source {source_id}: {e}')
                nparam = 0
            ndof = np.max([len(bands), ntotal_pix - nparam]).astype(np.int32)
            # Only include bands that were staged and have model_tracker entries
            tracked_bands = [band for band in bands if band in self.model_tracker[source_id][stage]]
            chi2 = np.sum(np.array([self.model_tracker[source_id][stage][band]['chisq'] for band in tracked_bands]))
            self.model_tracker[source_id][stage]['total'] = {}
            self.model_tracker[source_id][stage]['total']['rchisq'] = chi2 / ndof
            tot_rchi2_model = np.sum(rchi2_model_top) / np.sum(rchi2_model_bot)
            self.model_tracker[source_id][stage]['total']['rchisqmodel'] = tot_rchi2_model
            self.model_tracker[source_id][stage]['total']['chisq'] = chi2
            if not np.isnan(totchi).all():
                for pc, chi_npc in zip(q_pc, np.nanpercentile(totchi, q=q_pc)):
                    self.model_tracker[source_id][stage]['total'][f'chi_pc{pc}'] = chi_npc
            else:
                for pc in q_pc:
                    self.model_tracker[source_id][stage]['total'][f'chi_pc{pc}'] = np.nan
            if len(totchi) >= 8:
                self.model_tracker[source_id][stage]['total']['chi_k2'] = stats.normaltest(totchi)[0]
            else:
                self.model_tracker[source_id][stage]['total']['chi_k2'] = np.nan
            self.model_tracker[source_id][stage]['total']['ndata'] = ntotal_pix
            self.model_tracker[source_id][stage]['total']['nparam'] = nparam
            self.model_tracker[source_id][stage]['total']['ndof'] = ndof
            self.model_tracker[source_id][stage]['total']['nres'] = ntotalres_elem

            self.logger.debug(f'   Total: chi2/N = {chi2/ndof:2.2f} ({tot_rchi2_model:2.2f})')
            self.logger.debug(f'   Total: N(data) = {ntotal_pix} ({ntotalres_elem})')
            self.logger.debug(f'   Total: N(param) = {nparam}')
            self.logger.debug(f'   Total: N(DOF) = {ndof}')
            self.logger.debug(f'   Total: Med(chi) = {chi_pc[2]:2.2f}')
            self.logger.debug(f'   Total: Width(chi) = {chi_pc[3]-chi_pc[1]:2.2f}')

    def build_all_images(self, bands=None, source_id=None, overwrite=True, reconstruct=True, set_engine=True):
        if bands is None:
            bands = [band for band in self.bands if band != 'detection']
        elif np.isscalar(bands):
            bands = [bands,]

        # check all bands have group and segmaps
        for band in bands:
            # Skip bands that weren't loaded (e.g., all weight pixels are zero)
            if band not in self.data:
                self.logger.debug(f'Band {band} not in data. Skipping transfer_maps.')
                continue
            if 'segmap' not in self.data[band]:
                self.transfer_maps(band)

        self.stage_images(bands=bands) # assumes a single PSF!

        self.build_model_image(bands, source_id, overwrite=overwrite, reconstruct=reconstruct, set_engine=set_engine)
        self.build_residual_image(bands, overwrite=overwrite)
        self.build_chi_image(bands, overwrite=overwrite)

    def build_model_image(self, bands=None, source_id=None, overwrite=True, reconstruct=True, set_engine=True):
        if bands is None:
            bands = self.bands
        elif np.isscalar(bands):
            bands = [bands,]

        if np.isscalar(source_id):
            source_id = [source_id,]

        models = {}

        # check that the model_catalog is ok
        if not set_engine:
            use_sources = list(self.model_catalog.values())
            use_source_ids = list(self.model_catalog.keys())
        else:
            use_sources = []
            use_source_ids = []
            for source in self.get_catalog(self.catalog_band, self.catalog_imgtype):
                sid = source['id']

                if source_id is not None:
                    if sid not in source_id:
                        continue

                if sid not in self.model_catalog:
                    self.logger.warning(f'Source {sid} not in model catalog')
                    continue
                
                src = self.model_catalog[sid]

                # Only use models with the photometry bands requested (must have ALL)
                if not np.all([testband in src.getBrightness().order for testband in bands]):
                    self.logger.debug(f'Source {sid} does not have photometry in all requested bands')
                    continue

                # check for rejections
                if reconstruct & ((conf.RESIDUAL_BA_MIN != 'none') | (conf.RESIDUAL_REFF_MAX != 'none') | (conf.RESIDUAL_SHOW_NEGATIVE == False)):
                    if not hasattr(src, 'statistics'):
                        self.logger.warning(f'Source {sid} does not have statistics!')
                        continue
                    source = get_params(src)
                    if conf.RESIDUAL_SHOW_NEGATIVE == False:
                        for band in source['_bands']:
                            flux = src.getBrightness().getFlux(band)
                            if flux < 0: 
                                src.getBrightness().setFlux(band, 0)
                                self.logger.warning(f'Source {sid} has negative model flux in {band} and has been rejected from reconstruction')
                    if (conf.RESIDUAL_BA_MIN != 'none') & ('ba' in source):
                        if source['ba'] < conf.RESIDUAL_BA_MIN:
                            self.logger.warning(f'Source {sid} model is too narrow and has been rejected from reconstruction')
                            continue
                    if (conf.RESIDUAL_REFF_MAX != 'none') & ('reff' in source):
                        if source['reff'] > conf.RESIDUAL_REFF_MAX:
                            self.logger.warning(f'Source {sid} model is too large and has been rejected from reconstruction')
                            continue

                use_sources.append(src)
                use_source_ids.append(sid)
                
            self.logger.debug(f'Only including the {len(use_sources)} ({100*len(use_sources)/len(self.model_catalog):2.1f}%) valid models with photometry in all requested bands')

        for band in bands:
            # Skip bands that were not staged (e.g., all weight pixels are zero)
            if band not in self.images:
                self.logger.debug(f'Band {band} not in staged images. Skipping.')
                continue
                
            if (self.data[band]['psfcoords'] == 'none'):
                model = Tractor([self.images[band],], Catalog(*use_sources)).getModelImage(0)
            else: # Uses a different PSF for each group
                model = np.zeros_like(self.get_image('science', band))
                group_ids = list(self.data[band]['groupmap'].keys())
                catalog = self.catalogs[self.catalog_band]['science']
                wcs = self.get_wcs(band=band)
                for group_id in group_ids:
                    group_sources = np.array(catalog[catalog['group_id'] == group_id]['id'])
                    cy, cx = self.data[band]['groupmap'][group_id]
                    coord = wcs.pixel_to_world(np.mean(cy), np.mean(cx))
                    try:
                        psfmodel = self.get_psfmodel(band, coord)
                    except (KeyError, ValueError, IndexError) as e:
                        self.logger.debug(f'Failed to get PSF at coord for {band}, using global PSF: {e}')
                        psfmodel = self.get_psfmodel(band) # default to the global PSF for brick

                    group_models = []
                    for source_id in group_sources:
                        if source_id in use_source_ids:
                            group_models.append(self.model_catalog[source_id])
                    
                    group_image = self.images[band].copy()
                    group_image.psf = psfmodel
                    model += Tractor([group_image,], Catalog(*group_models)).getModelImage(0)

            self.set_image(model, 'model', band)
            self.headers[band]['model'] = self.headers[band]['science']

            self.logger.debug(f'Built model image for {band}')
            if len(bands) == 1:
                return model
            else:
                models[band] = model

        return models
        
    def build_residual_image(self, bands=None, source_id=None, imgtype='science', overwrite=True):
        if bands is None:
            bands = self.bands
        elif np.isscalar(bands):
            bands = [bands,]
        residuals = {}

        for band in bands:
            # Skip bands that were not staged (e.g., all weight pixels are zero)
            if band not in self.images:
                self.logger.debug(f'Band {band} not in staged images. Skipping.')
                continue
                
            if source_id is not None:
                model = self.build_model_image(band, source_id, overwrite, reconstruct=False)
                residual = self.get_image(imgtype, band) - model
            else:
                model = self.get_image('model', band)
                residual = self.get_image(imgtype, band) - model
                self.set_image(residual, 'residual', band)

            self.headers[band]['residual'] = self.headers[band][imgtype]
            self.logger.debug(f'Built residual image for {band}')
            if len(bands) == 1:
                return residual
            else:
                residuals[band] = residual
                    
        return residuals            

    def build_chi_image(self, bands=None, source_id=None, imgtype='science', overwrite=True):
        if bands is None:
            bands = self.bands
        elif np.isscalar(bands):
            bands = [bands,]
        chis = {}

        for band in bands:
            # Skip bands that were not staged (e.g., all weight pixels are zero)
            if band not in self.images:
                self.logger.debug(f'Band {band} not in staged images. Skipping.')
                continue
                
            if source_id is not None:
                residual = self.build_residual_image(band, source_id, imgtype, overwrite)
                chi = residual * np.sqrt(self.get_image('weight', band))
            else:
                chi = self.get_image('residual', band) * np.sqrt(self.get_image('weight', band))
                self.set_image(chi, 'chi', band)
            
            self.headers[band]['chi'] = self.headers[band][imgtype]
            self.logger.debug(f'Built chi image for {band}')
            if len(bands) == 1:
                return chi
            else:
                chis[band] = chi
                    
        return chis

    def get_catalog(self, catalog_band='detection', catalog_imgtype='science'):
        return self.catalogs[catalog_band][catalog_imgtype]
    
    def set_catalog(self, catalog, catalog_band='detection', catalog_imgtype='science'):
        self.catalogs[catalog_band][catalog_imgtype] = catalog

    def plot_image(self, band=None, imgtype=None, tag='', show_catalog=True, catalog_band='detection', catalog_imgtype='science', show_groups=True):
        # for each band, plot all available images: science, weight, mask, segmap, groupmap, background, rms
        if band is None:
            bands = self.get_bands()
        elif np.isscalar(band):
            bands = [band,]
        else:
            bands = band

        if self.type == 'group':
            show_groups = False

        if (tag != '') & (not tag.startswith('_')):
            tag = '_' + tag

        in_imgtype = imgtype

        outname = os.path.join(conf.PATH_FIGURES, self.filename.replace('.h5', f'_images{tag}.pdf'))
        pdf = matplotlib.backends.backend_pdf.PdfPages(outname)

        for band in bands:
            # Skip bands that weren't loaded (e.g., all weight pixels are zero)
            if (self.type != 'mosaic') and (band not in self.data):
                self.logger.debug(f'Band {band} not in data. Skipping plot_image.')
                continue
                
            if in_imgtype is None:
                if self.type == 'mosaic':
                    imgtypes = self.data.keys()
                else:
                    imgtypes = self.data[band].keys()
            elif np.isscalar(in_imgtype):
                imgtypes = [in_imgtype,]
            else:
                imgtypes = in_imgtype

            
            for imgtype in imgtypes:
                if imgtype.startswith('psf'):
                    continue

                if self.type == 'mosaic':
                    if imgtype not in self.data.keys():
                        continue
                else:
                    if imgtype not in self.data[band].keys():
                        continue

                fig = plt.figure(figsize=(20,20))
                ax = fig.add_subplot(projection=self.get_wcs(band))
                ax.set_title(f'{band} {imgtype} {tag}')


                self.logger.debug(f'Gathering image: {band} {imgtype}')
                image = self.get_image(band=band, imgtype=imgtype)

                background = 0
                if (imgtype in ('science',)) & self.get_property('subtract_background', band=band):
                    background = self.get_background(band)

                if imgtype in ('science', 'model', 'residual'):
                    # log-scaled
                    vmax, rms = np.nanpercentile(image, q=99), self.get_property('rms', band=band)
                    if vmax < rms:
                        vmax = 3*rms
                    norm = LogNorm(rms, vmax)
                    options = dict(cmap='Greys', norm=norm, origin='lower')
                    im = ax.imshow(image - background, **options)
                    # fig.colorbar(im, orientation="horizontal", pad=0.2)
                    pixscl = self.pixel_scales[band][0].to(u.deg).value, self.pixel_scales[band][1].to(u.deg).value
                    if self.type == 'brick':
                        brick_buffer_pix = conf.BRICK_BUFFER.to(u.deg).value / pixscl[0], conf.BRICK_BUFFER.to(u.deg).value / pixscl[1]
                        ax.add_patch(Rectangle(brick_buffer_pix, self.size[0].to(u.deg).value / pixscl[0], self.size[1].to(u.deg).value / pixscl[1],
                                        fill=False, alpha=0.3, edgecolor='purple', linewidth=1))
                    # show centroids
                    if show_catalog & (catalog_band in self.catalogs.keys()):
                        if catalog_imgtype in self.catalogs[catalog_band].keys():
                            coords = SkyCoord(self.catalogs[catalog_band][catalog_imgtype]['ra'], self.catalogs[catalog_band][catalog_imgtype]['dec'])
                            pos = self.wcs[band].world_to_pixel(coords)
                            for source_id, x, y in zip(self.catalogs[catalog_band][catalog_imgtype]['id'], pos[0], pos[1]):
                                if self.type == 'group':
                                    ax.annotate(source_id, (x, y), (x-2, y-4), color='r', alpha=0.8, fontsize=10, horizontalalignment='right')
                                    ax.hlines(y, x-5, x-2, color='r', alpha=0.8, lw=1)
                                    ax.vlines(x, y-5, y-2, color='r', alpha=0.8, lw=1)
                                elif imgtype not in ('residual',):
                                    ax.scatter(x, y, fc='none', ec='r', linewidths=1, marker='o', s=15)

                    # show group extents
                    if show_groups:
                        try:
                            groupmap = self.get_image(band=band, imgtype='groupmap')
                        except (KeyError, AttributeError) as e:
                            self.logger.debug(f'Groupmap not found for {band}, transferring maps: {e}')
                            self.transfer_maps(bands=band)
                            groupmap = self.get_image(band=band, imgtype='groupmap')

                        for group_id in tqdm(self.group_ids[catalog_band][catalog_imgtype]):

                            # use groupmap from brick to get position and buffsize
                            if group_id not in groupmap:
                                self.logger.debug(f'Group #{group_id} not found in groupmap! Skipping.')
                                continue
                            if band == 'detection':
                                idy, idx = (groupmap==group_id).nonzero()
                            else:
                                idy, idx = groupmap[group_id]
                            group_npix = len(idx)
                            if group_npix == 0:
                                self.logger.debug(f'No pixels belong to group #{group_id}! Skipping.')
                                continue
                            xlo, xhi = np.min(idx), np.max(idx) + 1
                            ylo, yhi = np.min(idy), np.max(idy) + 1
                            group_width = xhi - xlo
                            group_height = yhi - ylo

                            buffsize = (conf.GROUP_BUFFER.to(u.deg) / self.pixel_scales[band][0].to(u.deg)).value,\
                                         (conf.GROUP_BUFFER.to(u.deg) / self.pixel_scales[band][1].to(u.deg)).value

                            xlo -= buffsize[0]
                            ylo -= buffsize[1]
                            bdx, bdy = group_width + 2 * buffsize[0], group_height + 2 * buffsize[1]

                            rect = Rectangle((xlo, ylo), bdx, bdy, fill=False, alpha=0.3,
                                                    edgecolor='red', zorder=3, linewidth=1)
                            ax.add_patch(rect)
                            ax.annotate(str(group_id), (xlo, ylo), color='r', fontsize=2)

                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close()

                    # lin-scaled
                    fig = plt.figure(figsize=(20,20))
                    ax = fig.add_subplot(projection=self.get_wcs(band))
                    ax.set_title(f'{band} {imgtype} {tag}')
                    options = dict(cmap='RdGy', vmin=-5*self.get_property('rms', band=band), vmax=5*self.get_property('rms', band=band), origin='lower')
                    im = ax.imshow(image - background, **options)
                    # fig.colorbar(im, orientation="horizontal", pad=0.2)
                    if self.type == 'brick':
                        ax.add_patch(Rectangle(brick_buffer_pix, self.size[0].value, self.size[1].value,
                                     fill=False, alpha=0.3, edgecolor='purple', linewidth=1))
                    if show_catalog & (catalog_band in self.catalogs.keys()):
                        if catalog_imgtype in self.catalogs[catalog_band].keys():
                            coords = SkyCoord(self.catalogs[catalog_band][catalog_imgtype]['ra'], self.catalogs[catalog_band][catalog_imgtype]['dec'])
                            pos = self.wcs[band].world_to_pixel(coords)
                            for source_id, x, y in zip(self.catalogs[catalog_band][catalog_imgtype]['id'], pos[0], pos[1]):
                                if self.type == 'group':
                                    ax.annotate(source_id, (x, y), (x-2, y-4), color='r', alpha=0.8, fontsize=10, horizontalalignment='right')
                                    ax.hlines(y, x-5, x-2, color='r', alpha=0.8, lw=1)
                                    ax.vlines(x, y-5, y-2, color='r', alpha=0.8, lw=1)
                                # elif imgtype not in ('residual', 'chi'):
                                #     ax.scatter(x, y, fc='none', ec='r', linewidths=1, marker='.', s=15)
                    fig.tight_layout()

                if imgtype in ('chi'):
                    options = dict(cmap='RdGy', vmin=-3, vmax=3)
                    im = ax.imshow(image, **options)
                    # fig.colorbar(im, orientation="horizontal", pad=0.2)
                    if show_catalog & (catalog_band in self.catalogs.keys()):
                        if catalog_imgtype in self.catalogs[catalog_band].keys():
                            coords = SkyCoord(self.catalogs[catalog_band][catalog_imgtype]['ra'], self.catalogs[catalog_band][catalog_imgtype]['dec'])
                            pos = self.wcs[band].world_to_pixel(coords)
                            for source_id, x, y in zip(self.catalogs[catalog_band][catalog_imgtype]['id'], pos[0], pos[1]):
                                if self.type == 'group':
                                    ax.annotate(source_id, (x, y), (x-2, y-4), color='r', alpha=0.8, fontsize=10, horizontalalignment='right')
                                    ax.hlines(y, x-5, x-2, color='r', alpha=0.8, lw=1)
                                    ax.vlines(x, y-5, y-2, color='r', alpha=0.8, lw=1)
                                # else:
                                #     ax.scatter(x, y, fc='none', ec='r', linewidths=1, marker='.', s=15)
                    fig.tight_layout()
                
                if imgtype in ('weight', 'mask'):
                    options = dict(cmap='Greys', vmin=np.nanmin(image), vmax=np.nanmax(image))
                    im = ax.imshow(image, **options)
                    # fig.colorbar(im, orientation="horizontal", pad=0.2)
                    if show_catalog & (catalog_band in self.catalogs.keys()):
                        if catalog_imgtype in self.catalogs[catalog_band].keys():
                            coords = SkyCoord(self.catalogs[catalog_band][catalog_imgtype]['ra'], self.catalogs[catalog_band][catalog_imgtype]['dec'])
                            pos = self.wcs[band].world_to_pixel(coords)
                            for source_id, x, y in zip(self.catalogs[catalog_band][catalog_imgtype]['id'], pos[0], pos[1]):
                                if self.type == 'group':
                                    ax.annotate(source_id, (x, y), (x-2, y-4), color='r', alpha=0.8, fontsize=10, horizontalalignment='right')
                                    ax.hlines(y, x-5, x-2, color='r', alpha=0.8, lw=1)
                                    ax.vlines(x, y-5, y-2, color='r', alpha=0.8, lw=1)
                                else:
                                    ax.scatter(x, y, fc='none', ec='r', linewidths=1, marker='o', s=15)
                    fig.tight_layout()
            
                if imgtype in ('segmap', 'groupmap'):
                    if band != 'detection':
                    #    self.logger.warning(f'plot_image for {band} {imgtype} is NOT IMPLEMENTED YET.')
                        # get the map + identify pixels in fake image
                        groupmap = self.get_image('groupmap', band=band)
                        segmap = self.get_image('segmap', band=band)
                        source_ids = [sid for sid in segmap.keys()]
                        cmap = plt.get_cmap('rainbow', len(source_ids))
                        img = self.get_image('mask', band=band).copy().astype(np.int16)  #[src]
                        y, x = self.get_image(band=band, imgtype='groupmap')[self.group_id]
                        img[y, x] = 1
                        colors = ['white','grey']
                        bounds = [0, 1, 2]
                        for i, sid in enumerate(source_ids):
                            img[segmap[sid][0], segmap[sid][1]] = i + 2
                            colors.append(cmap(i))
                            bounds.append(i + 3)
                        # make a color map of fixed colors
                        cust_cmap = ListedColormap(colors)
                        norm = BoundaryNorm(bounds, cust_cmap.N)
                        ax.imshow(img, cmap=cust_cmap, norm=norm, origin='lower')
                        continue
                    if np.sum(image!=0) == 0:
                        continue
                    options = dict(cmap='prism', vmin=np.min(image[image!=0]), vmax=np.max(image))
                    image = image.copy().astype('float')
                    image[image==0] = np.nan
                    im = ax.imshow(image, **options)
                    # fig.colorbar(im, orientation="horizontal", pad=0.2)
                    if show_catalog & (catalog_band in self.catalogs.keys()):
                        if catalog_imgtype in self.catalogs[catalog_band].keys():
                            coords = SkyCoord(self.catalogs[catalog_band][catalog_imgtype]['ra'], self.catalogs[catalog_band][catalog_imgtype]['dec'])
                            pos = self.wcs[band].world_to_pixel(coords)
                            for source_id, x, y in zip(self.catalogs[catalog_band][catalog_imgtype]['id'], pos[0], pos[1]):
                                if self.type == 'group':
                                    ax.annotate(source_id, (x, y), (x-2, y-4), color='r', alpha=0.8, fontsize=10, horizontalalignment='right')
                                    ax.hlines(y, x-5, x-2, color='r', alpha=0.8, lw=1)
                                    ax.vlines(x, y-5, y-2, color='r', alpha=0.8, lw=1)
                                # else:
                                #     ax.scatter(x, y, color='r', marker='+', s=1)
                    fig.tight_layout()

                pdf.savefig(fig)
                plt.close()

        self.logger.info(f'Saving figure: {outname}') 
        pdf.close()

    def plot_psf(self, band=None):
          
        if band is None:
            bands = self.get_bands()
        else:
            bands = [band,]

        for band in bands:
            self.logger.debug(f'Plotting PSF for: {band}')    
            psfmodel = self.get_psfmodel(band).img

            pixscl = (self.pixel_scales[band][0]).to(u.arcsec).value
            fig, ax = plt.subplots(ncols=3, figsize=(30,10))
            norm = LogNorm(1e-8, 0.1*np.nanmax(psfmodel), clip='True')
            img_opt = dict(cmap='Blues', norm=norm, origin='lower')
            ax[0].imshow(psfmodel, **img_opt, extent=pixscl *np.array([-np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2, -np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2,]))
            ax[0].set(xlim=(-15,15), ylim=(-15, 15))
            ax[0].axvline(0, color='w', ls='dotted')
            ax[0].axhline(0, color='w', ls='dotted')

            xax = np.arange(-np.shape(psfmodel)[0]/2 + 0.5,  np.shape(psfmodel)[0]/2+0.5)
            [ax[1].plot(xax * pixscl, psfmodel[x], c='royalblue', alpha=0.5) for x in np.arange(0, np.shape(psfmodel)[1])]
            ax[1].axvline(0, ls='dotted', c='k')
            ax[1].set(xlim=(-15, 15), yscale='log', ylim=(1E-6, 1E-1), xlabel='arcsec')

            x = xax
            y = x.copy()
            xv, yv = np.meshgrid(x, y)
            radius = np.sqrt(xv**2 + xv**2)
            cumcurve = [np.sum(psfmodel[radius<i]) for i in np.arange(0, np.shape(psfmodel)[0]/2)]
            ax[2].plot(np.arange(0, np.shape(psfmodel)[0]/2) * pixscl, cumcurve)

            fig.suptitle(band)

            figname = os.path.join(conf.PATH_FIGURES, f'{band}_psf.pdf')
            self.logger.debug(f'Saving figure: {figname}')                
            fig.savefig(figname)
            plt.close(fig)

    def plot_summary(self, source_id=None, group_id=None, bands=None, stage=None, tag=None, catalog_band='detection', catalog_imgtype='science', overwrite=True):
        # show the group or source image, model, residuals, background, psf, distributions + statistics

        if bands is None:
            bands = self.get_bands()
            if 'detection' in bands:
                bands = bands[bands!='detection']
        elif np.isscalar(bands):
            bands = [bands,]

        if stage is None:
            stage = np.max(list(self.model_tracker['group'].keys()))

        catalog = self.get_catalog(catalog_band, catalog_imgtype)

        if source_id is None:
            if group_id is None:
                sources = list(self.model_tracker.keys())
            else:
                sources = [x for x in self.model_tracker.keys() if x in catalog['source_id'][catalog['group_id']==group_id]]
        else:
            if np.isscalar(source_id):
                sources = [source_id,]
            else:
                sources = source_id

        self.build_all_images(bands=bands, reconstruct=False)

        for source_id in sources:
            fnsrc = ''
            if source_id != 'group':
                fnsrc  = f'S{source_id}_'
            if tag is not None:
                fnsrc += f'{tag}_'
            outname = os.path.join(conf.PATH_FIGURES, fnsrc + self.filename.replace('.h5', '_summary.pdf'))
            pdf = matplotlib.backends.backend_pdf.PdfPages(outname)

            for band in self.model_tracker[source_id][stage]:
                if band not in conf.BANDS: continue # something else

                bandname = 'detection'
                if band != 'detection':
                    bandname = conf.BANDS[band]['name']

                # set up
                fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(15, 15))

                [ax.tick_params(labelbottom=False, labelleft=False) for ax in axes[0:3,:].flatten()]

                # information
                [ax.axis('off') for ax in axes[0, 1:-1]]
                if source_id == 'group':
                    axes[0,1].text(0, 1, f'Group #{self.group_id} (Brick #{self.brick_id})', transform=axes[0,1].transAxes) 
                    axes[0,1].text(0, 0.8, f'N = {len(self.source_ids)} sources {self.source_ids}', transform=axes[0,1].transAxes) 
                    stats = self.model_tracker['group'][stage]['total']
                    axes[0,1].text(0, 0.7, f'Total  $\chi^2_N$ = {stats["rchisq"]:2.2f} ({stats["rchisqmodel"]:2.2f}) N(DOF) = {stats["ndof"]} with {stats["nres"]:2.2f} resolution elements', transform=axes[0,1].transAxes)
                    axes[0,1].text(0, 0.6, f'        <$\chi$> = {stats["chi_pc50"]:2.2f} | $\sigma$($\chi$) = {stats["chi_pc84"] - stats["chi_pc16"]:2.2f} | $K^2$ = {stats["chi_k2"]:2.2f}', transform=axes[0,1].transAxes)
                    stats = self.model_tracker['group'][stage][band]
                    axes[0,1].text(0, 0.5, f'{conf.BANDS[band]["name"]}  $\chi^2_N$ = {stats["rchisq"]:2.2f} ({stats["rchisqmodel"]:2.2f}) N(DOF) = {stats["ndof"]} with {stats["nres"]:2.2f} resolution elements', transform=axes[0,1].transAxes)
                    axes[0,1].text(0, 0.4, f'        <$\chi$> = {stats["chi_pc50"]:2.2f} | $\sigma$($\chi$) = {stats["chi_pc84"] - stats["chi_pc16"]:2.2f} | $K^2$ = {stats["chi_k2"]:2.2f}', transform=axes[0,1].transAxes)

                else:
                    model = self.model_tracker[source_id][stage]['model']
                    axes[0,1].text(0, 1, f'Source #{source_id} (Group #{self.group_id}, Brick #{self.brick_id})', transform=axes[0,1].transAxes)
                    axes[0,1].text(0, 0.8, f'Model type: {model.name}', transform=axes[0,1].transAxes)
                    stats = self.model_tracker[source_id][stage]['total']
                    axes[0,1].text(0, 0.7, f'Total  $\chi^2_N$ = {stats["rchisq"]:2.2f} ({stats["rchisqmodel"]:2.2f}) N(DOF) = {stats["ndof"]} with {stats["nres"]:2.2f} resolution elements', transform=axes[0,1].transAxes)
                    axes[0,1].text(0, 0.6, f'        <$\chi$> = {stats["chi_pc50"]:2.2f} | $\sigma$($\chi$) = {stats["chi_pc84"] - stats["chi_pc16"]:2.2f} | $K^2$ = {stats["chi_k2"]:2.2f}', transform=axes[0,1].transAxes)
                    stats = self.model_tracker[source_id][stage][band]
                    axes[0,1].text(0, 0.5, f'{conf.BANDS[band]["name"]}  $\chi^2_N$ = {stats["rchisq"]:2.2f} ({stats["rchisqmodel"]:2.2f}) N(DOF) = {stats["ndof"]} with {stats["nres"]:2.2f} resolution elements', transform=axes[0,1].transAxes)
                    axes[0,1].text(0, 0.4, f'        <$\chi$> = {stats["chi_pc50"]:2.2f} | $\sigma$($\chi$) = {stats["chi_pc84"] - stats["chi_pc16"]:2.2f} | $K^2$ = {stats["chi_k2"]:2.2f}', transform=axes[0,1].transAxes)

                    source = get_params(model)
                    mag, mag_err = source[f'{band}_mag'].value, source[f'{band}_mag_err']
                    flux, flux_err = source[f'{band}_flux_ujy'].value, source[f'{band}_flux_ujy_err'].value
                    zpt = source[f'_{band}_zpt']
                    str_fracdev = ''
                    if isinstance(model, FixedCompositeGalaxy):
                        fracdev, fracdev_err = source['fracdev'], source['fracdev_err']
                        str_fracdev = r'$\mathcal{F}$(Dev)' + f'{fracdev:2.2f}+/-{fracdev_err:2.2f}'
                    axes[0,1].text(0, 0.3, f'{bandname}: {mag:2.2f}+/-{mag_err:2.2f} AB {flux:2.2f}+/-{flux_err:2.2f} uJy (zpt = {zpt}) {str_fracdev}', transform=axes[0,1].transAxes)
                    pos = source['ra'], source['dec']
                    axes[0,1].text(0, 0.2, f'Position:   ({pos[0]:2.2f}, {pos[1]:2.2f})', transform=axes[0,1].transAxes)
                    if isinstance(model, (ExpGalaxy, DevGalaxy)) & ~isinstance(model, SimpleGalaxy):
                        reff, reff_err = source['reff'].value, source['reff_err'].value
                        ba, ba_err = source['ba'], source['ba_err']
                        pa, pa_err = source['pa'].value, source['pa_err'].value
                        axes[0,1].text(0, 0.1, r'Shape:   $R_{\rm eff} = $' + f'{reff:2.2f}+/-{reff_err:2.2f}\" $b/a$ = {ba:2.2f}+/{ba_err:2.2f}  $\\theta$ = {pa:2.1f}+/-{pa_err:2.1f}'+r'$\degree$', transform=axes[0,1].transAxes)
                    elif isinstance(model, FixedCompositeGalaxy):
                        for skind, yloc in zip(('_exp', '_dev'), (0.1, 0.0)):
                            reff, reff_err = source[f'reff{skind}'].value, source[f'reff{skind}_err'].value
                            ba, ba_err = source[f'ba{skind}'], source[f'ba{skind}_err']
                            pa, pa_err = source[f'pa{skind}'].value, source[f'pa{skind}_err'].value
                            axes[0,1].text(0, yloc, f'Shape {skind[1:]}:   '+r'$R_{\rm eff} = $' + f'{reff:2.2f}+/-{reff_err}\" $b/a$ = {ba:2.2f}+/{ba_err:2.2f}  $\\theta$ = {pa:2.2f}+/-{pa_err:2.2f}', transform=axes[0,1].transAxes)

                

                groupmap = self.get_image('groupmap', band=band)
                segmap = self.get_image('segmap', band=band)
                source_ids = [sid for sid in segmap.keys()]
                cmap = plt.get_cmap('rainbow', len(source_ids))
                pixscl = self.pixel_scales[band]
                histbins = np.linspace(-3, 3, 20)
                hwhm = get_fwhm(self.get_psfmodel(band=band).img) / 2. * pixscl[0].to(u.arcsec).value
                
                cutout = self.data[band]['science']
                upper = self.wcs[band].pixel_to_world(cutout.shape[1], cutout.shape[0])
                lower = self.wcs[band].pixel_to_world(-1, -1)

                if source_id == 'group':
                    position = self.position
                    target_size = self.buffsize
                else:
                    ira, idec = catalog['ra'][catalog['id'] == source_id], catalog['dec'][catalog['id'] == source_id]
                    position = SkyCoord(ira, idec)
                    if source_id not in segmap:
                        self.logger.warning(f'Source {source_id} missing segmap for band {band}. Skipping plot segment.')
                        continue
                    idy, idx = segmap[source_id]
                    xlo, xhi = np.min(idx), np.max(idx)
                    ylo, yhi = np.min(idy), np.max(idy)
                    group_width = xhi - xlo
                    group_height = yhi - ylo
                    target_upper = self.wcs[band].pixel_to_world(group_height, group_width)
                    target_lower = self.wcs[band].pixel_to_world(0, 0)
                    target_size = ((target_lower.ra - target_upper.ra) * np.cos(np.deg2rad(self.position.dec.to(u.degree).value)) + 2 * conf.GROUP_BUFFER), \
                                    (target_upper.dec - target_lower.dec + 2 * conf.GROUP_BUFFER)

                extent = np.array([0.,0.,0.,0.])
                extent[0], extent[2] = dcoord_to_offset(lower, position)
                extent[1], extent[3] = dcoord_to_offset(upper, position)

                peakx = np.argmin(np.abs(np.linspace(extent[0], extent[1], cutout.shape[1])))
                peaky = np.argmin(np.abs(np.linspace(extent[2], extent[3], cutout.shape[0])))

                dx = target_size[0].to(u.arcsec).value / 2. # RA
                dy = target_size[1].to(u.arcsec).value / 2. # DEC
                ds = np.max([dx, dy])
                xlim, ylim = [-ds, ds],[-ds, ds]
                [ax.set(xlim=xlim, ylim=ylim) for ax in axes[:3].flatten()]

                dims = (xlim[1]-xlim[0], ylim[1]-ylim[0])

                pos = dims[0] * (-4/10.), dims[1] * (-4/10.)
                for ax in axes[0:3,:].flatten():
                    if ax in axes[0, 1:3].flatten():
                        continue
                    beam = Circle(pos, hwhm, color='grey')
                    ax.add_patch(beam)
                

                # science image
                img = self.get_image('science', band=band).copy()   #[src]
                background = 0
                if self.get_property('subtract_background', band=band):
                    background = self.get_background(band)
                if not np.isscalar(background):
                    background = background   #[src]
                img -= background
                rms = self.get_property('rms', band=band)
                vmax = np.nanpercentile(img, 99)
                vmin = np.nanpercentile(img, 50)
                axes[0,0].imshow(img, cmap='RdGy', norm=SymLogNorm(rms, 0.5, -vmax, vmax), extent=extent, origin='lower')
                axes[0,0].text(0.05, 0.90, bandname, transform=axes[0,0].transAxes, fontweight='bold')
                target_scale = np.round(self.pixel_scales[band][0].to(u.arcsec).value * dims[0] / 3.).astype(int) # arcsec
                target_scale = np.max([1, target_scale])
                target_center = 0.75
                xmin, xmax = target_center - target_scale/dims[0]/2., target_center + target_scale/dims[0]/2.
                axes[0,0].axhline(dims[1] * (-4/10.), xmin, xmax, c='k')
                axes[0,0].text(target_center, 0.12, f'{target_scale}\"', transform=axes[0,0].transAxes, fontweight='bold', horizontalalignment='center')
                
                axes[3,3].axvline(0, ls='dashed', c='grey')
                axes[3,3].axvline(-1, ls='dotted', c='grey')
                axes[3,3].axvline(1, ls='dotted', c='grey')
                axes[3,3].axhline(0.5, ls='dashed', c='grey')
                axes[3,3].axhline(0.16, ls='dotted', c='grey')
                axes[3,3].axhline(0.84, ls='dotted', c='grey')
                img = self.get_image('science', band=band).copy() * np.sqrt(self.get_image('weight', band=band).copy())
                # axes[3,3].hist(img[groupmap>0].flatten(), color='grey', histtype='step', bins=histbins)
                xcum, ycum = cumulative(img[groupmap[self.group_id][0], groupmap[self.group_id][1]])
                axes[3,3].plot(xcum, ycum, color='grey', alpha=0.3)
                model_patch = []
                for i, sid in enumerate(source_ids):
                    # axes[3,3].hist(img[segmap==sid].flatten(), color=cmap(i), histtype='step', bins=histbins)
                    if (source_id == 'group') or (sid == source_id):
                        xcum, ycum = cumulative(img[segmap[sid][0], segmap[sid][1]])
                        axes[3,3].plot(xcum, ycum, color=cmap(i), alpha=0.3)
                    if source_id == 'group':
                        center_position = self.position
                    else:
                        ira, idec = catalog['ra'][catalog['id'] == source_id], catalog['dec'][catalog['id'] == source_id]
                        center_position = SkyCoord(ira, idec)
                    ira, idec = catalog['ra'][catalog['id'] == sid], catalog['dec'][catalog['id'] == sid]
                    ixc, iyc = dcoord_to_offset(SkyCoord(ira, idec), center_position)
                    axes[0,0].scatter(ixc, iyc, facecolors='none', edgecolors=cmap(i))
                    model = self.model_catalog[sid] 
                    ra, dec = model.pos.ra, model.pos.dec
                    xc, yc = dcoord_to_offset(SkyCoord(ra*u.deg, dec*u.deg), center_position)

                    if isinstance(model, (PointSource, SimpleGalaxy)):
                        model_patch += [Circle((xc, yc), hwhm, fc="none", ec=cmap(i)),]
                    elif isinstance(model, (ExpGalaxy, DevGalaxy)) & ~isinstance(model, SimpleGalaxy):
                        shape = model.getShape()
                        width, height = shape.re * shape.ab, shape.re
                        angle = np.rad2deg(shape.theta)
                        model_patch += [Ellipse((xc, yc), width, height, angle=angle, fc="none", ec=cmap(i)),]
                    elif isinstance(model, (FixedCompositeGalaxy)):
                        shape = model.shapeExp
                        width, height = shape.re * shape.ab, shape.re
                        angle = np.rad2deg(shape.theta)
                        model_patch += [Ellipse((xc, yc), width, height, angle=angle, fc="none", ec=cmap(i)),]
                        shape = model.shapeDev
                        width, height = shape.re * shape.ab, shape.re
                        angle = np.rad2deg(shape.theta)
                        model_patch += [Ellipse((xc, yc), width, height, angle=angle, fc="none", ec=cmap(i)),]

                    
                    for ax in (axes[0,0], axes[0,3]):
                        ax.scatter(xc, yc, color=cmap(i), marker='+')
                        ax.annotate(int(sid), (xc, yc), (xc-0.1, yc-0.1), color=cmap(i), horizontalalignment='right', verticalalignment='top')
                    for ax in axes[1:3,:].flatten():
                        ax.scatter(xc, yc, color=cmap(i), marker='+')

                for ax in (axes[0,0], axes[0,3]):
                    [ax.add_patch(copy.copy(mp)) for mp in model_patch]
                for ax in axes[1:3,:].flatten():
                    [ax.add_patch(copy.copy(mp)) for mp in model_patch]



                # detection image
                if 'detection' in self.bands:
                    img = self.get_image('science', band='detection').copy()   #[src]
                    background = 0
                    if self.get_property('subtract_background', band='detection'):
                        background = self.get_background(band)
                    img -= background
                    srms = self.get_property('clipped_rms', band='detection')
                    svmax = np.nanmax(img)
                    axes[0,3].imshow(img, cmap='RdGy', norm=SymLogNorm(srms, 0.5, -svmax, svmax), extent=extent, origin='lower')
                    axes[0,3].text(0.05, 0.90, 'detection', transform=axes[0,3].transAxes, fontweight='bold')
                    axes[0,3].axhline(dims[1] * (-4/10.), xmin, xmax, c='k')
                    axes[0,3].text(target_center, 0.12, f'{target_scale}\"', transform=axes[0,3].transAxes, fontweight='bold', horizontalalignment='center')

                # science image
                img = self.get_image('science', band=band).copy()   #[src]
                axes[1,0].imshow(img, cmap='Greys', norm=LogNorm(vmin, vmax), extent=extent, origin='lower')
                axes[1,0].text(0.05, 0.90, 'Science', transform=axes[1,0].transAxes, fontweight='bold')
                axes[1,0].axhline(dims[1] * (-4/10.), xmin, xmax, c='k')
                axes[1,0].text(target_center, 0.12, f'{target_scale}\"', transform=axes[1,0].transAxes, fontweight='bold', horizontalalignment='center')

                px = np.linspace(extent[2], extent[3], np.shape(img)[0])
                py = img[:,peakx]
                wgt = self.get_image('weight', band=band).copy()   #[src]
                epy = np.where(wgt<=0, 0, 1/np.sqrt(wgt))[:,peakx]
                max1 = np.nanmax(py + 3*epy)

                axes[3,1].errorbar(px[epy>0], py[epy>0], yerr=epy[epy>0], c='k', capsize=0, marker='.', ls='')
                axes[3,1].errorbar(px[epy==0], py[epy==0], yerr=epy[epy==0], mfc='none', mec='k', capsize=0, marker='.', ls='')
                axes[3,1].axvline(0, ls='dashed', c='grey')
                axes[3,1].axhline(0, ls='solid', c='grey')
                axes[3,1].axhline(-rms, ls='dotted', c='grey')
                axes[3,1].axhline(rms, ls='dotted', c='grey')
                axes[3,1].text(0.05, 0.90, f'$X=0$ Slice', transform=axes[3,1].transAxes, fontweight='bold')
                
                px = np.linspace(extent[0], extent[1], np.shape(img)[1])
                py = img[peaky]
                wgt = self.get_image('weight', band=band).copy()   #[src]
                epy = np.where(wgt<=0, 0, 1/np.sqrt(wgt))[peaky]
                max2 = np.nanmax(py + 3*epy)
                axes[3,2].errorbar(px[epy>0], py[epy>0], yerr=epy[epy>0], c='k', capsize=0, marker='.', ls='')
                axes[3,2].errorbar(px[epy==0], py[epy==0], yerr=epy[epy==0], mfc='none', mec='k', capsize=0, marker='.', ls='')
                axes[3,2].axvline(0, ls='dashed', c='grey')
                axes[3,2].axhline(0, ls='solid', c='grey')
                axes[3,2].axhline(-rms, ls='dotted', c='grey')
                axes[3,2].axhline(rms, ls='dotted', c='grey')
                axes[3,2].text(0.05, 0.90, f'$Y=0$ Slice', transform=axes[3,2].transAxes, fontweight='bold')

                if np.isnan(max1): max1 = 0
                if np.isnan(max2): max2 = 0
                axes[3,1].set(xlim=xlim, xlabel='arcsec', ylim=(-3*rms, np.max([max1, max2])))
                axes[3,2].set(xlim=ylim, xlabel='arcsec', ylim=(-3*rms, np.max([max1, max2])))

                # model image
                img = self.get_image('model', band=band).copy()   #[src]
                axes[1,1].imshow(img, cmap='Greys', norm=LogNorm(vmin, vmax), extent=extent, origin='lower')
                axes[1,1].text(0.05, 0.90, 'Model', transform=axes[1,1].transAxes, fontweight='bold')
                axes[1,1].axhline(dims[1] * (-4/10.), xmin, xmax, c='k')
                axes[1,1].text(target_center, 0.12, f'{target_scale}\"', transform=axes[1,1].transAxes, fontweight='bold', horizontalalignment='center')

                # px = np.linspace(extent[2], extent[3], np.shape(img)[0])
                # py = img[:,peakx]
                # axes[3,1].plot(px, py, color='g')

                # px = np.linspace(extent[0], extent[1], np.shape(img)[1])
                # py = img[peaky]
                # axes[3,2].plot(px, py, color='g')

                nx, ny = np.shape(img)
                psf = self.get_psfmodel(band).img
                scl = 1
                zoomed_psf = ndimage.zoom(psf, scl)
                zoomed_psf /= np.sum(zoomed_psf)
                zoomed_wcs = read_wcs(self.wcs[band], scl)
                tr_img = Image(
                    data=np.zeros((scl*nx, scl*ny)),
                    invvar=np.ones((scl*nx, scl*ny)),
                    psf=PixelizedPSF(zoomed_psf),
                    wcs=zoomed_wcs,
                    photocal=FluxesPhotoCal(band),
                    sky=ConstantSky(0)
                    )
                srcs = []
                for i, sid in enumerate(self.source_ids):
                    isrc = self.model_catalog[sid]
                    srcs.append(isrc)

                    img = Tractor([tr_img,], Catalog(*[isrc,])).getModelImage(0) * scl**2
                    flux = isrc.getBrightness().getFlux(band)
                    eflux = np.sqrt(isrc.variance.getBrightness().getFlux(band))
                    frac = eflux / flux
                    inx, iny = np.shape(img)

                    ipeakx = np.argmin(np.abs(np.linspace(extent[0], extent[1], iny))) 
                    ipeaky = np.argmin(np.abs(np.linspace(extent[2], extent[3], inx)))
                    
                    px = np.linspace(extent[2], extent[3], inx)
                    py = img[:,ipeakx]
                    py_lo = py * (1 - frac)
                    py_hi = py * (1 + frac)
                    axes[3,1].errorbar(px, py, yerr=np.abs(py*frac), c=cmap(i), capsize=0, marker='.', ls='')
                    axes[3,1].plot(px, py, c=cmap(i), marker='none')
                    if (source_id == 'group') | (sid == source_id):
                        axes[3,1].fill_between(px, py_lo, py_hi, color=cmap(i), alpha=0.5)

                    px = np.linspace(extent[0], extent[1], iny)
                    py = img[ipeaky]
                    py_lo = py * (1 - frac)
                    py_hi = py * (1 + frac)
                    axes[3,2].errorbar(px, py, yerr=np.abs(py*frac), c=cmap(i), capsize=0, marker='.', ls='')
                    axes[3,2].plot(px, py, c=cmap(i), marker='none')
                    if (source_id == 'group') | (sid == source_id):
                        axes[3,2].fill_between(px, py_lo, py_hi, color=cmap(i), alpha=0.5)

                img = Tractor([tr_img,], Catalog(*srcs)).getModelImage(0) * scl**2
                inx, iny= np.shape(img)
                ipeakx = np.argmin(np.abs(np.linspace(extent[0], extent[1], iny))) 
                ipeaky = np.argmin(np.abs(np.linspace(extent[2], extent[3], inx)))

                px = np.linspace(extent[2], extent[3], inx)
                py = img[:,ipeakx]
                axes[3,1].plot(px, py, color='grey', zorder=-99)

                px = np.linspace(extent[0], extent[1], iny)
                py = img[ipeaky]
                axes[3,2].plot(px, py, color='grey', zorder=-99)

                # residual image
                img = self.get_image('residual', band=band).copy()   #[src]
                axes[1,2].imshow(img, cmap='RdGy', norm=Normalize(-3*rms, 3*rms), extent=extent, origin='lower')
                axes[1,2].text(0.05, 0.90, 'Residual', transform=axes[1,2].transAxes, fontweight='bold')
                axes[1,2].axhline(dims[1] * (-4/10.), xmin, xmax, c='k')
                axes[1,2].text(target_center, 0.12, f'{target_scale}\"', transform=axes[1,2].transAxes, fontweight='bold', horizontalalignment='center')

                px = np.linspace(extent[2], extent[3], np.shape(img)[0])
                py = img[:,peakx]
                wgt = self.get_image('weight', band=band).copy()   #[src]
                epy = np.where(wgt<=0, 0, 1/np.sqrt(wgt))[:,peakx]
                axes[3,1].errorbar(px, py, yerr=epy, c='g', capsize=0, marker='.', ls='')

                px = np.linspace(extent[0], extent[1], np.shape(img)[1])
                py = img[peaky]
                wgt = self.get_image('weight', band=band).copy()   #[src]
                epy = np.where(wgt<=0, 0, 1/np.sqrt(wgt))[peaky]
                axes[3,2].errorbar(px, py, yerr=epy, c='g', capsize=0, marker='.', ls='')

                img = self.get_image('chi', band=band).copy()
                xcum, ycum = cumulative(img[groupmap[self.group_id][0], groupmap[self.group_id][1]])
                axes[3,3].plot(xcum, ycum, color='grey')
                # axes[3,3].hist(img[groupmap>0].flatten(), color='grey', histtype='step', bins=histbins)
                for i, sid in enumerate(source_ids):
                    if (source_id == 'group') or (sid == source_id):
                        # axes[3,3].hist(img[segmap==sid].flatten(), color=cmap(i), histtype='step', bins=histbins)
                        xcum, ycum = cumulative(img[segmap[sid][0], segmap[sid][1]])
                        axes[3,3].plot(xcum, ycum, color=cmap(i))
                axes[3,3].set(xlim=(-3, 3), ylim=(0, 1), xlabel='$\chi$')
                axes[3,3].text(0.05, 0.90, 'CDF($\chi$)', transform=axes[3,3].transAxes, fontweight='bold')

                # science image
                img = self.get_image('science', band=band).copy()   #[src]
                axes[1,3].imshow(img, cmap='RdGy', norm=Normalize(-3*rms, 3*rms), extent=extent, origin='lower')
                axes[1,3].text(0.05, 0.90, 'Science', transform=axes[1,3].transAxes, fontweight='bold')
                axes[1,3].axhline(dims[1] * (-4/10.), xmin, xmax, c='k')
                axes[1,3].text(target_center, 0.12, f'{target_scale}\"', transform=axes[1,3].transAxes, fontweight='bold', horizontalalignment='center')

                # weight image
                img = self.get_image('weight', band=band).copy()   #[src]
                vmax = np.nanmax(img)
                vmin = np.nanmin(img)
                axes[2,0].imshow(img, cmap='Greys', norm=Normalize(vmin, vmax), extent=extent, origin='lower')
                axes[2,0].text(0.05, 0.90, 'Weight', transform=axes[2,0].transAxes, fontweight='bold')
                axes[2,0].axhline(dims[1] * (-4/10.), xmin, xmax, c='k')
                axes[2,0].text(target_center, 0.12, f'{target_scale}\"', transform=axes[2,0].transAxes, fontweight='bold', horizontalalignment='center')

                # background
                img = self.get_image('background', band=band).copy()   #[src]
                vmin, vmax = np.nanmin(img), np.nanmax(img)
                axes[2,1].imshow(img, cmap='Greys', norm=Normalize(vmin, vmax), extent=extent, origin='lower')
                backtype = self.get_property('backtype', band=band)
                backregion = self.get_property('backregion', band=band)
                axes[2,1].text(0.05, 0.90, f'Background ({backtype}, {backregion})', transform=axes[2,1].transAxes, fontweight='bold')
                axes[2,1].axhline(dims[1] * (-4/10.), xmin, xmax, c='k')
                axes[2,1].text(target_center, 0.12, f'{target_scale}\"', transform=axes[2,1].transAxes, fontweight='bold', horizontalalignment='center')

                # mask image
                img = self.get_image('mask', band=band).copy().astype(np.int16)  #[src]
                y, x = self.get_image(band=band, imgtype='groupmap')[self.group_id]
                img[y, x] = 1
                colors = ['white','grey']
                bounds = [0, 1, 2]
                for i, sid in enumerate(source_ids):
                    img[segmap[sid][0], segmap[sid][1]] = i + 2
                    colors.append(cmap(i))
                    bounds.append(i + 3)
                # make a color map of fixed colors
                cust_cmap = ListedColormap(colors)
                norm = BoundaryNorm(bounds, cust_cmap.N)
                axes[2,2].imshow(img, cmap=cust_cmap, norm=norm, extent=extent, origin='lower')
                axes[2,2].text(0.05, 0.90, 'Pixel Assignment', transform=axes[2,2].transAxes, fontweight='bold')
                axes[2,2].axhline(dims[1] * (-4/10.), xmin, xmax, c='k')
                axes[2,2].text(target_center, 0.12, f'{target_scale}\"', transform=axes[2,2].transAxes, fontweight='bold', horizontalalignment='center')

                # chi
                img = self.get_image('chi', band=band).copy()   #[src]
                axes[2,3].imshow(img, cmap='RdGy', norm=Normalize(-3, 3), extent=extent, origin='lower')
                axes[2,3].text(0.05, 0.90, r'$\chi$', transform=axes[2,3].transAxes, fontweight='bold')
                axes[2,3].axhline(dims[1] * (-4/10.), xmin, xmax, c='k')
                axes[2,3].text(target_center, 0.12, f'{target_scale}\"', transform=axes[2,3].transAxes, fontweight='bold', horizontalalignment='center')

                # PSF
                psfmodel = self.get_psfmodel(band=band).img
                pixscl = (self.pixel_scales[band][0]).to(u.arcsec).value
                xax = np.arange(-np.shape(psfmodel)[0]/2 + 0.5,  np.shape(psfmodel)[0]/2+0.5)
                [axes[3,0].plot(xax * pixscl, psfmodel[x], c='royalblue', alpha=0.5) for x in np.arange(0, np.shape(psfmodel)[1])]
                axes[3,0].axvline(0, ls='dashed', c='k')
                axes[3,0].text(0.05, 0.90, f'PSF', transform=axes[3,0].transAxes, fontweight='bold')
                
                axes[3,0].axvline(-hwhm, ls='dotted', c='grey')
                axes[3,0].axvline(hwhm, ls='dotted', c='grey')

                x = xax
                y = x.copy()
                xv, yv = np.meshgrid(x, y)
                radius = np.sqrt(xv**2 + xv**2)
                cumcurve = np.array([np.sum(psfmodel[radius<i]) for i in np.arange(0, np.shape(psfmodel)[0]/2)])
                hxax = np.arange(0, np.shape(psfmodel)[0]/2) * pixscl
                axes[3,0].plot(hxax, cumcurve, c='grey')
                allp = hxax[np.argmin(abs(cumcurve - 0.9999))]
                axes[3,0].set(xlim=(-allp, allp), yscale='log', ylim=(1e-6, 1), xlabel='arcsec')

                pdf.savefig(fig)
                plt.close()

            self.logger.info(f'Saving figure: {outname}') 
            pdf.close()

    def transfer_maps(self, bands=None, catalog_band='detection', overwrite=False):
        # rescale segmaps and groupmaps to other bands
        segmap = self.data[catalog_band]['segmap']
        groupmap = self.data[catalog_band]['groupmap']
        catalog_pixscl = np.array([self.pixel_scales[catalog_band][0].value, self.pixel_scales[catalog_band][1].value])

        if bands is None:
            bands = self.bands
        elif np.isscalar(bands):
            bands = [bands,]

        if overwrite:
            for band in bands:
                if band == catalog_band: continue
                if 'groupmap' in self.data[band]:
                    del self.data[band]['groupmap']
                if 'segmap' in self.data[band]:
                    del self.data[band]['segmap']

        # loop over bands
        for band in bands:
            if band == catalog_band:
                continue
            # Skip bands that weren't loaded (e.g., all weight pixels are zero)
            if band not in self.data:
                self.logger.warning(f'Band {band} not in data. Skipping transfer_maps.')
                continue
            if ('segmap' in self.data[band]) & ('groupmap' in self.data[band]) & (not overwrite):
                self.logger.debug(f'Segmap and groupmap for {band} already exists! Skipping.')
                continue
            pixscl = np.array([self.pixel_scales[band][0].value, self.pixel_scales[band][1].value])
            
            already_made = False
            for mband in self.bands:
                if mband == catalog_band: continue
                if mband == band: continue
                if 'groupmap' in self.data[mband]:
                    mpixscl = np.array([self.pixel_scales[mband][0].value, self.pixel_scales[mband][1].value])
                    # HACK -- this is a lot stupid. Should use WCS params instead.
                    if np.isclose(pixscl, mpixscl).all():
                        already_made = True
                        break
                    
            if already_made:
                self.logger.info(f'Mapping for segmap and groupmap of {mband} to {band} already exists!')
                self.logger.debug(f'Using the {mband} mapping for {band}')
                self.data[band]['segmap']= self.data[mband]['segmap']
                self.data[band]['groupmap'] = self.data[mband]['groupmap']
                self.logger.info(f'Copied maps for segmap and groupmap of {mband} to {band} by ({mpixscl} -> {pixscl})')
            
            else:
                self.logger.info(f'Creating mapping for segmap and groupmap of {catalog_band} to {band}')
                self.data[band]['segmap']= map_discontinuous((segmap.data, segmap.wcs), self.wcs[band], np.shape(self.data[band]['science'].data), force_simple=conf.FORCE_SIMPLE_MAPPING)
                self.data[band]['groupmap'] = map_discontinuous((groupmap.data, groupmap.wcs), self.wcs[band], np.shape(self.data[band]['science'].data), force_simple=conf.FORCE_SIMPLE_MAPPING)
                self.logger.debug(f'Created maps for segmap and groupmap of {catalog_band} to {band}')

    def write(self, filetype=None, allow_update=False, filename=None):
        if (filetype is None):
            filename = self.filename
            self.write_hdf5(allow_update=allow_update, filename=filename)
            self.write_fits(allow_update=allow_update, filename=filename.replace('.h5', '.fits'))
            if bool(self.catalogs):
                if np.sum([bool(self.catalogs[key]) for key in self.catalogs]) >= 1:
                    self.write_catalog(allow_update=allow_update, filename=filename.replace('.h5', '.cat'))
        elif filetype == 'hdf5':
            self.write_hdf5(allow_update=allow_update, filename=filename)
        elif filetype == 'fits':
            self.write_fits(allow_update=allow_update, filename=filename.replace('.h5', '.fits'))
        elif filetype == 'cat':
            if bool(self.catalogs):
                if np.sum([bool(self.catalogs[key]) for key in self.catalogs]) >= 1:
                    self.write_catalog(allow_update=allow_update, filename=filename.replace('.h5', '.cat'))
                else:
                    raise RuntimeError('Cannot write catalogs to disk as none are present!')
            else:
                raise RuntimeError('Cannot write catalogs to disk as none are present!')

    def write_fits(self, bands=None, imgtypes=None, allow_update=False, tag=None, filename=None, directory=conf.PATH_ANCILLARY):
        if filename is None:
            filename = self.filename.replace('.h5', '.fits')
        if tag is not None:
            filename = filename.replace('.fits', f'_{tag}.fits')
        self.logger.info(f'Writing to {filename} (allow_update = {allow_update})')
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            if not allow_update:
                raise RuntimeError(f'Cannot update {filename}! (allow_update = False)')
            else:
                # open files and add to them
                hdul = fits.open(path, mode='update')
                makenew = False
        else:
            # make new files
            hdul = fits.HDUList()
            hdul.append(fits.PrimaryHDU())
            makenew = True

        if bands is not None:
            write_bands = bands
        else:
            write_bands = self.data

        if np.isscalar(write_bands):
            write_bands = [write_bands,]
        
        self.logger.debug(f'... adding data to fits')
        for band in write_bands:
            for attr in self.data[band]:
                if imgtypes is not None:
                    if attr not in imgtypes: 
                        continue
                if attr.startswith('psf'): # skip this stuff.
                    continue
                if (band != 'detection') & ('map' in attr):
                    self.logger.warning(f'Writing {attr} for {band} is not possible.')
                    continue
                ext_name = f'{band}_{attr}'
                try:
                    hdul[ext_name]
                except KeyError:
                    if np.sum([(band.upper() in hdu.name) for hdu in hdul]) == 0:
                        hdul.append(fits.ImageHDU(name=ext_name))
                        self.logger.debug(f'Appended {attr} {band}')
                    else: 
                        if attr == 'model': # go after science
                            if f'{band}_SCIENCE' not in hdul:
                                idx = np.argmax([band.upper() in hdu.name for hdu in hdul])
                                hdul.insert(idx+1, fits.ImageHDU(name=ext_name))
                            else:
                                hdul.insert(hdul.index_of(f'{band}_SCIENCE')+1, fits.ImageHDU(name=ext_name))
                        elif attr == 'residual': # go after model
                            if f'{band}_MODEL' not in hdul:
                                idx = np.argmax([band.upper() in hdu.name for hdu in hdul])
                                hdul.insert(idx+1, fits.ImageHDU(name=ext_name))
                            else:
                                hdul.insert(hdul.index_of(f'{band}_MODEL')+1, fits.ImageHDU(name=ext_name))
                        elif attr == 'chi': # go after residual
                            if f'{band}_RESIDUAL' not in hdul:
                                idx = np.argmax([band.upper() in hdu.name for hdu in hdul])
                                hdul.insert(idx+1, fits.ImageHDU(name=ext_name))
                            else:
                                hdul.insert(hdul.index_of(f'{band}_RESIDUAL')+1, fits.ImageHDU(name=ext_name))
                        else:
                            idx = [hdul.index_of(hdu.name) for hdu in hdul if (band.upper() == '_'.join(hdu.name.split('_')[:-1]))][-1] + 1
                            if idx == len(hdul):
                                hdul.append(fits.ImageHDU(name=ext_name))
                            else:
                                hdul.insert(idx, fits.ImageHDU(name=ext_name))
                if not hasattr(self.data[band][attr], 'data'):
                    continue # segmap + groupmappings
                if self.data[band][attr].data.dtype == bool:
                    hdul[ext_name].data = self.data[band][attr].data.astype(int)
                else:
                    hdul[ext_name].data = self.data[band][attr].data
                # update WCS in header
                if attr in self.headers[band]:
                    for (key, value, comment) in self.headers[band]['science'].cards:
                        if key == 'EXTNAME': 
                            continue
                        hdul[ext_name].header[key] = (value, comment)
                for (key, value, comment) in self.data[band][attr].wcs.to_header().cards:
                    hdul[ext_name].header[key] = (value, comment)
                
                self.logger.debug(f'... added {attr} for {band}')

            if isinstance(self.catalogs[band], Table):
                ext_name = f'{band}_catalog'
                try:
                    hdul[ext_name]
                except KeyError:
                    hdul.append(fits.BinTableHDU(name=ext_name))
                hdul[ext_name].data = self.catalogs[band]
                
                self.logger.debug(f'... added catalog for {band}')


        if makenew:
            hdul.writeto(path, overwrite=conf.OVERWRITE)
            self.logger.info(f'Wrote to {filename} (allow_update = {allow_update})')
        else:
            hdul.flush()
            self.logger.info(f'Updated {filename} (allow_update = {allow_update})')
            

    def write_hdf5(self, allow_update=False, tag=None, filename=None, directory=conf.PATH_BRICKS):
        if filename is None:
            filename = self.filename
        if tag is not None:
            filename = filename.replace('.h5', f'_{tag}.h5')
        self.logger.debug(f'Writing to {filename} (allow_update = {allow_update})')
        path = os.path.join(directory, filename) 
        if os.path.exists(path):
            if not allow_update:
                raise RuntimeError(f'Cannot update {filename}! (allow_update = False)')
            else:
                # open files and add to them
                hf = h5py.File(path, 'r+')
        else:
            # make new files
            hf = h5py.File(path, 'a')

        self.logger.debug(f'... adding attributes to hdf5')
        recursively_save_dict_contents_to_group(hf, self.__dict__)

        hf.close()
        if os.path.exists(path):
            self.logger.info(f'Updated {filename} (allow_update = {allow_update})')
        else:
            self.logger.info(f'Wrote to {filename} (allow_update = {allow_update})')


    def read_hdf5(self, filename=None, directory=conf.PATH_BRICKS):
        if filename is None:
            filename = self.filename
        
        path = os.path.join(directory, filename) 
        if not os.path.exists(path):
            raise RuntimeError(f'Cannot find file at {path}!')
        hf = h5py.File(path, 'r')
        attr = recursively_load_dict_contents_from_group(hf)
        hf.close()
        return attr

    def write_catalog(self, catalog_imgtype=None, band_tag='phot', catalog_band=None, allow_update=False, tag=None, filename=None, directory=conf.PATH_CATALOGS, overwrite=False):

        if catalog_imgtype is None:
            catalog_imgtype = self.catalog_imgtype
        if catalog_band is None:
            catalog_band = self.catalog_band
        
        if filename is None:
            filename = self.filename.replace('.h5', '.cat')
        if tag is not None:
            filename = filename.replace('.cat', f'_{tag}.cat')
        self.logger.debug(f'Preparing catalog to be written to {filename} (allow_update = {allow_update})')
        path = os.path.join(directory, filename) 
        if os.path.exists(path):
            if overwrite:
                self.logger.warning(f'I will overwrite existing file {filename}')
                catalog = self.get_catalog(catalog_band, catalog_imgtype)
            elif not allow_update:
                raise RuntimeError(f'Cannot update {filename}! (allow_update = False)')
            else:
                # open files and add to them
                catalog = Table.read(path)
        else:
            # make new files from SEP base
            # NOTE What if there are new things in this run? i.e. residual sources?
            catalog = self.get_catalog(catalog_band, catalog_imgtype)

        # loop over set
        for source_id in self.model_catalog:
            source = self.model_catalog[source_id]
            if not hasattr(source, 'statistics'):
                self.logger.warning(f'Source {source_id} was not fit. Skipping.')
                continue
            group_id = catalog['group_id'][catalog['id'] == source_id][0]
            params = get_params(source)

            # for forced photometry, rename parameters if they are unfrozen
            if 11 in self.model_tracker[source_id]:
                if np.any([prior != 'freeze' for prior in self.phot_priors.values()]):
                    if len(params['_bands']) > 1:
                        band = band_tag
                    elif len(params['_bands']) == 1:
                        band = params['_bands'][0]
                    
                    pos_names = 'ra', 'dec', 'ra_err', 'dec_err'
                    reff_names = 'logre', 'logre_err', 'reff', 'reff_err'
                    reff_names += 'logre_exp', 'logre_exp_err', 'reff_exp', 'reff_exp_err'
                    reff_names += 'logre_dev', 'logre_dev_err', 'reff_dev', 'reff_dev_err'
                    shape_names = 'ellip', 'ellip_err', 'ee1', 'ee1_err', 'ee2', 'ee2_err', 'theta', 'theta_err', 'ba', 'ba_err', 'pa', 'pa_err'
                    shape_names += 'ellip_exp', 'ellip_exp_err', 'ee1_exp', 'ee1_exp_err', 'ee2_exp', 'ee2_exp_err', 'theta_exp', 'theta_exp_err', 'ba_exp', 'ba_exp_err', 'pa_exp', 'pa_exp_err'
                    shape_names += 'ellip_dev', 'ellip_dev_err', 'ee1_dev', 'ee1_dev_err', 'ee2_dev', 'ee2_dev_err', 'theta_dev', 'theta_dev_err', 'ba_dev', 'ba_dev_err', 'pa_dev', 'pa_dev_err'
                    fracdev_names = 'fracdev', 'fracdev_err', 'softfracdev', 'softfracdev_err'
                    
                    if self.phot_priors['pos'] != 'freeze':
                        for name in list(params.keys()):
                            if ((name in pos_names) & (self.phot_priors['pos'] != 'freeze')) \
                                | ((name in reff_names) & (self.phot_priors['reff'] != 'freeze')) \
                                | ((name in shape_names) & (self.phot_priors['shape'] != 'freeze')) \
                                | ((name in fracdev_names) & (self.phot_priors['fracDev'] != 'freeze')):
                                    value = params[name]
                                    params.pop(name)
                                    params[f'{band}_{name}'] = value
                    
            # Add group_time from statistics
            if hasattr(source, 'statistics') and 'group_time' in source.statistics:
                group_time = source.statistics['group_time']
                if 'group_time' not in catalog.colnames:
                    catalog.add_column(Column(length=len(catalog), name='group_time', dtype=float, unit=u.s))
                catalog['group_time'][catalog['id'] == source_id] = group_time
                self.logger.debug(f'G{group_id}.S{source_id} :: group_time = {group_time:2.2f} s')
            
            for name in params:
                if name.startswith('_') | (name == 'total_total'):
                    continue
                value = params[name]
                try:
                    unit = value.unit
                    value = value.value
                except AttributeError:
                    unit = None
                dtype = type(value)
                if type(value) == str:
                    dtype = 'S20'
                if name not in catalog.colnames:
                    catalog.add_column(Column(length=len(catalog), name=name, dtype=dtype, unit=unit))
                catalog[name][catalog['id'] == source_id] = value
                if type(value) == str:
                    self.logger.debug(f'G{group_id}.S{source_id} :: {name} = {value}')
                else:
                    self.logger.debug(f'G{group_id}.S{source_id} :: {name} = {value:2.2f}')

            if 'total_total' in params:
                for name in params['total_total']:
                    value = params['total_total'][name]
                    name = f'total_{name}'
                    try:
                        unit = value.unit
                        value = value.value
                    except AttributeError:
                        unit = None
                    dtype = type(value)
                    if type(value) == str:
                        dtype = 'S20'
                    if name not in catalog.colnames:
                        catalog.add_column(Column(length=len(catalog), name=name, dtype=dtype, unit=unit))
                    catalog[name][catalog['id'] == source_id] = value
                    if type(value) == str:
                        self.logger.debug(f'G{group_id}.S{source_id} :: {name} = {value}')
                    else:
                        self.logger.debug(f'G{group_id}.S{source_id} :: {name} = {value:2.2f}')

        # update catalog for self
        self.set_catalog(catalog, catalog_band=catalog_band, catalog_imgtype=catalog_imgtype)

        # write to disk
        if allow_update | overwrite:
            catalog.write(path, overwrite=conf.OVERWRITE, format='fits')
            self.logger.info(f'Updated {filename} (allow_update = {allow_update}, overwrite = {overwrite})')
        else:
            catalog.write(path, format='fits')
            self.logger.info(f'Wrote to {filename} (allow_update = {allow_update}, overwrite = {overwrite})')