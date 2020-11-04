# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Functions to handle command-line input

Known Issues
------------
None


"""

# General imports
import os
import sys
import time
from functools import partial
import shutil
sys.path.insert(0, os.path.join(os.getcwd(), 'config'))
import pickle
import dill

# Tractor imports
from tractor import NCircularGaussianPSF, PixelizedPSF, Image, Tractor, FluxesPhotoCal, NullWCS, ConstantSky, EllipseESoft, Fluxes, PixPos
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev, GalaxyShape
from tractor.sersic import SersicIndex, SersicGalaxy
from tractor.sercore import SersicCoreGalaxy
from tractor.pointsource import PointSource
from tractor.psfex import PixelizedPsfEx, PsfExModel
from tractor.psf import HybridPixelizedPSF

# Miscellaneous science imports
from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, join
from astropy.wcs import WCS
import astropy.units as u
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import weakref
from scipy import stats
import pathos as pa
from astropy.coordinates import SkyCoord
# import sfdmap

# Local imports
from .brick import Brick
from .mosaic import Mosaic
from .utils import header_from_dict, SimpleGalaxy
from .visualization import plot_background, plot_blob, plot_blobmap, plot_brick, plot_mask
try:
    import config as conf
except:
    raise RuntimeError('Cannot find configuration file!')

# m = sfdmap.SFDMap(conf.SFDMAP_DIR)

# Make sure no interactive plotting is going on.
plt.ioff()
import warnings
warnings.filterwarnings("ignore")

print(
f"""
====================================================================
 ________    _       _______     ____    ____  ________  _______        
|_   __  |  / \     |_   __ \   |_   \  /   _||_   __  ||_   __ \    
  | |_ \_| / _ \      | |__) |    |   \/   |    | |_ \_|  | |__) |   
  |  _|   / ___ \     |  __ /     | |\  /| |    |  _| _   |  __ /    
 _| |_  _/ /   \ \_  _| |  \ \_  _| |_\/_| |_  _| |__/ | _| |  \ \_ 
|_____||____| |____||____| |___||_____||_____||________||____| |___|
                                                                    
--------------------------------------------------------------------
M O D E L   P H O T O M E T R Y   W I T H   T H E   T R A C T O R   
--------------------------------------------------------------------
                                                                    
    (C) 2020 -- J. Weaver (DAWN, University of Copenhagen)          
====================================================================

CONSOLE_LOGGING_LEVEL ..... {conf.CONSOLE_LOGGING_LEVEL}			
LOGFILE_LOGGING_LEVEL ..... {conf.LOGFILE_LOGGING_LEVEL}												
PLOT ...................... {conf.PLOT}																		
NTHREADS .................. {conf.NTHREADS}																			
OVERWRITE ................. {conf.OVERWRITE} 
"""	
)

print('Starting up logging system...')

# Start the logging
import logging.config
logger = logging.getLogger('farmer')

if not len(logger.handlers):
    if conf.LOGFILE_LOGGING_LEVEL is not None:
        logging_level = logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL)
    else:
        logging_level = logging.DEBUG
    logger.setLevel(logging_level)
    formatter = logging.Formatter('[%(asctime)s] %(name)s :: %(levelname)s - %(message)s', '%H:%M:%S')

    # Logging to the console at logging level
    ch = logging.StreamHandler()
    ch.setLevel(logging.getLevelName(conf.CONSOLE_LOGGING_LEVEL))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (conf.LOGFILE_LOGGING_LEVEL is None) | (not os.path.exists(conf.LOGGING_DIR)):
        print('Logging information wills stream only to console.\n')
        
    else:
        # create file handler which logs even debug messages
        logging_path = os.path.join(conf.LOGGING_DIR, 'logfile.log')
        print(f'Logging information will stream to console and {logging_path}\n')
        # If overwrite is on, remove old logger
        if conf.OVERWRITE & os.path.exists(logging_path):
            print('WARNING -- Existing logfile will be overwritten.')
            os.remove(logging_path)

        fh = logging.FileHandler(logging_path)
        fh.setLevel(logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL))
        fh.setFormatter(formatter)
        logger.addHandler(fh)



# When a user invokes the interface, first check the translation file
# Optionally, tell the user.
# Try to import the translate file from it's usual spot first.
try:
    from translate import translate
    logger.info(f'interface.translation :: Imported translate file with {len(translate.keys())} entries.')
    if len(conf.BANDS) != len(translate.keys()):
        logger.warning(f'Configuration file only includes {len(conf.BANDS)} entries!')
    # I have nicknames in the config, I need the raw names for file I/O
    mask = np.ones_like(conf.BANDS, dtype=bool)
    for i, band in enumerate(conf.BANDS):
        if band not in translate.keys():
            logger.warning(f'Cound not find {band} in translate file!')
            mask[i] = False

    # Re-assign bands and rawbands in config object
    logger.debug(f'Assigning nicknames to raw image names:')
    conf.BANDS = list(np.array(conf.BANDS)[mask])
    conf.RAWBANDS = conf.BANDS.copy()
    for i, band in enumerate(conf.RAWBANDS):
        conf.RAWBANDS[i] = translate[band]
        logger.debug(f'     {i+1} :: {conf.RAWBANDS[i]} --> {conf.BANDS[i]}')

# The translation file could not be found, so make a scene.
except:
    logger.warning('interface.translation :: WARNING - Could not import translate file! Will use config instead.')
    logger.info('interface.translation :: Image names must be < 50 characters (FITS standard) - checking...')
    # I have raw names, I need shortened raw names (i.e. nicknames)
    conf.RAWBANDS = conf.BANDS.copy()
    count_short = 0
    for i, band in enumerate(conf.RAWBANDS):
        if len(band) > 50:  
            conf.BANDS[i] = band[:50]
            logger.debug(f'     {i+1} :: {band} --> {conf.BANDS[i]}')
            count_short += 1
    logger.info(f'interface.translation :: Done checking. Shortened {count_short} image names.')


def make_directories():
    """Uses the existing config file to set up the directories. Must call from config.py directory!
    """
    import pathlib
    logger.info('Making directories!')
    dir_dict = {'IMAGE_DIR': conf.IMAGE_DIR,
                'PSF_DIR': conf.PSF_DIR,
                'BRICK_DIR': conf.BRICK_DIR,
                'INTERIM_DIR': conf.INTERIM_DIR,
                'PLOT_DIR': conf.PLOT_DIR,
                'CATALOG_DIR': conf.CATALOG_DIR,
                'LOGGING_DIR': conf.LOGGING_DIR
    }
    for key in dir_dict.keys():
        path = dir_dict[key]
        if os.path.exists(path):  # too important to allow overwrite...
            logger.warning(f'{key} already exists under {path}!')
            for i in dir_dict.keys():
                if path == dir_dict[i]:
                    logger.info(f'{key} was already created for {i}...OK')
                    break
        else:
            logger.info(f'{key} --> {path}')
            pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
            

def make_psf(image_type=conf.MULTIBAND_NICKNAME, band=None, sextractor_only=False, psfex_only=False, override=conf.OVERWRITE):
    """ This is where we automatically construct the PSFs for Farmer.

    Step 1. Run sextractor_only=True to obtain the PSF candidates
    Step 2. Using the output plot, determine the selection box for the stars
    Step 3. Run psfex_only=True to construct the PSF.

    See config file to set box dimensions, psf spatial sampling, etc.

    """

    # If the user asked to make a PSF for the detection image, tell them we don't do that
    if image_type is conf.DETECTION_NICKNAME:
        raise ValueError('Farmer does not use a PSF to perform detection!')

    # Else if the user asks for a PSF to be made for the modeling band
    elif image_type is conf.MODELING_NICKNAME:
        # Make the Mosaic
        logger.info(f'Making PSF for {conf.MODELING_NICKNAME}')
        modmosaic = Mosaic(conf.MODELING_NICKNAME, modeling=True, mag_zeropoint=conf.MODELING_ZPT, skip_build=True)

        # Make the PSF
        logger.info(f'Mosaic loaded for {conf.MODELING_NICKNAME}')
        modmosaic._make_psf(xlims=conf.MOD_REFF_LIMITS, ylims=conf.MOD_VAL_LIMITS, override=override, sextractor_only=sextractor_only, psfex_only=psfex_only)

        logger.info(f'PSF made successfully for {conf.MODELING_NICKNAME}')

    # Else if the user asks for a PSF in one of the bands
    elif image_type is conf.MULTIBAND_NICKNAME:
        
        # Sanity check
        if band not in conf.BANDS:
            raise ValueError(f'{band} is not a valid band nickname!')

        # Use all bands or just one?
        if band is not None:
            sbands = [band,]
        else:
            sbands = conf.BANDS

        # Loop over bands
        for i, band in enumerate(sbands):

            # Figure out PS selection box position and zeropoint
            idx_band = np.array(conf.BANDS) == band
            multi_xlims = np.array(conf.MULTIBAND_REFF_LIMITS)[idx_band][0]
            multi_ylims = np.array(conf.MULTIBAND_VAL_LIMITS)[idx_band][0]
            mag_zpt = np.array(conf.MULTIBAND_ZPT)[idx_band][0]

            # Make the Mosaic
            logger.info(f'Making PSF for {band}')
            bandmosaic = Mosaic(band, mag_zeropoint = mag_zpt, skip_build=True)

            # Make the PSF
            logger.info(f'Mosaic loaded for {band}')
            bandmosaic._make_psf(xlims=multi_xlims, ylims=multi_ylims, override=override, sextractor_only=sextractor_only, psfex_only=psfex_only)

            if not sextractor_only:
                logger.info(f'PSF made successfully for {band}')
            else:
                logger.info(f'interface.make_psf :: SExtraction complete for {band}')
    
    return


def make_bricks(image_type=conf.MULTIBAND_NICKNAME, band=None, brick_id=None, insert=False, skip_psf=True, max_bricks=None, make_new_bricks=False):
    """ Stage 1. Here we collect the detection, modelling, and multiband images for processing. We may also cut them up! 
    
    NB: PSFs can be automatically made at this stage too, assuming you've determined your PSF selection a priori.
    
    """

    # Make bricks for the detection image
    if (image_type==conf.DETECTION_NICKNAME) | (image_type is None):
        # Detection
        logger.info('Making mosaic for detection...')
        detmosaic = Mosaic(conf.DETECTION_NICKNAME, detection=True)

        if conf.NTHREADS > 1:
            logger.warning('Parallelization of brick making is currently disabled')
            # BUGGY DUE TO MEM ALLOC
            # logger.info('Making bricks for detection (in parallel)')
            # pool = mp.ProcessingPool(processes=conf.NTHREADS)
            # pool.map(partial(detmosaic._make_brick, detection=True, overwrite=True), np.arange(0, detmosaic.n_bricks()))

        logger.info('Making bricks for detection (in serial)')
        for bid in np.arange(1, detmosaic.n_bricks()+1):
            detmosaic._make_brick(bid, detection=True, overwrite=True)

    # Make bricks for the modeling image
    elif (image_type==conf.MODELING_NICKNAME) | (image_type is None):
        # Modeling
        logger.info('Making mosaic for modeling...')
        modmosaic = Mosaic(conf.MODELING_NICKNAME, modeling=True)

        # The user wants PSFs on the fly
        if not skip_psf: 

            mod_xlims = np.array(conf.MOD_REFF_LIMITS)
            mod_ylims = np.array(conf.MOD_VAL_LIMITS)
                
            modmosaic._make_psf(xlims=mod_xlims, ylims=mod_ylims)

        # Make bricks in parallel
        if (conf.NTHREADS > 1) & (brick_id is None):
            logger.warning('Parallelization of brick making is currently disabled')
            # BUGGY DUE TO MEM ALLOC
            # if conf.VERBOSE: print('Making bricks for detection (in parallel)')
            # pool = mp.ProcessingPool(processes=conf.NTHREADS)
            # pool.map(partial(modmosaic._make_brick, detection=True, overwrite=True), np.arange(0, modmosaic.n_bricks()))

        # Make bricks in serial
        else:
            if brick_id is not None:
                logger.info(f'Making brick #{brick_id} for modeling (in serial)')
                modmosaic._make_brick(brick_id, modeling=True, overwrite=True)
            else:
                logger.info('Making bricks for modeling (in serial)')
                if max_bricks is None:
                    max_bricks = modmosaic.n_bricks()
                for bid in np.arange(1, max_bricks+1):
                    modmosaic._make_brick(bid, modeling=True, overwrite=True)
    
    # Make bricks for one or more multiband images
    elif (image_type==conf.MULTIBAND_NICKNAME) | (image_type is None):

        # One variable list
        if band is not None:
            try:
                if len(band) > 0:
                    sbands = band
                else:
                    sbands = conf.BANDS
            except:
                sbands = [band,]
        else:
            sbands = conf.BANDS

        # In serial, loop over images
        for i, sband in enumerate(sbands):

            # Assume we can overwrite files unless insertion is explicit
            # First image w/o insertion will make new file anyways
            if make_new_bricks:
                overwrite = True
                if insert | (i > 0):
                    overwrite = False
            else:
                overwrite=False

            # Build the mosaic
            logger.info(f'Making mosaic for image {sband}...')
            bandmosaic = Mosaic(sband)

            # The user wants PSFs made on the fly
            if not skip_psf: 

                idx_band = np.array(conf.BANDS) == sband
                multi_xlims = np.array(conf.MULTIBAND_REFF_LIMITS)[idx_band][0]
                multi_ylims = np.array(conf.MULTIBAND_VAL_LIMITS)[idx_band][0]
                    
                bandmosaic._make_psf(xlims=multi_xlims, ylims=multi_ylims)

            # Make bricks in parallel
            if (conf.NTHREADS > 1)  & (brick_id is None):
                logger.info(f'Making bricks for band {sband} (in parallel)')
                with pa.pools.ProcessPool(ncpus=conf.NTHREADS) as pool:
                    logger.info(f'Parallel processing pool initalized with {conf.NTHREADS} threads.')
                    pool.uimap(partial(bandmosaic._make_brick, detection=False, overwrite=overwrite), np.arange(0, bandmosaic.n_bricks()))
                    logger.info('Parallel processing complete.')
            # Make bricks in serial
            else:
                if brick_id is not None:
                    logger.info(f'Making brick #{brick_id} for multiband (in serial)')
                    bandmosaic._make_brick(brick_id, detection=False, overwrite=overwrite)
                else:
                    logger.info(f'Making bricks for band {sband} (in serial)')
                    if max_bricks is None:
                        max_bricks = bandmosaic.n_bricks()
                    for bid in np.arange(1, max_bricks+1):
                        bandmosaic._make_brick(bid, detection=False, overwrite=overwrite)


    # image type is invalid
    else:
        raise RuntimeError(f'{image_type} is an unrecognized nickname (see {conf.DETECTION_NICKNAME}, {conf.MODELING_NICKNAME}, {conf.MULTIBAND_NICKNAME})')

    return

  
def runblob(blob_id, blobs, modeling=None, catalog=None, plotting=0, source_id=None, source_only=False):
    """ Essentially a private function. Runs each individual blob and handles the bulk of the work. """

    # if conf.NTHREADS != 0:
    #     fh = logging.FileHandler(f'B{blob_id}.log')
    #     fh.setLevel(logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL))
    #     formatter = logging.Formatter('[%(asctime)s] %(name)s :: %(levelname)s - %(message)s', '%H:%M:%S')
    #     fh.setFormatter(formatter)

    #     logger = pathos.logger(level=logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL), handler=fh)

    logger = logging.getLogger(f'farmer.blob.{blob_id}')
    logger.info(f'Starting on Blob #{blob_id}')

    modblob = None
    fblob = None
    tstart = time.time()
    logger.debug('Making weakref proxies of blobs')
    if modeling is None:
        modblob, fblob = weakref.proxy(blobs[0]), weakref.proxy(blobs[1])
    elif modeling:
        modblob = weakref.proxy(blobs)
    else:
        fblob = weakref.proxy(blobs)
    logger.debug(f'Weakref made ({time.time() - tstart:3.3f})s')


    # Make blob with modeling image 
    if modblob is not None:
        logger.debug(f'Making blob with {conf.MODELING_NICKNAME}')
        modblob.logger = logger

        if modblob.rejected:
            logger.info('Blob has been rejected!')
            # if conf.NTHREADS != 0:
            #     logger.removeHandler(fh)
            catout = modblob.bcatalog.copy()
            del modblob
            return catout

        # If the user wants to just model a specific source...
        if source_only & (source_id is not None):
            logger.info(f'Preparing to model single source: {source_id}')
            sid = modblob.bcatalog['source_id']
            modblob.bcatalog = modblob.bcatalog[sid == source_id]
            modblob.n_sources = len(modblob.bcatalog)
            modblob.mids = np.ones(modblob.n_sources, dtype=int)
            modblob.model_catalog = np.zeros(modblob.n_sources, dtype=object)
            modblob.solution_catalog = np.zeros(modblob.n_sources, dtype=object)
            modblob.solved_chisq = np.zeros(modblob.n_sources)
            modblob.solved_bic = np.zeros(modblob.n_sources)
            modblob.solution_chisq = np.zeros(modblob.n_sources)
            modblob.tr_catalogs = np.zeros((modblob.n_sources, 3, 2), dtype=object)
            modblob.chisq = np.zeros((modblob.n_sources, 3, 2))
            modblob.rchisq = np.zeros((modblob.n_sources, 3, 2))
            modblob.bic = np.zeros((modblob.n_sources, 3, 2))
            assert(len(modblob.bcatalog) > 0)

        if (conf.MODEL_PHOT_MAX_NBLOB > 0) & (modblob.n_sources > conf.MODEL_PHOT_MAX_NBLOB):
            logger.info('Number of sources exceeds set limit. Skipping!')
            # if conf.NTHREADS != 0:
            #     logger.removeHandler(fh)
            catout = modblob.bcatalog.copy()
            catout['x'] += modblob.subvector[1]
            catout['y'] += modblob.subvector[0]
            del modblob
            return catout

        # Run models
        if conf.ITERATIVE_SUBTRACTION_THRESH is None:
            iter_thresh = 1E31
        else:
            iter_thresh = conf.ITERATIVE_SUBTRACTION_THRESH
        if (conf.ITERATIVE_SUBTRACTION_THRESH is not None) & (modblob.n_sources >= iter_thresh):
            logger.debug(f'Performing iterative subtraction for {conf.MODELING_NICKNAME}')
            astart = time.time()

            for i, band in enumerate(modblob.bands):
                band_name = band[len(conf.MODELING_NICKNAME)+1:]
                zpt = conf.MULTIBAND_ZPT[modblob._band2idx(band_name)]

            # sorting order
            avg_flux = np.zeros(modblob.n_sources)
            for i, item in enumerate(modblob.bcatalog):
                rawfluxes = np.array([np.sum(img[modblob.segmap == item['source_id']]) for img in modblob.images])
                fluxes = rawfluxes * 10**(-0.4 * (zpt - 23.9))
                avg_flux[i] = np.mean(fluxes, 0)

            index = np.argsort(avg_flux)[::-1] # sort by brightness

            copy_images = modblob.images.copy()
            import copy

            modblob.solution_model_images = np.zeros_like(modblob.images)

        
            for i, idx in enumerate(index):
                logger.debug(f" ({i+1}/{modblob.n_sources}) Attemping to model source #{item['source_id']}")
                itemblob = copy.deepcopy(modblob)
                itemblob.bcatalog = Table(modblob.bcatalog[idx])
                itemblob.n_sources = 1
                itemblob.mids = np.ones(itemblob.n_sources, dtype=int)
                itemblob.model_catalog = np.zeros(itemblob.n_sources, dtype=object)
                itemblob.solution_catalog = np.zeros(itemblob.n_sources, dtype=object)
                itemblob.solved_chisq = np.zeros(itemblob.n_sources)
                itemblob.solved_bic = np.zeros(itemblob.n_sources)
                itemblob.solution_chisq = np.zeros(itemblob.n_sources)
                itemblob.tr_catalogs = np.zeros((itemblob.n_sources, 3, 2), dtype=object)
                itemblob.chisq = np.zeros((itemblob.n_sources, 3, 2))
                itemblob.rchisq = np.zeros((itemblob.n_sources, 3, 2))
                itemblob.bic = np.zeros((itemblob.n_sources, 3, 2))

                itemblob.images = copy_images

                itemblob._is_itemblob = True

                
            
                logger.debug(f'Staging images for {conf.MODELING_NICKNAME} -- blob #{modblob.blob_id}')
                itemblob.stage_images()
                logger.debug(f'Images staged. ({time.time() - astart:3.3f})s')

                astart = time.time()
                logger.debug(f'Modeling images for {conf.MODELING_NICKNAME} -- blob #{modblob.blob_id}')
                status = itemblob.tractor_phot()

                if status:

                    logger.debug(f'Morphology determined. ({time.time() - astart:3.3f})s')

                    logger.debug(f'Transferring results back to parent blob...')
                    #transfer back
                    modblob.bcatalog[idx] = itemblob.bcatalog[0]
                    modblob.solution_model_images += itemblob.solution_model_images
                    
                    # subtract model from image
                    copy_images -= itemblob.solution_model_images

                else:
                    logger.warning(f'Morphology failed! ({time.time() - astart:3.3f})s')

                    # # if conf.NTHREADS != 0:
                    # #     logger.removeHandler(fh)
                    # catout = modblob.bcatalog.copy()
                    # catout['x'] += modblob.subvector[1]
                    # catout['y'] += modblob.subvector[0]
                    # del modblob
                    # return catout


        else:
            astart = time.time()
            logger.debug(f'Staging images for {conf.MODELING_NICKNAME}')
            modblob.stage_images()
            logger.debug(f'Images staged. ({time.time() - astart:3.3f})s')

            astart = time.time()
            logger.debug(f'Modeling images for {conf.MODELING_NICKNAME}')
            status = modblob.tractor_phot()

            if not status:
                logger.warning(f'Morphology failed! ({time.time() - astart:3.3f})s')

                # if conf.NTHREADS != 0:
                #     logger.removeHandler(fh)
                catout = modblob.bcatalog.copy()
                catout['x'] += modblob.subvector[1]
                catout['y'] += modblob.subvector[0]
                del modblob
                return catout

            logger.debug(f'Morphology determined. ({time.time() - astart:3.3f})s')


        # Run follow-up phot
        if conf.DO_APPHOT:
            for img_type in ('image', 'model', 'isomodel', 'residual',):
                for band in modblob.bands:
                    try:
                        modblob.aperture_phot(band, img_type, sub_background=conf.SUBTRACT_BACKGROUND)
                    except:
                        logger.warning(f'Aperture photmetry FAILED for {band} {img_type}. Likely a bad blob.')
        if conf.DO_SEPHOT:
            for img_type in ('image', 'model', 'isomodel', 'residual',):
                for band in modblob.bands:
                    if True:
                        modblob.sep_phot(band, img_type, centroid='MODEL', sub_background=conf.SUBTRACT_BACKGROUND)
                        modblob.sep_phot(band, img_type, centroid='DETECTION', sub_background=conf.SUBTRACT_BACKGROUND)
                    if False: #except:
                        logger.warning(f'SEP photometry FAILED for {band} {img_type}. Likely a bad blob.')

        if conf.DO_SEXPHOT:
            for band in modblob.bands:
                try:
                    modblob.residual_phot(band, sub_background=conf.SUBTRACT_BACKGROUND)
                except:
                    logger.warning(f'SEP residual photmetry FAILED. Likely a bad blob.)')

        duration = time.time() - tstart
        logger.info(f'Solution for Blob #{modblob.blob_id} (N={modblob.n_sources}) arrived at in {duration:3.3f}s ({duration/modblob.n_sources:2.2f}s per src)')
    
        catout = modblob.bcatalog.copy()
        del modblob


    #################### FORCED PHOTOMETRY ################################
    if fblob is not None:
        # make new blob with band information
        logger.debug(f'Making blob with {conf.MULTIBAND_NICKNAME}')
        fblob.logger = logger

        if fblob.rejected:
            logger.info('Blob has been rejected!')
            # if conf.NTHREADS != 0:
            #     logger.removeHandler(fh)
            catout = fblob.bcatalog.copy()
            del fblob
            return catout

        astart = time.time() 
        status = fblob.stage_images()
        if not status:
            # if conf.NTHREADS != 0:
            #     logger.removeHandler(fh)
            catout = fblob.bcatalog.copy()
            del fblob
            return catout
            
        logger.info(f'{len(fblob.bands)} images staged. ({time.time() - astart:3.3f})s')

        
        astart = time.time() 
        if modblob is not None:
            fblob.model_catalog = modblob.solution_catalog.copy()
            fblob.position_variance = modblob.position_variance.copy()
            fblob.parameter_variance = modblob.parameter_variance.copy()
            logger.info(f'Solution parameters transferred. ({time.time() - astart:3.3f})s')

        else:
            if catalog is None:
                raise ValueError('Input catalog not supplied!')
            else:
                blobmask = np.ones(len(catalog))
                if source_id is not None:
                    # If the user wants to just model a specific source...
                    logger.info(f'Preparing to force single source: {source_id}')
                    sid = catalog['source_id']
                    bid = catalog['blob_id']
                    fblob.bcatalog = catalog[(sid == source_id) & (bid == blob_id)]
                    fblob.n_sources = len(fblob.bcatalog)
                    fblob.mids = np.ones(fblob.n_sources, dtype=int)
                    fblob.model_catalog = np.zeros(fblob.n_sources, dtype=object)
                    fblob.solution_catalog = np.zeros(fblob.n_sources, dtype=object)
                    fblob.solved_chisq = np.zeros(fblob.n_sources)
                    fblob.solved_bic = np.zeros(fblob.n_sources)
                    fblob.solution_chisq = np.zeros(fblob.n_sources)
                    fblob.tr_catalogs = np.zeros((fblob.n_sources, 3, 2), dtype=object)
                    fblob.chisq = np.zeros((fblob.n_sources, 3, 2))
                    fblob.rchisq = np.zeros((fblob.n_sources, 3, 2))
                    fblob.bic = np.zeros((fblob.n_sources, 3, 2))
                    assert(len(fblob.bcatalog) > 0)
                else:
                    if blob_id is not None:
                        blobmask = catalog['blob_id'] == blob_id
                    fblob.bcatalog = catalog[blobmask]
                    fblob.n_sources = len(fblob.bcatalog)
                    catalog = catalog[blobmask]

                catalog['X_MODEL'] -= fblob.subvector[1] + fblob.mosaic_origin[1] - conf.BRICK_BUFFER + 1
                catalog['Y_MODEL'] -= fblob.subvector[0] + fblob.mosaic_origin[0] - conf.BRICK_BUFFER + 1

                fblob.model_catalog, good_sources = models_from_catalog(catalog, fblob)
                if (good_sources == False).all():
                    logger.warning('All sources are invalid!')
                    catalog['X_MODEL'] += fblob.subvector[1] + fblob.mosaic_origin[1] - conf.BRICK_BUFFER + 1
                    catalog['Y_MODEL'] += fblob.subvector[0] + fblob.mosaic_origin[0] - conf.BRICK_BUFFER + 1
                    return catalog

                fblob.position_variance = None
                fblob.parameter_variance = None
                fblob.bcatalog = catalog[good_sources]
                fblob.n_sources = len(catalog)

        if fblob.rejected:
            logger.info('Blob has been rejected!')
            # if conf.NTHREADS != 0:
            #     logger.removeHandler(fh)
            catout = fblob.bcatalog.copy()
            del fblob
            return catout

        # Forced phot
        

        astart = time.time() 
        logger.info(f'Starting forced photometry...')
        status = fblob.forced_phot()

        if not status:
            # if conf.NTHREADS != 0:
            #     logger.removeHandler(fh)
            catout = fblob.bcatalog.copy()
            del fblob
            return catout

        logger.info(f'Force photometry complete. ({time.time() - astart:3.3f})s')

        # Run follow-up phot
        if conf.DO_APPHOT:
            for img_type in ('image', 'model', 'isomodel', 'residual',):
                for band in fblob.bands:
                    try:
                        fblob.aperture_phot(band, img_type, sub_background=conf.SUBTRACT_BACKGROUND)
                    except:
                        logger.warning(f'Aperture photmetry FAILED for {band} {img_type}. Likely a bad blob.')
        if conf.DO_SEPHOT:
            for img_type in ('image', 'model', 'isomodel', 'residual',):
                for band in fblob.bands:
                    if True:
                        fblob.sep_phot(band, img_type, centroid='MODEL', sub_background=conf.SUBTRACT_BACKGROUND)
                        fblob.sep_phot(band, img_type, centroid='DETECTION', sub_background=conf.SUBTRACT_BACKGROUND)
                    if False: #except:
                        logger.warning(f'SEP photometry FAILED for {band} {img_type}. Likely a bad blob.')

        if conf.DO_SEXPHOT:
            for band in fblob.bands:
                try:
                    fblob.residual_phot(band, sub_background=conf.SUBTRACT_BACKGROUND)
                except:
                    logger.warning(f'SEP residual photmetry FAILED. Likely a bad blob.)')

        duration = time.time() - tstart
        logger.info(f'Solution for blob {fblob.blob_id} (N={fblob.n_sources}) arrived at in {duration:3.3f}s ({duration/fblob.n_sources:2.2f}s per src)')


        catout = fblob.bcatalog.copy()
        del fblob

    # if conf.NTHREADS != 0:
    #     logger.removeHandler(fh)
    return catout


def detect_sources(brick_id, catalog=None, segmap=None, blobmap=None, use_mask=True):
    """Now we can detect stuff and be rid of it!

    Parameters
    ----------
    brick_id : [type]
        [description]
    catalog : [type], optional
        [description], by default None
    segmap : [type], optional
        [description], by default None
    blobmap : [type], optional
        [description], by default None
    catalog : [type], optional
        [description], by default None
    use_mask : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    RuntimeError
        [description]
    ValueError
        [description]
    ValueError
        [description]
    ValueError
        [description]
    """

    if conf.LOGFILE_LOGGING_LEVEL is not None:
        brick_logging_path = os.path.join(conf.LOGGING_DIR, f"B{brick_id}_logfile.log")
        logging.info(f'Logging information will be streamed to console and to {brick_logging_path}\n')
        # If overwrite is on, remove old logger                                                                                                                                                                                 
        if conf.OVERWRITE & os.path.exists(brick_logging_path):
            logging.warning('Existing logfile will be overwritten.')
            os.remove(brick_logging_path)

        # close and remove the old file handler
        #fh.close()
        #logger.removeHandler(fh)

        # we will add an additional file handler to keep track of brick_id specific information
        # set up the new file handler
        shutil.copy(logging_path, brick_logging_path)
        new_fh = logging.FileHandler(brick_logging_path,mode='a')
        new_fh.setLevel(logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL))
        new_fh.setFormatter(formatter)
            
        logger.addHandler(new_fh)

    # Create detection brick
    tstart = time.time()


    detbrick = stage_brickfiles(brick_id, nickname=conf.DETECTION_NICKNAME, modeling=True, is_detection=True)
    if detbrick is None:
        return

    logger.info(f'Detection brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    # Sextract sources 
    tstart = time.time()
    if (segmap is None) & (catalog is None):
        try:
            detbrick.sextract(conf.DETECTION_NICKNAME, sub_background=conf.DETECTION_SUBTRACT_BACKGROUND, use_mask=use_mask, incl_apphot=conf.DO_APPHOT)
            logger.info(f'Detection brick #{brick_id} sextracted {detbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')
            detbrick.is_borrowed = False
        except:
            raise RuntimeError(f'Detection brick #{brick_id} sextraction FAILED. ({time.time() - tstart:3.3f}s)')
            return

    # or find existing catalog/segmap info
    elif (catalog == 'auto') | ((segmap is not None) & (catalog is not None) & (segmap is not None)):
        if (catalog == 'auto'):
            search_fn = os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat')
            if os.path.exists(search_fn):
                catalog = Table(fits.open(search_fn)[1].data)
            else:
                raise ValueError(f'No valid catalog was found for {brick_id}')
            logger.info(f'Overriding SExtraction with external catalog. ({search_fn})')
            search_fn = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits')
            if os.path.exists(search_fn):
                hdul_seg = fits.open(search_fn)
                segmap = hdul_seg['SEGMAP'].data
                blobmap = hdul_seg['BLOBMAP'].data
            else:
                raise ValueError(f'No valid segmentation map was found for {brick_id}')
        if conf.X_COLNAME is not 'x':
            if 'x' in catalog.colnames:
                if 'x_borrowed' in catalog.colnames:
                    catalog.remove_column('x_borrowed')
                catalog['x'].name = 'x_borrowed'
            catalog[conf.X_COLNAME].name = 'x'
        if conf.Y_COLNAME is not 'y':
            if 'y' in catalog.colnames:
                if 'y_borrowed' in catalog.colnames:
                    catalog.remove_column('y_borrowed')
                catalog['y'].name = 'y_borrowed'
            catalog[conf.Y_COLNAME].name = 'y'
        # catalog['x'] = catalog['x'] - detbrick.mosaic_origin[1] + conf.BRICK_BUFFER - 1
        # catalog['y'] = catalog['y'] - detbrick.mosaic_origin[0] + conf.BRICK_BUFFER - 1
        detbrick.catalog = catalog
        detbrick.n_sources = len(catalog)
        detbrick.n_blobs = len(np.unique(catalog['blob_id']))
        detbrick.is_borrowed = True
        detbrick.segmap = segmap
        detbrick.segmask = segmap.copy()
        detbrick.segmask[segmap!=0] = 1 
        detbrick.blobmap = blobmap
        
    else:
        raise ValueError('No valid segmap, blobmap, and catalog provided to override SExtraction!')
        return

    if (~detbrick.is_borrowed):
        detbrick.cleanup()

    if conf.PLOT > 2:
        plot_blobmap(detbrick, image=detbrick.images[0], band=conf.DETECTION_NICKNAME)

    # Save segmap and blobmaps
    if (~detbrick.is_borrowed):
        tstart = time.time()
        logger.info('Saving segmentation and blob maps...')
        outpath = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits')
        if os.path.exists(outpath) & (~conf.OVERWRITE):
            logger.warning('Segmentation file exists and I will not overwrite it!')
        else:
            hdul = fits.HDUList()
            hdul.append(fits.PrimaryHDU())
            hdul.append(fits.ImageHDU(data=detbrick.segmap, name='SEGMAP', header=detbrick.wcs.to_header()))
            hdul.append(fits.ImageHDU(data=detbrick.blobmap, name='BLOBMAP', header=detbrick.wcs.to_header()))
            outpath = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits')
            hdul.writeto(outpath, overwrite=conf.OVERWRITE)
            hdul.close()
            logger.info(f'Saved to {outpath} ({time.time() - tstart:3.3f}s)')

            tstart = time.time()
        
    else:
        logger.info(f'You gave me a catalog and segmap, so I am not saving it again.')

    filen = open(os.path.join(conf.INTERIM_DIR, f'detbrick_N{brick_id}.pkl'), 'wb')
    dill.dump(detbrick, filen)
    return detbrick


def make_models(brick_id, detbrick='auto', band=None, source_id=None, blob_id=None, multiband_model=len(conf.BANDS)>1, source_only=False):
    """ Stage 2. Detect your sources and determine the best model parameters for them """

    # Warn user that you cannot plot while running multiprocessing...
    if (source_id is None) & (blob_id is None):
        if (conf.NBLOBS == 0) & (conf.NTHREADS > 1) & ((conf.PLOT > 0)):
            conf.PLOT = 0
            logger.warning('Plotting not supported while modeling in parallel!')

    if detbrick=='auto':
        filen = open(os.path.join(conf.INTERIM_DIR, f'detbrick_N{brick_id}.pkl'), 'rb')
        detbrick = dill.load(filen)

    # Create modbrick
    if band is None:
        if not multiband_model:
            img_names = [conf.MODELING_NICKNAME,]
            mod_nickname = conf.MODELING_NICKNAME        

        elif multiband_model:
            img_names = conf.MODELING_BANDS
            for iname in img_names:
                if iname not in conf.BANDS:
                    raise ValueError(f'{iname} is listed as a band to model, but is not found in conf.BANDS!')
            mod_nickname = conf.MULTIBAND_NICKNAME

    else:
        if type(band) == list:
            img_names = band
        else:
            img_names = [band,]
        mod_nickname = conf.MULTIBAND_NICKNAME

    # Loop over bands to do the modelling on -- if model in series!
    eff_area = None
    if not multiband_model:
        for band_num, mod_band in enumerate(img_names):
            tstart = time.time()
            if band_num > 0:
                n_blobs, n_sources, segmap, segmask, blobmap, catalog = detbrick.n_blobs, detbrick.n_sources, detbrick.segmap, detbrick.segmask, detbrick.blobmap, detbrick.catalog
                catalog['x'] = catalog['x'] - detbrick.mosaic_origin[1] + conf.BRICK_BUFFER - 1
                catalog['y'] = catalog['y'] - detbrick.mosaic_origin[0] + conf.BRICK_BUFFER - 1
            modbrick = stage_brickfiles(brick_id, band=mod_band, nickname=mod_nickname, modeling=True)
            if modbrick is None:
                return
            if (band is not None) & (band != conf.MODELING_NICKNAME):
                modbrick.bands = [f'{conf.MODELING_NICKNAME}_{mod_band}',]
                modbrick.n_bands = len(modbrick.bands)
            else:
                mod_band = conf.MODELING_NICKNAME
            logger.info(f'Modeling brick #{brick_id} created ({time.time() - tstart:3.3f}s)')
            
            if conf.PLOT > 2:
                plot_brick(modbrick, 0, band=mod_band)
                plot_background(modbrick, 0, band=mod_band)
                plot_mask(modbrick, 0, band=mod_band)

            if conf.SAVE_BACKGROUND:
                outpath = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_BACKGROUNDS.fits')
                logger.info('Saving background and RMS maps...')
                if os.path.exists(outpath):
                    hdul = fits.open(outpath)
                else:
                    hdul = fits.HDUList()
                    hdul.append(fits.PrimaryHDU())
                for m, mband in enumerate(modbrick.bands):
                    hdul.append(fits.ImageHDU(data=modbrick.background_images[m], name=f'BACKGROUND_{mband}', header=modbrick.wcs.to_header()))
                    hdul[f'BACKGROUND_{mband}'].header['BACK_GLOBAL'] = modbrick.backgrounds[m,0]
                    hdul[f'BACKGROUND_{mband}'].header['BACK_RMS'] = modbrick.backgrounds[m,1]
                    if (conf.SUBTRACT_BACKGROUND_WITH_MASK|conf.SUBTRACT_BACKGROUND_WITH_DIRECT_MEDIAN):
                        hdul[f'BACKGROUND_{mband}'].header['MASKEDIMAGE_GLOBAL'] = modbrick.masked_median[m]
                        hdul[f'BACKGROUND_{mband}'].header['MASKEDIMAGE_RMS'] = modbrick.masked_std[m]
                    hdul.append(fits.ImageHDU(data=modbrick.background_rms_images[m], name=f'RMS_{mband}', header=modbrick.wcs.to_header()))
                    hdul.append(fits.ImageHDU(data=1/np.sqrt(modbrick.weights[m]), name=f'UNC_{mband}', header=modbrick.wcs.to_header()))
                hdul.writeto(outpath, overwrite=conf.OVERWRITE)
                hdul.close()
                logger.info(f'Saved to {outpath} ({time.time() - tstart:3.3f}s)')

            logger.debug(f'Brick #{brick_id} -- Image statistics for {mod_band}')
            shape, minmax, mean, var = stats.describe(modbrick.images[0], axis=None)[:4]
            logger.debug(f'    Limits: {minmax[0]:6.6f} - {minmax[1]:6.6f}')
            logger.debug(f'    Mean: {mean:6.6f}+/-{np.sqrt(var):6.6f}\n')
            logger.debug(f'Brick #{brick_id} -- Weight statistics for {mod_band}')
            shape, minmax, mean, var = stats.describe(modbrick.weights[0], axis=None)[:4]
            logger.debug(f'    Limits: {minmax[0]:6.6f} - {minmax[1]:6.6f}')
            logger.debug(f'    Mean: {mean:6.6f}+/-{np.sqrt(var):6.6f}\n')
            logger.debug(f'Brick #{brick_id} -- Error statistics for {mod_band}')
            shape, minmax, mean, var = stats.describe(1/np.sqrt(np.nonzero(modbrick.weights[0].flatten())), axis=None)[:4]
            logger.debug(f'    Limits: {minmax[0]:6.6f} - {minmax[1]:6.6f}')
            logger.debug(f'    Mean: {mean:6.6f}+/-{np.sqrt(var):6.6f}\n')
            logger.debug(f'Brick #{brick_id} -- Background statistics for {mod_band}')
            logger.debug(f'    Global: {modbrick.backgrounds[0, 0]:6.6f}')
            logger.debug(f'    RMS: {modbrick.backgrounds[0, 1]:6.6f}\n')

            modbrick.catalog = detbrick.catalog.copy()
            modbrick.segmap = detbrick.segmap
            modbrick.n_sources = detbrick.n_sources
            modbrick.is_modeling = True
            # if detbrick.is_borrowed:
            modbrick.blobmap = detbrick.blobmap
            modbrick.n_blobs = detbrick.n_blobs
            modbrick.segmask = detbrick.segmask

            # Transfer to MODBRICK
            tstart = time.time()
            if band_num > 0:
                modbrick.n_blobs, modbrick.n_sources, modbrick.segmap, modbrick.segmask, modbrick.blobmap, modbrick.catalog = n_blobs, n_sources, segmap, segmask, blobmap, catalog
            if modbrick.n_blobs <= 0:
                logger.critical(f'Modeling brick #{brick_id} gained {modbrick.n_blobs} blobs! Quiting.')
                return

            modbrick.run_weights()
            
            modbrick.add_columns(modbrick_name=mod_band, multiband_model = False) # doing on detbrick gets column names wrong
            logger.info(f'Modeling brick #{brick_id} gained {modbrick.n_blobs} blobs with {modbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')

            if source_only:
                if source_id is None:
                    raise ValueError('Source only is set True, but no source is has been provided!')
            
            # Run a specific source or blob
            if (source_id is not None) | (blob_id is not None):
                # conf.PLOT = True
                outcatalog = modbrick.catalog.copy()
                # print('AHHHHH   ', outcatalog['x', 'y'])
                mosaic_origin = modbrick.mosaic_origin
                # print('MOSAIC ORIGIN ', mosaic_origin)
                brick_id = modbrick.brick_id
                if source_id is not None:
                    blob_id = np.unique(modbrick.blobmap[modbrick.segmap == source_id])
                    if len(blob_id) == 1:
                        blob_id = blob_id[0]
                    else:
                        raise ValueError('Requested source is not in brick!')
                if blob_id is not None:
                    if blob_id not in outcatalog['blob_id']:
                        raise ValueError(f'No blobs exist for requested blob id {blob_id}')

                logger.info(f'Running single blob {blob_id}')
                modblob = modbrick.make_blob(blob_id)
                modblob.is_modeling=True

                # if source_id is set, then look at only that source



                
                if modblob.rejected:
                    raise ValueError('Requested blob is invalid')
                if source_only & (source_id not in modblob.bcatalog['source_id']):
                    logger.warning('Requested source is not in blob!')
                    for source in modblob.bcatalog:
                        logger.warning(source['source_id'], source['cflux'])
                    raise ValueError('Requested source is not in blob!')

                output_rows = runblob(blob_id, modblob, modeling=True, plotting=conf.PLOT, source_id=source_id)

                output_cat = vstack(output_rows)
                        
                for colname in output_cat.colnames:
                    if colname not in outcatalog.colnames:
                        colshape = output_cat[colname].shape
                        if colname.startswith('FLUX_APER'):
                            outcatalog.add_column(Column(length=len(outcatalog), dtype=float, shape=(len(conf.APER_PHOT)), name=colname))
                        else:
                            outcatalog.add_column(Column(length=len(outcatalog), dtype=output_cat[colname].dtype, shape=(1,), name=colname))

                #outcatalog = join(outcatalog, output_cat, join_type='left', )
                for row in output_cat:
                    outcatalog[np.where(outcatalog['source_id'] == row['source_id'])[0]] = row

                # vs = outcatalog['VALID_SOURCE']
                # scoords = SkyCoord(ra=outcatalog[vs]['RA'], dec=outcatalog[vs]['DEC'], unit='degree')
                # ebmv = m.ebv(scoords)
                # col_ebmv = Column(np.zeros_like(outcatalog['RA']), name='EBV')
                # col_ebmv[vs] = ebmv
                # outcatalog.add_column(col_ebmv)
                modbrick.catalog = outcatalog
            
            # Else, production mode -- all objects in brick are to be run.
            else:
                
                if conf.NBLOBS > 0:
                    run_n_blobs = conf.NBLOBS
                else:
                    run_n_blobs = modbrick.n_blobs
                logger.info(f'Preparing to run {run_n_blobs} blobs.')
                
                outcatalog = modbrick.catalog.copy()
                mosaic_origin = modbrick.mosaic_origin
                brick_id = modbrick.brick_id

                logger.info('Generating blobs...')
                astart = time.time()
                modblobs = (modbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1))
                logger.info(f'{run_n_blobs} blobs generated ({time.time() - astart:3.3f}s)')
                #del modbrick

                tstart = time.time()

                if conf.NTHREADS > 1:

                    with pa.pools.ProcessPool(ncpus=conf.NTHREADS) as pool:
                        logger.info(f'Parallel processing pool initalized with {conf.NTHREADS} threads.')
                        result = pool.uimap(partial(runblob, modeling=True, plotting=conf.PLOT, source_only=source_only), np.arange(1, run_n_blobs+1), modblobs)
                        output_rows = list(result)
                        logger.info('Parallel processing complete.')


                else:
                    logger.info('Serial processing initalized.')
                    output_rows = [runblob(kblob_id+1, kblob, modeling=True, plotting=conf.PLOT, source_only=source_only) for kblob_id, kblob in enumerate(modblobs)]

                output_cat = vstack(output_rows)

                # estimate effective area
                if conf.ESTIMATE_EFF_AREA:
                    eff_area = np.zeros(len(img_names))
                    for b, bname in enumerate(img_names):
                        eff_area[b] = modbrick.estimate_effective_area(output_cat, bname, modeling=True)[0]

                ttotal = time.time() - tstart
                logger.info(f'Completed {run_n_blobs} blobs with {len(output_cat)} sources in {ttotal:3.3f}s (avg. {ttotal/len(output_cat):2.2f}s per source)')
                        
                for colname in output_cat.colnames:
                    if colname not in outcatalog.colnames:
                        colshape = output_cat[colname].shape
                        if colname.startswith('FLUX_APER'):
                            outcatalog.add_column(Column(length=len(outcatalog), dtype=float, shape=(len(conf.APER_PHOT)), name=colname))
                        else:
                            outcatalog.add_column(Column(length=len(outcatalog), dtype=output_cat[colname].dtype, shape=(1,), name=colname))
                #outcatalog = join(outcatalog, output_cat, join_type='left', )
                for row in output_cat:
                    outcatalog[np.where(outcatalog['source_id'] == row['source_id'])[0]] = row

                # vs = outcatalog['VALID_SOURCE']
                # scoords = SkyCoord(ra=outcatalog[vs]['RA'], dec=outcatalog[vs]['DEC'], unit='degree')
                # ebmv = m.ebv(scoords)
                # col_ebmv = Column(np.zeros_like(outcatalog['RA']), name='EBV')
                # col_ebmv[vs] = ebmv
                # outcatalog.add_column(col_ebmv)
                modbrick.catalog = outcatalog

                # open again and add

                # If user wants model and/or residual images made:
                if conf.MAKE_RESIDUAL_IMAGE:
                    cleancatalog = outcatalog[outcatalog[f'VALID_SOURCE_{modbrick.bands[0]}']]
                    modbrick.make_residual_image(catalog=cleancatalog, use_band_position=False, modeling=True)
                elif conf.MAKE_MODEL_IMAGE:
                    cleancatalog = outcatalog[outcatalog[f'VALID_SOURCE_{modbrick.bands[0]}']]
                    modbrick.make_model_image(catalog=cleancatalog, use_band_position=False, modeling=True)

            # Reconstuct mosaic positions of invalid sources 
            invalid = ~modbrick.catalog[f'VALID_SOURCE_{modbrick.bands[0]}']
            # modbrick.catalog[invalid][f'X_MODEL_{modbrick.bands[0]}'] = modbrick.catalog[invalid]['x_orig'] + modbrick.mosaic_origin[1] - conf.BRICK_BUFFER
            # modbrick.catalog[invalid][f'Y_MODEL_{modbrick.bands[0]}'] = modbrick.catalog[invalid]['y_orig'] + modbrick.mosaic_origin[0] - conf.BRICK_BUFFER

            # print(np.sum(invalid), len(invalid))
            # plt.pause(10)
            # idx = np.argwhere(invalid)[:20]
            # print(modbrick.catalog[idx][f'X_MODEL_{modbrick.bands[0]}'],  np.array(modbrick.catalog[idx]['x_orig']) + modbrick.mosaic_origin[1] - conf.BRICK_BUFFER)

   
    # if multiband model is enabled...
    elif multiband_model:
        tstart = time.time()
        n_sources, segmap, catalog = detbrick.n_sources, detbrick.segmap, detbrick.catalog
        if detbrick.is_borrowed:
            catalog['x'] = catalog['x'] - detbrick.mosaic_origin[1] + conf.BRICK_BUFFER - 1
            catalog['y'] = catalog['y'] - detbrick.mosaic_origin[0] + conf.BRICK_BUFFER - 1
        modbrick = stage_brickfiles(brick_id, band=img_names, nickname=mod_nickname, modeling=True)
        if modbrick is None:
            return
        modbrick.bands = [f'{conf.MODELING_NICKNAME}_{b}' for b in img_names]
        modbrick.n_bands = len(modbrick.bands)
        logger.info(f'Multi-band Modeling brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

        for i, mod_band in enumerate(modbrick.bands):
            if conf.PLOT > 3:
                plot_brick(modbrick, 0, band=mod_band)
                plot_background(modbrick, 0, band=mod_band)
                plot_mask(modbrick, 0, band=mod_band)

            logger.debug(f'Brick #{brick_id} -- Image statistics for {mod_band}')
            shape, minmax, mean, var = stats.describe(modbrick.images[i], axis=None)[:4]
            logger.debug(f'    Limits: {minmax[0]:6.6f} - {minmax[1]:6.6f}')
            logger.debug(f'    Mean: {mean:6.6f}+/-{np.sqrt(var):6.6f}\n')
            logger.debug(f'Brick #{brick_id} -- Weight statistics for {mod_band}')
            shape, minmax, mean, var = stats.describe(modbrick.weights[i], axis=None)[:4]
            logger.debug(f'    Limits: {minmax[0]:6.6f} - {minmax[1]:6.6f}')
            logger.debug(f'    Mean: {mean:6.6f}+/-{np.sqrt(var):6.6f}\n')
            logger.debug(f'Brick #{brick_id} -- Error statistics for {mod_band}')
            shape, minmax, mean, var = stats.describe(1/np.sqrt(np.nonzero(modbrick.weights[i].flatten())), axis=None)[:4]
            logger.debug(f'    Limits: {minmax[0]:6.6f} - {minmax[1]:6.6f}')
            logger.debug(f'    Mean: {mean:6.6f}+/-{np.sqrt(var):6.6f}\n')
            logger.debug(f'Brick #{brick_id} -- Background statistics for {mod_band}')
            logger.debug(f'    Global: {modbrick.backgrounds[i, 0]:6.6f}')
            logger.debug(f'    RMS: {modbrick.backgrounds[i, 1]:6.6f}\n')

        modbrick.catalog = catalog.copy()
        modbrick.segmap = segmap
        modbrick.n_sources = n_sources
        modbrick.is_modeling = True
        modbrick.blobmap = detbrick.blobmap
        modbrick.n_blobs = detbrick.n_blobs
        modbrick.segmask = detbrick.segmask


        # Cleanup on MODBRICK
        tstart = time.time()
        modbrick.shared_params = True ## CRITICAL THING TO DO HERE!
        modbrick.add_columns(multiband_model=True) # doing on detbrick gets column names wrong
        logger.info(f'Modeling brick #{brick_id} has {modbrick.n_blobs} blobs with {modbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')

        modbrick.run_weights()

        if conf.PLOT > 2:
            plot_blobmap(modbrick)

        # Save segmap and blobmaps
        if not detbrick.is_borrowed:
            tstart = time.time()
            logger.info('Saving segmentation and blob maps...')
            hdul = fits.HDUList()
            hdul.append(fits.PrimaryHDU())
            hdul.append(fits.ImageHDU(data=modbrick.segmap, name='SEGMAP', header=modbrick.wcs.to_header()))
            hdul.append(fits.ImageHDU(data=modbrick.blobmap, name='BLOBMAP', header=modbrick.wcs.to_header()))
            outpath = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits')
            hdul.writeto(outpath, overwrite=conf.OVERWRITE)
            hdul.close()
            logger.info(f'Saved to {outpath} ({time.time() - tstart:3.3f}s)')

            if conf.SAVE_BACKGROUND:
                outpath = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_BACKGROUNDS.fits')
                logger.info('Saving background and RMS maps...')
                if os.path.exists(outpath):
                    hdul = fits.open(outpath)
                else:
                    hdul = fits.HDUList()
                    hdul.append(fits.PrimaryHDU())
                for m, mband in enumerate(modbrick.bands):
                    hdul.append(fits.ImageHDU(data=modbrick.background_images[m], name=f'BACKGROUND_{mband}', header=modbrick.wcs.to_header()))
                    hdul[f'BACKGROUND_{mband}'].header['BACK_GLOBAL'] = modbrick.backgrounds[m,0]
                    hdul[f'BACKGROUND_{mband}'].header['BACK_RMS'] = modbrick.backgrounds[m,1]
                    if (conf.SUBTRACT_BACKGROUND_WITH_MASK|conf.SUBTRACT_BACKGROUND_WITH_DIRECT_MEDIAN):
                        hdul[f'BACKGROUND_{mband}'].header['MASKEDIMAGE_GLOBAL'] = modbrick.masked_median[m]
                        hdul[f'BACKGROUND_{mband}'].header['MASKEDIMAGE_RMS'] = modbrick.masked_std[m]
                    hdul.append(fits.ImageHDU(data=modbrick.background_rms_images[m], name=f'RMS_{mband}', header=modbrick.wcs.to_header()))
                    hdul.append(fits.ImageHDU(data=1/np.sqrt(modbrick.weights[m]), name=f'UNC_{mband}', header=modbrick.wcs.to_header()))
                hdul.writeto(outpath, overwrite=conf.OVERWRITE)
                hdul.close()
                logger.info(f'Saved to {outpath} ({time.time() - tstart:3.3f}s)')

            tstart = time.time()
        else:
            logger.info(f'You gave me a catalog and segmap, so I am not saving it again.')

        if source_only:
            if source_id is None:
                raise ValueError('Source only is set True, but no source is has been provided!')
        
        # Run a specific source or blob
        if (source_id is not None) | (blob_id is not None):
            # conf.PLOT = True
            outcatalog = modbrick.catalog.copy()
            # print('AHHHHH   ', outcatalog['x', 'y'])
            mosaic_origin = modbrick.mosaic_origin
            # print('MOSAIC ORIGIN ', mosaic_origin)
            brick_id = modbrick.brick_id
            if source_id is not None:
                blob_id = np.unique(modbrick.blobmap[modbrick.segmap == source_id])
                if len(blob_id) == 1:
                    blob_id = blob_id[0]
                else:
                    raise ValueError('Requested source is not in brick!')
            if blob_id is not None:
                if blob_id not in outcatalog['blob_id']:
                    raise ValueError(f'No blobs exist for requested blob id {blob_id}')

            logger.info(f'Running single blob for blob {blob_id}')
            modblob = modbrick.make_blob(blob_id)
            

            # if source_id is set, then look at only that source



            
            if modblob.rejected:
                raise ValueError('Requested blob is invalid')
            if source_only & (source_id not in modblob.bcatalog['source_id']):
                logger.warning('Requested source is not in blob!')
                for source in modblob.bcatalog:
                    logger.warning(source['source_id'], source['cflux'])
                raise ValueError('Requested source is not in blob!')

            output_rows = runblob(blob_id, modblob, modeling=True, plotting=conf.PLOT, source_id=source_id)

            output_cat = vstack(output_rows)
                    
            for colname in output_cat.colnames:
                if colname not in outcatalog.colnames:
                    colshape = output_cat[colname].shape
                    outcatalog.add_column(Column(length=len(outcatalog), dtype=output_cat[colname].dtype, shape=colshape, name=colname))

            #outcatalog = join(outcatalog, output_cat, join_type='left', )
            for row in output_cat:
                outcatalog[np.where(outcatalog['source_id'] == row['source_id'])[0]] = row

            # vs = outcatalog['VALID_SOURCE']
            # scoords = SkyCoord(ra=outcatalog[vs]['RA'], dec=outcatalog[vs]['DEC'], unit='degree')
            # ebmv = m.ebv(scoords)
            # col_ebmv = Column(np.zeros_like(outcatalog['RA']), name='EBV')
            # col_ebmv[vs] = ebmv
            # outcatalog.add_column(col_ebmv)
            modbrick.catalog = outcatalog
        
        # Else, production mode -- all objects in brick are to be run.
        else:
            
            if conf.NBLOBS > 0:
                run_n_blobs = conf.NBLOBS
            else:
                run_n_blobs = modbrick.n_blobs
            logger.info(f'Preparing to run {run_n_blobs} blobs.')
            
            outcatalog = modbrick.catalog.copy()
            mosaic_origin = modbrick.mosaic_origin
            brick_id = modbrick.brick_id

            logger.info('Generating blobs...')
            astart = time.time()
            modblobs = (modbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1))
            logger.info(f'{run_n_blobs} blobs generated ({time.time() - astart:3.3f}s)')
            #del modbrick

            tstart = time.time()

            if conf.NTHREADS > 1:

                with pa.pools.ProcessPool(ncpus=conf.NTHREADS) as pool:
                    logger.info(f'Parallel processing pool initalized with {conf.NTHREADS} threads.')
                    result = pool.uimap(partial(runblob, modeling=True, plotting=conf.PLOT, source_only=source_only), np.arange(1, run_n_blobs+1), modblobs)
                    output_rows = list(result)
                    logger.info('Parallel processing complete.')


            else:
                logger.info('Serial processing initalized.')
                output_rows = [runblob(kblob_id+1, kblob, modeling=True, plotting=conf.PLOT, source_only=source_only) for kblob_id, kblob in enumerate(modblobs)]

            
            output_cat = vstack(output_rows)

            ttotal = time.time() - tstart
            logger.info(f'Completed {run_n_blobs} blobs with {len(output_cat)} sources in {ttotal:3.3f}s (avg. {ttotal/len(output_cat):2.2f}s per source)')

            # estimate effective area
            if conf.ESTIMATE_EFF_AREA:
                eff_area = dict(zip(img_names, np.zeros(len(img_names))))
                for b, bname in enumerate(img_names):
                    eff_area[bname] = modbrick.estimate_effective_area(output_cat, bname, modeling=True)[0]
            else:
                eff_area = None
                        
            for colname in output_cat.colnames:
                if colname not in outcatalog.colnames:
                    colshape = np.shape(output_cat[colname])
                    if len(colshape) == 2:
                        colshape = colshape[1]
                    else:
                        colshape = 1
                    print(colname, colshape)
                    outcatalog.add_column(Column(length=len(outcatalog), dtype=output_cat[colname].dtype, shape=colshape, name=colname))
            #outcatalog = join(outcatalog, output_cat, join_type='left', )
            for row in output_cat:
                outcatalog[np.where(outcatalog['source_id'] == row['source_id'])[0]] = row

            # vs = outcatalog['VALID_SOURCE']
            # scoords = SkyCoord(ra=outcatalog[vs]['RA'], dec=outcatalog[vs]['DEC'], unit='degree')
            # ebmv = m.ebv(scoords)
            # col_ebmv = Column(np.zeros_like(outcatalog['RA']), name='EBV')
            # col_ebmv[vs] = ebmv
            # outcatalog.add_column(col_ebmv)
            modbrick.catalog = outcatalog
           

        # Reconstuct mosaic positions of invalid sources 
        invalid = ~modbrick.catalog[f'VALID_SOURCE']

   
    modbrick.catalog['x'] = modbrick.catalog['x'] + modbrick.mosaic_origin[1] - conf.BRICK_BUFFER
    modbrick.catalog['y'] = modbrick.catalog['y'] + modbrick.mosaic_origin[0] - conf.BRICK_BUFFER
    modbrick.catalog['x_orig'] = modbrick.catalog['x_orig'] + modbrick.mosaic_origin[1] - conf.BRICK_BUFFER
    modbrick.catalog['y_orig'] = modbrick.catalog['y_orig'] + modbrick.mosaic_origin[0] - conf.BRICK_BUFFER


    # If model bands is more than one, choose best one
    # Choose based on min chisq
    if (len(img_names) > 1) & ~multiband_model:
        logger.info(f'Selecting best-fit models within {len(img_names)} bands')
        name_arr = np.ones(shape=(len(modbrick.catalog), len(img_names)), dtype='U11')
        score_arr = np.zeros(shape=(len(modbrick.catalog), len(img_names)))
        valid_arr = np.zeros(shape=(len(modbrick.catalog), len(img_names)))
        xmodel_arr = np.zeros(shape=(len(modbrick.catalog), len(img_names)))
        ymodel_arr = np.zeros(shape=(len(modbrick.catalog), len(img_names)))
        for i, mod_band in enumerate(img_names):
            name_arr[:, i] = mod_band
            score_arr[:, i] = modbrick.catalog[f'CHISQ_{conf.MODELING_NICKNAME}_{mod_band}']
            xmodel_arr[:, i] = modbrick.catalog[f'X_MODEL_{conf.MODELING_NICKNAME}_{mod_band}']
            ymodel_arr[:, i] = modbrick.catalog[f'Y_MODEL_{conf.MODELING_NICKNAME}_{mod_band}']
            valid_arr[:, i] = modbrick.catalog[f'VALID_SOURCE_{conf.MODELING_NICKNAME}_{mod_band}']
            score_arr[np.logical_not(valid_arr[:,i]), i] = 1E31
        argmin_score = np.argmin(score_arr, 1)
        argmin_zero = np.min(score_arr, 1) == 1E31
        argmin_zero = np.zeros_like(argmin_zero)
        modbrick.catalog['BEST_MODEL_BAND'][~argmin_zero] = [modband_opt[k] for modband_opt, k in zip(name_arr[~argmin_zero], argmin_score[~argmin_zero])]
        modbrick.catalog['X_MODEL'][~argmin_zero] = [modband_opt[k] for modband_opt, k in zip(xmodel_arr[~argmin_zero], argmin_score[~argmin_zero])]
        modbrick.catalog['Y_MODEL'][~argmin_zero] = [modband_opt[k] for modband_opt, k in zip(ymodel_arr[~argmin_zero], argmin_score[~argmin_zero])]
        modbrick.catalog['VALID_SOURCE'][~argmin_zero] = [modband_opt[k] for modband_opt, k in zip(valid_arr[~argmin_zero], argmin_score[~argmin_zero])]
        # if modbrick.wcs is not None:
        #         skyc = self.brick_wcs.all_pix2world(modbrick.catalog[f'X_MODEL'] - modbrick.mosaic_origin[0] + conf.BRICK_BUFFER, modbrick.catalog[f'Y_MODEL'] - modbrick.mosaic_origin[1] + conf.BRICK_BUFFER, 0)
        #         modbrick.bcatalog[row][f'RA'] = skyc[0]
        #         modbrick.bcatalog[row][f'DEC'] = skyc[1]
        #         logger.info(f"    Sky Model RA, Dec:   {skyc[0]:6.6f} deg, {skyc[1]:6.6f} deg")

    elif (len(img_names) > 1) & multiband_model:
        modbrick.catalog['BEST_MODEL_BAND'] = conf.MODELING_NICKNAME
        # modbrick.catalog['X_MODEL']
        # modbrick.catalog['Y_MODEL'] # ???? WHAT
        # modbrick.catalog['VALID_SOURCE']

    
    elif img_names[0] != conf.MODELING_NICKNAME:
        modbrick.catalog['BEST_MODEL_BAND'] = f'{conf.MODELING_NICKNAME}_{img_names[0]}'
        modbrick.catalog['X_MODEL'] = modbrick.catalog[f'X_MODEL_{conf.MODELING_NICKNAME}_{img_names[0]}']
        modbrick.catalog['Y_MODEL'] = modbrick.catalog[f'Y_MODEL_{conf.MODELING_NICKNAME}_{img_names[0]}']
        modbrick.catalog['VALID_SOURCE'] = modbrick.catalog[f'VALID_SOURCE_{conf.MODELING_NICKNAME}_{img_names[0]}']
    else:
        modbrick.catalog['BEST_MODEL_BAND'] = f'{conf.MODELING_NICKNAME}'
        modbrick.catalog['X_MODEL'] = modbrick.catalog[f'X_MODEL_{conf.MODELING_NICKNAME}']
        modbrick.catalog['Y_MODEL'] = modbrick.catalog[f'Y_MODEL_{conf.MODELING_NICKNAME}']
        modbrick.catalog['VALID_SOURCE'] = modbrick.catalog[f'VALID_SOURCE_{conf.MODELING_NICKNAME}']

    # write out cat
    if conf.OUTPUT:
        hdr = header_from_dict(conf.__dict__)
        if eff_area is not None:
            for b, band in enumerate(conf.BANDS):
                if band in img_names:
                    eff_area_deg = eff_area[b] * (conf.PIXEL_SCALE / 3600)**2
                    hdr.set(f'AREA{b}', eff_area_deg, f'{conf.MODELING_NICKNAME} {band} EFF_AREA (deg2)')
        hdu_info = fits.ImageHDU(header=hdr, name='CONFIG')
        hdu_table = fits.table_to_hdu(modbrick.catalog)
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu_table, hdu_info])
        outpath = os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat')
        hdul.writeto(outpath, output_verify='ignore', overwrite=conf.OVERWRITE)
        logger.info(f'Wrote out catalog to {outpath}')


    # If user wants model and/or residual images made:
    if conf.MAKE_RESIDUAL_IMAGE:
        cleancatalog = outcatalog[outcatalog[f'VALID_SOURCE']]
        modbrick.make_residual_image(catalog=cleancatalog, use_band_position=False, modeling=True)
    elif conf.MAKE_MODEL_IMAGE:
        cleancatalog = outcatalog[outcatalog[f'VALID_SOURCE']]
        modbrick.make_model_image(catalog=cleancatalog, use_band_position=False, modeling=True)

    # close the brick_id specific file handlers         
    if conf.LOGFILE_LOGGING_LEVEL is not None:                                                                                
        new_fh.close()
        logger.removeHandler(new_fh)


def force_photometry(brick_id, band=None, source_id=None, blob_id=None, insert=True, source_only=False, unfix_bandwise_positions=(not conf.FREEZE_FORCED_POSITION), unfix_bandwise_shapes=(not conf.FREEZE_FORCED_SHAPE), rao_cramer_only=False):

    if band is None:
        fband = conf.BANDS
        addName = conf.MULTIBAND_NICKNAME
    else:
        if (type(band) == list) | (type(band) == np.ndarray):
            fband = band
        elif (type(band) == str) | (type(band) == np.str_):
            fband = [band,]
        else:
            sys.exit('ERROR -- Input band is not a list, array, or string!')
        
        addName = '_'.join(fband)

    # create new logging file
    if conf.LOGFILE_LOGGING_LEVEL is not None:
        brick_logging_path = os.path.join(conf.LOGGING_DIR, f"B{brick_id}_{addName}_logfile.log")
        logger.info(f'Logging information will be streamed to console and to {brick_logging_path}\n')
        # If overwrite is on, remove old logger                                                                                           
        if conf.OVERWRITE & os.path.exists(brick_logging_path):
            logger.warning('Existing logfile will be overwritten.')
            os.remove(brick_logging_path)
        # close and remove the old file handler                                                                                           
        #fh.close()                                                                                                         
        #logger.removeHandler(fh)                                                                                                  
        
        # we will add an additional file handler to keep track of brick_id specific information                                                                   
        # set up the new file handler                                                                                                
        shutil.copy(logging_path, brick_logging_path)
        new_fh = logging.FileHandler(brick_logging_path,mode='a')
        new_fh.setLevel(logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL))
        new_fh.setFormatter(formatter)
        logger.addHandler(new_fh)

    # TODO Check if the catalog will be too big...


    if ((not unfix_bandwise_positions) & (not unfix_bandwise_shapes)) | (len(fband) == 1):
        
        force_models(brick_id=brick_id, band=band, source_id=source_id, blob_id=blob_id, insert=insert, source_only=source_only, force_unfixed_pos=False, use_band_shape=unfix_bandwise_shapes, rao_cramer_only=rao_cramer_only)

    else:
        if conf.FREEZE_FORCED_POSITION:
            logger.warning('Setting FREEZE_FORCED_POSITION to False!')
            conf.FREEZE_FORCED_POSITION = False
       
        for b in fband:
            tstart = time.time()
            logger.critical(f'Running Forced Photometry on {b}')
            if rao_cramer_only:
                logger.critical('WARNING -- ONLY COMPUTING RAO-CRAMER FLUX ERRORS! THIS IS NOT A NORMAL MODE!')
                logger.critical('ENSURE PLOTTING IS TURNED OFF!!!')
            force_models(brick_id=brick_id, band=b, source_id=source_id, blob_id=blob_id, insert=insert, source_only=source_only, force_unfixed_pos=True, use_band_shape=unfix_bandwise_shapes, rao_cramer_only=rao_cramer_only)
            logger.critical(f'Forced Photometry for {b} finished in {time.time() - tstart:3.3f}s')
            # TODO -- compare valid source_band and add to catalog!

        if conf.PLOT >  0: # COLLECT SRCPROFILES
            logger.info('Collecting srcprofile diagnostic plots...')
            import glob
            # find sids
            files = glob.glob(os.path.join(conf.PLOT_DIR, f'T{brick_id}_B*_S*_*_srcprofile.pdf'))
            sids= []
            for f in files:
                tsid = int(f[len(conf.PLOT_DIR):].split('S')[1].split('_')[0])
                if tsid not in sids:
                    sids.append(tsid)
            for sid in sids:
                logger.debug(f' * source {sid}')
                fnames = []
                files = glob.glob(os.path.join(conf.PLOT_DIR, f'T{brick_id}_B*_S{sid}_*_srcprofile.pdf'))
                if len(files) == 0:
                    logger.error('Source {source_id} does not have any srcprofile plots to collect!')
                    return
                bid = int(files[0][len(conf.PLOT_DIR):].split('B')[1].split('_')[0])
                for b in fband:
                    logger.debug(f' *** adding {b}')
                    fname = os.path.join(conf.PLOT_DIR, f'T{brick_id}_B{bid}_S{sid}_{b}_srcprofile.pdf')
                    if os.path.exists(fname):
                        fnames.append(fname)
                    else:
                        logger.warning(f' *** {b} does not exist!')

                # collect
                from PyPDF2 import PdfFileMerger
                merger = PdfFileMerger()

                for pdf in fnames:
                    merger.append(pdf)

                logger.debug(f'Writing out combined srcprofile...')
                merger.write(os.path.join(conf.PLOT_DIR, f'T{brick_id}_B{bid}_S{sid}_srcprofile.pdf'))
                merger.close()

                # remove
                logger.debug(f'Removing individual srcprofiles...')
                [os.system(f'rm {fname}') for fname in fnames]


def force_models(brick_id, band=None, source_id=None, blob_id=None, insert=True, source_only=False, force_unfixed_pos=(not conf.FREEZE_FORCED_POSITION), use_band_shape=(not conf.FREEZE_FORCED_SHAPE), rao_cramer_only=False):
    """ Stage 3. Force the models on the other images and solve only for flux. """

    # Create and update multiband brick
    tstart = time.time()
    eff_area = None

    if source_only:
            if source_id is None:
                raise ValueError('Source only is set True, but no source is has been provided!')

    if (source_id is None) & (blob_id is None):
        if (conf.NBLOBS == 0) & (conf.NTHREADS > 1) & (conf.PLOT > 0):
            conf.PLOT = 0
            logger.warning('Plotting not supported while forcing models in parallel!')

    if band is None:
        fband = conf.BANDS
    else:
        if (type(band) == list) | (type(band) == np.ndarray):
            fband = band
        elif (type(band) == str) | (type(band) == np.str_):
            fband = [band,]
        else:
            sys.exit('ERROR -- Input band is not a list, array, or string!')
            

    fbrick = stage_brickfiles(brick_id, nickname=conf.MULTIBAND_NICKNAME, band=fband, modeling=False)
    if fbrick is None:
        return

    search_fn = os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat')
    if os.path.exists(search_fn):
        fbrick.catalog = Table(fits.open(search_fn)[1].data)
        fbrick.n_sources = len(fbrick.catalog)
        fbrick.n_blobs = fbrick.catalog['blob_id'].max()
    else:
        logger.critical(f'No valid catalog was found for {brick_id}')
        return

    search_fn = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits')
    if os.path.exists(search_fn):
        hdul_seg = fits.open(search_fn)
        fbrick.segmap = hdul_seg['SEGMAP'].data
        fbrick.blobmap = hdul_seg['BLOBMAP'].data
        fbrick.segmask = fbrick.segmap.copy()
        fbrick.segmask[fbrick.segmap>0] = 1
    else:
        logger.critical(f'No valid segmentation map was found for {brick_id}')
        return

    if (~fbrick.catalog['VALID_SOURCE_MODELING']).all():
        logger.critical(f'All sources in brick #{brick_id} are invalid. Quitting!')
        return

    uniq_src, index_src = np.unique(fbrick.catalog['source_id'], return_index=True)
    if len(uniq_src) != len(fbrick.catalog):
        n_nonuniq = len(fbrick.catalog) - len(uniq_src)
        logger.warning(f'Removing {n_nonuniq} non-unique sources from catalog!')
        fbrick.catalog = fbrick.catalog[index_src]

    if not rao_cramer_only:
        fbrick.add_columns(modeling=False)
    else:
        filler = np.zeros(len(fbrick.catalog))
        for colname in fbrick.bands:
            colname = colname.replace(' ', '_')
            fbrick.catalog.add_column(Column(filler, name=f'RAW_DIRECTFLUX_{colname}'))
            fbrick.catalog.add_column(Column(filler, name=f'RAW_DIRECTFLUXERR_{colname}'))
            fbrick.catalog.add_column(Column(filler, name=f'DIRECTFLUX_{colname}'))
            fbrick.catalog.add_column(Column(filler, name=f'DIRECTFLUXERR_{colname}'))
    fbrick.run_background()
    fbrick.run_weights()

    logger.info(f'{conf.MULTIBAND_NICKNAME} brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    if conf.PLOT > 2:
        for plt_band in fband:
            if (len(fband) == 1) | force_unfixed_pos:
                idx = 0
            else:
                idx = np.argwhere(np.array(fband)==plt_band)[0][0]
            plot_brick(fbrick, idx, band=plt_band)
            plot_background(fbrick, idx, band=plt_band)
            plot_mask(fbrick, idx, band=plt_band)
            fcat = fbrick.catalog.copy()
            fcat['x'] -= fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1
            fcat['y'] -= fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1
            plot_blobmap(fbrick, image=fbrick.images[idx], band=plt_band, catalog=fcat)


    for i, vb_band in enumerate(fband):
        logger.debug(f'Brick #{brick_id} -- Image statistics for {vb_band}')
        shape, minmax, mean, var = stats.describe(fbrick.images[i], axis=None)[:4]
        logger.debug(f'    Limits: {minmax[0]:6.6f} - {minmax[1]:6.6f}')
        logger.debug(f'    Mean: {mean:6.6f}+/-{np.sqrt(var):6.6f}\n')
        logger.debug(f'Brick #{brick_id} -- Weight statistics for {vb_band}')
        shape, minmax, mean, var = stats.describe(fbrick.weights[i], axis=None)[:4]
        logger.debug(f'    Limits: {minmax[0]:6.6f} - {minmax[1]:6.6f}')
        logger.debug(f'    Mean: {mean:6.6f}+/-{np.sqrt(var):6.6f}\n')
        logger.debug(f'Brick #{brick_id} -- Error statistics for {vb_band}')
        shape, minmax, mean, var = stats.describe(1/np.sqrt(np.nonzero(fbrick.weights[i].flatten())), axis=None)[:4]
        logger.debug(f'    Limits: {minmax[0]:6.6f} - {minmax[1]:6.6f}')
        logger.debug(f'    Mean: {mean:6.6f}+/-{np.sqrt(var):6.6f}\n')
        logger.debug(f'Brick #{brick_id} -- Background statistics for {vb_band}')
        logger.debug(f'    Global: {fbrick.backgrounds[i, 0]:6.6f}')
        logger.debug(f'    RMS: {fbrick.backgrounds[i, 1]:6.6f}')

    if conf.SAVE_BACKGROUND:
        outpath = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_BACKGROUNDS.fits')
        logger.info('Saving background and RMS maps...')
        if os.path.exists(outpath):
            hdul = fits.open(outpath)
        else:
            hdul = fits.HDUList()
            hdul.append(fits.PrimaryHDU())

        for m, mband in enumerate(fbrick.bands):
            hdul.append(fits.ImageHDU(data=fbrick.background_images[m], name=f'BACKGROUND_{mband}', header=fbrick.wcs.to_header()))
            hdul[f'BACKGROUND_{mband}'].header['BACK_GLOBAL'] = fbrick.backgrounds[m,0]
            hdul[f'BACKGROUND_{mband}'].header['BACK_RMS'] = fbrick.backgrounds[m,1]
            if (conf.SUBTRACT_BACKGROUND_WITH_MASK|conf.SUBTRACT_BACKGROUND_WITH_DIRECT_MEDIAN):
                hdul[f'BACKGROUND_{mband}'].header['MASKEDIMAGE_GLOBAL'] = fbrick.masked_median[m]
                hdul[f'BACKGROUND_{mband}'].header['MASKEDIMAGE_RMS'] = fbrick.masked_std[m]
            hdul.append(fits.ImageHDU(data=fbrick.background_rms_images[m], name=f'RMS_{mband}', header=fbrick.wcs.to_header()))
            hdul.append(fits.ImageHDU(data=1/np.sqrt(fbrick.weights[m]), name=f'UNC_{mband}', header=fbrick.wcs.to_header()))
        hdul.writeto(outpath, overwrite=conf.OVERWRITE)
        hdul.close()
        logger.info(f'Saved to {outpath} ({time.time() - tstart:3.3f}s)')
            

    logger.info(f'Forcing models on {len(fband)} {conf.MULTIBAND_NICKNAME} bands')

    tstart = time.time()
    if (source_id is not None) | (blob_id is not None):
        # conf.PLOT = True
        if source_id is not None:
            blob_id = np.unique(fbrick.blobmap[fbrick.segmap == source_id])
            assert(len(blob_id) == 1)
            blob_id = blob_id[0]
        fblob = fbrick.make_blob(blob_id)
        if source_only & (source_id not in fbrick.catalog['source_id']):
            logger.warning('Requested source is not in blob!')
            for source in fbrick.catalog:
                logger.debug(source['source_id'], source['cflux'])
            raise ValueError('Requested source is not in blob!')
        
        if rao_cramer_only:
            output_rows = runblob_rc(blob_id, fblob, catalog=fbrick.catalog, source_id=source_id)
        else:
            output_rows = runblob(blob_id, fblob, modeling=False, catalog=fbrick.catalog, plotting=conf.PLOT, source_id=source_id)

        if not conf.OUTPUT:
            logging.warning('OUTPUT is DISABLED! Quitting...')
        else:
            output_cat = vstack(output_rows)

            if insert & conf.OVERWRITE & (conf.NBLOBS==0):
                # open old cat
                path_mastercat = os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat')
                if os.path.exists(path_mastercat):
                    mastercat = Table.read(path_mastercat, format='fits')

                    # find new columns
                    newcols = np.in1d(output_cat.colnames, mastercat.colnames, invert=True)
                    # make fillers
                    for colname in np.array(output_cat.colnames)[newcols]:
                        #mastercat.add_column(output_cat[colname])
                        if colname not in mastercat.colnames:
                            if colname.startswith('FLUX_APER') | colname.startswith('MAG_APER'):
                                mastercat.add_column(Column(length=len(mastercat), dtype=float, shape=(len(conf.APER_PHOT),), name=colname))
                            else:
                                mastercat.add_column(Column(length=len(mastercat), dtype=output_cat[colname].dtype, shape=(1,), name=colname))

                    for row in output_cat:
                        mastercat[np.where(mastercat['source_id'] == row['source_id'])[0]] = row
                    # coordinate correction
                    # fbrick.catalog['x'] = fbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
                    # fbrick.catalog['y'] = fbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
                    # save
                    mastercat.write(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat'), format='fits', overwrite=conf.OVERWRITE)
                    logger.info(f'Saving results for brick #{fbrick.brick_id} to existing catalog file.')
            else:
                    
                for colname in output_cat.colnames:
                    if colname not in fbrick.catalog.colnames:
                        
                        if colname.startswith('FLUX_APER') | colname.startswith('MAG_APER'):
                            fbrick.catalog.add_column(Column(length=len(fbrick.catalog), dtype=float, shape=(len(conf.APER_PHOT),), name=colname))
                        else:
                            fbrick.catalog.add_column(Column(length=len(fbrick.catalog), dtype=output_cat[colname].dtype, shape=(1,), name=colname))

                #fbrick.catalog = join(fbrick.catalog, output_cat, join_type='left', )
                for row in output_cat:
                    fbrick.catalog[np.where(fbrick.catalog['source_id'] == row['source_id'])[0]] = row

                mode_ext = conf.MULTIBAND_NICKNAME
                if fband is not None:
                    if len(fband) == 1:
                        mode_ext = fband[0].replace(' ', '_')
                    else:
                        mode_ext = conf.MULTIBAND_NICKNAME

                # write out cat
                # fbrick.catalog['x'] = fbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
                # fbrick.catalog['y'] = fbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
                if conf.OUTPUT:
                    fbrick.catalog.write(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}_{mode_ext}.cat'), format='fits', overwrite=conf.OVERWRITE)
                    logger.info(f'Saving results for brick #{fbrick.brick_id} to new {mode_ext} catalog file.')


    else:

        if conf.NBLOBS > 0:
            run_n_blobs = conf.NBLOBS
        else:
            run_n_blobs = fbrick.n_blobs

        fblobs = (fbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1))

        if conf.NTHREADS > 1:

            with pa.pools.ProcessPool(ncpus=conf.NTHREADS) as pool:
                logger.info(f'Parallel processing pool initalized with {conf.NTHREADS} threads.')
                if rao_cramer_only:
                    result = pool.uimap(partial(runblob_rc, catalog=fbrick.catalog), np.arange(1, run_n_blobs+1), fblobs)
                else:
                    result = pool.uimap(partial(runblob, modeling=False, catalog=fbrick.catalog, plotting=conf.PLOT), np.arange(1, run_n_blobs+1), fblobs)
                output_rows = list(result)
                logger.info('Parallel processing complete.')


        else:
            if rao_cramer_only:
                output_rows = [runblob_rc(kblob_id, fbrick.make_blob(kblob_id), catalog=fbrick.catalog) for kblob_id in np.arange(1, run_n_blobs+1)]
            else:
                output_rows = [runblob(kblob_id, fbrick.make_blob(kblob_id), modeling=False, catalog=fbrick.catalog, plotting=conf.PLOT) for kblob_id in np.arange(1, run_n_blobs+1)]

        logger.info(f'Completed {run_n_blobs} blobs in {time.time() - tstart:3.3f}s')

        #output_rows = [x for x in output_rows if x is not None]

        output_cat = vstack(output_rows)  # HACK -- at some point this should just UPDATE the bcatalog with the new photoms. IF the user sets NBLOBS > 0, the catalog is truncated!
        uniq_src, idx_src = np.unique(output_cat['source_id'], return_index=True)
        # if len(idx_src) != len(fbrick.catalog):
        #     raise RuntimeError(f'Output catalog is truncated! {len(idx_src)} out of {len(fbrick.catalog)}')
        if len(uniq_src) < len(output_cat):
            logger.warning(f'Found {len(uniq_src)} unique sources, out of {len(output_cat)} -- CLEANING!')
            output_cat = output_cat[idx_src]
        else:
            logger.debug(f'Found {len(uniq_src)} unique sources, out of {len(output_cat)}')


        # estimate effective area
        if conf.ESTIMATE_EFF_AREA:
            eff_area = dict(zip(fband, np.zeros(len(fband))))
            for b, bname in enumerate(fband):
                eff_area[bname] = fbrick.estimate_effective_area(output_cat, bname, modeling=False)[0]
        else:
            eff_area = None

        if not conf.OUTPUT:
            logging.warning('OUTPUT is DISABLED! Quitting...')
        else:
            if insert & conf.OVERWRITE & (conf.NBLOBS==0) & (not force_unfixed_pos):
                # open old cat
                path_mastercat = os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat')
                if os.path.exists(path_mastercat):
                    mastercat = Table.read(path_mastercat, format='fits')

                    # find new columns
                    newcols = np.in1d(output_cat.colnames, mastercat.colnames, invert=True)
                    # make fillers
                    for colname in np.array(output_cat.colnames)[newcols]:
                        if colname not in mastercat.colnames:
                            if colname.startswith('FLUX_APER') | colname.startswith('MAG_APER'):
                                mastercat.add_column(Column(length=len(mastercat), dtype=float, shape=(len(conf.APER_PHOT),), name=colname))
                            else:
                                mastercat.add_column(Column(length=len(mastercat), dtype=output_cat[colname].dtype, shape=(1,), name=colname))
                    for row in output_cat:
                        mastercat[np.where(mastercat['source_id'] == row['source_id'])[0]] = row
                    # coordinate correction
                    # fbrick.catalog['x'] = fbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
                    # fbrick.catalog['y'] = fbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
                    # save
                    hdr = fits.open(path_mastercat)['CONFIG'].header
                    lastb = 0
                    for b in np.arange(99):
                        if 'AREA{b}' not in hdr.keys():
                            lastb = b
                    if eff_area is not None:
                        for b, band in enumerate(conf.BANDS):
                            if band in fband:
                                eff_area_deg = eff_area[band]  * (conf.PIXEL_SCALE / 3600)**2
                                hdr.set(f'AREA{b+lastb}', eff_area_deg, f'{band} EFF_AREA (deg2)')
                    hdu_info = fits.ImageHDU(header=hdr, name='CONFIG')
                    hdu_table = fits.table_to_hdu(mastercat)
                    hdul = fits.HDUList([fits.PrimaryHDU(), hdu_table, hdu_info])
                    hdul.writeto(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat'), overwrite=conf.OVERWRITE)
                    logger.info(f'Saving results for brick #{fbrick.brick_id} to existing catalog file.')

                    outcatalog = mastercat

                else:
                    logger.critical(f'Catalog file for brick #{fbrick.brick_id} could not be found!')
                    return

            elif (not insert) & force_unfixed_pos:
                # make a new MULITBAND catalog or add to it!
                path_mastercat = os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}_{conf.MULTIBAND_NICKNAME}.cat')
                if os.path.exists(path_mastercat):
                    mastercat = Table.read(path_mastercat, format='fits')

                    # find new columns
                    newcols = np.in1d(output_cat.colnames, mastercat.colnames, invert=True)

                    if np.sum(newcols) == 0:
                        logger.warning('Columns exist in catalog -- defaulting to separate file output!')
                        hdr = fits.open(path_mastercat)['CONFIG'].header
                        lastb = 0
                        for b in np.arange(99):
                            if 'AREA{b}' in hdr.keys():
                                lastb = b
                        if eff_area is not None:
                            for b, band in enumerate(conf.BANDS):
                                if band in fband:
                                    eff_area_deg = eff_area[band] * (conf.PIXEL_SCALE / 3600)**2
                                    hdr.set(f'AREA{b+lastb}', eff_area_deg, f'{band} EFF_AREA (deg2)')
                        hdu_info = fits.ImageHDU(header=hdr, name='CONFIG')
                        hdu_table = fits.table_to_hdu(mastercat)
                        hdul = fits.HDUList([fits.PrimaryHDU(), hdu_table, hdu_info])
                        hdul.writeto(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}_{conf.MULTIBAND_NICKNAME}.cat'), overwrite=conf.OVERWRITE)
                        logger.info(f'Saving results for brick #{fbrick.brick_id} to new catalog file.')

                    else:

                        join_cat = output_cat[list(np.array(output_cat.colnames)[newcols])] 
                        join_cat.add_column(output_cat['source_id'])
                        mastercat = join(mastercat, join_cat, keys='source_id', join_type='left')

                        # # add new columns, filled.
                        # newcolnames = []
                        # for colname in np.array(output_cat.colnames)[newcols]:
                        #     if colname not in mastercat.colnames:
                        #         if colname.startswith('FLUX_APER') | colname.startswith('MAG_APER'):
                        #             mastercat.add_column(Column(length=len(mastercat), dtype=float, shape=(len(conf.APER_PHOT),), name=colname))
                        #         else:
                        #             mastercat.add_column(Column(length=len(mastercat), dtype=output_cat[colname].dtype, shape=(1,), name=colname))
                        #         newcolnames.append(colname)
                        # #         if colname.startswith('FLUX_APER') | colname.startswith('MAG_APER'):
                        # #             mastercat.add_column(Column(length=len(mastercat), dtype=float, shape=(len(conf.APER_PHOT),), name=colname))
                        # #         else:
                        # #             mastercat.add_column(Column(length=len(mastercat), dtype=output_cat[colname].dtype, shape=(1,), name=colname))
                        # # [print(j) for j in mastercat.colnames]
                        # # [print(j) for j in output_cat.colnames]


                        # # count = 0
                        # # for row in output_cat:
                        # #     idx = np.where(mastercat['source_id'] == row['source_id'])[0]
                        # for colname in newcolnames:
                        #     mastercat[colname][idx] = output_cat[colname]

                        #     # print(mastercat[np.where(mastercat['source_id'] == row['source_id'])[0]][newcolnames])
                        #     # print(newcolnames)
                        #     # print(row[newcolnames])
                        #     # print(np.where(mastercat['source_id'] == row['source_id'])[0])
                            

                        #     mastercat[newcolnames][idx] = row[newcolnames]

                        #     count+=1

                        hdr = fits.open(path_mastercat)['CONFIG'].header
                        lastb = 0
                        for b in np.arange(99):
                            if 'AREA{b}' not in hdr.keys():
                                lastb = b
                        if eff_area is not None:
                            for b, band in enumerate(conf.BANDS):
                                if band in fband:
                                    eff_area_deg = eff_area[band]  * (conf.PIXEL_SCALE / 3600)**2
                                    hdr.set(f'AREA{b+lastb}', eff_area_deg, f'{band} EFF_AREA (deg2)')
                        hdu_info = fits.ImageHDU(header=hdr, name='CONFIG')
                        hdu_table = fits.table_to_hdu(mastercat)
                        hdul = fits.HDUList([fits.PrimaryHDU(), hdu_table, hdu_info])
                        hdul.writeto(path_mastercat, overwrite=conf.OVERWRITE)
                        logger.info(f'Saving results for brick #{fbrick.brick_id} to existing catalog file.')

                else:
                    mastercat = output_cat
                    hdr = header_from_dict(conf.__dict__)
                    # hdr = fits.open(path_mastercat)['CONFIG'].header
                    # lastb = 0
                    # for b in np.arange(99):
                    #     if 'AREA{b}' not in hdr.keys():
                    lastb = 0
                    if eff_area is not None:
                        for b, band in enumerate(conf.BANDS):
                            if band in fband:
                                eff_area_deg = eff_area[band] * (conf.PIXEL_SCALE / 3600)**2
                                hdr.set(f'AREA{b+lastb}', eff_area_deg, f'{band} EFF_AREA (deg2)')
                    hdu_info = fits.ImageHDU(header=hdr, name='CONFIG')
                    hdu_table = fits.table_to_hdu(mastercat)
                    hdul = fits.HDUList([fits.PrimaryHDU(), hdu_table, hdu_info])
                    hdul.writeto(path_mastercat, overwrite=conf.OVERWRITE)
                    logger.info(f'Saving results for brick #{fbrick.brick_id} to new catalog file.')
                

                outcatalog = mastercat


            else:
                    
                for colname in output_cat.colnames:
                    if colname not in fbrick.catalog.colnames:
                        if colname.startswith('FLUX_APER') | colname.startswith('MAG_APER'):
                            fbrick.catalog.add_column(Column(length=len(fbrick.catalog), dtype=float, shape=(len(conf.APER_PHOT),), name=colname))
                        elif colname.endswith('FLUX_RADIUS'):
                            fbrick.catalog.add_column(Column(length=len(fbrick.catalog), dtype=float, shape=(len(conf.PHOT_FLUXFRAC),), name=colname))
                        else:
                            fbrick.catalog.add_column(Column(length=len(fbrick.catalog), dtype=output_cat[colname].dtype, shape=(1,), name=colname))
                #fbrick.catalog = join(fbrick.catalog, output_cat, join_type='left', )
                for row in output_cat:
                    fbrick.catalog[np.where(fbrick.catalog['source_id'] == row['source_id'])[0]] = row

                mode_ext = conf.MULTIBAND_NICKNAME
                if fband is not None:
                    if len(fband) == 1:
                        mode_ext = fband[0].replace(' ', '_')

                # write out cat
                mastercat = fbrick.catalog
                # fbrick.catalog['x'] = fbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
                # fbrick.catalog['y'] = fbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
                hdr = header_from_dict(conf.__dict__)
                
                lastb = 0
                for b in np.arange(99):
                    if 'AREA{b}' not in hdr.keys():
                        lastb = b
                if eff_area is not None:
                    for b, band in enumerate(conf.BANDS):
                        if band in fband:
                            eff_area_deg = eff_area[band] * (conf.PIXEL_SCALE / 3600)**2
                            hdr.set(f'AREA{b+lastb}', eff_area_deg, f'{band} EFF_AREA (deg2)')
                hdu_info = fits.ImageHDU(header=hdr, name='CONFIG')
                hdu_table = fits.table_to_hdu(fbrick.catalog)
                hdul = fits.HDUList([fits.PrimaryHDU(), hdu_table, hdu_info])
                hdul.writeto(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}_{mode_ext}.cat'), overwrite=conf.OVERWRITE)
                logger.info(f'Saving results for brick #{fbrick.brick_id} to new {mode_ext} catalog file.')

                outcatalog = fbrick.catalog


            # If user wants model and/or residual images made:
            if conf.MAKE_RESIDUAL_IMAGE:
                fbrick.make_residual_image(catalog=outcatalog, use_band_position=force_unfixed_pos, use_band_shape=use_band_shape, modeling=False)
            elif conf.MAKE_MODEL_IMAGE:
                fbrick.make_model_image(catalog=outcatalog, use_band_position=force_unfixed_pos, use_band_shape=use_band_shape, modeling=False)
            
    del fbrick
    return 


def make_model_image(brick_id, band, catalog=None, use_band_position=(not conf.FREEZE_FORCED_POSITION), use_band_shape=(not conf.FREEZE_FORCED_SHAPE), modeling=False):
    # USE BAND w/ MODELING NICKNAME FOR MODELING RESULTS!

    if band.startswith(conf.MODELING_NICKNAME):
        nickname = conf.MULTIBAND_NICKNAME
        sband = band[len(conf.MODELING_NICKNAME)+1:]
        modeling=True
    elif band == conf.MODELING_NICKNAME:
        nickname = conf.MODELING_NICKNAME
        sband = conf.MODELING_NICKNAME
        modeling=True
    else:
        nickname = conf.MULTIBAND_NICKNAME
        sband = band
        modeling=False

    brick = stage_brickfiles(brick_id, nickname=nickname, band=sband)

    if catalog is not None:
        brick.catalog = catalog
        brick.n_sources = len(brick.catalog)
        brick.n_blobs = brick.catalog['blob_id'].max()
        if use_single_band_run:
            use_band_position=True
        else:
            use_band_position=False
    else:
        search_fn = os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat')
        search_fn2 = os.path.join(conf.CATALOG_DIR, f'B{brick_id}_{band}.cat') # this means the band was run by itself!
        if os.path.exists(search_fn) & ~use_single_band_run:
            brick.logger.info(f'Adopting catalog from {search_fn}')
            brick.catalog = Table(fits.open(search_fn)[1].data)
            brick.n_sources = len(brick.catalog)
            brick.n_blobs = brick.catalog['blob_id'].max()
            use_band_position=False
        elif os.path.exists(search_fn2) & use_single_band_run:
            brick.logger.info(f'Adopting catalog from {search_fn2}')
            brick.catalog = Table(fits.open(search_fn2)[1].data)
            brick.n_sources = len(brick.catalog)
            brick.n_blobs = brick.catalog['blob_id'].max()
            use_band_position=True
        else:
            raise ValueError(f'No valid catalog was found for {brick_id}')

    search_fn = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits')
    if os.path.exists(search_fn):
        hdul_seg = fits.open(search_fn)
        brick.segmap = hdul_seg['SEGMAP'].data
        brick.blobmap = hdul_seg['BLOBMAP'].data
        brick.segmask = brick.segmap.copy()
        brick.segmask[brick.segmap>0] = 1
    else:
        raise ValueError(f'No valid segmentation map was found for {brick_id}')

    brick.run_background()

    brick.make_model_image(brick.catalog, use_band_position=use_band_position, modeling=modeling)


def make_residual_image(brick_id, band, catalog=None, use_band_position=(not conf.FREEZE_FORCED_POSITION), use_band_shape=(not conf.FREEZE_FORCED_SHAPE), modeling=False):
    # USE BAND w/ MODELING NICKNAME FOR MODELING RESULTS!

    if band.startswith(conf.MODELING_NICKNAME) | ((modeling==True) & (band != conf.MODELING_NICKNAME)):
        nickname = conf.MULTIBAND_NICKNAME
        if band.startswith(conf.MODELING_NICKNAME):
            sband = band[len(conf.MODELING_NICKNAME)+1:]
        else:
            sband = band
        modeling=True
    elif band == conf.MODELING_NICKNAME:
        nickname = conf.MODELING_NICKNAME
        sband = conf.MODELING_NICKNAME
        modeling=True
    else:
        nickname = conf.MULTIBAND_NICKNAME
        sband = band
        modeling=False

    brick = stage_brickfiles(brick_id, nickname=nickname, band=sband)

    if catalog is not None:
        brick.catalog = catalog
        brick.n_sources = len(brick.catalog)
        brick.n_blobs = brick.catalog['blob_id'].max()
  
    else:
        search_fn = os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat')
        search_fn2 = os.path.join(conf.CATALOG_DIR, f'B{brick_id}_{conf.MULTIBAND_NICKNAME}.cat') # this means the band was run by itself!
        search_fn3 = os.path.join(conf.CATALOG_DIR, f'B{brick_id}_{band}.cat')
        if os.path.exists(search_fn) & ~(use_band_position | use_band_shape):
            brick.logger.info(f'Adopting catalog from {search_fn}')
            brick.catalog = Table(fits.open(search_fn)[1].data)
            brick.n_sources = len(brick.catalog)
            brick.n_blobs = brick.catalog['blob_id'].max()
            use_band_position=False
        elif os.path.exists(search_fn2) & (use_band_position | use_band_shape):
            brick.logger.info(f'Adopting catalog from {search_fn2}')   # Tries to find BXXX_MULTIBAND.fits
            brick.catalog = Table(fits.open(search_fn2)[1].data)
            brick.n_sources = len(brick.catalog)
            brick.n_blobs = brick.catalog['blob_id'].max()
            use_band_position=True
        elif os.path.exists(search_fn3) & (use_band_position | use_band_shape):
            brick.logger.info(f'Adopting catalog from {search_fn3}')  # Tries to find BXXX_BAND.fits
            brick.catalog = Table(fits.open(search_fn3)[1].data)
            brick.n_sources = len(brick.catalog)
            brick.n_blobs = brick.catalog['blob_id'].max()
            use_band_position=True
        else:
            raise ValueError(f'No valid catalog was found for {brick_id}')

    search_fn = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits')
    if os.path.exists(search_fn):
        hdul_seg = fits.open(search_fn)
        brick.segmap = hdul_seg['SEGMAP'].data
        brick.blobmap = hdul_seg['BLOBMAP'].data
        brick.segmask = brick.segmap.copy()
        brick.segmask[brick.segmap>0] = 1
    else:
        raise ValueError(f'No valid segmentation map was found for {brick_id}')

    brick.run_background()

    brick.make_residual_image(brick.catalog, use_band_position=use_band_position, use_band_shape=use_band_shape, modeling=modeling)


def estimate_effective_area(brick_id, band, catalog=None, save=False, use_band_position=(not conf.FREEZE_FORCED_POSITION), use_band_shape=(not conf.FREEZE_FORCED_SHAPE), modeling=False):
    if band.startswith(conf.MODELING_NICKNAME) | ((modeling==True) & (band != conf.MODELING_NICKNAME)):
        nickname = conf.MULTIBAND_NICKNAME
        if band.startswith(conf.MODELING_NICKNAME):
            sband = band[len(conf.MODELING_NICKNAME)+1:]
        else:
            sband = band
        modeling=True
    elif band == conf.MODELING_NICKNAME:
        nickname = conf.MODELING_NICKNAME
        sband = conf.MODELING_NICKNAME
        modeling=True
    else:
        nickname = conf.MULTIBAND_NICKNAME
        sband = band
        modeling=False

    brick = stage_brickfiles(brick_id, nickname=nickname, band=sband)

    if catalog is not None:
        brick.catalog = catalog[catalog['brick_id']==brick_id]
        brick.n_sources = len(brick.catalog)
        brick.n_blobs = brick.catalog['blob_id'].max()
  
    else:
        search_fn = os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat')
        search_fn2 = os.path.join(conf.CATALOG_DIR, f'B{brick_id}_{conf.MULTIBAND_NICKNAME}.cat') # this means the band was run by itself!
        search_fn3 = os.path.join(conf.CATALOG_DIR, f'B{brick_id}_{band}.cat')
        if os.path.exists(search_fn) & ~(use_band_position | use_band_shape):
            brick.logger.info(f'Adopting catalog from {search_fn}')
            brick.catalog = Table(fits.open(search_fn)[1].data)
            brick.n_sources = len(brick.catalog)
            brick.n_blobs = brick.catalog['blob_id'].max()
            use_band_position=False
        elif os.path.exists(search_fn2) & (use_band_position | use_band_shape):
            brick.logger.info(f'Adopting catalog from {search_fn2}')   # Tries to find BXXX_MULTIBAND.fits
            brick.catalog = Table(fits.open(search_fn2)[1].data)
            brick.n_sources = len(brick.catalog)
            brick.n_blobs = brick.catalog['blob_id'].max()
            use_band_position=True
        elif os.path.exists(search_fn3) & (use_band_position | use_band_shape):
            brick.logger.info(f'Adopting catalog from {search_fn3}')  # Tries to find BXXX_BAND.fits
            brick.catalog = Table(fits.open(search_fn3)[1].data)
            brick.n_sources = len(brick.catalog)
            brick.n_blobs = brick.catalog['blob_id'].max()
            use_band_position=True
        else:
            raise ValueError(f'No valid catalog was found for {brick_id}')

    import os
    search_fn = os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits')
    if os.path.exists(search_fn):
        hdul_seg = fits.open(search_fn)
        brick.segmap = hdul_seg['SEGMAP'].data
        brick.blobmap = hdul_seg['BLOBMAP'].data
        brick.segmask = brick.segmap.copy()
        brick.segmask[brick.segmap>0] = 1
    else:
        raise ValueError(f'No valid segmentation map was found for {brick_id}')

    # brick.run_background()

    good_area_pix, inner_area_pix = brick.estimate_effective_area(brick.catalog, sband, modeling=modeling)

    if save:
        import os
        outF = open(os.path.join(conf.INTERIM_DIR, f"effarea_{band}_{brick_id}.dat"), "w")
        outF.write(f'{good_area_pix}\n{inner_area_pix}')
        outF.close()

    return good_area_pix, inner_area_pix


def stage_brickfiles(brick_id, nickname='MISCBRICK', band=None, modeling=False, is_detection=False):
    """ Essentially a private function. Pre-processes brick files and relevant catalogs """

    # Wraps Brick with a single parameter call
    # THIS ASSUMES YOU HAVE IMG, WGT, and MSK FOR ALL BANDS!

    path_brickfile = os.path.join(conf.BRICK_DIR, f'B{brick_id}_N{nickname}_W{conf.BRICK_WIDTH}_H{conf.BRICK_HEIGHT}.fits')
    logger.info(f'Staging brickfile ({path_brickfile}')

    if modeling & (band is None):
        sbands = [nickname,]
    elif band is None:
        sbands = conf.BANDS
    else:
        if type(band) == list:
            sbands = band
        else:
            sbands = [band,]
        # conf.BANDS = sbands
    
    [logger.debug(f' *** {i}') for i in sbands]

    if os.path.exists(path_brickfile):
        # Stage things
        images = np.zeros((len(sbands), conf.BRICK_WIDTH + 2 * conf.BRICK_BUFFER, conf.BRICK_HEIGHT + 2 * conf.BRICK_BUFFER))
        weights = np.zeros_like(images)
        masks = np.zeros_like(images, dtype=bool)

        # Loop over expected bands
        with fits.open(path_brickfile) as hdul_brick:

            # Attempt a WCS
            wcs = WCS(hdul_brick[1].header)

            # Stuff data into arrays
            for i, tband in enumerate(sbands):
                logger.info(f'Adding {tband} IMAGE, WEIGHT, and MASK arrays to brick')
                images[i] = hdul_brick[f"{tband}_IMAGE"].data
                weights[i] = hdul_brick[f"{tband}_WEIGHT"].data
                masks[i] = hdul_brick[f"{tband}_MASK"].data

                if (images[i] == 0).all():
                    logger.critical('HACK: All-zero image found. Cannot perform modelling. Skipping brick!')
                    return None
    else:
        raise ValueError(f'Brick file not found for {path_brickfile}')

    psfmodels = np.zeros((len(sbands)), dtype=object)
    for i, band in enumerate(sbands):
        if band == conf.DETECTION_NICKNAME:
            continue

        if band in conf.PSFGRID:
            # open up gridpnt file
            trypath = os.path.join(conf.PSFGRID_OUT_DIR, f'{band}_OUT')
            if os.path.exists(trypath):
                pathgrid = os.path.join(trypath, f'{band}_GRIDPT.dat')
                if os.path.exists(pathgrid):
                    psftab_grid = ascii.read(pathgrid)
                    psftab_ra = psftab_grid['RA']
                    psftab_dec = psftab_grid['Dec']
                    psfcoords = SkyCoord(ra=psftab_ra*u.degree, dec=psftab_dec*u.degree)
                    psffname = psftab_grid['FILE_ID']
                    psfmodels[i] = (psfcoords, psffname)
                    logger.info(f'Adopted PSFGRID PSF.')
                    continue    
                else:
                    raise RuntimeError(f'{band} is in PRFGRID but does NOT have an gridpoint file!')
            else:
                raise RuntimeError(f'{band} is in PRFGRID but does NOT have an output directory!')

        if band in conf.PRFMAP_PSF:
            if band in conf.PRFMAP_GRID_FILENAME.keys():
                # read in prfmap table
                prftab = ascii.read(conf.PRFMAP_GRID_FILENAME[band])
                prftab_ra = prftab[conf.PRFMAP_COLUMNS[1]]
                prftab_dec = prftab[conf.PRFMAP_COLUMNS[2]]
                prfcoords = SkyCoord(ra=prftab_ra*u.degree, dec=prftab_dec*u.degree)
                prfidx = prftab[conf.PRFMAP_COLUMNS[0]]
                psfmodels[i] = (prfcoords, prfidx)
                logger.info(f'Adopted PRFMap PSF.')
                continue    
            else:
                raise RuntimeError(f'{band} is in PRFMAP_PS but does NOT have a PRFMAP grid filename!')

        path_psffile = os.path.join(conf.PSF_DIR, f'{band}.psf')    
        if os.path.exists(path_psffile) & (not conf.FORCE_GAUSSIAN_PSF):
            try:
                psfmodels[i] = PixelizedPsfEx(fn=path_psffile)
                logger.info(f'PSF model for {band} adopted as PixelizedPsfEx. ({path_psffile})')

            except:
                img = fits.open(path_psffile)[0].data
                img = img.astype('float32')
                img[img<=0.] = 1E-31
                psfmodels[i] = PixelizedPSF(img)
                logger.info(f'PSF model for {band} adopted as PixelizedPSF. ({path_psffile})')
        
        elif os.path.exists(os.path.join(conf.PSF_DIR, f'{band}.fits')) & (not conf.FORCE_GAUSSIAN_PSF):
                path_psffile = os.path.join(conf.PSF_DIR, f'{band}.fits') 
                img = fits.open(path_psffile)[0].data
                img = img.astype('float32')
                img[img<=0.] = 1E-31
                psfmodels[i] = PixelizedPSF(img)
                logger.info(f'PSF model for {band} adopted as PixelizedPSF. ({path_psffile})')

        else:
            if conf.USE_GAUSSIAN_PSF:
                psfmodels[i] = None
                logger.warning(f'PSF model not found for {band} -- using {conf.PSF_SIGMA}" gaussian! ({path_psffile})')
            else:
                raise ValueError(f'PSF model not found for {band}! ({path_psffile})')

    if modeling & (len(sbands) == 1):
        images, weights, masks = images[0], weights[0], masks[0]

    newbrick = Brick(images=images, weights=weights, masks=masks, psfmodels=psfmodels, wcs=wcs, bands=np.array(sbands), brick_id=brick_id)

    return newbrick


def models_from_catalog(catalog, fblob, unit_flux=False):
    """ Given an input catalog, construct models """
    # make multiband catalog from det output
    logger.info('Adopting sources from existing catalog.')
    model_catalog = -99 * np.ones(len(catalog), dtype=object)
    good_sources = np.ones(len(catalog), dtype=bool)

    band, rmvector = fblob.bands, fblob.mosaic_origin

    for i, src in enumerate(catalog):

        best_band = conf.MODELING_NICKNAME #src['BEST_MODEL_BAND']

        if src['BEST_MODEL_BAND'] == '':
            logger.warning(f'Source #{src["source_id"]}: no best-fit model chosen, trying out {conf.MODELING_NICKNAME}')
            best_band = conf.MODELING_NICKNAME

        if (src[f'X_MODEL_{best_band}'] < 0) | (src[f'Y_MODEL_{best_band}'] < 0):
            good_sources[i] = False
            logger.warning(f'Source #{src["source_id"]}: {src[f"SOLMODEL_{best_band}"]} model at ({src[f"X_MODEL_{best_band}"]}, {src[f"Y_MODEL_{best_band}"]}) is INVALID.')
            continue

        inpos = [src[f'X_MODEL_{best_band}'], src[f'Y_MODEL_{best_band}']]

        # for col in src.colnames:
        #     print(f"  {col} :: {src[col]}")
        # print(inpos)
        # print(fblob.subvector)
        # print(fblob.mosaic_origin)

        inpos[0] -= (fblob.subvector[1] + fblob.mosaic_origin[1] - conf.BRICK_BUFFER)
        inpos[1] -= (fblob.subvector[0] + fblob.mosaic_origin[0] - conf.BRICK_BUFFER)

        position = PixPos(inpos[0], inpos[1])
        
        # src.pos[0] + self.subvector[1] + self.mosaic_origin[1] - conf.BRICK_BUFFER
        #     self.bcatalog[row][f'Y_MODEL_{mod_band}'] = src.pos[1] + self.subvector[0] + self.mosaic_origin[0] - conf.BRICK_BUFFER
       
        idx_bands = [fblob._band2idx(b) for b in fblob.bands]
        target_zpt = np.array(conf.MULTIBAND_ZPT)[idx_bands]
        
        if unit_flux:
            flux = Fluxes(**dict(zip(fblob.bands, np.ones(len(fblob.bands)))))
        else:
            try:
                # Make initial guess at flux using PSF!
                # This will UNDERESTIMATE for exp/dev models!
                qflux = np.zeros(len(fblob.bands))
                src_seg = fblob.segmap==src['source_id']
                for j, (img, band) in enumerate(zip(fblob.images, fblob.bands)):
                    max_img = np.nanmax(img * src_seg)                                              # TODO THESE ARENT ALWAYS THE SAME SHAPE!
                    max_psf = np.nanmax(fblob.psfimg[band])
                    qflux[j] = max_img / max_psf

                flux = Fluxes(**dict(zip(fblob.bands, qflux)))


            except:
                try:
                    original_zpt = conf.MODELING_ZPT
                    logger.info(f'Converting fluxes from zerpoint {original_zpt} to {target_zpt}')
                    qflux = src[f'RAWFLUX_{best_band}'] * 10 ** (0.4 * (target_zpt - original_zpt))
                except:
                    # IF I TRIED MULTIBAND MODELING, THEN I STILL NEED AN INITIAL FLUX. START WITH 0 idx!
                    init_band = f'{conf.MODELING_NICKNAME}_{conf.INIT_FLUX_BAND}'
                    if conf.INIT_FLUX_BAND is None:
                        conf.INIT_FLUX_BAND = fblob.bands[0]
                    logger.warning(f'Coming from multiband model, so using flux from {init_band}')
                    original_zpt = fblob._band2idx(conf.INIT_FLUX_BAND)
                    qflux = src[f'RAWFLUX_{init_band}'] * 10 ** (0.4 * (target_zpt - original_zpt))
                    
                flux = Fluxes(**dict(zip(band, qflux)))

        # Check if valid source
        if not src[f'VALID_SOURCE_{best_band}']:
            good_sources[i] = False
            logger.warning(f'Source #{src["source_id"]}: {src[f"SOLMODEL_{best_band}"]} is INVALID.')
            continue

        if (not conf.FREEZE_FORCED_POSITION) & conf.USE_FORCE_POSITION_PRIOR:
            ffps_x = conf.FORCE_POSITION_PRIOR_SIG
            ffps_y = conf.FORCE_POSITION_PRIOR_SIG
            if conf.FORCE_POSITION_PRIOR_SIG in ('auto', 'AUTO'):

                # find position of peak in segment... OR just make a gaussian hit the edge of the segment at 5sigma
                npix = src['npix']
                snr = np.nanmedian((qflux / npix) / fblob.backgrounds[:,1])
                snr_thresh = 1.
                pos_sig_under = 0.1
                if snr < snr_thresh:
                    ffps_x, ffps_y = pos_sig_under, pos_sig_under
                else:
                    seg = fblob.segmap == src['source_id']
                    xpix, ypix = np.nonzero(seg)
                    dx, dy = (np.max(xpix) - np.min(xpix)) / 2., (np.max(ypix) - np.min(ypix)) / 2.
                    ffps_x, ffps_y = dx / 1, dy / 1
                # print(snr, ffps)
                # conf.FORCE_POSITION_PRIOR_SIG = 1 - np.exp(-0.5*src[f'CHISQ_{conf.MODELING_NICKNAME}_{conf.INIT_FLUX_BAND}'])
            logger.info(f'Setting position prior. X = {inpos[0]:2.2f}+/-{ffps_x}; Y = {inpos[1]:2.2f}+/-{ffps_y}')
            position.addGaussianPrior('x', inpos[0], ffps_x)
            position.addGaussianPrior('y', inpos[1], ffps_y)

        #shape = GalaxyShape(src['REFF'], 1./src['AB'], src['theta'])
        if src[f'SOLMODEL_{best_band}'] not in ('PointSource', 'SimpleGalaxy'):
            #shape = EllipseESoft.fromRAbPhi(src['REFF'], 1./src['AB'], -src['THETA'])  # Reff, b/a, phi
            shape = EllipseESoft(src[f'REFF_{best_band}'], src[f'EE1_{best_band}'], src[f'EE2_{best_band}'])
            nre = SersicIndex(src[f'N_{best_band}'])
            fluxcore = flux # Eh, ok.... #HACK

            if conf.USE_FORCE_SHAPE_PRIOR:
                shape.addGaussianPrior('logre', src[f'REFF_{best_band}'], conf.FORCE_REFF_PRIOR_SIG/conf.PIXEL_SCALE )
                shape.addGaussianPrior('ee1', src[f'EE1_{best_band}'], conf.FORCE_EE_PRIOR_SIG/conf.PIXEL_SCALE )
                shape.addGaussianPrior('ee2', src[f'EE2_{best_band}'], conf.FORCE_EE_PRIOR_SIG/conf.PIXEL_SCALE )

        if src[f'SOLMODEL_{best_band}'] == 'PointSource':
            model_catalog[i] = PointSource(position, flux)
            model_catalog[i].name = 'PointSource' # HACK to get around Dustin's HACK.
        elif src[f'SOLMODEL_{best_band}'] == 'SimpleGalaxy':
            model_catalog[i] = SimpleGalaxy(position, flux)
        elif src[f'SOLMODEL_{best_band}'] == 'ExpGalaxy':
            model_catalog[i] = ExpGalaxy(position, flux, shape)
        elif src[f'SOLMODEL_{best_band}'] == 'DevGalaxy':
            model_catalog[i] = DevGalaxy(position, flux, shape)
        elif src[f'SOLMODEL_{best_band}'] == 'SersicGalaxy':
            model_catalog[i] = SersicGalaxy(position, flux, shape, nre)
        elif src[f'SOLMODEL_{best_band}'] == 'SersicCoreGalaxy':
            model_catalog[i] = SersicCoreGalaxy(position, flux, shape, nre, fluxcore)
        elif src[f'SOLMODEL_{best_band}'] == 'FixedCompositeGalaxy':
            #expshape = EllipseESoft.fromRAbPhi(src['EXP_REFF'], 1./src['EXP_AB'],  -src['EXP_THETA'])
            #devshape = EllipseESoft.fromRAbPhi(src['DEV_REFF'], 1./src['DEV_AB'],  -src['DEV_THETA'])
            expshape = EllipseESoft(src[f'EXP_REFF_{best_band}'], src[f'EXP_EE1_{best_band}'], src[f'EXP_EE2_{best_band}'])
            devshape = EllipseESoft(src[f'DEV_REFF_{best_band}'], src[f'DEV_EE1_{best_band}'], src[f'DEV_EE2_{best_band}'])
            model_catalog[i] = FixedCompositeGalaxy(
                                            position, flux,
                                            SoftenedFracDev(src[f'FRACDEV_{best_band}']),
                                            expshape, devshape)
        else:
            raise RuntimeError('Blob is valid but it is somehow missing a model for a source! Bug...')

        logger.debug(f'Source #{src["source_id"]}: {src[f"SOLMODEL_{best_band}"]} model at {position}')
        logger.debug(f'               {flux}') 
        if src[f'SOLMODEL_{best_band}'] not in ('PointSource', 'SimpleGalaxy'):
            if src[f'SOLMODEL_{best_band}'] != 'FixedCompositeGalaxy':
                logger.debug(f"    Reff:               {src[f'REFF_{best_band}']:3.3f}")
                logger.debug(f"    a/b:                {src[f'AB_{best_band}']:3.3f}")
                logger.debug(f"    pa:                 {src[f'THETA_{best_band}']:3.3f}")
            
            if src[f'SOLMODEL_{best_band}'] == 'SersicGalaxy':
                logger.debug(f"    Nsersic:            {src[f'N_{best_band}']:3.3f}")
            
            if src[f'SOLMODEL_{best_band}'] == 'SersicCoreGalaxy':
                logger.debug(f"    Nsersic:            {src[f'N_{best_band}']:3.3f}")
                # logger.debug(f"    FluxCore:           {src[f'FLUXCORE_{best_band}']:3.3f}")

            if src[f'SOLMODEL_{best_band}'] == 'FixedCompositeGalaxy':
                logger.debug(f"EXP|Reff:               {src[f'EXP_REFF_{best_band}']:3.3f}")
                logger.debug(f"    a/b:                {src[f'EXP_AB_{best_band}']:3.3f}")
                logger.debug(f"    pa:                 {src[f'EXP_THETA_{best_band}']:3.3f}")
                logger.debug(f"DEV|Reff:               {src[f'DEV_REFF_{best_band}']:3.3f}")
                logger.debug(f"    a/b:                {src[f'DEV_AB_{best_band}']:3.3f}")
                logger.debug(f"    pa:                 {src[f'DEV_THETA_{best_band}']:3.3f}")


    if (conf.FORCED_PHOT_MAX_NBLOB > 0) & (np.sum(good_sources) > conf.FORCED_PHOT_MAX_NBLOB):
        logger.warning(f'Number of good sources in blob ({np.sum(good_sources)}) exceeded limit of {conf.FORCED_PHOT_MAX_NBLOB}.')
        good_sources = np.zeros_like(good_sources, dtype=bool)

    return model_catalog[good_sources], good_sources


def runblob_rc(blob_id, fblob, catalog=None, source_id=None):
    """ Essentially a private function. Runs each individual blob and handles the bulk of the work. """

    # if conf.NTHREADS != 0:
    #     fh = logging.FileHandler(f'B{blob_id}.log')
    #     fh.setLevel(logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL))
    #     formatter = logging.Formatter('[%(asctime)s] %(name)s :: %(levelname)s - %(message)s', '%H:%M:%S')
    #     fh.setFormatter(formatter)

    #     logger = pathos.logger(level=logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL), handler=fh)

    logger = logging.getLogger(f'farmer.blob.{blob_id}')
    logger.info(f'Starting on Blob #{blob_id}')

    tstart = time.time()
    logger.debug('Making weakref proxies of blobs')
    fblob = weakref.proxy(fblob)
    logger.debug(f'Weakref made ({time.time() - tstart:3.3f})s')

    if fblob is not None:
        # make new blob with band information
        logger.debug(f'Making blob with {conf.MULTIBAND_NICKNAME}')
        fblob.logger = logger

        if fblob.rejected:
            logger.info('Blob has been rejected!')
            # if conf.NTHREADS != 0:
            #     logger.removeHandler(fh)
            catout = fblob.bcatalog.copy()
            del fblob
            return catout

        astart = time.time() 
        status = fblob.stage_images()
        if not status:
            # if conf.NTHREADS != 0:
            #     logger.removeHandler(fh)
            catout = fblob.bcatalog.copy()
            del fblob
            return catout
            
        logger.info(f'{len(fblob.bands)} images staged. ({time.time() - astart:3.3f})s')

        
        astart = time.time() 

        if catalog is None:
            raise ValueError('Input catalog not supplied!')
        else:
            # remove non-unique sources!
            uniq_src, index_src = np.unique(catalog['source_id'], return_index=True)
            if len(uniq_src) != len(catalog):
                n_nonuniq = len(catalog) - len(uniq_src)
                logger.warning(f'Removing {n_nonuniq} non-unique sources from catalog!')
                catalog = catalog[index_src]

            blobmask = np.ones(len(catalog))
            if source_id is not None:
                # If the user wants to just model a specific source...
                logger.info(f'Preparing to force single source: {source_id}')
                sid = catalog['source_id']
                bid = catalog['blob_id']
                fblob.bcatalog = catalog[(sid == source_id) & (bid == blob_id)]
                fblob.n_sources = len(fblob.bcatalog)
                fblob.mids = np.ones(fblob.n_sources, dtype=int)
                fblob.model_catalog = np.zeros(fblob.n_sources, dtype=object)
                fblob.solution_catalog = np.zeros(fblob.n_sources, dtype=object)
                fblob.solved_chisq = np.zeros(fblob.n_sources)
                fblob.solved_bic = np.zeros(fblob.n_sources)
                fblob.solution_chisq = np.zeros(fblob.n_sources)
                fblob.tr_catalogs = np.zeros((fblob.n_sources, 3, 2), dtype=object)
                fblob.chisq = np.zeros((fblob.n_sources, 3, 2))
                fblob.rchisq = np.zeros((fblob.n_sources, 3, 2))
                fblob.bic = np.zeros((fblob.n_sources, 3, 2))
                assert(len(fblob.bcatalog) > 0)
            else:
                if blob_id is not None:
                    blobmask = catalog['blob_id'] == blob_id
                fblob.bcatalog = catalog[blobmask]
                fblob.n_sources = len(fblob.bcatalog)
                catalog = catalog[blobmask]

            band = fblob.bands[0] # HACK!
            # replace the main x/y columns with the forced phot solution position!
            orig_xcol, orig_ycol = catalog[f'X_MODEL'].copy(), catalog[f'Y_MODEL'].copy()
            catalog[f'X_MODEL'] = catalog[f'X_MODEL_{band}'] - fblob.subvector[1] + fblob.mosaic_origin[1] - conf.BRICK_BUFFER + 1
            catalog[f'Y_MODEL'] = catalog[f'Y_MODEL_{band}'] - fblob.subvector[0] + fblob.mosaic_origin[0] - conf.BRICK_BUFFER + 1

            fblob.model_catalog, good_sources = models_from_catalog(catalog, fblob, unit_flux=True) # Gets us unit models!
            if (good_sources == False).all():
                logger.warning('All sources are invalid!')
                catalog[f'X_MODEL_{band}'] = orig_xcol
                catalog[f'Y_MODEL_{band}'] = orig_ycol
                return catalog

            fblob.position_variance = None
            fblob.parameter_variance = None
            fblob.bcatalog = catalog[good_sources]
            fblob.n_sources = len(catalog)

        if fblob.rejected:
            logger.info('Blob has been rejected!')
            # if conf.NTHREADS != 0:
            #     logger.removeHandler(fh)
            catout = fblob.bcatalog.copy()
            del fblob
            return catout

        # Forced phot
    
        astart = time.time() 
        logger.info(f'Starting rao_cramer computation...')

        status = fblob.rao_cramer()

        if not status:
            # if conf.NTHREADS != 0:
            #     logger.removeHandler(fh)
            catout = fblob.bcatalog.copy()
            del fblob
            return catout

        logger.info(f'Force photometry complete. ({time.time() - astart:3.3f})s')

        duration = time.time() - tstart
        logger.info(f'Rao-Cramer computed for blob {fblob.blob_id} (N={fblob.n_sources}) in {duration:3.3f}s ({duration/fblob.n_sources:2.2f}s per src)')


        catout = fblob.bcatalog.copy()
        del fblob

    return catout