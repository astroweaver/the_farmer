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


import os
import sys
import time
from functools import partial
sys.path.insert(0, os.path.join(os.getcwd(), 'config'))

from tractor import NCircularGaussianPSF, PixelizedPSF, Image, Tractor, FluxesPhotoCal, NullWCS, ConstantSky, EllipseESoft, Fluxes, PixPos
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev, GalaxyShape
from tractor.pointsource import PointSource
from tractor.psfex import PixelizedPsfEx, PsfExModel
from tractor.psf import HybridPixelizedPSF

from astropy.io import fits
from astropy.table import Table, Column, vstack, join
from astropy.wcs import WCS
import numpy as np
import pathos.multiprocessing as mp 
import pathos.parallel as pp
from functools import partial
import matplotlib.pyplot as plt
import weakref
from scipy import stats

from .brick import Brick
from .mosaic import Mosaic
from .utils import plot_background, plot_blob, SimpleGalaxy, plot_blobmap, plot_brick, plot_mask, header_from_dict

import config as conf
plt.ioff()

# The logo
if conf.VERBOSE:
    print('====================================================================')
    print(' ________    _       _______     ____    ____  ________  _______    ')    
    print('|_   __  |  / \     |_   __ \   |_   \  /   _||_   __  ||_   __ \   ') 
    print('  | |_ \_| / _ \      | |__) |    |   \/   |    | |_ \_|  | |__) |  ') 
    print('  |  _|   / ___ \     |  __ /     | |\  /| |    |  _| _   |  __ /   ') 
    print(' _| |_  _/ /   \ \_  _| |  \ \_  _| |_\/_| |_  _| |__/ | _| |  \ \_ ')
    print('|_____||____| |____||____| |___||_____||_____||________||____| |___|')
    print('                                                                    ')
    print('--------------------------------------------------------------------')
    print('  M O D E L   P H O T O M E T R Y   W I T H   T H E   T R A C T O R ')
    print('--------------------------------------------------------------------')
    print('                                                                    ')
    print('    (C) 2019 -- J. Weaver (DAWN, University of Copenhagen)          ')
    print('====================================================================')
    print()


# When a user invokes the interface, first check the translation file
# Optionally, tell the user.
# Try to import the translate file from it's usual spot first.
try:
    from translate import translate
    if conf.VERBOSE: print(f'interface.translation :: Imported translate file with {len(translate.keys())} entries.')
    if len(conf.BANDS) != len(translate.keys()):
        if conf.VERBOSE: print(f'interface.translation :: WARNING - Configuration file only includes {len(conf.BANDS)} entries!')
    # I have nicknames in the config, I need the raw names for file I/O
    mask = np.ones_like(conf.BANDS, dtype=bool)
    for i, band in enumerate(conf.BANDS):
        if band not in translate.keys():
            if conf.VERBOSE: print(f'interface.translation :: WARNING - Cound not find {band} in translate file!')
            mask[i] = False

    # Re-assign bands and rawbands in config object
    if conf.VERBOSE: print(f'interface.translation :: Assigning nicknames to raw image names:')
    conf.BANDS = list(np.array(conf.BANDS)[mask])
    conf.RAWBANDS = conf.BANDS.copy()
    for i, band in enumerate(conf.RAWBANDS):
        conf.RAWBANDS[i] = translate[band]
        if conf.VERBOSE: print(f'     {i+1} :: {conf.RAWBANDS[i]} --> {conf.BANDS[i]}')

# The translation file could not be found, so make a scene.
except:
    if conf.VERBOSE: print('interface.translation :: WARNING - Could not import translate file! Will use config instead.')
    if conf.VERBOSE: print('interface.translation :: Image names must be < 50 characters (FITS standard) - checking...')
    # I have raw names, I need shortened raw names (i.e. nicknames)
    conf.RAWBANDS = conf.BANDS.copy()
    count_short = 0
    for i, band in enumerate(conf.RAWBANDS):
        if len(band) > 50:  
            conf.BANDS[i] = band[:50]
            if conf.VERBOSE: print(f'     {i+1} :: {band} --> {conf.BANDS[i]}')
            count_short += 1
    if conf.VERBOSE: print(f'interface.translation :: Done checking. Shortened {count_short} image names.')


def make_psf(image_type=conf.MULTIBAND_NICKNAME, band=None, override=False, sextractor_only=False, psfex_only=False):

    # If the user asked to make a PSF for the detection image, tell them we don't do that
    if image_type is conf.DETECTION_NICKNAME:
        raise ValueError('interface.make_psf :: ERROR - Farmer does not use a PSF to perform detection!')

    # Else if the user asks for a PSF to be made for the modeling band
    elif image_type is conf.MODELING_NICKNAME:
        # Make the Mosaic
        if conf.VERBOSE: print(f'interface.make_psf :: Making PSF for {conf.MODELING_NICKNAME}')
        modmosaic = Mosaic(conf.MODELING_NICKNAME, modeling=True, mag_zeropoint=conf.MODELING_ZPT)

        # Make the PSF
        if conf.VERBOSE: print(f'interface.make_psf :: Mosaic loaded for {conf.MODELING_NICKNAME}')
        modmosaic._make_psf(xlims=conf.MOD_REFF_LIMITS, ylims=conf.MOD_VAL_LIMITS, override=override, sextractor_only=sextractor_only, psfex_only=psfex_only)

        if conf.VERBOSE: print(f'interface.make_psf :: PSF made successfully for {conf.MODELING_NICKNAME}')

    # Else if the user asks for a PSF in one of the bands
    elif image_type is conf.MULTIBAND_NICKNAME:
        
        # Sanity check
        if band not in conf.BANDS:
            raise ValueError(f'interface.make_psf :: ERROR - {band} is not a valid band nickname!')

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
            if conf.VERBOSE: print(f'interface.make_psf :: Making PSF for {band}')
            bandmosaic = Mosaic(band, mag_zeropoint = mag_zpt)

            # Make the PSF
            if conf.VERBOSE: print(f'interface.make_psf :: Mosaic loaded for {band}')
            bandmosaic._make_psf(xlims=multi_xlims, ylims=multi_ylims, override=override, sextractor_only=sextractor_only, psfex_only=psfex_only)

            if not sextractor_only:
                if conf.VERBOSE: print(f'interface.make_psf :: PSF made successfully for {band}')
            else:
                if conf.VERBOSE: print(f'interface.make_psf :: SExtraction complete for {band}')
    
    return


def make_bricks(image_type=conf.MULTIBAND_NICKNAME, band=None, insert=False, skip_psf=False):

    # Make bricks for the detection image
    if image_type==conf.DETECTION_NICKNAME:
        # Detection
        if conf.VERBOSE: print('interface.make_bricks :: Making mosaic for detection')
        detmosaic = Mosaic(conf.DETECTION_NICKNAME, detection=True)

        if conf.NTHREADS > 0:
            print('interface.make_bricks :: WARNING - Parallelization of brick making is currently disabled')
            # BUGGY DUE TO MEM ALLOC
            # if conf.VERBOSE: print('Making bricks for detection (in parallel)')
            # pool = mp.ProcessingPool(processes=conf.NTHREADS)
            # pool.map(partial(detmosaic._make_brick, detection=True, overwrite=True), np.arange(0, detmosaic.n_bricks()))

        if conf.VERBOSE: print('interface.make_bricks :: Making bricks for detection (in serial)')
        for brick_id in np.arange(1, detmosaic.n_bricks()+1):
            detmosaic._make_brick(brick_id, detection=True, overwrite=True)

    # Make bricks for the modeling image
    elif image_type==conf.MODELING_NICKNAME:
        # Modeling
        if conf.VERBOSE: print('interface.make_bricks :: Making mosaic for modeling')
        modmosaic = Mosaic(conf.MODELING_NICKNAME, modeling=True)

        # The user wants PSFs on the fly
        if not skip_psf: 

            mod_xlims = np.array(conf.MOD_REFF_LIMITS)
            mod_ylims = np.array(conf.MOD_VAL_LIMITS)
                
            modmosaic._make_psf(xlims=mod_xlims, ylims=mod_ylims)

        # Make bricks in parallel
        if conf.NTHREADS > 0:
            print('interface.make_bricks :: WARNING - Parallelization of brick making is currently disabled')
            # BUGGY DUE TO MEM ALLOC
            # if conf.VERBOSE: print('Making bricks for detection (in parallel)')
            # pool = mp.ProcessingPool(processes=conf.NTHREADS)
            # pool.map(partial(modmosaic._make_brick, detection=True, overwrite=True), np.arange(0, modmosaic.n_bricks()))

        # Make bricks in serial
        else:
            if conf.VERBOSE: print('interface.make_bricks :: Making bricks for modeling (in serial)')
            for brick_id in np.arange(1, modmosaic.n_bricks()+1):
                modmosaic._make_brick(brick_id, modeling=True, overwrite=True)
    
    # Make bricks for one or more multiband images
    elif image_type==conf.MULTIBAND_NICKNAME:

        # One variable list
        if band is not None:
            sbands = [band,]
        else:
            sbands = conf.BANDS

        # In serial, loop over images
        for i, sband in enumerate(sbands):

            # Assume we can overwrite files unless insertion is explicit
            # First image w/o insertion will make new file anyways
            overwrite = True
            if insert | (i > 0):
                overwrite = False

            # Build the mosaic
            if conf.VERBOSE: print(f'interface.make_bricks :: Making mosaic for image {sband}')
            bandmosaic = Mosaic(sband)

            # The user wants PSFs made on the fly
            if not skip_psf: 

                idx_band = np.array(conf.BANDS) == sband
                multi_xlims = np.array(conf.MULTIBAND_REFF_LIMITS)[idx_band][0]
                multi_ylims = np.array(conf.MULTIBAND_VAL_LIMITS)[idx_band][0]
                    
                bandmosaic._make_psf(xlims=multi_xlims, ylims=multi_ylims)

            # Make bricks in parallel
            if conf.NTHREADS > 0:
                if conf.VERBOSE: print(f'interface.make_bricks :: Making bricks for band {sband} (in parallel)')
                pool = mp.ProcessingPool(processes=conf.NTHREADS)
                pool.map(partial(bandmosaic._make_brick, detection=False, overwrite=overwrite), np.arange(0, bandmosaic.n_bricks()))

            # Make bricks in serial
            else:
                if conf.VERBOSE: print(f'interface.make_bricks :: Making bricks for band {sband} (in serial)')
                for brick_id in np.arange(1, bandmosaic.n_bricks()+1):
                    bandmosaic._make_brick(brick_id, detection=False, overwrite=overwrite)

    # image type is invalid
    else:
        raise RuntimeError(f'interface.make_bricks :: ERROR {image_type} is an unrecognized nickname (see {conf.DETECTION_NICKNAME}, {conf.MODELING_NICKNAME}, {conf.MULTIBAND_NICKNAME})')

    return

        
def runblob(blob_id, blobs, detection=None, catalog=None, plotting=False):

    if conf.VERBOSE: print()
    if conf.VERBOSE: print(f'interface.runblob :: Starting on Blob #{blob_id}')
    tstart = time.time()

    modblob = None
    fblob = None
    if detection is None:
        modblob, fblob = weakref.proxy(blobs[0]), weakref.proxy(blobs[1])
    elif detection:
        modblob = weakref.proxy(blobs)
    else:
        fblob = weakref.proxy(blobs)

    conf.PLOT = plotting

    # Make blob with detection image 

    if modblob is not None:

        # Run models
        astart = time.time()
        modblob.stage_images()
        if conf.VERBOSE2: print(f'interface.runblob :: Images staged. ({time.time() - astart:3.3f})s')

        astart = time.time()
        status = modblob.tractor_phot()

        if not status:
            return modblob.bcatalog.copy()

        if conf.VERBOSE2: print(f'interface.runblob :: Morphology determined. ({time.time() - astart:3.3f})s')


        # Run follow-up phot
        if conf.DO_APPHOT:
            for img_type in ('image', 'model', 'residual'):
                # for band in modblob.bands:
                # try:
                modblob.aperture_phot(image_type=img_type, sub_background=conf.SUBTRACT_BACKGROUND)
                # except:
                #     if conf.VERBOSE: print(f'interface.runblob :: WARNING - Aperture photmetry FAILED for {conf.MODELING_NICKNAME} {img_type}')
        if conf.DO_SEXPHOT:
            try:
                modblob.sextract_phot()
            except:
                if conf.VERBOSE: print(f'interface.runblob :: WARNING - Source extraction on the residual blob FAILED for {conf.MODELING_NICKNAME} {img_type}')    

        duration = time.time() - tstart
        if conf.VERBOSE: print(f'interface.runblob :: Solution for Blob #{modblob.blob_id} (N={modblob.n_sources}) arrived at in {duration:3.3f}s ({duration/modblob.n_sources:2.2f}s per src)')
    
        catout = modblob.bcatalog.copy()
        del modblob

    if fblob is not None:
        # make new blob with band information
        astart = time.time() 

        if modblob is not None:
            fblob.model_catalog = modblob.solution_catalog.copy()
            fblob.position_variance = modblob.position_variance.copy()
            fblob.parameter_variance = modblob.parameter_variance.copy()
            if conf.VERBOSE2: print(f'interface.runblob :: Solution parameters transferred. ({time.time() - astart:3.3f})s')

        else:
            if catalog is None:
                raise ValueError('interface.runblob :: ERROR - Input catalog not supplied!')
            else:
                blobmask = np.ones(len(catalog))
                if blob_id is not None:
                    blobmask = catalog['blob_id'] == blob_id
                fblob.bcatalog = catalog[blobmask]
                fblob.n_sources = len(fblob.bcatalog)
                catalog = catalog[blobmask]

                catalog['X_MODEL'] -= fblob.subvector[1] + fblob.mosaic_origin[1] - conf.BRICK_BUFFER + 1
                catalog['Y_MODEL'] -= fblob.subvector[0] + fblob.mosaic_origin[0] - conf.BRICK_BUFFER + 1

                fblob.model_catalog, good_sources = models_from_catalog(catalog, fblob.bands, fblob.mosaic_origin)
                if (good_sources == False).all():
                    if conf.VERBOSE: print('interface.runblob :: WARNING - All sources are invalid!')
                    catalog['X_MODEL'] += fblob.subvector[1] + fblob.mosaic_origin[1] - conf.BRICK_BUFFER + 1
                    catalog['Y_MODEL'] += fblob.subvector[0] + fblob.mosaic_origin[0] - conf.BRICK_BUFFER + 1
                    return fblob.bcatalog.copy()

                fblob.position_variance = None
                fblob.parameter_variance = None
                fblob.bcatalog = catalog[good_sources]
                fblob.n_sources = len(catalog)


        # Forced phot
        astart = time.time() 
        fblob.stage_images()
        if conf.VERBOSE2: print(f'interface.runblob :: {fblob.bands} images staged. ({time.time() - astart:3.3f})s')

        astart = time.time() 
        status = fblob.forced_phot()

        if not status:
            return fblob.bcatalog.copy()

        if conf.VERBOSE2: print(f'interface.runblob :: Force photometry complete. ({time.time() - astart:3.3f})s')


        # Run follow-up phot
        if conf.DO_APPHOT:
            for img_type in ('image', 'model', 'isomodel', 'residual',):
                for band in fblob.bands:
                    try:
                        fblob.aperture_phot(band, img_type, sub_background=conf.SUBTRACT_BACKGROUND)
                    except:
                        if conf.VERBOSE: print(f'interface.runblob :: Aperture photmetry FAILED for {band} {img_type}. Likely a bad blob.')
        if conf.DO_SEXPHOT:
            try:
                [fblob.sextract_phot(band) for band in fblob.bands]
            except:
                if conf.VERBOSE: print(f'interface.runblob :: Residual Sextractor photmetry FAILED. Likely a bad blob.)')

        duration = time.time() - tstart
        if conf.VERBOSE: print(f'interface.runblob :: Solution for blob {fblob.blob_id} (N={fblob.n_sources}) arrived at in {duration:3.3f}s ({duration/fblob.n_sources:2.2f}s per src)')


        catout = fblob.bcatalog.copy()
        del fblob

    return catout


def make_models(brick_id, source_id=None, blob_id=None, segmap=None, catalog=None, use_mask=True):
    # Create detection brick
    tstart = time.time()
    detbrick = stage_brickfiles(brick_id, nickname=conf.DETECTION_NICKNAME, detection=True)

    if conf.VERBOSE: print(f'Detection brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    # Sextract sources
    tstart = time.time()
    if (segmap is None) & (catalog is None):
        try:
            detbrick.sextract(conf.DETECTION_NICKNAME, sub_background=conf.DETECTION_SUBTRACT_BACKGROUND, use_mask=use_mask, incl_apphot=conf.DO_APPHOT)
            if conf.VERBOSE: print(f'Detection brick #{brick_id} sextracted {detbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')
        except:
            raise RuntimeError(f'Detection brick #{brick_id} sextraction FAILED. ({time.time() - tstart:3.3f}s)')
            return
    elif (segmap is not None) & (catalog is not None):
        catalog[conf.X_COLNAME].colname = 'x'
        catalog[conf.Y_COLNAME].colname = 'y'
        detbrick.catalog = catalog
        detbrick.segmap = segmap
        if conv.VERBOSE: print(f'Overriding SExtraction with external catalog.')
    else:
        raise ValueError('No valid segmap and catalog provided to override SExtraction!')
        return

    # Create modbrick
    tstart = time.time()
    modbrick = stage_brickfiles(brick_id, nickname=conf.MODELING_NICKNAME, detection=True)
    if conf.VERBOSE: print(f'Modeling brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    if conf.PLOT2:
        plot_brick(modbrick, 0, band=conf.MODELING_NICKNAME)
        plot_background(modbrick, 0, band=conf.MODELING_NICKNAME)
        plot_mask(modbrick, 0, band=conf.MODELING_NICKNAME)

    if conf.VERBOSE:
        print()
        print(f'Brick #{brick_id} -- Image statistics for {conf.MODELING_NICKNAME}')
        shape, minmax, mean, var = stats.describe(modbrick.images[0], axis=None)[:4]
        print(f'    Limits: {minmax[0]:3.3f} - {minmax[1]:3.3f}')
        print(f'    Mean: {mean:3.3f}+/-{np.sqrt(var):3.3f}')
        print()
        print(f'Brick #{brick_id} -- Weight statistics for {conf.MODELING_NICKNAME}')
        shape, minmax, mean, var = stats.describe(modbrick.weights[0], axis=None)[:4]
        print(f'    Limits: {minmax[0]:3.3f} - {minmax[1]:3.3f}')
        print(f'    Mean: {mean:3.3f}+/-{np.sqrt(var):3.3f}')
        print()
        print(f'Brick #{brick_id} -- Background statistics for {conf.MODELING_NICKNAME}')
        print(f'    Global: {modbrick.backgrounds[0, 0]:3.3f}')
        print(f'    RMS: {modbrick.backgrounds[0, 1]:3.3f}')

    modbrick.catalog = detbrick.catalog
    modbrick.segmap = detbrick.segmap
    modbrick.n_sources = detbrick.n_sources

    # Cleanup on MODBRICK
    tstart = time.time()
    modbrick.cleanup()
    modbrick.add_columns() # doing on detbrick gets column names wrong
    if conf.VERBOSE: print(f'Modeling brick #{brick_id} gained {modbrick.n_blobs} blobs with {modbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')

    if conf.PLOT2:
        plot_blobmap(modbrick)

    # Save segmap and blobmaps
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul.append(fits.ImageHDU(data=modbrick.segmap, name='SEGMAP'))
    hdul.append(fits.ImageHDU(data=modbrick.blobmap, name='BLOBMAP'))
    hdul.writeto(os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits'), overwrite=conf.OVERWRITE)
    hdul.close()

    tstart = time.time()
    
    # Run a specific source or blob
    if (source_id is not None) | (blob_id is not None):
        # conf.PLOT = True
        outcatalog = modbrick.catalog.copy()
        mosaic_origin = modbrick.mosaic_origin
        brick_id = modbrick.brick_id
        if source_id is not None:
            blob_id = np.unique(modbrick.blobmap[modbrick.segmap == source_id])
            assert(len(blob_id) == 1)
            blob_id = blob_id[0]
        if blob_id is not None:
            if blob_id not in outcatalog['blob_id']:
                raise ValueError(f'ERROR :: Requested blob id {blob_id} is NOT valid!')
        modblob = modbrick.make_blob(blob_id)
        output_rows = runblob(blob_id, modblob, detection=True, plotting=conf.PLOT)

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

        # write out cat
        hdr = header_from_dict(conf.__dict__, verbose=conf.VERBOSE)
        hdu_info = fits.ImageHDU(header=hdr, name='CONFIG')
        hdu_table = fits.table_to_hdu(outcatalog)
        hdul = fits.HDUList([hdu_table, hdu_info])
        outpath = os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat')
        hdul.writeto(outpath, output_verify='ignore', overwrite=conf.OVERWRITE)
        if conf.VERBOSE: print(f'interface.make_models :: Wrote out catalog to {outpath}')

        plt.pause(100)
        return None
    
    # Else, production mode -- all objects in brick are to be run.
    else:

        if conf.NBLOBS > 0:
            run_n_blobs = conf.NBLOBS
        else:
            run_n_blobs = modbrick.n_blobs
        
        outcatalog = modbrick.catalog.copy()
        mosaic_origin = modbrick.mosaic_origin
        brick_id = modbrick.brick_id

        #detblobs = [modbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1)]
        modblobs = (modbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1))
        #del modbrick

        if conf.NTHREADS > 0:

            #pool = pp.ParallelPool(processes=conf.NTHREADS)
            with mp.ProcessPool(processes=conf.NTHREADS) as pool:
                result = pool.map(partial(runblob, detection=True), np.arange(1, run_n_blobs+1), modblobs)

                output_rows = [res for res in result]

            #rows = pool.map(runblob, zip(detblobs, fblobs))
            # output_rows = pool.map(partial(runblob, detection=True), np.arange(1, run_n_blobs+1), detblobs)
            # while not results.ready():
            #     time.sleep(10)
            #     if not conf.VERBOSE2: print(".", end=' ')
            # pool.close()
            # pool.join()
            # pool.terminate()
            # output_rows = results.get()
        else:
            output_rows = [runblob(kblob_id+1, kblob, detection=True, plotting=conf.PLOT) for kblob_id, kblob in enumerate(modblobs)]

        if conf.VERBOSE: print(f'Completed {run_n_blobs} blobs in {time.time() - tstart:3.3f}s')

        #output_rows = [x for x in output_rows if x is not None]

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

        # write out cat
        # outcatalog['x'] += outcatalog['x'] + mosaic_origin[1] - conf.BRICK_BUFFER + 1.
        # outcatalog['y'] += outcatalog['y'] + mosaic_origin[0] - conf.BRICK_BUFFER + 1.
        hdr = header_from_dict(conf.__dict__, verbose=conf.VERBOSE)
        hdu_info = fits.ImageHDU(header=hdr, name='CONFIG')
        hdu_table = fits.table_to_hdu(outcatalog)
        hdul = fits.HDUList([hdu_table, hdu_info])
        hdul.writeto(os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat'), overwrite=conf.OVERWRITE)
        
        # open again and add

        # If user wants model and/or residual images made:
        cleancatalog = outcatalog[outcatalog['VALID_SOURCE']]
        if conf.MAKE_MODEL_IMAGE:
            modbrick.make_model_image(catalog=cleancatalog)
        if conf.MAKE_RESIDUAL_IMAGE:
            modbrick.make_residual_image(catalog=cleancatalog)

        return
    

def force_models(brick_id, band=None, source_id=None, blob_id=None, insert=True):
    # Create and update multiband brick
    tstart = time.time()

    if (conf.NTHREADS > 0) & ((conf.PLOT == False) | (conf.PLOT2 == False)):
        conf.PLOT = False
        conf.PLOT2 = False
        if conf.VERBOSE: print('WARNING - Plotting not supported while forcing models in parallel!')

    # for fband in band:
    
    if band is None:
        fband = conf.BANDS
    else:
        if (type(band) == list) | (type(band) == np.ndarray):
            fband = band
        elif (type(band) == str) | (type(band) == np.str_):
            fband = [band,]
        else:
            sys.exit('ERROR -- Input band is not a list, array, or string!')
            

    fbrick = stage_brickfiles(brick_id, nickname=conf.MULTIBAND_NICKNAME, band=fband, detection=False)

    if conf.VERBOSE: print(f'{fband} brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    if conf.PLOT2:
        for plt_band in fband:
            if len(fband) == 1:
                idx = 0
            else:
                idx = np.argwhere(np.array(conf.BANDS)==plt_band)[0][0]
            plot_brick(fbrick, idx, band=plt_band)
            plot_background(fbrick, idx, band=plt_band)
            plot_mask(fbrick, idx, band=plt_band)

    if conf.VERBOSE:
        for i, vb_band in enumerate(fband):
            print()
            print(f'Brick #{brick_id} -- Image statistics for {vb_band}')
            shape, minmax, mean, var = stats.describe(fbrick.images[i], axis=None)[:4]
            print(f'    Limits: {minmax[0]:3.3f} - {minmax[1]:3.3f}')
            print(f'    Mean: {mean:3.3f}+/-{np.sqrt(var):3.3f}')
            print()
            print(f'Brick #{brick_id} -- Weight statistics for {vb_band}')
            shape, minmax, mean, var = stats.describe(fbrick.weights[i], axis=None)[:4]
            print(f'    Limits: {minmax[0]:3.3f} - {minmax[1]:3.3f}')
            print(f'    Mean: {mean:3.3f}+/-{np.sqrt(var):3.3f}')
            print()
            print(f'Brick #{brick_id} -- Background statistics for {vb_band}')
            print(f'    Global: {fbrick.backgrounds[i, 0]:3.3f}')
            print(f'    RMS: {fbrick.backgrounds[i, 1]:3.3f}')
            

    if conf.VERBOSE: print(f'Forcing models on {fband}')

    tstart = time.time()
    if (source_id is not None) | (blob_id is not None):
        # conf.PLOT = True
        if source_id is not None:
            blob_id = np.unique(fbrick.blobmap[fbrick.segmap == source_id])
            assert(len(blob_id) == 1)
            blob_id = blob_id[0]
        fblob = fbrick.make_blob(blob_id)
        output_rows = runblob(blob_id, fblob, detection=False, catalog=fbrick.catalog, plotting=conf.PLOT)

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
                        if colname.startswith('FLUX_APER'):
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
                if conf.VERBOSE: print(f'Saving results for brick #{fbrick.brick_id} to existing catalog file.')
        else:
                
            for colname in output_cat.colnames:
                if colname not in fbrick.catalog.colnames:
                    
                    if colname.startswith('FLUX_APER'):
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
            fbrick.catalog.write(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}_{mode_ext}.cat'), format='fits', overwrite=conf.OVERWRITE)
            if conf.VERBOSE: print(f'Saving results for brick #{fbrick.brick_id} to new {mode_ext} catalog file.')


    else:

        if conf.NBLOBS > 0:
            run_n_blobs = conf.NBLOBS
        else:
            run_n_blobs = fbrick.n_blobs

        fblobs = (fbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1))

        if conf.NTHREADS > 0:

            with mp.ProcessPool(processes=conf.NTHREADS) as pool:
                result = pool.map(partial(runblob, detection=False, catalog=fbrick.catalog, plotting=conf.PLOT), np.arange(1, run_n_blobs+1), fblobs)

                output_rows = [res for res in result]

            # pool = mp.ProcessingPool(processes=conf.NTHREADS)
            # #rows = pool.map(partial(runblob, fbrick=fbrick, fbrick=fbrick), np.arange(1, fbrick.n_blobs))
            
            # #rows = pool.map(runblob, zip(fblobs, fblobs))
            # output_rows = pool.map(partial(runblob, detection=False), np.arange(1, run_n_blobs+1), fblobs)
            # while not results.ready():
            #     time.sleep(10)
            #     if not conf.VERBOSE2: print(".", end=' ')
            # pool.close()
            # pool.join()
            # pool.terminate()
            # output_rows = results.get()
        else:
            output_rows = [runblob(kblob_id, fbrick.make_blob(kblob_id), detection=False, catalog=fbrick.catalog, plotting=conf.PLOT) for kblob_id in np.arange(1, run_n_blobs+1)]

        if conf.VERBOSE: print(f'Completed {run_n_blobs} blobs in {time.time() - tstart:3.3f}s')

        #output_rows = [x for x in output_rows if x is not None]

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
                    if colname not in mastercat.colnames:
                        if colname.startswith('FLUX_APER'):
                            mastercat.add_column(Column(length=len(mastercat), dtype=float, shape=(len(conf.APER_PHOT),), name=colname))
                        else:
                            mastercat.add_column(Column(length=len(mastercat), dtype=output_cat[colname].dtype, shape=(1,), name=colname))
                for row in output_cat:
                    mastercat[np.where(mastercat['source_id'] == row['source_id'])[0]] = row
                # coordinate correction
                # fbrick.catalog['x'] = fbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
                # fbrick.catalog['y'] = fbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
                # save
                hdr = header_from_dict(conf.__dict__, verbose=conf.VERBOSE)
                hdu_info = fits.ImageHDU(header=hdr, name='CONFIG')
                hdu_table = fits.table_to_hdu(mastercat)
                hdul = fits.HDUList([hdu_table, hdu_info])
                hdul.writeto(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat'), overwrite=conf.OVERWRITE)
                if conf.VERBOSE: print(f'Saving results for brick #{fbrick.brick_id} to existing catalog file.')

                outcatalog = mastercat

        else:
                
            for colname in output_cat.colnames:
                if colname not in fbrick.catalog.colnames:
                    if colname.startswith('FLUX_APER'):
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
            hdr = header_from_dict(conf.__dict__, verbose=conf.VERBOSE)
            hdu_info = fits.ImageHDU(header=hdr, name='CONFIG')
            hdu_table = fits.table_to_hdu(fbrick.catalog)
            hdul = fits.HDUList([hdu_table, hdu_info])
            hdul.writeto(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}_{mode_ext}.cat'), overwrite=conf.OVERWRITE)
            if conf.VERBOSE: print(f'Saving results for brick #{fbrick.brick_id} to new {mode_ext} catalog file.')

            outcatalog = fbrick.catalog

        # If user wants model and/or residual images made:
        if conf.MAKE_MODEL_IMAGE:
            fbrick.make_model_image(outcatalog)
        if conf.MAKE_RESIDUAL_IMAGE:
            fbrick.make_residual_image()

    return


def stage_brickfiles(brick_id, nickname='MISCBRICK', band=None, detection=False):
    # Wraps Brick with a single parameter call
    # THIS ASSUMES YOU HAVE IMG, WGT, and MSK FOR ALL BANDS!

    path_brickfile = os.path.join(conf.BRICK_DIR, f'B{brick_id}_N{nickname}_W{conf.BRICK_WIDTH}_H{conf.BRICK_HEIGHT}.fits')

    if detection:
        sbands = [nickname,]
    elif band is None:
        sbands = conf.BANDS
    else:
        if type(band) == list:
            sbands = band
        else:
            sbands = [band,]

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
                images[i] = hdul_brick[f"{tband}_IMAGE"].data
                weights[i] = hdul_brick[f"{tband}_WEIGHT"].data
                masks[i] = hdul_brick[f"{tband}_MASK"].data
    else:
        raise ValueError(f'Brick file not found for {path_brickfile}')

    psfmodels = np.zeros((len(sbands)), dtype=object)
    for i, band in enumerate(sbands):
        if band == conf.DETECTION_NICKNAME:
            continue
        path_psffile = os.path.join(conf.PSF_DIR, f'{band}.psf')
        if os.path.exists(path_psffile) & (not conf.FORCE_GAUSSIAN_PSF):
            psfmodels[i] = PixelizedPsfEx(fn=path_psffile)
        else:
            if conf.USE_GAUSSIAN_PSF:
                psfmodels[i] = None
                print(f'PSF model not found for {band} -- using {conf.PSF_SIGMA}" gaussian! ({path_psffile})')
            else:
                raise ValueError(f'PSF model not found for {band}! ({path_psffile})')

    if detection:
        images, weights, masks = images[0], weights[0], masks[0]

    newbrick = Brick(images=images, weights=weights, masks=masks, psfmodels=psfmodels, wcs=wcs, bands=np.array(sbands), brick_id=brick_id)

    if not detection:
        catalog = Table.read(os.path.join(conf.CATALOG_DIR, f'B{brick_id}.cat'), format='fits')
        newbrick.catalog = catalog
        newbrick.n_sources = len(catalog)
        newbrick.n_blobs = catalog['blob_id'].max()
        try:
            newbrick.add_columns(band_only=True)
        except:
            if conf.VERBOSE: print('WARNING - could not add new columns. Overwrting old ones!')
        hdul_seg = fits.open(os.path.join(conf.INTERIM_DIR, f'B{brick_id}_SEGMAPS.fits'))
        newbrick.segmap = hdul_seg['SEGMAP'].data
        newbrick.blobmap = hdul_seg['BLOBMAP'].data 

    return newbrick


def models_from_catalog(catalog, band, rmvector):
        # make multiband catalog from det output
        if conf.VERBOSE2: print()
        if conf.VERBOSE2: print('Adopting sources from existing catalog.')
        model_catalog = -99 * np.ones(len(catalog), dtype=object)
        good_sources = np.ones(len(catalog), dtype=bool)

        

        for i, src in enumerate(catalog):

            position = PixPos(src['X_MODEL'], src['Y_MODEL'])
            flux = Fluxes(**dict(zip(band, src['FLUX_'+conf.MODELING_NICKNAME] * np.ones(len(band)))))

            # Check if valid source
            if not src['VALID_SOURCE']:
                good_sources[i] = False
                if conf.VERBOSE2: print(f'Source #{src["source_id"]}: {src["SOLMODEL"]} model at {position} is INVALID.')
                continue

            #shape = GalaxyShape(src['REFF'], 1./src['AB'], src['theta'])
            if src['SOLMODEL'] not in ('PointSource', 'SimpleGalaxy'):
                shape = EllipseESoft.fromRAbPhi(src['REFF'], 1./src['AB'], -src['THETA'])  # Reff, b/a, phi

            if src['SOLMODEL'] == 'PointSource':
                model_catalog[i] = PointSource(position, flux)
                model_catalog[i].name = 'PointSource' # HACK to get around Dustin's HACK.
            elif src['SOLMODEL'] == 'SimpleGalaxy':
                model_catalog[i] = SimpleGalaxy(position, flux)
            elif src['SOLMODEL'] == 'ExpGalaxy':
                model_catalog[i] = ExpGalaxy(position, flux, shape)
            elif src['SOLMODEL'] == 'DevGalaxy':
                model_catalog[i] = DevGalaxy(position, flux, shape)
            elif src['SOLMODEL'] == 'FixedCompositeGalaxy':
                expshape = EllipseESoft.fromRAbPhi(src['EXP_REFF'], 1./src['EXP_AB'],  -src['EXP_THETA'])
                devshape = EllipseESoft.fromRAbPhi(src['DEV_REFF'], 1./src['DEV_AB'],  -src['DEV_THETA'])
                model_catalog[i] = FixedCompositeGalaxy(
                                                position, flux,
                                                SoftenedFracDev(src['FRACDEV']),
                                                expshape, devshape)

            if conf.VERBOSE2: print(f'Source #{src["source_id"]}: {src["SOLMODEL"]} model at {position}')
            if conf.VERBOSE2: print(f'               {flux}') 
            if src['SOLMODEL'] not in ('PointSource', 'SimpleGalaxy'):
                if src['SOLMODEL'] != 'FixedCompositeGalaxy':
                    if conf.VERBOSE2: print(f'               {shape}')
                else:
                    if conf.VERBOSE2: print(f'               {expshape}')
                    if conf.VERBOSE2: print(f'               {devshape}')


        if (conf.FORCED_PHOT_MAX_NBLOB > 0) & (np.sum(good_sources) > conf.FORCED_PHOT_MAX_NBLOB):
            print(f'interface.force_models :: WARNING -- Number of good sources in blob ({np.sum(good_sources)}) exceeded limit of {conf.FORCED_PHOT_MAX_NBLOB}.')
            good_sources = np.zeros_like(good_sources, dtype=bool)

        return model_catalog[good_sources], good_sources