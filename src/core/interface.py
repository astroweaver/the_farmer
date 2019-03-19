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

from astropy.io import fits
from astropy.table import Table, Column, vstack, join
from astropy.wcs import WCS
import numpy as np
import pathos.multiprocessing as mp 
from functools import partial
import matplotlib.pyplot as plt

from .brick import Brick
from .mosaic import Mosaic
from .utils import plot_blob

import config as conf

def makebricks(multiband_only=False, single_band=None, insert=False, skip_psf=False):

    if not multiband_only:
        # Detection
        if conf.VERBOSE: print('Making mosaic for detection')
        detmosaic = Mosaic(conf.DETECTION_NICKNAME, detection=True)
        if not skip_psf: 
            detmosaic._make_psf()

        if conf.NTHREADS > 0:
            pass
            # BUGGY DUE TO MEM ALLOC
            # if conf.VERBOSE: print('Making bricks for detection (in parallel)')
            # pool = mp.ProcessingPool(processes=conf.NTHREADS)
            # pool.map(partial(detmosaic._make_brick, detection=True, overwrite=True), np.arange(0, detmosaic.n_bricks()))

        else:
            if conf.VERBOSE: print('Making bricks for detection (in serial)')
            for brick_id in np.arange(1, detmosaic.n_bricks()+1):
                detmosaic._make_brick(brick_id, detection=True, overwrite=True)


    # Bands
    if single_band is not None:
        sbands = [single_band,]
    else:
        sbands = conf.BANDS

    for i, band in enumerate(sbands):

        overwrite = True
        if insert:
            overwrite=False
        if i > 0:
            overwrite = False

        if conf.VERBOSE: print(f'Making mosaic for band {band}')
        bandmosaic = Mosaic(band)
        if not skip_psf: 
            bandmosaic._make_psf()

        if conf.NTHREADS > 0:
            if conf.VERBOSE: print(f'Making bricks for band {band} (in parallel)')
            pool = mp.ProcessingPool(processes=conf.NTHREADS)
            pool.map(partial(bandmosaic._make_brick, detection=False, overwrite=overwrite), np.arange(0, detmosaic.n_bricks()))

        else:
            if conf.VERBOSE: print(f'Making bricks for band {band} (in serial)')
            for brick_id in np.arange(1, bandmosaic.n_bricks()+1):
                bandmosaic._make_brick(brick_id, detection=False, overwrite=overwrite)

    return


def tractor(brick_id, source_id=None, blob_id=None): # need to add overwrite args!

    # Send out list of bricks to each node to call this function!

    # Create detection brick
    tstart = time.time()
    detbrick = stage_brickfiles(brick_id, nickname=conf.DETECTION_NICKNAME, detection=True)

    if conf.VERBOSE: print(f'Detection brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    # Sextract sources
    tstart = time.time()
    try:
        detbrick.sextract(conf.DETECTION_NICKNAME, sub_background=True)
        if conf.VERBOSE: print(f'Detection brick #{brick_id} sextracted {detbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')
    except:
        if conf.VERBOSE: print(f'Detection brick #{brick_id} sextraction FAILED. ({time.time() - tstart:3.3f}s)')
        return

    # Cleanup
    tstart = time.time()
    detbrick.cleanup()
    if conf.VERBOSE: print(f'Detection brick #{brick_id} gained {detbrick.n_blobs} blobs with {detbrick.n_sources} objects ({time.time() - tstart:3.3f}s)')

    # Create and update multiband brick
    tstart = time.time()
    fbrick = stage_brickfiles(brick_id, nickname=conf.MULTIBAND_NICKNAME, detection=False)
    fbrick.blobmap = detbrick.blobmap
    fbrick.segmap = detbrick.segmap
    fbrick.catalog = detbrick.catalog.copy()
    fbrick.add_columns()
    if conf.VERBOSE: print(f'Multiband brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    tstart = time.time()
    if (source_id is not None) | (blob_id is not None):
        conf.PLOT = True
        if source_id is not None:
            blob_id = np.unique(fbrick.blobmap[fbrick.segmap == source_id])
            assert(len(blob_id) == 1)
            blob_id = blob_id[0]
        detblob, fblob = detbrick.make_blob(blob_id), fbrick.make_blob(blob_id)
        runblob(blob_id, detblob, fblob, plotting=conf.PLOT)

    else:

        if conf.NBLOBS > 0:
            run_n_blobs = conf.NBLOBS
        else:
            run_n_blobs = detbrick.n_blobs
        
        itstart = time.time()
        detblobs = [detbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1)]
        fblobs = [fbrick.make_blob(i) for i in np.arange(1, run_n_blobs+1)]
        if conf.VERBOSE: print(f'Blobs created in {time.time() - itstart:3.3f}s')

        if conf.NTHREADS > 0:
            pool = mp.ProcessingPool(processes=conf.NTHREADS)
            #rows = pool.map(partial(runblob, detbrick=detbrick, fbrick=fbrick), np.arange(1, detbrick.n_blobs))
            
            #rows = pool.map(runblob, zip(detblobs, fblobs))
            
            output_rows = pool.map(runblob, np.arange(1, run_n_blobs+1), detblobs, fblobs)
            # while not results.ready():
            #     time.sleep(10)
            #     if not conf.VERBOSE2: print(".", end=' ')
            # pool.close()
            # pool.join()
            # pool.terminate()
            # output_rows = results.get()
        else:
            output_rows = [runblob(kblob_id, detblobs[kblob_id-1], fblobs[kblob_id-1], plotting=conf.PLOT) for kblob_id in np.arange(1, run_n_blobs+1)]
    
        if conf.VERBOSE: print(f'Completed {run_n_blobs} blobs in {time.time() - tstart:3.3f}s')

        output_rows = [x for x in output_rows if x is not None]

        output_cat = vstack(output_rows)
                
        for colname in output_cat.colnames:
           if colname not in fbrick.catalog.colnames:
               fbrick.catalog.add_column(Column(np.zeros(len(output_cat[colname]), dtype=output_cat[colname].dtype), name=colname))
        #fbrick.catalog = join(fbrick.catalog, output_cat, join_type='left', )
        for row in output_cat:
            fbrick.catalog[np.where(fbrick.catalog['sid'] == row['sid'])[0]] = row

        # write out cat
        fbrick.catalog['x'] = detbrick.catalog['x'] + fbrick.mosaic_origin[1] - conf.BRICK_BUFFER + 1.
        fbrick.catalog['y'] = detbrick.catalog['y'] + fbrick.mosaic_origin[0] - conf.BRICK_BUFFER + 1.
        fbrick.catalog.write(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat'), format='fits')

        return

def runblob(blob_id, detblob, fblob, plotting=False):

    if conf.VERBOSE: print()
    if conf.VERBOSE: print(f'Starting on Blob #{blob_id}')
    tstart = time.time()

    # Make blob with detection image 

    if detblob is None:
        if conf.VERBOSE2: print('BLOB REJECTED!')
        return None

    # Run models

    astart = time.time()
    detblob.stage_images()
    if conf.VERBOSE2: print(f'Images staged. ({time.time() - astart:3.3f})s')

    astart = time.time()
    status = detblob.tractor_phot()
    if conf.VERBOSE2: print(f'Morphology determined. ({time.time() - astart:3.3f})s')

    if not status:
        return None


    # make new blob with band information
    astart = time.time() 

    fblob.model_catalog = detblob.solution_catalog.copy()
    fblob.position_variance = detblob.position_variance.copy()
    fblob.parameter_variance = detblob.parameter_variance.copy()
    if conf.VERBOSE2: print(f'Solution parameters transferred. ({time.time() - astart:3.3f})s')

    # Forced phot
    astart = time.time() 
    fblob.stage_images()
    if conf.VERBOSE2: print(f'Multiband images staged. ({time.time() - astart:3.3f})s')

    astart = time.time() 
    status = fblob.forced_phot()
    if conf.VERBOSE2: print(f'Force photometry complete. ({time.time() - astart:3.3f})s')

    if not status:
        return None


    if plotting:
        plot_blob(detblob, fblob)

    # Run follow-up phot
    if conf.DO_APPHOT:
        try:
            [[fblob.aperture_phot(band, img_type) for band in fblob.bands] for img_type in ('image', 'model', 'residual')]
        except:
            if conf.VERBOSE2: print(f'Aperture photmetry FAILED. Likely a bad blob.')
    if conf.DO_SEXPHOT:
        try:
            [fblob.sextract_phot(band) for band in fblob.bands]
        except:
            if conf.VERBOSE2: print(f'Residual Sextractor photmetry FAILED. Likely a bad blob.)')

    duration = time.time() - tstart
    if conf.VERBOSE: print(f'Solution for blob {detblob.blob_id} (N={detblob.n_sources}) arrived at in {duration:3.3f}s ({duration/detblob.n_sources:2.2f}s per src)')
    
    catout = fblob.bcatalog.copy()
    del detblob, fblob

    return catout



def stage_brickfiles(brick_id, nickname='MISCBRICK', detection=False):
    # Wraps Brick with a single parameter call
    # THIS ASSUMES YOU HAVE IMG, WGT, and MSK FOR ALL BANDS!

    path_brickfile = os.path.join(conf.BRICK_DIR, f'B{brick_id}_N{nickname}_W{conf.BRICK_WIDTH}_H{conf.BRICK_HEIGHT}.fits')

    if detection:
        sbands = [nickname,]
    else:
        sbands = conf.BANDS

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
                images[i] = hdul_brick[f"{tband}_{conf.IMAGE_EXT.upper()}"].data
                weights[i] = hdul_brick[f"{tband}_{conf.WEIGHT_EXT.upper()}"].data
                masks[i] = hdul_brick[f"{tband}_{conf.MASK_EXT.upper()}"].data
    else:
        raise ValueError(f'Brick file not found for {path_brickfile}')

    psfmodels = np.zeros((len(sbands)))
    for i, band in enumerate(sbands):
        path_psffile = os.path.join(conf.PSF_DIR, f'snap_{band}.fits')
        if os.path.exists(path_psffile):
            with fits.open(path_psffile) as hdul:
                psfmodel = hdul[0].data
                if i == 0:
                    psfmodels = np.zeros((len(sbands), psfmodel.shape[0], psfmodel.shape[1]))
                psfmodels[i] = psfmodel
        else:
            psfmodels[i] = -99

    if detection:
        images, weights, masks = images[0], weights[0], masks[0]

    return Brick(images=images, weights=weights, masks=masks, psfmodels=psfmodels, wcs=wcs, bands=np.array(sbands), brick_id=brick_id)
