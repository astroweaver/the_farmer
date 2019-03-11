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

def makebricks(multiband_only=False, single_band=None, skip_psf=False):

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
            for brick_id in np.arange(1, detmosaic.n_bricks()):
                detmosaic._make_brick(brick_id, detection=True, overwrite=True)


    # Bands
    if single_band is not None:
        sbands = [single_band,]
    else:
        sbands = conf.BANDS

    for i, band in enumerate(sbands):

        overwrite = True
        if i > 0:
            overwrite = False

        if conf.VERBOSE: print(f'Making mosaic for band {band}')
        bandmosaic = Mosaic(band)
        if not skip_psf: 
            bandmosaic._make_psf(forced_psf=True)

        if conf.NTHREADS > 0:
            if conf.VERBOSE: print(f'Making bricks for band {band} (in parallel)')
            pool = mp.ProcessingPool(processes=conf.NTHREADS)
            pool.map(partial(bandmosaic._make_brick, detection=False, overwrite=overwrite), np.arange(0, detmosaic.n_bricks()))

        else:
            if conf.VERBOSE: print(f'Making bricks for band {band} (in serial)')
            for brick_id in np.arange(1, bandmosaic.n_bricks()):
                bandmosaic._make_brick(brick_id, detection=False, overwrite=overwrite)

    return


def tractor(brick_id, source_id=None): # need to add overwrite args!

    # Send out list of bricks to each node to call this function!

    # Create detection brick
    tstart = time.time()
    detbrick = stage_brickfiles(brick_id, nickname=conf.DETECTION_NICKNAME, detection=True)

    if conf.VERBOSE: print(f'Detection brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    # Sextract sources
    tstart = time.time()
    try:
        detbrick.sextract(conf.DETECTION_NICKNAME)
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
    fbrick.catalog = detbrick.catalog
    fbrick.add_columns()
    fbrick.catalog
    if conf.VERBOSE: print(f'Multiband brick #{brick_id} created ({time.time() - tstart:3.3f}s)')

    tstart = time.time()
    if source_id is not None:
        blob_id = np.unique(fbrick.blobmap[fbrick.segmap == source_id])
        assert(len(blob_id) == 1, 'More than one blob inhabits that segment!')
        runblob(blob_id[0], detbrick, fbrick, plotting=conf.PLOT)

    else:
        if conf.NTHREADS > 0:
            pool = mp.ProcessingPool(processes=conf.NTHREADS)
            #rows = pool.map(partial(runblob, detbrick=detbrick, fbrick=fbrick), np.arange(1, detbrick.n_blobs))
            # detblobs = [detbrick.make_blob(i) for i in np.arange(1, detbrick.n_blobs+1)]
            #fblobs = [fbrick.make_blob(i) for i in np.arange(1, detbrick.n_blobs+1)]
            #rows = pool.map(runblob, zip(detblobs, fblobs))
            
            output_rows = pool.map(partial(runblob, detbrick=detbrick, fbrick=fbrick), np.arange(1, detbrick.n_blobs))
            # while not results.ready():
            #     time.sleep(10)
            #     if not conf.VERBOSE2: print(".", end=' ')
            pool.close()
            # output_rows = results.get()
        else:
            output_rows = [runblob(blob_id, detbrick, fbrick, plotting=conf.PLOT) for blob_id in np.arange(1, detbrick.n_blobs+1)]
    
        if conf.VERBOSE: print(f'Completed {detbrick.n_blobs} blobs in {time.time() - tstart:3.3f}s')

    output_cat = vstack(output_rows)
    #for colname in output_cat.colnames:
    #    if colname not in fbrick.catalog.colnames:
    #        fbrick.catalog.add_column(Column(np.zeros_like(output_cat[colname], dtype=output_cat[colname].dtype), name=colname))
    #fbrick.catalog = join(fbrick.catalog, output_cat, join_type='left', )
    for row in output_cat:
         fbrick.catalog[np.where(fbrick.catalog['sid'] == row['sid'])[0]] = row

    # write out cat
    fbrick.catalog.write(os.path.join(conf.CATALOG_DIR, f'B{fbrick.brick_id}.cat'), format='fits')

    return

def runblob(blob_id, detbrick, fbrick, plotting=False):

    if conf.VERBOSE2: print()
    if conf.VERBOSE2: print(f'Starting on Blob #{blob_id}')
    tstart = time.time()

    # Make blob with detection image    
    myblob = detbrick.make_blob(blob_id)

    if myblob is None:
        if conf.VERBOSE2: print('BLOB REJECTED!')
        return

    # Run models
    myblob.stage_images()

    status = myblob.tractor_phot()

    if not status:
        return

    # make new blob with band information
    myfblob = fbrick.make_blob(blob_id)

    myfblob.model_catalog = myblob.solution_catalog
    myfblob.position_variance = myblob.position_variance
    myfblob.parameter_variance = myblob.parameter_variance

    # Forced phot
    myfblob.stage_images()
    status = myfblob.forced_phot()

    # if plotting:
    #     plot_blob(myblob, myfblob)

    # Run follow-up phot
    if conf.DO_APPHOT:
        try:
            [[myfblob.aperture_phot(band, img_type) for band in myfblob.bands] for img_type in ('image', 'model', 'residual')]
        except:
            if conf.VERBOSE2: print(f'Aperture photmetry FAILED. Likely a bad blob.')
    if conf.DO_SEXPHOT:
        try:
            [myfblob.sextract_phot(band) for band in myfblob.bands]
        except:
            if conf.VERBOSE2: print(f'Residual Sextractor photmetry FAILED. Likely a bad blob.)')

    duration = time.time() - tstart
    if conf.VERBOSE2: print(f'Solution for {myblob.n_sources} sources arrived at in {duration}s ({duration/myblob.n_sources:2.2f}s per src)')
    
    return myfblob.catalog



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
            wcs = WCS(hdul_brick[0].header)

            # Stuff data into arrays
            for i, tband in enumerate(sbands):
                images[i] = hdul_brick[f"{tband}_{conf.IMAGE_EXT.upper()}"].data
                weights[i] = hdul_brick[f"{tband}_{conf.WEIGHT_EXT.upper()}"].data
                masks[i] = hdul_brick[f"{tband}_{conf.MASK_EXT.upper()}"].data
    else:
        raise ValueError(f'Brick file not found for {path_brickfile}')

    psfmodels = np.zeros((len(sbands), 101, 101))
    for i, band in enumerate(sbands):
        path_psffile = os.path.join(conf.PSF_DIR, f'snap_{band}.fits')
        if os.path.exists(path_psffile):
            with fits.open(path_psffile) as hdul:
                psfmodels[i] = hdul[0].data
        else:
            psfmodels[i] = -99

    if detection:
        images, weights, masks = images[0], weights[0], masks[0]

    return Brick(images=images, weights=weights, masks=masks, psfmodels=psfmodels, wcs=wcs, bands=np.array(sbands), brick_id=brick_id)
