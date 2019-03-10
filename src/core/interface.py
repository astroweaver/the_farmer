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
from time import time
from functools import partial
sys.path.insert(0, os.path.join(os.getcwd(), 'config'))

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pathos.multiprocessing as mp 
from functools import partial
import matplotlib.pyplot as plt

from .brick import Brick
from .mosaic import Mosaic

import config as conf

def makebricks(multiband_only=False, single_band=None):

    if not multiband_only:
        # Detection
        if conf.VERBOSE: print('Making mosaic for detection')
        detmosaic = Mosaic(conf.DETECTION_NICKNAME, detection=True)
        #detmosaic._make_psf()

        if conf.NTHREADS > 0:
            if conf.VERBOSE: print('Making bricks for detection (in parallel)')
            pool = mp.ProcessingPool(processes=conf.NTHREADS)
            pool.map(partial(detmosaic._make_brick, detection=True, overwrite=True), np.arange(0, detmosaic.n_bricks()))

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
        #bandmosaic._make_psf(forced_psf=True)

        if conf.NTHREADS > 0:
            if conf.VERBOSE: print(f'Making bricks for band {band} (in parallel)')
            pool = mp.ProcessingPool(processes=conf.NTHREADS)
            pool.map(partial(bandmosaic._make_brick, detection=False, overwrite=overwrite), np.arange(0, detmosaic.n_bricks()))

        else:
            if conf.VERBOSE: print(f'Making bricks for band {band} (in serial)')
            for brick_id in np.arange(1, bandmosaic.n_bricks()):
                bandmosaic._make_brick(brick_id, detection=False, overwrite=overwrite)


def tractor(brick_id): # need to add overwrite args!

    # Send out list of bricks to each node to call this function!

    # Create detection brick
    tstart = time()
    detbrick = _stage_brickfiles(brick_id, nickname=conf.DETECTION_NICKNAME, detection=True)

    if conf.VERBOSE: print(f'Detection brick #{brick_id} created ({time() - tstart:3.3f}s)')

    # Sextract sources
    tstart = time()
    try:
        detbrick.sextract(conf.DETECTION_NICKNAME)
        if conf.VERBOSE: print(f'Detection brick #{brick_id} sextracted {detbrick.n_sources} objects ({time() - tstart:3.3f}s)')
    except:
        if conf.VERBOSE: print(f'Detection brick #{brick_id} sextraction FAILED. ({time() - tstart:3.3f}s)')
        return

    # Cleanup
    tstart = time()
    detbrick.cleanup()
    if conf.VERBOSE: print(f'Detection brick #{brick_id} gained {detbrick.n_blobs} blobs with {detbrick.n_sources} objects ({time() - tstart:3.3f}s)')

    # Create and update multiband brick
    tstart = time()
    fbrick = _stage_brickfiles(brick_id, nickname=conf.MULTIBAND_NICKNAME, detection=False)
    fbrick.blobmap = detbrick.blobmap
    fbrick.segmap = detbrick.segmap
    fbrick.catalog = detbrick.catalog
    fbrick.add_columns()
    fbrick.catalog

    if conf.VERBOSE: print(f'Multiband brick #{brick_id} created ({time() - tstart:3.3f}s)')

    if conf.NTHREADS > 0:
        pool = mp.ProcessingPool(processes=conf.NTHREADS)
        rows = pool.map(partial(_runblob, detbrick=detbrick, fbrick=fbrick), np.arange(1, detbrick.n_blobs))
    else:
        [_runblob(blob_id, detbrick, fbrick) for blob_id in np.arange(1, detbrick.n_blobs)]


def _runblob(blob_id, detbrick, fbrick):

    #####################3
    print()
    print(f'Starting on Blob #{blob_id}')
    tstart = time()

    # Make blob with detection image
    fig, ax = plt.subplots(nrows = 1, ncols = 3,
                       sharex=True, sharey=True,
                       figsize=(15, 5))

    band = detbrick.bands
    i=0
    back = detbrick.backgrounds[i]
    mean, rms = back.globalback, back.globalrms
    
    img_opt = dict(cmap='magma', vmin = mean + 5 * rms, vmax = mean + 10 * rms)
    #wgt_opt = dict(cmap='magma', vmin = mean - rms, vmax = mean + rms)
    
    ax[0].imshow(detbrick.images[i], **img_opt)
    ax[1].imshow(detbrick.weights[i])
    ax[2].imshow(detbrick.masks[i])
    ##########################
    
    myblob = detbrick.make_blob(blob_id)
    if myblob is None:
        print('BLOB REJECTED!')
        return

    ##############3
    fig, ax = plt.subplots(nrows = 1, ncols = 3,
                       sharex=True, sharey=True,
                       figsize=(15, 5))

    band = myblob.bands
    i=0
    back = myblob.backgrounds[i]
    mean, rms = back.globalback, back.globalrms
    
    img_opt = dict(cmap='magma', vmin = mean + 5 * rms, vmax = mean + 10 * rms)
    #wgt_opt = dict(cmap='magma', vmin = mean - rms, vmax = mean + rms)
    
    ax[0].imshow(myblob.images[i], **img_opt)
    ax[1].imshow(myblob.weights[i])
    ax[2].imshow(myblob.masks[i])
    ###################

    # Run models
    myblob.stage_images()

        ##############3
    fig, ax = plt.subplots(nrows = 1, ncols = 3,
                       sharex=True, sharey=True,
                       figsize=(15, 5))

    band = myblob.bands
    i=0
    back = myblob.backgrounds[i]
    mean, rms = back.globalback, back.globalrms
    
    img_opt = dict(cmap='magma', vmin = mean + 5 * rms, vmax = mean + 10 * rms)
    #wgt_opt = dict(cmap='magma', vmin = mean - rms, vmax = mean + rms)
    
    tweight = myblob.weights[i].copy()
    tweight[myblob.masks[i]] = 0
    ax[0].imshow(myblob.images[i], **img_opt)
    ax[1].imshow(tweight)
    ax[2].imshow(myblob.masks[i])
    ###################
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

    # Run follow-up phot
    [[myfblob.aperture_phot(band, img_type) for band in myfblob.bands] for img_type in ('image', 'model', 'residual')]
    [myfblob.sextract_phot(band) for band in myfblob.bands]

    duration = time() - tstart
    print(f'Solution for {myblob.n_sources} sources arrived at in {duration}s ({duration/myblob.n_sources:2.2f}s per src)')


def _stage_brickfiles(brick_id, nickname='MISCBRICK', detection=False):
    # Wraps Brick with a single parameter call
    # THIS ASSUMES YOU HAVE IMG, WGT, and MSK FOR ALL BANDS!

    if nickname == 'MULTIBAND':
        conf.IMAGE_EXT = 'img'
        conf.WEIGHT_EXT = 'wgt'
    else:
        conf.IMAGE_EXT = 'sci'
        conf.WEIGHT_EXT = 'weight'

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

            hdul_brick.info()

            # Attempt a WCS
            wcs = WCS(hdul_brick[0].header)

            # Stuff data into arrays
            for i, tband in enumerate(sbands):
                images[i] = hdul_brick[f"{tband}_{conf.IMAGE_EXT.upper()}"].data[0] # QUICK FIX. The extra dimension SHOULD NOT EXIST!
                weights[i] = hdul_brick[f"{tband}_{conf.WEIGHT_EXT.upper()}"].data[0]
                masks[i] = hdul_brick[f"{tband}_{conf.MASK_EXT.upper()}"].data[0]
    else:
        raise ValueError(f'Brick file not found for {path_brickfile}')

    psfmodels = np.zeros((len(sbands), 101, 101))
    for i, band in enumerate(sbands):
        path_psffile = os.path.join(conf.PSF_DIR, f'snap_{band}.fits')
        if os.path.exists(path_psffile):
            with fits.open(path_psffile) as hdul:
                psfmodels[i] = hdul[0].data

    print('shape of images: ', np.shape(images))
    print(wcs)
    return Brick(images=images, weights=weights, masks=masks, psfmodels=psfmodels, wcs=wcs, bands=np.array(sbands))
