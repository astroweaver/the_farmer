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
from time import time
from functools import partial

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pathos.multiprocessing as mp

from .brick import Brick

#TODO: import config


def tractor(brick_id): # need to add overwrite args!

    # Send out list of bricks to each node to call this function!

    # Create detection brick
    tstart = time()
    kwargs = stage_brickfiles(brick_id, detection=True)
    detbrick = Brick(**kwargs)

    if VERBOSE: print(f'Detection brick #{brick_id} created ({time() - tstart:3.3f}s)')

    # Sextract sources
    tstart = time()
    detbrick.sextract(DETECTION_NICKNAME)
    if VERBOSE: print(f'Detection brick #{brick_id} sextracted {detbrick.n_sources} objects ({time() - tstart:3.3f}s)')

    # Cleanup
    tstart = time()
    detbrick.cleanup()
    if VERBOSE: print(f'Detection brick #{brick_id} cleaned with {detbrick.n_sources} remaining ({time() - tstart:3.3f}s)')

    # Create and update multiband brick
    tstart = time()
    kwargs = stage_brickfiles(brick_id, detection=False)
    fbrick = Brick(kwargs)
    fbrick.blobmap = detbrick.blobmap
    fbrick.segmap = detbrick.segmap
    fbrick.catalog = detbrick.catalog
    fbrick.add_columns()
    fbrick.catalog

    if VERBOSE: print(f'Multiband brick #{brick_id} created ({time() - tstart:3.3f}s)')

    if NTHREADS > 0:
        pool = mp.ProcessingPool(processes=NTHREADS)
        rows = pool.map(partial(runblob, detbrick=detbrick, fbrick=fbrick), np.arange(1, detbrick.n_blobs))
    else:
        [runblob(blob_id, detbrick, fbrick) for blob_id in np.arange(1, detbrick.n_blobs)]


def runblob(blob_id, detbrick, fbrick):
    print()
    print(f'Starting on Blob #{blob_id}')
    tstart = time()

    # Make blob with detection image
    myblob = detbrick.make_blob(blob_id)
    if myblob is None:
        print('BLOB REJECTED!')
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

    # Run follow-up phot
    [[myfblob.aperture_phot(band, img_type) for band in myfblob.bands] for img_type in ('image', 'model', 'residual')]
    [myfblob.sextract_phot(band) for band in myfblob.bands]

    duration = time() - tstart
    print(f'Solution for {myblob.n_sources} sources arrived at in {duration}s ({duration/myblob.n_sources:2.2f}s per src)')


def stage_brickfiles(brick_id, nickname='MISCBRICK', detection=False):
    # Wraps Brick with a single parameter call
    # THIS ASSUMES YOU HAVE IMG, WGT, and MSK FOR ALL BANDS!

    path_brickfile = os.path.join(BRICK_DIR, f'B{brick_id}_N{nickname}_W{BRICK_WIDTH}_H{BRICK_HEIGHT}.fits')

    if detection:
        sbands = [nickname,]
    else:
        sbands = BANDS

    if os.path.exists(path_brickfile):
        # Stage things
        images = np.zeros((len(sbands), BRICK_WIDTH + 2 * BRICK_BUFFER, BRICK_HEIGHT + 2 * BRICK_BUFFER))
        weights = np.zeros_like(images)
        masks = np.zeros_like(images, dtype=bool)

        # Loop over expected bands
        with fits.open(path_brickfile) as hdul_brick:

            # Attempt a WCS
            wcs = WCS(hdul_brick[0].header)

            # Stuff data into arrays
            for i, tband in enumerate(sbands):
                images[i] = hdul_brick[f"{tband}_{IMAGE_EXT}"].data
                weights[i] = hdul_brick[f"{tband}_{WEIGHT_EXT}"].data
                masks[i] = hdul_brick[f"{tband}_{MASK_EXT}"].data

    return dict(images=images, weights=weights, masks=masks, psfmodels=None, wcs=wcs, bands=np.array(sbands))
