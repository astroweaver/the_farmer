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

# ------------------------------------------------------------------------------
# Standard Packages
# ------------------------------------------------------------------------------
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt

from core.brick import Brick, Subimage
from time import time


def tractor(): # need to add overwrite args!

    # Create detection brick
    tstart = time()
    mysub = Brick(images=detimg, weights=detwgt, psfmodels=psfmodel, bands=np.array(['hsc i',]), wcs=mywcs)
    print(time() - tstart)


    tstart = time()
    mysub.sextract('hsc i')

    print(time() - tstart)

    tstart = time()
    mysub.cleanup()
    print(time() - tstart)


    mybrick = Brick(images=rawimgs, weights=rawwgts, psfmodels=psfmodels, bands=np.array(['hsc i', 'hsc z',]), wcs=mywcs)
    mybrick.blobmap = mysub.blobmap
    mybrick.segmap = mysub.segmap
    mybrick.catalog = mysub.catalog
    mybrick.add_columns()
    mybrick.catalog

  print()
    print(f'Starting on Blob #{blob_id}')
    tstart = time()
    
    # Make blob with detection image
    myblob = mysub.make_blob(blob_id)
    if myblob is None:
        print('BLOB REJECTED!')
        return
    myblob.stage_images()
#     print('Starting Model')
    plt.imshow(myblob.images[0])
    status = myblob.tractor_phot()

    if not status:
        return
    
    # make new blob with band information
    myfblob = mybrick.make_blob(blob_id)
    myfblob.model_catalog = myblob.solution_catalog
    myfblob.position_variance = myblob.position_variance
    myfblob.parameter_variance = myblob.parameter_variance

    myfblob.stage_images()
    
#     print('Starting Forced Phot')
    status = myfblob.forced_phot()
    
    # Run follow-up phot
    [[myfblob.aperture_phot(band, img_type) for band in myfblob.bands] for img_type in ('image', 'model', 'residual')]
    [myfblob.sextract_phot(band) for band in myfblob.bands]

    duration = time() - tstart
    print(f'Solution for {myblob.n_sources} sources arrived at in {duration}s ({duration/myblob.n_sources:2.2f}s per src)')


def stage_brickfiles(brick_id, nickname='MISCBRICK'):
    # Wraps Brick with a single parameter call
    # THIS ASSUMES YOU HAVE IMG, WGT, and MSK FOR ALL BANDS!

    path_brickfile = os.path.join(BRICK_DIR, f'B{brick_id}_N{nickname}_W{BRICK_WIDTH}_H{BRICK_HEIGHT}.fits')

    if os.path.exists(path_brickfile):
        # Stage things
        images = np.zeros((len(BANDS), BRICK_WIDTH + 2 * BRICK_BUFFER, BRICK_HEIGHT + 2 * BRICK_BUFFER))
        weights = np.zeros_like(images)
        masks = np.zeros_like(images, dtype=bool)
        
        # Loop over expected bands
        with fits.open(path_brickfile) as hdul_brick:

            # Attempt a WCS
            wcs = WCS(hdul_brick[0].header)

            # Stuff data into arrays
            for i, tband in enumerate(BANDS):
                images[i] = hdul_brick[f"{BAND}_{IMAGE_EXT}"].data
                weights[i] = hdul_brick[f"{BAND}_{WEIGHT_EXT}"].data
                masks[i] = hdul_brick[f"{BAND}_{MASK_EXT}"].data

    return dict(images=images, weights=weights, masks=masks, psfmodels=None, wcs=wcs, bands=BANDS)
