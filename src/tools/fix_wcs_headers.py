import numpy as np
from astropy.io import fits
import os

# take in directory to look through
WORKING_DIR = '.'
TYPE = 'DETECTION' #'MULTIBAND'
SAVE = True
BRICKS = (0, 600) #(0, 600)
BRICK_WIDTH = 2004
BRICK_HEIGHT = 2004
DIMS = (48096, 48096)

def get_origin(brick_id):
    x0 = int(((brick_id - 1) * BRICK_WIDTH) % DIMS[0])
    y0 = int(((brick_id - 1) * BRICK_HEIGHT) / DIMS[1]) * BRICK_HEIGHT
    return (x0, y0)

# look through directory
for bid in np.arange(BRICKS[0], BRICKS[1]):

    fn = f'B{bid}_N{TYPE}_W2004_H2004.fits'
    path = os.path.join(WORKING_DIR, fn)
    if os.path.exists(path):
        # open file
        with fits.open(path, mode='update') as hdul:
            print(f'Opened brick {bid}')
            # calculate mosaic_origin
            mosaic_origin = get_origin(bid)
            print(mosaic_origin)

            # find extensions with _IMAGE
            for i in np.arange(len(hdul)):
                for ext in ('IMAGE', 'WEIGHT', 'MASK'):
                    if ext in hdul[i].name:
                        print(f'*** Correcing CRPIX in {hdul[i].name}')
                        # perform fix
                        # CRPIX1, CRPIX2 += brick - brick[::-1]
                        
                        corr1 = mosaic_origin[0] - mosaic_origin[1]
                        corr2 = mosaic_origin[1] - mosaic_origin[0]
                        print(f"{hdul[i].header['CRPIX1']} + {corr1:5.5f} --> {hdul[i].header['CRPIX1'] + corr1:5.5f}")
                        print(f"{hdul[i].header['CRPIX2']} + {corr2:5.5f} --> {hdul[i].header['CRPIX2'] + corr2:5.5f}")
                        hdul[i].header['CRPIX1'] += corr1
                        hdul[i].header['CRPIX2'] += corr2
                    else:
                        continue

            # Save it! Move on!
            if SAVE:
                print('*** SAVING.')
                hdul.flush()
            print()

    else:
        print(f'{fn} does not exist here!')
        print()
