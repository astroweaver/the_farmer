# Combine all aux images
import os
import sys
sys.path.insert(0, os.path.join('/n07data/weaver/COSMOS2020/config'))
import config as conf
import numpy as np
from astropy.io import fits




n_bricks = int(conf.MOSAIC_WIDTH / conf.BRICK_WIDTH * conf.MOSAIC_HEIGHT / conf.BRICK_HEIGHT)
dir_aux = '.'

def combine(band, img_type):
    print(f'Starting aux combine on {band} {img_type}')
    img_total = np.zeros((conf.MOSAIC_WIDTH, conf.MOSAIC_HEIGHT))
    N = 0
    for i in np.arange(n_bricks):
        fn = f'B{i+1}_AUXILLARY_MAPS.fits'
        brick_id = i + 1
        path = os.path.join(dir_aux, fn)
        if not os.path.exists(path):
            continue
        with fits.open(path) as hdul:
            print(f'{i} -- {fn}')
            N+=1
            try:
                extension = f'{band}_{img_type}'
                extension = extension.upper()
                img = hdul[extension].data
            except:
                print(f'Could not find extension: {extension}')
            
            img_crop = img[conf.BRICK_BUFFER:-conf.BRICK_BUFFER,conf.BRICK_BUFFER:-conf.BRICK_BUFFER]
            
            x0 = int(((brick_id - 1) * conf.BRICK_WIDTH) % conf.MOSAIC_WIDTH)
            y0 = int(((brick_id - 1) * conf.BRICK_HEIGHT) / conf.MOSAIC_HEIGHT) * conf.BRICK_HEIGHT
            img_total[x0:x0+conf.BRICK_WIDTH, y0:y0+conf.BRICK_HEIGHT] = img_crop

    if N > 0:
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data = img_total)])
        hdul.writeto(f'AUX_{band}_{img_type}.fits', overwrite=conf.OVERWRITE)
        print(f'Wrote out {N} bricks to AUX_{band}_{img_type}.fits')

for band in conf.BANDS:
    for img_type in ('MODEL', 'RESIDUAL'):
        combine(band, img_type)