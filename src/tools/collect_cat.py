# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Master catalog

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
from astropy.table import Table, vstack
from astropy.io import ascii, fits


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
out_dir = sys.argv[1]

cat_prefix = 'B'
cat_suffix = 'cat'
overwrite = True

def walk_through_files(path, file_prefix = 'B', file_extension='.fits'):
    """
    Generate a list of filenames at given path
    """
    output = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.startswith(file_prefix) & filename.endswith(file_extension):
                output.append(os.path.join(dirpath, filename))
    return output


nsources = 0
skip_count = 0
total_count = 0
first_stack = True
for i, fname in enumerate(walk_through_files(out_dir, cat_prefix, cat_suffix)):
    print('addding {}'.format(fname))
    try:
        cat = Table.read(fname, format='fits')
        total_count += 1
    except:
        print('COULD NOT READ FILE.')
        continue
    
    if i == 0:
        hdu_info = fits.open(fname)['CONFIG']

    if (cat['SOLMODEL']=='').all():
        print('BAD SOL MODEL. SKIPPING.')
        skip_count += 1
        continue

    # for band in conf.BANDS:

    #     cat[f'MAG_{band}'] = cat[f'MAG_{band}'][:, 0]
    #     cat[f'MAGERR_{band}'] = cat[f'MAGERR_{band}'][:, 0]
    #     cat[f'FLUX_{band}'] = cat[f'FLUX_{band}'][:, 0]
    #     cat[f'FLUXERR_{band}'] = cat[f'FLUXERR_{band}'][:, 0]
    #     cat[f'CHISQ_{band}'] = cat[f'CHISQ_{band}'][:, 0]

        # print(cat[f'MAG_{band}'])

        # for i in np.arange(len(cat)):
        #     cat[f'MAG_{band}'][i] = cat[f'MAG_{band}'][i][0]
        #     cat[f'MAGERR_{band}'][i] = cat[f'MAGERR_{band}'][i][0]
        #     cat[f'FLUX_{band}'][i] = cat[f'FLUX_{band}'][i][0]
        #     cat[f'FLUXERR_{band}'][i] = cat[f'FLUXERR_{band}'][i][0]
        #     cat[f'CHISQ_{band}'][i] = cat[f'CHISQ_{band}'][i][0]

        # cat[f'MAG_{band}'].dtype = float
        # cat[f'MAGERR_{band}'].dtype = float
        # cat[f'FLUX_{band}'].dtype = float
        # cat[f'FLUXERR_{band}'].dtype = float
        # cat[f'CHISQ_{band}'].dtype = float

    # # FIX x AND Y
    # brick_id = int(fname.split('/')[-1][:-4])
    # H, W = (2000, 2000)
    # dim_img = (50000, 50000)
    # x0 = (((brick_id - 1) * W) % dim_img[0])
    # y0 = int(((brick_id - 1) * H) / dim_img[1]) * H

    # cat['x'] = cat['x'] + x0 - 74.5
    # cat['y'] = cat['y']


    nsources += len(cat)

    if first_stack:
        tab = cat
        print('Stack started.')
        first_stack = False
    else:
        try:
            tab = vstack((tab, cat))
            print('stack successful!')
            continue
        except:
            print('Failed to stack!')
            continue
        
    
    

outfname = 'master_catalog.fits'
print('Writing {} sources to {}'.format(nsources, outfname))
print(f'Skipped {skip_count}/{total_count} tiles.')

hdu_table = fits.table_to_hdu(tab)
hdul = fits.HDUList([hdu_table, hdu_info])
hdul.writeto(os.path.join(out_dir, outfname), overwrite=overwrite)
