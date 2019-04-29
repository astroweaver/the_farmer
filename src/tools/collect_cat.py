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
from astropy.io import ascii

sys.path.insert(0, os.path.join(os.getcwd(), 'config'))
import config as conf

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
out_dir = conf.CATALOG_DIR
out_dir = '/Volumes/WD4/Current/tractor_pipeline/data/catalogs/candide/grizy'
cat_prefix = 'B'
cat_suffix = 'cat'

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
for fname in walk_through_files(out_dir, cat_prefix, cat_suffix):
    print('addding {}'.format(fname))
    cat = Table.read(fname)

    # # FIX x AND Y
    # brick_id = int(fname.split('/')[-1][:-4])
    # H, W = (2000, 2000)
    # dim_img = (50000, 50000)
    # x0 = (((brick_id - 1) * W) % dim_img[0])
    # y0 = int(((brick_id - 1) * H) / dim_img[1]) * H

    # cat['x'] = cat['x'] + x0 - 74.5
    # cat['y'] = cat['y']


    nsources += len(cat)
    try:
        tab = vstack((tab, cat))
        print('stack successful!')
    except:
        tab = cat

outfname = 'master_catalog.fits'
print('Writing {} sources to {}'.format(nsources, outfname))
tab.write(os.path.join(out_dir, outfname), format='fits', overwrite=True)
