
import os
import sys
import numpy as np

from astropy.table import Table
from astropy.io import fits

fname_tab = sys.argv[1]
N_try = int(sys.argv[2])

# open catalog
tab = Table.read(fname_tab)

# get brick ids
brick_id = tab['brick_id']

masked_px_deep = 0
masked_px_udeep = 0

if N_try > len(np.unique(brick_id)):
        N_try = len(np.unique(brick_id))

print(f'Will try {N_try} bricks!')

count = 0
for i, bid in enumerate(np.unique(brick_id)):
        print()
        print(bid)

        if (~tab['VALID_SOURCE'][tab['brick_id']==bid]).all():
                print('ALL INVALID!')
                continue

        count += 1
        if count > N_try:
            break

        # open segmap
        with fits.open(f'B{bid}_SEGMAPS.fits') as hdul:
                blobmap = hdul['BLOBMAP'].data

                # deep regions
                brick_deep = brick_id[tab['FLAG_deepstripes'] & (tab['brick_id'] == bid)]
                print(f'Number of sources in deep region: {len(brick_deep)}')

                inval_brick_deep = brick_id[tab['VALID_SOURCE'] & tab['FLAG_deepstripes'] & (tab['brick_id'] == bid)]$

                area_bad = np.sum(blobmap[np.isin(blobmap, inval_brick_deep)])
                pc_bad  = area_bad / (2004*2004)  # estimation!

                print(f'Area of invalid sources: {area_bad} ({pc_bad*100}%)')

                masked_px_deep += area_bad


                # udeep regions
                brick_udeep = brick_id[tab['FLAG_shallowstripes'] & (tab['brick_id'] == bid)]
                print(f'Number of sources in udeep region: {len(brick_udeep)}')

                inval_brick_udeep = brick_id[tab['VALID_SOURCE'] & tab['FLAG_shallowstripes'] & (tab['brick_id'] == b$

                area_bad = np.sum(blobmap[np.isin(blobmap, inval_brick_udeep)])
                pc_bad  = area_bad / (2004*2004)  # estimation!

                print(f'Area of invalid sources: {area_bad} ({pc_bad*100}%)')

                masked_px_udeep += area_bad


print(f'Final bad area over {i} bricks in deep: {masked_px_deep} px')
print(f'Final bad area over {i} bricks in udeep: {masked_px_udeep} px')
