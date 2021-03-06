# imports
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import wcs
from regions import read_ds9

# fake the mask structure
mask_dir = '/n07data/weaver/COSMOS2020/data/images/masks/'
masks = {
    'brightstars': [('hsc_S18mask_griz.reg',), (1,)],
    'ultravista': [('border_uv_2.reg'), (1, )],
    'deep': [('ultravista', 'deepstripes-hjmcc.reg'), (1, 0)],
    'ultradeep': [('deepstripes-hjmcc.reg',), (1, )],
    'suprimecam': [('COSMOS.Peterr2.dd.reg',), (1, )]
}
filename = 'COSMOS_14_01_20_withflags.fits'

table = Table.read(filename)
ra, dec = table['RA'], table['DEC']
coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

# Create a new WCS object.  The number of axes must be set
# from the start
w = wcs.WCS(naxis=2)

w.wcs.crpix = [0, 0]
w.wcs.cdelt = np.array([-1, 1])
w.wcs.crval = [0, 0]
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]


# make a quick function to handle the column making
def mask_select(regfile, coords, opt=1):
    print('    - reading file...')
    regions = read_ds9(regfile)
    print('    - read file OK')

    reg_total = regions[0]
    # make composite
    if len(regions) > 1:
        for reg in regions:
            reg_total &= reg
    print('    - composite made OK')

    # find sources
    if opt == 1: # is in
        col = reg_total.contains(coords, w)
    elif opt == 0: # is outside of
        col = reg_total.contains(coords, w)
    print('    - column made OK')

    return col

# open up the region files and loop and add columns
# keys are the names
# then we have the filenames as a list with their operations (+, -, |, &, ~)
# then we have their good or bad values

for name in masks.keys():
    fnames, opts = masks[name]

    print(fnames[0], opts[0])
    col = mask_select(os.path.join(mask_dir, fnames[0]), coords, opt=opts[0])
    if len(fnames) > 1:
        for fname, opt in zip(fnames[1:], opts[1:]):
            print(fname, opt)
            col &= mask_select(os.path.join(mask_dir, fname), coords, opt=opt)

    print(f'{np.sum(col)} sources selected in {name} ({100*np.sum(col) / len(col)}%)')

    table.add_column(Column(col, name='flag_'+name))


# save it
table.write(filename.split('.')[0]+'_withflags.fits')
        





