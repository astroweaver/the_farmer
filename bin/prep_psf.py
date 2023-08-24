BAND = 'HSC-I'

import astropy.units as u
import os, glob

from farmer.utils import prepare_psf

path = f'/Volumes/2TB_Weaver/Projects/Current/farmer_test/EDFN_farmer2_test/PSFs/{BAND}'
out = path.replace(BAND, f'{BAND}_proc')

if not os.path.exists(out):
    os.mkdir(out)

for fn in glob.glob(path+'/*'):

    name = fn.split('/')[-1]
    prepare_psf(fn, out+'/'+name, 
                pixel_scale=0.17 * u.arcsec, 
                mask_radius = 18 * 0.17 * u.arcsec,
                clip_radius = 20 * 0.17 * u.arcsec,
                norm = None
                )
    print(name)

