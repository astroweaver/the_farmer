BAND = 'HSC-I'

# HSC at 3" radius
# HSC-R 0.8660520150634863
# HSC-I 0.9036244689222772
# HSC-Z 0.9223879278478824

# IRAC at 10" radius
# CH1 0.9371
# CH2 0.9249

import astropy.units as u
import os, glob

from farmer.utils import prepare_psf

path = f'/n25data1/dawn_cats/farmer_test/data/interim/psfmodels/{BAND}'
out = path.replace(BAND, f'{BAND}_proc/')

if not os.path.exists(out):
    os.mkdir(out)

for fn in glob.glob(path+'/*'):

    name = fn.split('/')[-1]

    # HSC
    if 'HSC' in BAND:
        if BAND == 'HSC-R':
            NORM = 0.8660520150634863
        if BAND == "HSC-I":
            NORM = 0.9036244689222772
        if BAND == "HSC-Z":
            NORM = 0.9223879278478824      

        prepare_psf(fn, out+name, 
                    pixel_scale = 0.17 * u.arcsec, 
                    mask_radius = 18 * 0.17 * u.arcsec,
                    clip_radius = 18 * 0.17 * u.arcsec,
                    norm = NORM
                    )
    # IRAC OVERSAMP --> NATIVE
    elif 'IRAC' in BAND:
        if 'ch1' in BAND:
            NORM = 0.9371
        elif 'ch2' in BAND:
            NORM = 0.9249
        name = fn.split('/')[-1].replace('oversamp', 'native')
        psf_proc = prepare_psf(fn, out+name, 
                    pixel_scale =  0.012 * u.arcsec,
                    target_pixel_scale = 0.6 * u.arcsec, 
                    clip_radius = 10 * u.arcsec,
                    norm = 0.9371 # ch1 - 0.9371 # ch2 - 0.9249 at 10" radius
        )

        # IRAC OVERSAMP --> HSC scale
        name = fn.split('/')[-1].replace('oversamp', 'resamp')
        psf_proc = prepare_psf(fn, out+name, 
            pixel_scale =  0.012 * u.arcsec,
            target_pixel_scale = 0.17 * u.arcsec, 
            clip_radius = 10 * u.arcsec,
            norm = 0.9371 # ch1 - 0.9371 # ch2 - 0.9249 at 10" radius
            )

    print(name)

