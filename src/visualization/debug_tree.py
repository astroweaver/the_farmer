# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Debug the decision tree

Known Issues
------------
None


"""

# ------------------------------------------------------------------------------
# Standard Packages
# ------------------------------------------------------------------------------
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u
import pathos.multiprocessing as mp

import sys
from time import time

from matplotlib.patches import Rectangle

# ------------------------------------------------------------------------------
# Additional Packages
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
params = {
'bands': ['hsc_i', 'hsc_z'],
'figure_dir': '/Volumes/WD4/Current/tractor_pipeline/figures',
'save_fits': True,
'verbose': True,
'Nproc': 0
}



# ------------------------------------------------------------------------------
# Declarations and Functions
# ------------------------------------------------------------------------------
verbose = params['verbose']

brick_id = sys.argv[1]
source_id = sys.argv[2]
morph_filter = 'HSC_I'

# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------



# find file
fname = 'B{}_fitsrc_{}.fits'.format(brick_id, source_id)
fpath = os.path.join(params['figure_dir'], fname)

# Read in file
try:
    hdul = fits.open(fpath)
    hdul.info()
except:
    print('{} could not be opened.'.format(fname))
    sys.exit()

back = np.median(hdul[morph_filter].data)
noisesigma = 0.8*np.std(hdul[morph_filter].data)
noise = np.random.normal(back, noisesigma, size=np.shape(hdul[morph_filter].data))


fig, ax = plt.subplots(ncols=4, nrows = 6, figsize=(8, 12))

ima = dict(interpolation=None, origin='lower', cmap='Greys',
    vmin=back, vmax=3*noisesigma)

imchi = dict(interpolation=None, origin='lower', cmap='RdGy',
      vmin=-5, vmax=5)

# Input
ax[0,0].imshow(hdul[morph_filter].data, **ima)
ax[0,1].imshow(hdul[morph_filter+'_IVAR'].data, **ima)


# Round 1
r1_text = 'Round 1: PointSource ($\chi{{2}}$={}) vs. SimpleGalaxy ($\chi{{2}}$={})'.format('N/A', 'N/A')
ax[1,0].text(0, 1.05, r1_text, transform=ax[1,0].transAxes )
ax[1,0].imshow(hdul[morph_filter+'_MODS0_1'].data + noise, **ima)
ax[1,1].imshow(hdul[morph_filter+'_CHIS0_1'].data, **imchi)

ax[1,2].imshow(hdul[morph_filter+'_MODS_1'].data + noise, **ima)
ax[1,3].imshow(hdul[morph_filter+'_CHIS_1'].data, **imchi)


ax[2,0].imshow(hdul[morph_filter+'_MODS0_2'].data + noise, **ima)
ax[2,1].imshow(hdul[morph_filter+'_CHIS0_2'].data, **imchi)

ax[2,2].imshow(hdul[morph_filter+'_MODS_2'].data + noise, **ima)
ax[2,3].imshow(hdul[morph_filter+'_CHIS_2'].data, **imchi)

r1_text = 'Round 2: ExpGalaxy ($\chi{{2}}$={}) vs. DevGalaxy ($\chi{{2}}$={})'.format('N/A', 'N/A')
ax[3,0].text(0, 1.05, r1_text, transform=ax[3,0].transAxes )
try:
    ax[3,0].imshow(hdul[morph_filter+'_MODS0_3'].data + noise, **ima)
    ax[3,1].imshow(hdul[morph_filter+'_CHIS0_3'].data, **imchi)

    ax[3,2].imshow(hdul[morph_filter+'_MODS_3'].data + noise, **ima)
    ax[3,3].imshow(hdul[morph_filter+'_CHIS_3'].data, **imchi)


    ax[4,0].imshow(hdul[morph_filter+'_MODS0_4'].data + noise, **ima)
    ax[4,1].imshow(hdul[morph_filter+'_CHIS0_4'].data, **imchi)

    ax[4,2].imshow(hdul[morph_filter+'_MODS_4'].data + noise, **ima)
    ax[4,3].imshow(hdul[morph_filter+'_CHIS_4'].data, **imchi)

except:
    ax[3,0].text(0.3, 0.5, 'N/A', transform=ax[3,0].transAxes)
    ax[3,1].text(0.3, 0.5, 'N/A', transform=ax[3,1].transAxes)

    ax[3,2].text(0.3, 0.5, 'N/A', transform=ax[3,2].transAxes)
    ax[3,3].text(0.3, 0.5, 'N/A', transform=ax[3,3].transAxes)


    ax[4,0].text(0.3, 0.5, 'N/A', transform=ax[4,0].transAxes)
    ax[4,1].text(0.3, 0.5, 'N/A', transform=ax[4,1].transAxes)

    ax[4,2].text(0.3, 0.5, 'N/A', transform=ax[4,2].transAxes)
    ax[4,3].text(0.3, 0.5, 'N/A', transform=ax[4,3].transAxes)

r1_text = 'Round 3: CompositeGalaxy ($\chi{{2}}$={})'.format('N/A')
ax[5,0].text(0, 1.05, r1_text, transform=ax[5,0].transAxes )
try:
    ax[5,0].imshow(hdul[morph_filter+'_MODS0_5'].data + noise, **ima)
    ax[5,1].imshow(hdul[morph_filter+'_CHIS0_5'].data, **imchi)

    ax[5,2].imshow(hdul[morph_filter+'_MODS_5'].data + noise, **ima)
    ax[5,3].imshow(hdul[morph_filter+'_CHIS_5'].data, **imchi)

except:
    ax[5,0].text(0.3, 0.5, 'N/A', transform=ax[5,0].transAxes)
    ax[5,1].text(0.3, 0.5, 'N/A', transform=ax[5,1].transAxes)

    ax[5,2].text(0.3, 0.5, 'N/A', transform=ax[5,2].transAxes)
    ax[5,3].text(0.3, 0.5, 'N/A', transform=ax[5,3].transAxes)

[[ax[m, n].axis('off') for n in np.arange(4)] for m in np.arange(6)]

fig.tight_layout()
fig.savefig(os.path.join(params['figure_dir'], 'B{}_tree_{}.png'.format(brick_id, source_id)))
