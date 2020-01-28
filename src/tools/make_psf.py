# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Quick script to make PSFs using SExtractor + PSFEx

Known Issues
------------
Extensive testing not yet performed. Be wary of bugs!


"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
import subprocess

from matplotlib.patches import Rectangle


####################

PSF_DIR = '.'
PLOT = True
OVERRIDE = True
BAND = 'hsc_i' # unique nickname of band
MAG_ZEROPOINTS = [23.93,]
PATH_IMAGE = '/Volumes/LaCie12/Projects_offsite/Current/splash_xsdf/data/external/mosaic_hsc/mosaic_hsc_i.img.fits'
XLIMS = (2.5, 3.2) # Reff
YLIMS = (21.0, 22.5) # Mag

#####################

def plot_ldac(tab_ldac, band, xlims=None, ylims=None, box=False):
    fig, ax = plt.subplots()
    ax.scatter(tab_ldac['FLUX_RADIUS'], tab_ldac['MAG_AUTO'], c='k', s=0.5)
    if box:
        rect = Rectangle((xlims[0], ylims[0]), xlims[1] - xlims[0], ylims[1] - ylims[0], fill=True, alpha=0.3,
                                edgecolor='r', facecolor='r', zorder=3, linewidth=1)
        ax.add_patch(rect)
    fig.subplots_adjust(bottom = 0.15)
    ax.set(xlabel='Flux Radius (px)', xlim=(1, 10),
            ylabel='Mag Auto (AB)', ylim=(26, 16))
    fig.savefig(os.path.join(conf.PLOT_DIR, f'{band}_ldac.pdf'), overwrite=True)

#####################


psf_dir = PSF_DIR
psf_cat = os.path.join(PSF_DIR, f'{BAND}_clean.ldac')
path_savexml = PSF_DIR
path_savechkimg = ','.join([os.path.join(PSF_DIR, ext) for ext in ('chi', 'proto', 'samp', 'resi', 'snap')])
path_savechkplt = ','.join([os.path.join(PSF_DIR, ext) for ext in ('fwhm', 'ellipticity', 'counts', 'countfrac', 'chi2', 'resi')])

# run SEXTRACTOR in LDAC mode
if (not os.path.exists(psf_cat)) | OVERRIDE:
    try:
        subprocess.Popen(f'sex {PATH_IMAGE} -c config/config_psfex.sex -PARAMETERS_NAME config/param_psfex.sex -CATALOG_NAME {psf_cat} -CATALOG_TYPE FITS_LDAC -MAG_ZEROPOINT {MAG_ZEROPOINTS}', shell=True).wait()
        if VERBOSE & os.path.exists(psf_cat): print('SExtractor succeded!')
        else: raise RuntimeError('SExtractor failed!')
    except:
        raise RuntimeError('SExtractor failed!')

    hdul_ldac = fits.open(psf_cat, ignore_missing_end=True, mode='update')
    tab_ldac = hdul_ldac['LDAC_OBJECTS'].data

    mask_ldac = (tab_ldac['MAG_AUTO'] > YLIMS[0]) &\
            (tab_ldac['MAG_AUTO'] < YLIMS[1]) &\
            (tab_ldac['FLUX_RADIUS'] > XLIMS[0]) &\
            (tab_ldac['FLUX_RADIUS'] < XLIMS[1])

    if PLOT:
        if VERBOSE: print('Plotting LDAC for pointsource bounding box')
        plot_ldac(tab_ldac, BAND, xlims=XLIMS, ylims=YLIMS, box=True)

    idx_exclude = np.arange(1, len(tab_ldac) + 1)[~mask_ldac]

    hdul_ldac['LDAC_OBJECTS'].data = tab_ldac[mask_ldac]
    hdul_ldac.flush()

    # RUN PSF
    subprocess.call(f'psfex {psf_cat} -c config/config.psfex -BASIS_TYPE PIXEL -PSF_SIZE 101,101 -PSF_DIR {psf_dir} -WRITE_XML Y -XML_NAME {path_savexml} -CHECKIMAGE_NAME {path_savechkimg} -CHECKPLOT_NAME {path_savechkplt}')
    