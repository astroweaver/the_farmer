# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Class to handle mosaics (PSF + bricking)

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
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table, Column
from scipy.ndimage import label, binary_dilation, binary_fill_holes
from time import time
from astropy.wcs import WCS

from .utils import create_circular_mask
from .subimage import Subimage
from .blob import Blob
from .config import *


class Mosaic(Subimage):
    
    def __init__(self, band, detection=False, psfmodel=None, wcs=None, header=None, mag_zeropoint=None,
                ):

        if detection:
            self.path_image = os.path.join(IMAGE_DIR, DETECTION_FILENAME.replace('EXT', IMAGE_EXT))
            self.path_weight = os.path.join(IMAGE_DIR, DETECTION_FILENAME.replace('EXT', WEIGHT_EXT))
            self.path_mask = os.path.join(IMAGE_DIR, DETECTION_FILENAME.replace('EXT', MASK_EXT))
        else:
            self.path_image = os.path.join(IMAGE_DIR, FORCED_FILENAME.replace('EXT', IMAGE_EXT).replace('BAND', band))
            self.path_weight = os.path.join(IMAGE_DIR, FORCED_FILENAME.replace('EXT', WEIGHT_EXT).replace('BAND', band))
            self.path_mask = os.path.join(IMAGE_DIR, FORCED_FILENAME.replace('EXT', MASK_EXT).replace('BAND', band))

        # open the files
        if self.path_image.exists():
            hdu_image = fits.open(self.path_image)
            self.image = hdu_image[f'{band}_{IMAGE_EXT}'].data
            self.master_head = hdu_image[f'{band}_{IMAGE_EXT}'].header
            self.wcs = WCS(self.master_header)
        else:
            self.image = None
            self.master_head = None
            self.wcs = None

        if self.path_weight.exists():
            hdu_weight = fits.open(self.path_weight)
        else:
            self.weight = None

        if self.path_mask.exists():
            hdu_mask = fits.open(self.path_mask)    
        else:
            self.mask = None    

        self.psfmodel = psfmodel
        self.band = band
        self.mag_zeropoint = mag_zeropoint
        
        super().__init__()
    
    # def about(self):
    #     print(f'*** Mosaic')
    #     print(f' Shape: {self.shape}')
    #     print(f' Nbands: {self.n_bands}')
    #     print(f' Origin: {self.subvector}')
    #     print()

    def _make_psf(self, forced_psf=False):

        # Set filenames
        psf_dir = PSF_DIR
        psf_cat = os.path.join(PSF_DIR, f'{self.band}_clean.ldac')
        path_savexml = PSF_DIR
        path_savechkimg = ','.join([os.path.join(dir_psf, ext) for ext in ('chi', 'proto', 'samp', 'resi', 'snap')])
        path_savechkplt = ','.join([os.path.join(dir_psf, ext) for ext in ('fwhm', 'ellipticity', 'counts', 'countfrac', 'chi2', 'resi')])
        path_segmap = os.path.join(IMAGE_DIR, f'{self.band}_segmap')

        if forced_psf:
            self.path_image = os.path.join(IMAGE_DIR, DETECTION_FILENAME.replace('EXT', IMAGE_EXT)) + f',{self.path_image}'

        # run SEXTRACTOR in LDAC mode (either forced or not)(can this be done with sep?!)
        os.system(f'sextractor {self.path_image} -c config/config_psfex.sex -PARAMETERS_NAME config/param_psfex.sex -CATALOG_NAME {psf_cat} -CATALOG_TYPE FITS_LDAC -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME {path_segmap} -MAG_ZEROPOINT {self.mag_zeropoint}')

        # if not forced, then the cleaned segmap should be saved as the weight for the dual mode!

        # LDAC is only cleaned if not forced mode! (otherwise should be clean already)
        if not forced_psf:

            hdul_ldac = fits.open(psf_cat)
            tab_ldac = hdul_ldac['LDAC_OBJECTS'].data

            mask_ldac = (tab_ldac['MAG_AUTO'] > mag_lims[0]) &\
                    (tab_ldac['MAG_AUTO'] < mag_lims[1]) &\
                    (tab_ldac['FLUX_RADIUS'] > reff_lims[0]) &\
                    (tab_ldac['FLUX_RADIUS'] < reff_lims[1])

            idx_exclude = np.arange(1, len(tab_ldac) + 1)[~mask_ldac]

            hdul_ldac['LDAC_OBJECTS'].data = tab_ldac[mask_ldac]
            hdul_ldac.writeto(psf_cat, overwrite=True)

            # clean segmap
            segmap = fits.open(path_segmap)[1].data
            for idx in idx_exclude:
                segmap[segmap == idx['id']] = 0
            # this is just to remove non-PS from the other LDACs

            segmap.writeto('detection_segmap_clean.fits')

        else:
            segmap = fits.open('detection_segmap_clean.fits')[1].data
            idx_obj = np.unique(segmap)[1:] - 1 # rm zero, shift to index-zero
            segmap[segmap > 0] = 1

            hdul_ldac = fits.open(psf_cat)
            tab_ldac = hdul_ldac['LDAC_OBJECTS'].data[idx_obj]

            tab_ldac.writeto(psf_cat, overwrite=True)

        # RUN PSF
        os.system(f'psfex {psf_cat} -c config/config.psfex -BASIS_TYPE PIXEL -PSF_SIZE 101,101 -PSF_DIR {psf_dir} -WRITE_XML Y -XML_NAME {path_savexml} -CHECKIMAGE_NAME {path_savechkimg} -CHECKPLOT_NAME {path_savechkplt}')
        
    
    def make_brick(self, brick_id, overwrite=False, nickname='MISCBRICK', 
            brick_width=BRICK_WIDTH, brick_height=BRICK_HEIGHT, brick_buffer=BRICK_BUFFER):
        save_fitsname = f'B{brick_id}_N{nickname}_W{brick_width}_H{self.brick_height}.fits'

        x0, y0 = self._get_origin(brick_id)
        subinfo = self._get_subimage(self, x0, y0, brick_width, brick_height, brick_buffer)
        subimage, subweight, submask, psfmodel, band, subwcs, subvector, slicepix, subslice = subinfo
        
        # Make hdus
        head_image = self.master_head.copy()
        head_image.update(subwcs.to_header())
        hdu_image = fits.ImageHDU(subimage, head_image, f'{self.band}_SCI')
        hdu_weight = fits.ImageHDU(subweight, head_image, f'{self.band}_WEIGHT')
        hdu_mask = fits.ImageHDU(submask, head_image, f'{self.band}_MASK')
        
        # if overwrite, make it
        if overwrite:
            hdu_prim = fits.PrimaryHDU()
            hdul_new = fits.HDUList([hdu_prim, hdu_image, hdu_weight, hdu_mask])
            hdul_new.writeto(save_fitsname, mode='append')
        else:
        # otherwise add to it
            exist_hdul = fits.open(save_fitsname, mode='append')
            exist_hdul.append(hdu_image)
            exist_hdul.append(hdu_weight)
            exist_hdul.append(hdu_mask)
            exist_hdul.flush()
            exist_hdul.close()

    def _get_origin(self, brick_id):
        x0 = (((brick_id - 1) * brick_width) % self.dims[0])
        y0 = int(((brick_id - 1) * brick_height) / self.dims[1]) * brick_height
        return np.array([x0, y0])


        



