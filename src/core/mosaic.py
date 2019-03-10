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

from .subimage import Subimage
import config as conf


class Mosaic(Subimage):
    
    def __init__(self, band, detection=False, psfmodel=None, wcs=None, header=None, mag_zeropoint=None,
                ):

        if detection:
            self.path_image = os.path.join(conf.IMAGE_DIR, conf.DETECTION_FILENAME.replace('EXT', conf.IMAGE_EXT))
            self.path_weight = os.path.join(conf.IMAGE_DIR, conf.DETECTION_FILENAME.replace('EXT', conf.WEIGHT_EXT))
            self.path_mask = os.path.join(conf.IMAGE_DIR, conf.DETECTION_FILENAME.replace('EXT', conf.MASK_EXT))
        else:
            self.path_image = os.path.join(conf.IMAGE_DIR, conf.MULTIBAND_FILENAME.replace('EXT', conf.IMAGE_EXT).replace('BAND', band))
            self.path_weight = os.path.join(conf.IMAGE_DIR, conf.MULTIBAND_FILENAME.replace('EXT', conf.WEIGHT_EXT).replace('BAND', band))
            self.path_mask = os.path.join(conf.IMAGE_DIR, conf.MULTIBAND_FILENAME.replace('EXT', conf.MASK_EXT).replace('BAND', band))

        # open the files
        tstart = time()
        if os.path.exists(self.path_image):
            with fits.open(self.path_image, memmap=True) as hdu_image:
                self.images = hdu_image['PRIMARY'].data
                self.master_head = hdu_image['PRIMARY'].header
                self.wcs = WCS(self.master_head)
        else:
            raise ValueError(f'No image found at {self.path_image}')
        if conf.VERBOSE: print(f'Added image in {time()-tstart:3.3f}s.')

        tstart = time()
        if os.path.exists(self.path_weight):
            with fits.open(self.path_weight) as hdu_weight:
                self.weights = hdu_weight['PRIMARY'].data
        else:
            raise ValueError(f'No weight found at {self.path_image}')
            self.weights = None
        if conf.VERBOSE: print(f'Added weight in {time()-tstart:3.3f}s.')


        tstart = time()
        if os.path.exists(self.path_mask):
            with fits.open(self.path_mask) as hdu_mask:
                self.masks = hdu_mask['PRIMARY'].data
        else:
            self.masks = None    
        if conf.VERBOSE: print(f'Added mask in {time()-tstart:3.3f}s.')

        self.psfmodels = psfmodel
        self.bands = band
        self.mag_zeropoints = mag_zeropoint
        
        super().__init__()

    def _make_psf(self, forced_psf=False):

        # Set filenames
        psf_dir = conf.PSF_DIR
        psf_cat = os.path.join(conf.PSF_DIR, f'{self.bands}_clean.ldac')
        path_savexml = conf.PSF_DIR
        path_savechkimg = ','.join([os.path.join(conf.PSF_DIR, ext) for ext in ('chi', 'proto', 'samp', 'resi', 'snap')])
        path_savechkplt = ','.join([os.path.join(conf.PSF_DIR, ext) for ext in ('fwhm', 'ellipticity', 'counts', 'countfrac', 'chi2', 'resi')])
        path_segmap = os.path.join(conf.PSF_DIR, f'{self.bands}_segmap.fits')

        if forced_psf:
            self.path_image = os.path.join(conf.IMAGE_DIR, conf.DETECTION_FILENAME.replace('EXT', conf.IMAGE_EXT)) + f',{self.path_image}'

        # run SEXTRACTOR in LDAC mode (either forced or not)(can this be done with sep?!)
        if not os.path.exists(psf_cat):
            try:
                #os.system('sextractor {} -c config/config_psfex.sex -PARAMETERS_NAME config/param_psfex.sex -CATALOG_NAME {} -CATALOG_TYPE FITS_LDAC -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE {} -MAG_ZEROPOINT {}'.format(path_im, path_outcat, path_wt, zpt))
                os.system(f'sex {self.path_image} -c config/config_psfex.sex -PARAMETERS_NAME config/param_psfex.sex -CATALOG_NAME {psf_cat} -CATALOG_TYPE FITS_LDAC -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME {path_segmap} -MAG_ZEROPOINT {self.mag_zeropoints}')
                #print('RUNNING SEXTRACTOR WITHOUT SEGMAP')
                #os.system(f'sex {self.path_image} -c config/config_psfex.sex -PARAMETERS_NAME config/param_psfex.sex -CATALOG_NAME {psf_cat} -CATALOG_TYPE FITS_LDAC -MAG_ZEROPOINT {self.mag_zeropoints}')
                sys.exit()
                if conf.VERBOSE: print('SExtractor succeded!')
            except:
                raise ValueError('SExtractof failed!')
        # if not forced, then the cleaned segmap should be saved as the weight for the dual mode!

        # LDAC is only cleaned if not forced mode! (otherwise should be clean already)
        if not forced_psf:

            hdul_ldac = fits.open(psf_cat, ignore_missing_end=True, mode='update')
            tab_ldac = hdul_ldac['LDAC_OBJECTS'].data

            mask_ldac = (tab_ldac['MAG_AUTO'] > conf.DET_VAL_LIMITS[0]) &\
                    (tab_ldac['MAG_AUTO'] < conf.DET_VAL_LIMITS[1]) &\
                    (tab_ldac['FLUX_RADIUS'] > conf.DET_REFF_LIMITS[0]) &\
                    (tab_ldac['FLUX_RADIUS'] < conf.DET_REFF_LIMITS[1])

            idx_exclude = np.arange(1, len(tab_ldac) + 1)[~mask_ldac]

            hdul_ldac['LDAC_OBJECTS'].data = tab_ldac[mask_ldac]
            hdul_ldac.flush()

            # clean segmap
            segmap = fits.open(path_segmap)[0].data
            for idx in idx_exclude:
                segmap[segmap == idx['id']] = 0
            # this is just to remove non-PS from the other LDACs

            segmap.writeto('detection_segmap_clean.fits')
            hdul_ldac.close()

        else:
            segmap = fits.open('detection_segmap_clean.fits')[0].data
            idx_obj = np.unique(segmap)[1:] - 1 # rm zero, shift to index-zero
            segmap[segmap > 0] = 1

            hdul_ldac = fits.open(psf_cat, ignore_missing_end=True)
            tab_ldac = hdul_ldac['LDAC_OBJECTS'].data[idx_obj]

            tab_ldac.writeto(psf_cat, overwrite=True)
            tab_ldac.close()

        # RUN PSF
        os.system(f'psfex {psf_cat} -c config/config.psfex -BASIS_TYPE PIXEL -PSF_SIZE 101,101 -PSF_DIR {psf_dir} -WRITE_XML Y -XML_NAME {path_savexml} -CHECKIMAGE_NAME {path_savechkimg} -CHECKPLOT_NAME {path_savechkplt}')
        
    
    def _make_brick(self, brick_id, overwrite=False, detection=False, 
            brick_width=conf.BRICK_WIDTH, brick_height=conf.BRICK_HEIGHT, brick_buffer=conf.BRICK_BUFFER):

        if conf.VERBOSE: print(f'Making brick {brick_id}/{self.n_bricks()}')

        if detection:
            nickname = conf.DETECTION_NICKNAME
        else:
            nickname = conf.MULTIBAND_NICKNAME

        save_fitsname = f'B{brick_id}_N{nickname}_W{brick_width}_H{brick_height}.fits'
        path_fitsname = os.path.join(conf.BRICK_DIR, save_fitsname)

        if (not overwrite) & (not os.path.exists(path_fitsname)):
            raise ValueError(f'No existing file found for {path_fitsname}. Will not write new one.')

        x0, y0 = self._get_origin(brick_id, brick_width, brick_height)
        subinfo = self._get_subimage(x0, y0, brick_width, brick_height, brick_buffer)
        subimage, subweight, submask, psfmodel, band, subwcs, subvector, slicepix, subslice = subinfo

        if detection:
            sbands = nickname
        else:
            sbands = self.bands

        # Remove n_bands = 1 dimension
        subimage, subweight, submask = subimage[0], subweight[0], submask[0]
        
        # Make hdus
        head_image = self.master_head.copy()
        head_image.update(subwcs.to_header())
        hdu_image = fits.ImageHDU(subimage, head_image, f'{sbands}_{conf.IMAGE_EXT}')
        hdu_weight = fits.ImageHDU(subweight, head_image, f'{sbands}_{conf.WEIGHT_EXT}')
        hdu_mask = fits.ImageHDU(submask.astype(int), head_image, f'{sbands}_{conf.MASK_EXT}')
        
        # if overwrite, make it
        if overwrite:
            hdu_prim = fits.PrimaryHDU()
            hdul_new = fits.HDUList([hdu_prim, hdu_image, hdu_weight, hdu_mask])
            hdul_new.writeto(path_fitsname)
        else:
        # otherwise add to it
            exist_hdul = fits.open(path_fitsname, mode='append')
            exist_hdul.append(hdu_image)
            exist_hdul.append(hdu_weight)
            exist_hdul.append(hdu_mask)
            exist_hdul.flush()
            exist_hdul.close()

    def _get_origin(self, brick_id, brick_width=conf.BRICK_WIDTH, brick_height=conf.BRICK_HEIGHT):
        x0 = int(((brick_id - 1) * brick_width) % self.dims[0])
        y0 = int(((brick_id - 1) * brick_height) / self.dims[1]) * brick_height
        return np.array([x0, y0])


    def n_bricks(self, brick_width=conf.BRICK_WIDTH, brick_height=conf.BRICK_HEIGHT):
        n_xbricks = self.dims[0] / brick_width
        n_ybricks = self.dims[1] / brick_height
        return int(n_xbricks * n_ybricks)



