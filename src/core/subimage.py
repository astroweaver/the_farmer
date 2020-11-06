# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Basic class for images. Mainly used for creating further subimages.

Known Issues
------------
None


"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table, Column
from astropy.wcs import WCS
import sep
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clip

import config as conf

import logging
class Subimage():
    """
    Parent superclass for all images in Farmer. Does useful calculations:

    * WCS
    * Sextraction
    * Weights/Masks
    * Data validation
    * PSF models
    * Image chopping
    """

    def __init__(self): 

        self.logger = logging.getLogger('farmer.subimage')

        self.subvector = None
        self.catalog = None
        self.n_sources = None
        # If I make these self._X, the setters break.

    @property
    def wcs(self):
        return self._wcs

    @wcs.setter
    def wcs(self, value):
        """Sets the World-Coordinate System attribute

        Parameters
        ----------

        wcs : WCS Object
            Astropy.WCS object

        Raises
        ------

        TypeError
            Input WCS is not an Astropy.WCS object

        """
        if value is None:
            self._wcs = None
            self.pixel_scale = conf.PIXEL_SCALE
        elif isinstance(value, WCS):
            self._wcs = value
            self.pixel_scale = proj_plane_pixel_scales(self._wcs)[0] * 3600.
        else:
            raise TypeError('WCS is not an astropy WCS object.')

    ### DATA VALIDATION - IMAGES
    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, array):
        try:
            array = np.array(array)
            ndim = array.ndim

        except:
            raise TypeError('Not a valid image array.')

        if ndim == 2:
            self.ndim = ndim
            if array.dtype.byteorder == '>':
                array = array.byteswap().newbyteorder()
            self._images = array[None, :, :]
            self.shape = self._images.shape

        elif ndim == 3:
            self.ndim = ndim
            if array.dtype.byteorder == '>':
                array = array.byteswap().newbyteorder()
            self._images = array
            self.shape = self._images.shape

        else:
            raise ValueError(f'Images found with invalid dimensions (ndim = {ndim})')

        self.dims = self.shape[1:]
        self.n_bands = self.shape[0]

    def generate_backgrounds(self):
        # Generate backgrounds
        self.logger.info('Generating image backgrounds')
        if ((self.shape[1] * self.shape[2]) < 1E8):
            self.backgrounds = np.zeros((self.n_bands, 2), dtype=float)
            self.background_images = np.zeros_like(self._images) 
            self.background_rms_images = np.zeros_like(self._images)
            for i, img in enumerate(self._images):
                background = sep.Background(img, bw = conf.SUBTRACT_BW, bh = conf.SUBTRACT_BH, fw = conf.SUBTRACT_FW, fh = conf.SUBTRACT_FH)
                self.backgrounds[i] = background.globalback, background.globalrms
                self.background_images[i] = background.back()
                self.background_rms_images[i] = background.rms()

        else:
            self.backgrounds = None
            self.background_images = None
            self.background_rms_images = None

        # if conf.VERBOSE2:
        #     print('--- Image Details ---')
        #     print(f'Mean = {np.mean(self._images, (1,2))}')
        #     print(f'Std = {np.std(self._images, (1,2))}')

        #     print('--- Background Details ---')
        #     print(f'Mesh size = ({conf.SUBTRACT_BW}, {conf.SUBTRACT_BH})')
        #     print(f'Mean = {np.mean(self.background_images, (1,2))}')
        #     print(f'Std = {np.std(self.background_images, (1,2))}')
        #     print(f'Global = {self.backgrounds[:,0]}')
        #     print(f'RMS = {self.backgrounds[:,1]}')


    ### DATA VALIDATION - WEIGHTS
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, array):
        if array is None:
            self._weights = np.ones(self.shape)
        else:
            try:
                array = np.array(array)
                ndim = np.ndim(array)

            except:
                raise TypeError('Not a valid image array.')

            if ndim == 2:
                self._weights = array[None, :, :]

            elif ndim == 3:
                self._weights = array

            else:
                raise ValueError(f'Weights found with invalid dimensions (ndim = {ndim})')

            shape = np.shape(self._weights)
            if shape != self.shape:
                raise ValueError(f'Weights found with invalid shape (shape = {shape})')

    @property
    def masks(self):
        """### DATA VALIDATION - MASKS"""
        return self._masks

    @masks.setter
    def masks(self, array):
        if array is None:
            # self._masks = np.zeros_like(self._images, dtype=bool)
            self.masks = self.weights == 0
        else:
            try:
                ndim = np.ndim(array)
                array = np.array(array)
            except:
                raise TypeError('Not a valid image array.')

            if ndim == 2:
                self._masks = array[None, :, :] | (self.weights == 0)

            elif ndim == 3:
                self._masks = array | (self.weights == 0)

            else:
                raise ValueError(f'Masks found with invalid dimensions (ndim = {ndim})')

            shape = np.shape(self._masks)
            if shape != self.shape:
                raise ValueError(f'Masks found with invalid shape (shape = {shape}')


    ### DATA VALIDATION - PSFS

    @property
    def psfmodels(self):
        return self._psfmodels

    @psfmodels.setter
    def psfmodels(self, psfmodels):
        if psfmodels is None:
            self._psfmodels = -99 * np.ones(self.n_bands)
        else:
            self._psfmodels = psfmodels

    ### METHODS
    def _get_subimage(self, x0, y0, w, h, buffer):
        # Make a cut-out
        # Handle buffer zones
        subdims = (w + 2*buffer, h + 2*buffer)
        left = x0 - buffer
        right = x0 + subdims[0] - buffer
        bottom = y0 - buffer
        top = y0 + subdims[1] - buffer

        subvector = (left, bottom)

        # Check if corrections are necessary
        subshape = (self.n_bands, subdims[0], subdims[1])
        subimages = np.zeros(subshape)
        subweights = np.zeros(subshape)
        submasks = np.ones(subshape, dtype=bool)

        if left < 0:
            leftpix = abs(left)
        else:
            leftpix = 0

        if right > self.dims[0]:
            rightpix = self.dims[0] - right
        else:
            rightpix = subshape[1]

        if bottom < 0:
            bottompix = abs(bottom)
        else:
            bottompix = 0

        if top > self.dims[1]:
            toppix = self.dims[1] - top
        else:
            toppix = subshape[2]

        leftpos = np.max([left, 0])
        rightpos = np.min([right, self.dims[0]])
        bottompos = np.max([bottom, 0])
        toppos = np.min([top, self.dims[1]])

        self.slice = [slice(leftpos, rightpos), slice(bottompos, toppos)]
        self.slicepos = tuple([slice(0, self.n_bands),] + self.slice)
        self.slicepix = tuple([slice(0, self.n_bands), slice(leftpix, rightpix), slice(bottompix, toppix)])
        self.slice = tuple(self.slice)

        # print(self.slicepix, self.slicepos)
        subimages[self.slicepix] = self.images[self.slicepos]
        subweights[self.slicepix] = self.weights[self.slicepos]
        submasks[self.slicepix]= self.masks[self.slicepos]

        if self.wcs is not None:
            # subwcs = self.wcs.slice(self.slice[::-1])
            subwcs = self.wcs.slice(self.slice)

            # subwcs.wcs.crpix -= (left, bottom)
            # subwcs.array_shape = subshape[1:]
        else:
            subwcs = None

        # FIXME: too many return values; maybe try a namedtuple? a class?
        return subimages, subweights, submasks, self.psfmodels, self.bands, subwcs, subvector, self.slicepix, self.slice

    def _band2idx(self, band, bands=conf.BANDS):
        # Convert band to index for arrays
        # TODO: Handle more than one band at a time
        if type(bands) != list:
            bands = list(bands)
        if band in bands:
            idx = bands.index(band)
            # self.logger.debug(f'subimage._band2idx :: {band} returns idx={idx}')
            return idx
        else:
            raise ValueError(f"{band} is not a valid band.")

    def sextract(self, band, sub_background=False, force_segmap=None, use_mask=False, incl_apphot=False):
        # perform sextractor on single band only (may expand for matched source phot)
        # Generate segmap and segmask
        
        if (band == conf.DETECTION_NICKNAME) | (band == conf.MODELING_NICKNAME):
            idx = 0
        else:
            idx = self._band2idx(band)
        image = self.images[idx].copy()
        wgts = self.weights[idx].copy()
        wgts[wgts==0] = -1
        var = 1. / wgts
        var[wgts==-1] = 0
        mask = self.masks[idx].copy()
        background = self.backgrounds[idx]

        # Supply a segmap to "match" detection
        if force_segmap is not None:
            var[force_segmap] = 0

        if (self.weights == 1).all() | (conf.USE_DETECTION_WEIGHT == False) :
            # No weight supplied by user
            var = None
            thresh = conf.THRESH #* background[1]  # WTF is this about?!
            # if not sub_background:
            #     thresh += background[0]

            self.logger.debug(f'Detection is to be performed with weights? {conf.USE_DETECTION_WEIGHT}')
        else:
            thresh = conf.THRESH

        convfilt = None
        if conf.FILTER_KERNEL is not None:
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, '../../config/conv_filters/'+conf.FILTER_KERNEL)
            if os.path.exists(filename):
                convfilt = np.array(np.array(ascii.read(filename, data_start=2)).tolist())
            else:
                raise FileExistsError(f"Convolution file at {filename} does not exist!")

        if use_mask:
            mask = mask
        else:
            mask = None

        self.logger.debug(f'Detection is to be performed with thresh = {thresh}')

        # Set extraction pixel limit buffer
        sep.set_extract_pixstack(conf.PIXSTACK_SIZE)

        if sub_background:
            self.logger.debug('Background will be subtracted.')
            background = sep.Background(self.images[idx], bw = conf.DETECT_BW, bh = conf.DETECT_BH,
                                fw = conf.DETECT_FW, fh = conf.DETECT_FH)
            if conf.USE_FLAT:
                image -= background.globalback
                self.logger.debug(f'Subtracted flat background level ({background.globalback:4.4f})')
            else:
                image -= background.back()

        var = np.ones_like(var)
        kwargs = dict(var=var, mask=mask, minarea=conf.MINAREA, filter_kernel=convfilt, 
                filter_type=conf.FILTER_TYPE, segmentation_map=True, 
                deblend_nthresh=conf.DEBLEND_NTHRESH, deblend_cont=conf.DEBLEND_CONT)
        catalog, segmap = sep.extract(image, thresh, **kwargs)
        catalog['y'] -= 0.75 # HACK
        catalog['x'] -= 0.1 #HACK

        # plt.ion()
        # fig, ax = plt.subplots(ncols=2, figsize=(40,20))
        # from matplotlib.colors import LogNorm
        # ax[0].imshow(image, norm=LogNorm(), vmin=1E-8, vmax=10, cmap='Greys')
        # ax[0].scatter(catalog['x'], catalog['y'], s=1)
        # ax[1].imshow(segmap)
        # ax[1].scatter(catalog['x'], catalog['y'], s=1)
        # # plt.show()
        # fig.savefig(conf.PLOT_DIR + '/checksep.pdf')
        # plt.close('all')

        
        if len(catalog) != 0:
            catalog = Table(catalog)
            catalog.add_column(Column(catalog['x'], name='x_orig' ))
            catalog.add_column(Column(catalog['y'], name='y_orig' ))
            
            if self.wcs is not None:
                wx = catalog['x_orig'] + self.mosaic_origin[1] - conf.BRICK_BUFFER
                wy = catalog['y_orig'] + self.mosaic_origin[0] - conf.BRICK_BUFFER
                wwx, wwy = wx - self.mosaic_origin[0] + conf.BRICK_BUFFER, wy - self.mosaic_origin[1] + conf.BRICK_BUFFER
                skyc = self.wcs.all_pix2world(wwx, wwy, 0)
                catalog.add_column(Column(skyc[0], name=f'RA_{conf.DETECTION_NICKNAME}'))
                catalog.add_column(Column(skyc[1], name=f'DEC_{conf.DETECTION_NICKNAME}'))

            # Aperture Photometry
            if incl_apphot:
                flux = np.zeros(len(catalog), dtype=(float, len(conf.APER_PHOT)))
                flux_err = flux.copy()
                flag = np.zeros_like(catalog['x'], dtype=bool)
                for i, radius in enumerate(conf.APER_PHOT): # in arcsec
                    flux[:,i], flux_err[:,i], flag = sep.sum_circle(image,
                                        x=catalog['x'],
                                        y=catalog['y'],
                                        r=radius / conf.PIXEL_SCALE,
                                        var=var)

                mag = -2.5 * np.log10(flux) + conf.MODELING_ZPT
                mag_err = 1.09 * flux_err / flux
                catalog.add_column(Column(flux, name='flux_aper' ))
                catalog.add_column(Column(flux_err, name='fluxerr_aper' ))
                catalog.add_column(Column(mag, name='mag_aper' ))
                catalog.add_column(Column(mag_err, name='magerr_aper' ))
                catalog.add_column(Column(flag, name='flag_aper' ))

            self.catalog = catalog
            self.n_sources = len(catalog)
            self.segmap = segmap
            return catalog, segmap
        else:
            raise ValueError('No objects found by SExtractor.')

    def subtract_background(self, idx=None, flat=False, use_masked=False, use_direct_median=False, apply=True):
        if apply:
            self.logger.info(f'Subtracting background. (flat={flat}, use_masked={use_masked}, use_direct_median={use_direct_median})')

        self.masked_median = np.zeros((self.n_bands), dtype=float)
        self.masked_std = np.zeros((self.n_bands), dtype=float)

        if use_masked: # need to redo everything!
            # apply segmap
            if idx is None:
                for i, (img, mask) in enumerate(zip(self.images, self.masks)):
                    img_masked = img.copy()
                    # fill with 1st pass background median
                    img_masked[(self.segmask == 1) | mask] = np.nan 
                    # get direct median
                    img_masked_stat = sigma_clip(img_masked.flatten(), 3, masked=False)
                    self.masked_median[i] = np.nanmedian(img_masked_stat)
                    self.masked_std[i] = np.nanstd(img_masked_stat)
                    # re-derive
                    background = sep.Background(img_masked, bw = conf.SUBTRACT_BW, bh = conf.SUBTRACT_BH, fw = conf.SUBTRACT_FW, fh = conf.SUBTRACT_FH)
                    # re-assign!
                    self.backgrounds[i] = background.globalback, background.globalrms
                    self.background_images[i] = background.back()
                    self.background_rms_images[i] = background.rms()      
                    self.logger.info(f'    Band: {conf.BANDS[i]}')
                    self.logger.info(f'    Mesh size = ({conf.SUBTRACT_BW}, {conf.SUBTRACT_BH})')
                    self.logger.info(f'    Back Median = {np.nanmedian(self.background_images[i], (0,1))}')
                    self.logger.info(f'    Back Std = {np.nanstd(self.background_images[i], (0,1))}')
                    self.logger.info(f'    Back Global = {self.backgrounds[i,0]}')
                    self.logger.info(f'    RMS Global = {self.backgrounds[i,1]}')
                    if use_direct_median:
                        self.logger.info(f'    Direct Median = {self.masked_median}')
                        self.logger.info(f'    Direct RMS = {self.masked_std}') 

            else:
                img, mask = self.images[idx], self.masks[idx]
                img_masked = img.copy()
                # fill with 1st pass background median
                img_masked[(self.segmask==1) | mask] = np.nan 
                img_masked_stat = sigma_clip(img_masked.flatten(), 3, masked=False)
                self.masked_median[idx] = np.nanmedian(img_masked_stat)
                self.masked_std[idx] = np.nanstd(img_masked_stat)
                # re-derive
                background = sep.Background(img_masked, bw = conf.SUBTRACT_BW, bh = conf.SUBTRACT_BH, fw = conf.SUBTRACT_FW, fh = conf.SUBTRACT_FH)
                # re-assign!
                self.backgrounds[idx] = background.globalback, background.globalrms
                self.background_images[idx] = background.back()
                self.background_rms_images[idx] = background.rms()  

                self.logger.info(f'    Band: {conf.BANDS[idx]}')
                self.logger.info(f'    Mesh size = ({conf.SUBTRACT_BW}, {conf.SUBTRACT_BH})')
                self.logger.info(f'    Back Median = {np.nanmedian(self.background_images[idx], (0,1))}')
                self.logger.info(f'    Back Std = {np.nanstd(self.background_images[idx], (0,1))}')
                self.logger.info(f'    Back Global = {self.backgrounds[idx,0]}')
                self.logger.info(f'    RMS Global = {self.backgrounds[idx,1]}')
                if use_direct_median:
                    self.logger.info(f'    Direct Median = {self.masked_median}')
                    self.logger.info(f'    Direct RMS = {self.masked_std}') 

        if apply:
            if use_direct_median:
                if idx is None:
                    self.images -= self.masked_median
                else:
                    self.images[idx] -= self.masked_median[idx]

            else:

                if idx is None:
                    if flat:
                        self.images -= self.backgrounds[:,0]
                    else:
                        self.images -= self.background_images

                    self.logger.info(f'    Band: {conf.BANDS}')
                    self.logger.info(f'    Mesh size = ({conf.SUBTRACT_BW}, {conf.SUBTRACT_BH})')
                    self.logger.info(f'    Back Median = {np.nanmedian(self.background_images, (1,2))}')
                    self.logger.info(f'    Back Std = {np.nanstd(self.background_images, (1,2))}')
                    self.logger.info(f'    Back Global = {self.backgrounds[:,0]}')
                    self.logger.info(f'    RMS Global = {self.backgrounds[:,1]}')
                    if use_direct_median:
                        self.logger.info(f'    Direct Median = {self.masked_median}')
                        self.logger.info(f'    Direct RMS = {self.masked_std}') 

                else:
                    if flat:
                        self.images[idx] -= self.backgrounds[idx][0]
                    else:
                        self.images[idx] -= self.background_images[idx]

                    self.logger.info(f'    Band: {conf.BANDS[idx]}')
                    self.logger.info(f'    Mesh size = ({conf.SUBTRACT_BW}, {conf.SUBTRACT_BH})')
                    self.logger.info(f'    Back Median = {np.nanmedian(self.background_images[idx], (0,1))}')
                    self.logger.info(f'    Back Std = {np.nanstd(self.background_images[idx], (0,1))}')
                    self.logger.info(f'    Back Global = {self.backgrounds[idx,0]}')
                    self.logger.info(f'    RMS Global = {self.backgrounds[idx,1]}')
                    if use_direct_median:
                        self.logger.info(f'    Direct Median = {self.masked_median}')
                        self.logger.info(f'    Direct RMS = {self.masked_std}') 
                        




