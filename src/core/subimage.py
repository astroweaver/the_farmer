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
from astropy.table import Table
from astropy.wcs import WCS
import sep
from astropy.wcs.utils import proj_plane_pixel_scales

from .config import *

class Subimage():

    def __init__(self):


        self.subvector = None
        self.catalog = None
        self.n_sources = None
        # If I make these self._X, the setters break.

    ### DATA VALIDATION - WCS
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
            self.pixel_scale = PIXEL_SCALE
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
            self._images = array[None, :, :]
            self.shape = self._images.shape

        elif ndim == 3:
            self.ndim = ndim
            self._images = array
            self.shape = self._images.shape

        else:
            raise ValueError(f'Images found with invalid dimensions (ndim = {ndim})')

        self.dims = self.shape[1:]
        self.n_bands = self.shape[0]

        # Generate backgrounds
        if (self.shape[1] * self.shape[2]) < 1E8:
            self.backgrounds = np.zeros(self.n_bands, dtype=object)
            for i, img in enumerate(self._images):
                self.backgrounds[i] = sep.Background(img, bw = BW, bh = BH)
        else:
            self.backgrounds = None


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
    


    ### DATA VALIDATION - MASKS
    @property
    def masks(self):
        return self._masks

    @masks.setter
    def masks(self, array):
        if array is None:
            self._masks = np.zeros_like(self._images, dtype=bool)
        else:
            try:
                ndim = np.ndim(array)
                array = np.array(array)
            except:
                raise TypeError('Not a valid image array.')

            if ndim == 2:
                self._masks = array[None, :, :]
            
            if ndim == 3:
                self._masks = array
        
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

       # leftpix = np.max([bottom, 0])  

        if left < 0:
            leftpix = abs(left) 
        else:
            leftpix = 0

        if right > self.dims[0]:
            rightpix = self.dims[0] - right
        else:
            rightpix = subshape[1]

        #rightpix = np.max([right, self.dims[0]])
    
        if bottom < 0:
            bottompix = abs(bottom)
        else:
            bottompix = 0

        if top > self.dims[1]:
            toppix = self.dims[1] - top
        else:
            toppix = subshape[2]

        # toppix = np.max([top, subdims[1]])

        leftpos = np.max([left, 0])
        rightpos = np.min([right, self.dims[0]])
        bottompos = np.max([bottom, 0])
        toppos = np.min([top, self.dims[1]])

        self.slice = [slice(leftpos, rightpos), slice(bottompos, toppos)]
        self.slicepos = tuple([slice(0, self.n_bands),] + self.slice)
        self.slicepix = (slice(0, self.n_bands), slice(leftpix, rightpix), slice(bottompix, toppix))
        
        subimages[self.slicepix] = self.images[self.slicepos]
        subweights[self.slicepix] = self.weights[self.slicepos]
        submasks[self.slicepix]= self.masks[self.slicepos]

        if self.wcs is not None:
            subwcs = self.wcs.deepcopy()
            subwcs.wcs.crpix -= (left, bottom)
            subwcs.array_shape = subshape[1:]

        return subimages, subweights, submasks, self.psfmodels, self.bands, subwcs, subvector, self.slicepix, self.slice
    

    def _band2idx(self, band):
        # Convert band to index for arrays
        # TODO: Handle more than one band at a time
        if band in self.bands:
            return np.arange(self.n_bands)[self.bands == band][0]
        else:
            raise ValueError(f"{band} is not a valid band.")

    def sextract(self, band, include_mask = True, sub_background = False):
        # perform sextractor on single band only (may expand for matched source phot)
        # Generate segmap and segmask
        idx = self._band2idx(band)
        image = self.images[idx]
        var = 1. / self.weights[idx] # TODO: WRITE TO UTILS
        mask = self.masks[idx]
        background = self.backgrounds[idx]

        if (self.weights == 1).all():
            # No weight given - kinda
            var = None
            thresh = THRESH * background.globalrms
            if not sub_background:
                thresh += background.globalback 
        
        else:
            thresh = THRESH

        if sub_background:
            image -= background.back()
        
        kwargs = dict(var=var, minarea=MINAREA, segmentation_map=True, deblend_nthresh=DEBLEND_NTHRESH, deblend_cont=DEBLEND_CONT)
        catalog, segmap = sep.extract(image, thresh, **kwargs)

        if len(catalog) != 0:
            catalog = Table(catalog)
            self.catalog = catalog
            self.n_sources = len(catalog)
            self.segmap = segmap
            return catalog, segmap
        else:
            raise ValueError('No objects found by SExtractor.')

    
