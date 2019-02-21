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

# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table
from astropy.wcs import WCS
import sep
 
# local imports from this project

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
BW = 16
BH = 16
THRESH = 5 # <sigma>

MINAREA = 5
DEBLEND_NTHRESH = 32
DEBLEND_CONT = 0.001

# ------------------------------------------------------------------------------
# Declarations and Functions
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------

class Subimage():

    def __init__(self, images, weights = None, masks = None,
                 bands = None, wcs = None
                 ):

        self.wcs = wcs
        self.images = images
        self.weights = weights
        self.masks = masks
        self.bands = bands
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
        elif isinstance(value, WCS):
            self._wcs = value
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
            ndim = np.ndim(array)

        except:
            raise TypeError('Not a valid image array.')

        if ndim == 2:
            self.ndim = ndim
            self._images = array[None]
            self.shape = np.shape(self._images)

        elif ndim == 3:
            self.ndim = ndim
            self._images = array
            self.shape = np.shape(self._images)
        else:
            raise ValueError(f'Images found with invalid dimensions (ndim = {ndim})')

        self.dims = self.shape[1:]
        self.n_bands = self.shape[0]

        # Generate backgrounds
        self.backgrounds = np.zeros(self.n_bands, dtype=object)
        for i, img in enumerate(self._images):
            self.backgrounds[i] = sep.Background(img, bw = BW, bh = BH)


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
                self._weights = array[None]
                
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
            self._masks = np.zeros_like(self._images)
        else:
            try:
                ndim = np.ndim(array)
                array = np.array(array)
            except:
                raise TypeError('Not a valid image array.')

            if ndim == 2:
                self._masks = array[None]
            
            if ndim == 3:
                self._masks = array
        
            else:
                raise ValueError(f'Masks found with invalid dimensions (ndim = {ndim})')

            shape = np.shape(self._masks)
            if shape != self.shape:
                raise ValueError(f'Masks found with invalid shape (shape = {shape}')


    ### METHODS
    def get_subimage(self, x0, y0, w, h, buffer):
        # Make a cut-out
        # Handle buffer zones
        subdims = (w + 2*buffer, h + 2*buffer)
        left = x0 - buffer
        right = x0 + subdims[0] - buffer
        bottom = y0 - buffer
        top = y0 + subdims[1] - buffer

        # Check if corrections are necessary
        self.subshape = (self.n_bands, subdims[0], subdims[1])
        self.subimages = np.zeros(self.subshape)
        self.subweights = self.subimages.copy()
        self.submasks = np.ones(self.subshape)

       # leftpix = np.max([bottom, 0])  

        if left < 0:
            leftpix = abs(left) 
        else:
            leftpix = 0

        if right > self.dims[0]:
            rightpix = right - self.dims[0]
        else:
            rightpix = self.subshape[1]

        #rightpix = np.max([right, self.dims[0]])
    
        if bottom < 0:
            bottompix = abs(bottom)
        else:
            bottompix = 0

        if top > self.dims[1]:
            toppix = top - self.dims[1]
        else:
            toppix = self.subshape[2]

        # toppix = np.max([top, subdims[1]])

        leftpos = np.max([left, 0])
        rightpos = np.min([right, self.dims[0]])
        bottompos = np.max([bottom, 0])
        toppos = np.min([top, self.dims[1]])
        
        print(leftpix, rightpix, bottompix, toppix)
        print(left, right, bottom, top)
        self.subimages[:, leftpix:rightpix, bottompix:toppix] = self.images[:, leftpos:rightpos, bottompos:toppos]
        self.subweights[:, leftpix:rightpix, bottompix:toppix] = self.weights[:, leftpos:rightpos, bottompos:toppos]
        self.submasks[:, leftpix:rightpix, bottompix:toppix]= self.masks[:, leftpos:rightpos, bottompos:toppos]

        if self.wcs is not None:

            self.subwcs = self.wcs.deepcopy()
            self.subwcs.wcs.crpix -= (left, bottom)
            self.subwcs.array_shape = self.subshape[1:]

        return self.subimages, self.subweights, self.submasks

    def _band2idx(self, band):
        # Convert band to index for arrays
        # TODO: Handle more than one band at a time
        if band in self.bands:
            return np.argwhere(self.bands == band)[0][0]
        else:
            raise ValueError(f'{band} is not a valid band.')

    def sextract(self, band, include_mask = True, sub_background = True):
        # perform sextractor on single band only (may expand for matched source phot)
        # Generate segmap and segmask
        idx = self._band2idx(band)
        image = self.images[idx]
        var = 1. / self.weights[idx] # TODO: WRITE TO UTILS
        mask = self.masks[idx]

        thresh = THRESH * self.backgrounds[idx].globalrms

        if sub_background:
            image -= bkg.back()
        
        # TODO : Args should come into SExtractor as KWARGS!
        objects, segmap = sep.extract(image, thresh, var = var, mask = mask, minarea = MINAREA, segmentation_map = True,
                                    deblend_nthresh = DEBLEND_NTHRESH, deblend_cont = DEBLEND_CONT
                                    )
        if len(objects) != 0:
            return objects, segmap
        else:
            raise ValueError('No objects found by SExtractor.')

    def dilate(self, segmap, radius = 5, fill_holes = True, clean = True):

        # Make binary
        segmask = segmap.copy() # I don't see how we can get around this.
        segmask[segmask != 0] = 1
        # Dilate
        struct2 = utils.create_circular_mask(2*dsize, 2*dsize, radius=dsize)
        segmask = binary_dilation(segmask, structure = struct2 )

        if fill_holes:
            segmask = binary_fill_holes(segmask).astype(int)

        return segmask

    def relabel(self, segmask):
        newmask, Nblobs = label(segmask)

        return newmask, Nblobs
