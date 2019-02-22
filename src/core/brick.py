# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Class to handle mosaic subimages (i.e. bricks)

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

from .utils import create_circular_mask
from .subimage import Subimage
from .blob import Blob
from .config import *

# ------------------------------------------------------------------------------
# Additional Packages
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Declarations and Functions
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------
class Brick(Subimage):
    
    def __init__(self, images, weights=None, masks=None, wcs=None,
                bands=None, buffer=BRICK_BUFFER, brick_id=-99
                ):
        super().__init__(images, weights, masks, bands, wcs)

        self._buffer = buffer
        self.brick_id = brick_id


        self._buff_left = self._buffer
        self._buff_right = self.dims[0] - self._buffer
        self._buff_bottom = self._buffer
        self._buff_top = self.dims[1] - self._buffer
        
        # Replace mask
        self._masks[:, :, :self._buff_left] = True
        self._masks[:, :, self._buff_right:] = True
        self._masks[:, :self._buff_bottom] = True
        self._masks[:, self._buff_top:] = True

        #self._masks[:, self._buff_right:, self._buff_top: ] = True
    
    @property
    def buffer(self):
        return self._buffer

    def cleanup(self):
        self.clean_segmap()

        self.add_sid()

        self.clean_catalog()

        self.dilate()

        self.relabel()


    def about(self):
        print(f'*** Brick {self.brick_id}')
        print(f' Shape: {self.shape}')
        print(f' Nbands: {self.n_bands}')
        print(f' Origin: {self.subvector}')
        print()
        
    def clean_segmap(self):
        # Clean segmap
        coords = np.array([self.catalog['x'], self.catalog['y']]).T
        self._allowed_sources = (coords[:,0] > self._buff_left) & (coords[:,0] < self._buff_right )\
                        & (coords[:,1] > self._buff_bottom) & (coords[:,1] < self._buff_top)
        idx = np.where(~self._allowed_sources)[0] + 1
        for i in idx:
            self.segmap[self.segmap == i] = 0

    def clean_catalog(self):
        # Clean cat
        self.catalog = self.catalog[self._allowed_sources]
        self.n_sources = len(self.catalog)

    def add_sid(self):        
        sid_col = np.arange(1, self.n_sources+1, dtype=int)
        self.catalog.add_column(Column(sid_col, name='sid'), 0)

        brick_col = self.brick_id * np.ones(self.n_sources, dtype=int)
        self.catalog.add_column(Column(brick_col, name='bid'), 0)

    def dilate(self, radius = DILATION_RADIUS, fill_holes = True, clean = True):

        # Make binary
        segmask = self.segmap.copy() # I don't see how we can get around this.
        segmask[segmask != 0] = 1
        # Dilate
        struct2 = create_circular_mask(2*radius, 2*radius, radius=radius)
        segmask = binary_dilation(segmask, structure = struct2 )

        if fill_holes:
            segmask = binary_fill_holes(segmask).astype(int)

        self.segmask = segmask
        return segmask

    def relabel(self):
        self.blobmap, self.n_blobs = label(self.segmask)

        return self.blobmap, self.n_blobs


    def make_blob(self, blob_id):

        if blob_id < 1:
            raise ValueError('Blob id must be greater than 0.')

        # Grab blob
        blobmask = np.array(self.blobmap == blob_id, bool)
        blob_sources = np.unique(self.segmap[blobmask])

        # Dimensions
        idx, idy = blobmask.nonzero()
        xlo, xhi = np.min(idx), np.max(idx) + 1
        ylo, yhi = np.min(idy), np.max(idy) + 1
        w = xhi - xlo
        h = yhi - ylo

        # Make cutout
        blob_kwargs = self._get_subimage(xlo, ylo, w, h, buffer=BLOB_BUFFER)
        blob = Blob(**blob_kwargs)
        blob.masks[self.slicepix] = np.logical_not(blobmask[self.slice])
        blob.segmap = self.segmap[self.slice]

        # Clean
        blob_sourcemask = np.in1d(self.catalog['sid'], blob_sources)
        blob.catalog = self.catalog[blob_sourcemask]
        blob.catalog['x'] -= blob.subvector[1]
        blob.catalog['y'] -= blob.subvector[0]

        return blob

        



