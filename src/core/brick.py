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

import sys
import numpy as np

from astropy.table import Column
from scipy.ndimage import label, binary_dilation, binary_fill_holes

from .utils import create_circular_mask
from .subimage import Subimage
from .blob import Blob
import config as conf


class Brick(Subimage):
    """TODO: docstring"""

    def __init__(self,
                 images,
                 weights=None,
                 masks=None,
                 psfmodels=None,
                 wcs=None,
                 bands=None,
                 buffer=conf.BRICK_BUFFER,
                 brick_id=-99):
        """TODO: docstring"""

        self.wcs = wcs
        self.images = images
        self.weights = weights
        self.masks = masks
        self.psfmodels = psfmodels
        self.bands = np.array(bands)


        super().__init__()

        self._buffer = buffer
        self.brick_id = brick_id

        self.segmap = None
        self.blobmap = None

        self._buff_left = self._buffer
        self._buff_right = self.dims[0] - self._buffer
        self._buff_bottom = self._buffer
        self._buff_top = self.dims[1] - self._buffer

        # Replace mask
        self._masks[:, :, :self._buff_left] = True
        self._masks[:, :, self._buff_right:] = True
        self._masks[:, :self._buff_bottom] = True
        self._masks[:, self._buff_top:] = True

        x0 = int(((brick_id - 1) * conf.BRICK_WIDTH) % conf.MOSAIC_WIDTH)
        y0 = int(((brick_id - 1) * conf.BRICK_HEIGHT) / conf.MOSAIC_HEIGHT) * conf.BRICK_HEIGHT
        self.mosaic_origin = np.array([x0, y0])

    @property
    def buffer(self):
        return self._buffer

    def cleanup(self):
        """TODO: docstring"""
        self.clean_segmap()

        self.clean_catalog()

        self.dilate()

        self.relabel()

        self.add_ids()

    def clean_segmap(self):
        """TODO: docstring"""
        coords = np.array([self.catalog['x'], self.catalog['y']]).T
        self._allowed_sources = (coords[:,0] > self._buff_left) & (coords[:,0] < self._buff_right )\
                        & (coords[:,1] > self._buff_bottom) & (coords[:,1] < self._buff_top)
        
        idx = np.where(~self._allowed_sources)[0]
        for i in idx:
            self.segmap[self.segmap == i+1] = 0

    

    def clean_catalog(self):
        """TODO: docstring"""
        sid_col = np.arange(1, self.n_sources+1, dtype=int)
        self.catalog.add_column(Column(sid_col.astype(int), name='source_id'), 0)
        self.catalog = self.catalog[self._allowed_sources]
        self.n_sources = len(self.catalog)

    def add_columns(self, band_only=False):
        """TODO: docstring"""
        filler = np.zeros(len(self.catalog))
        for colname in self.bands:
            colname = colname.replace(' ', '_')
            self.catalog.add_column(Column(filler, name=f'MAG_{colname}'))
            self.catalog.add_column(Column(filler, name=f'MAGERR_{colname}'))
            self.catalog.add_column(Column(filler, name=f'FLUX_{colname}'))
            self.catalog.add_column(Column(filler, name=f'FLUXERR_{colname}'))
            self.catalog.add_column(Column(filler, name=f'CHISQ_{colname}'))
        if not band_only:
            for colname in ('X_MODEL', 'Y_MODEL', 'XERR_MODEL', 'YERR_MODEL', 'RA', 'DEC'):
                self.catalog.add_column(Column(filler, name=colname))
            self.catalog.add_column(Column(np.zeros(len(self.catalog), dtype='S20'), name='SOLMODEL'))
            self.catalog.add_column(Column(np.zeros(len(self.catalog), dtype=bool), name='VALID_SOURCE'))
            for colname in ('REFF', 'REFF_ERR', 'AB', 'AB_ERR', 'THETA', 'THETA_ERR',
                        'FRACDEV', 'EXP_REFF', 'EXP_REFF_ERR', 'EXP_AB', 'EXP_AB_ERR', 'EXP_THETA', 'EXP_THETA_ERR', 
                        'DEV_REFF', 'DEV_REFF_ERR', 'DEV_AB', 'DEV_AB_ERR', 'DEV_THETA', 'DEV_THETA_ERR' ):
                self.catalog.add_column(Column(filler, name=colname))

    def dilate(self, radius=conf.DILATION_RADIUS, fill_holes=True):
        """TODO: docstring"""
        # Make binary
        segmask = self.segmap.copy()  # I don't see how we can get around this.
        segmask[segmask != 0] = 1
        # Dilate
        struct2 = create_circular_mask(2*radius, 2*radius, radius=radius)
        segmask = binary_dilation(segmask, structure=struct2)

        if fill_holes:
            segmask = binary_fill_holes(segmask).astype(int)

        self.segmask = segmask
        return segmask

    def relabel(self):
        self.blobmap, self.n_blobs = label(self.segmask)

        return self.blobmap, self.n_blobs

    def add_ids(self):
        """TODO: docstring. rename sid and bid throughout"""
        brick_col = float(self.brick_id) * np.ones(self.n_sources, dtype=int)
        self.catalog.add_column(Column(brick_col.astype(int), name='brick_id'), 1)

        blob_col = np.array([np.unique(self.blobmap[self.segmap == sid])[0] for sid in self.catalog['source_id']])
        self.catalog.add_column(Column(blob_col.astype(int), name='blob_id'), 1)

    def make_blob(self, blob_id):

        if blob_id < 1:
            raise ValueError('Blob id must be greater than 0.')

        blob = Blob(self, blob_id)

        return blob





