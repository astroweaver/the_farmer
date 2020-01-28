"""

Filename: utils.py

Purpose: Set of utility functions

Author: John Weaver
Date created: 28.11.2018
Possible problems:
1.

"""
import os
import numpy as np
from tractor.galaxy import ExpGalaxy
from tractor import EllipseE
from tractor.galaxy import ExpGalaxy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from skimage.segmentation import find_boundaries

import config as conf
import matplotlib.cm as cm
import random
from time import time
from astropy.io import fits

import logging
logger = logging.getLogger('farmer.utils')

def header_from_dict(params):
    """ Take in dictionary and churn out a header. Never forget configs again. """
    hdr = fits.Header()
    total_public_entries = np.sum([ not k.startswith('__') for k in params.keys()])
    logger.debug(f'header_from_dict :: Dictionary has {total_public_entries} entires')
    tstart = time()
    for i, attr in enumerate(params.keys()):
        if not attr.startswith('__'):
            logger.debug(f'header_from_dict ::   {attr}')
            value = params[attr]
            if type(value) == str:
                # store normally
                hdr.set(attr[:8], value, attr)
            if type(value) in (float, int):
                # store normally
                hdr.set(attr[:8], value, attr)
            if type(value) in (list, tuple):
                # freak out.
                for j, val in enumerate(value):
                    nnum = len(f'{j+1}') + 1
                    hdr.set(f'{attr[:(8-nnum)]}_{j+1}', str(val), f'{attr}_{j+1}')
            
    logger.debug(f'header_from_dict :: Completed writing header ({time() - tstart:2.3f}s)')
    return hdr

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = np.zeros((h, w), dtype=int)
    mask[dist_from_center <= radius] = 1
    return mask
class SimpleGalaxy(ExpGalaxy):
    '''This defines the 'SIMP' galaxy profile -- an exponential profile
    with a fixed shape of a 0.45 arcsec effective radius and spherical
    shape.  It is used to detect marginally-resolved galaxies.
    '''
    shape = EllipseE(0.45 / conf.PIXEL_SCALE, 0., 0.)

    def __init__(self, *args):
        super(SimpleGalaxy, self).__init__(*args)

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) + ')')

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1)

    def getName(self):
        return 'SimpleGalaxy'

    ### HACK -- for Galaxy.getParamDerivatives()
    def isParamFrozen(self, pname):
        if pname == 'shape':
            return True
        return super(SimpleGalaxy, self).isParamFrozen(pname)   