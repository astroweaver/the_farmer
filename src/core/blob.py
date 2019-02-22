# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----
Class function to handle potentially blended sources (i.e. blobs)

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
from astropy.table import Table
from tractor import *

from .subimage import Subimage
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
class Blob(Subimage):

    def __init__(self, images=None, weights=None, masks=None, wcs=None,
                bands=None, subvector=None, buffer=None, brick_id=-99):
        super().__init__(images, weights, masks, bands, wcs, subvector)

        