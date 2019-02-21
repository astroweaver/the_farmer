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
from astropy.table import Table

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
    
    def __init__(images, weights, masks, 
                detection_images=None, detection_weights=None, detection_masks=None
                ):
        pass

        self._catalog = None

    @property
    def catalog():
        return self._catalog

    @property.setter
    def validate():
        # Run validation on brick input
        pass
