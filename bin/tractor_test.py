# -*- coding: utf-8 -*-
"""

Authors
-------
John Weaver <john.weaver.astro@gmail.com>


About
-----


Known Issues
------------
None


"""

# ------------------------------------------------------------------------------
# Standard Packages
# ------------------------------------------------------------------------------
from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from tractor import *

# ------------------------------------------------------------------------------
# Additional Packages
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
W, H = 50, 50
psfsigma = 1
noisesigma = 1E-2

# ------------------------------------------------------------------------------
# Declarations and Functions
# ------------------------------------------------------------------------------
# Track images
keep_mods = np.zeros(shape=(30, W, H))

np.random.seed(42)

fig, ax = plt.subplots(ncols = 3, nrows = 9, figsize=(4,10), sharex=True, sharey=True)

ima = dict(interpolation=None, origin='lower', cmap='magma',
    vmin=-2*noisesigma, vmax=10*noisesigma)
imchi = dict(interpolation=None, origin='lower', cmap='Blues',
      vmin=0, vmax=1)

# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------
print('Tractor Test: Starting')

# Make an image
tims = []
img = Image(data=np.zeros((H, W)),
            invvar=np.ones(shape=(H, W)) / noisesigma**2,
            psf=NCircularGaussianPSF([psfsigma], [1.]),
            wcs = NullWCS(),
            photocal=NullPhotoCal(),
            sky=ConstantSky(0.)
            )

print('Tractor Test: Blank image made')

# Make a model Galaxy
position = PixPos(25.5, 24.5)
flux = Flux(4.1)
shape = GalaxyShape(3.5, 0.6, 25)

testmodel = ExpGalaxy(position, flux, shape)
testimg = testmodel.getModelPatch(img)
testimg.addTo(img.data)
img.data += np.random.normal(0, noisesigma, size=(W,H))

# Replace data with image patch
tims.append(img)

print('Tractor Test: Target source added')

# Make a good guess
cat = []
position = PixPos(23, 26)
flux = Flux(2.)
shape = GalaxyShape(4, 1, 0)
cat.append(ExpGalaxy(position, flux, shape))

print('Tractor Test: Exponential model created')

# Make tractor object
tractor = Tractor(tims, cat)

# Render the models
mods0 = tractor.getModelImage(0)
chis0 = tractor.getChiImage(0)

print('Tractor Test: Models rendered')

# Freeze calibration Parameters
tractor.freezeParam('images')

# Plot
ax[0,0].imshow(img.getImage(), **ima)
ax[0,1].axis('off')
ax[0,2].axis('off')
ax[1,0].imshow(mods0, **ima)
ax[1,1].imshow(img.getImage() - mods0, **ima)
ax[1,2].imshow(chis0, **imchi)

# Grab derivates
derivs = tractor.getDerivs()

print('Tractor Test: About to begin optimization...')

# Take several linearized least squares steps
for k in range(10):
    try:
        dlnp, X, alpha, var = tractor.optimize(variance=True)
        print('dlnp {}'.format(dlnp))
    except:
        print('Optimization failed!')
        break
    if dlnp < 1E-6:
        break

    # Render the solution models
    mods = tractor.getModelImage(0)
    chis = tractor.getChiImage(0)

    # Plot
    ax[k+2,0].imshow(mods, **ima)
    ax[k+2,1].imshow(img.getImage() - mods, **ima)
    ax[k+2,2].imshow(chis, **imchi)

fig.tight_layout()
fig.savefig('testoutput.png', dpi=300)

if np.isclose(np.sum(chis**2), 2408.781998):
    print('Tractor Test: Success!')
else:
    print('Tractor Test: Failed!')

# Plot stuff!
