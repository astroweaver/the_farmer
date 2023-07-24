# This is a barebones example of what a small job with Farmer
# might look like. Note that more sophisticated jobs are
# possible, e.g. using multiple computing nodes.
# See the how_to_farm.ipynb notebook for further details.

# Questions? Contact John Weaver (john.weaver.astro@gmail.com)

# After configuration via the config file...

# Import Farmer
# Make sure the 'the_farmer' directory is findable in your python path
# to see your path, do 'import sys; print(sys.path)'
from the_farmer import interface

# Make detection bricks
# This will make stacks of regions of your detection image
interface.make_bricks(image_type='detection')

# Make multiband bricks
# Ditto, but for the rest of your bands of interest
interface.make_bricks(image_type='multiband')

# Create PSFs with PSFEx, if not already provided
# Note: requires both Source Extractor and PSFEx to be installed
# First run with sextractor_only=True
interface.make_psf(sextractor_only=True)

# Inspect the PSF diagnostic file and enter a suitable selection box
# in the configuration file. Then, run with psfex_only=True
interface.make_psf(psfex_only=True)

# Run source detection
# Runs pythonic Source Extractor (SEP) over the detection
# For this demo, let's detect on brick 1 only
interface.detect_sources(brick_id=1)

# Determine the most appropriate models based on your model bands
# Performance is highly dependent on the decision tree, so
# tuning the relevant parameters is highly recommended
# Note that the model bands should be the same as those used to
# make the detection image
interface.make_models(brick_id=1)

# Force the models on all of the bands
# This is where the photometry is measured and written as catalogs
interface.force_models(brick_id=1)

# After this, there area number of loosely tested scripts
# for combining the brick catalogs into one main catalog
# as well as for making residual images of bricks
