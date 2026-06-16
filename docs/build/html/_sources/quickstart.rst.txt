Quick Start
===========

This page walks through a minimal end-to-end run: building bricks, detecting sources, fitting morphological models, and extracting multi-band photometry. It assumes you have completed :doc:`installation` and have FITS images on disk.

Before you begin
-----------------

Have the following ready before starting:

- A **high-resolution detection image** in FITS format. This is the image used for source detection and morphology fitting — ideally a chi-mean stack or a deep optical image. It does not need to be one of the photometric bands.
- **Science and weight images** for every photometric band you want to measure.
- **PSF models** for each photometric band (PSFEx ``.fits`` format). See :doc:`configuration` for details; ``bin/prep_psf.py`` can help prepare raw PSF stamps.
- Enough disk space: each brick HDF5 file is typically 50–500 MB depending on image size and the number of bands.

The Farmer does **not** reproject images — all bands must already be on the same pixel grid as the detection image, or you must accept that the multi-resolution mapping will introduce small positional uncertainties.

Step 1 — Configure your survey
--------------------------------

Copy ``config/config.py`` into the root of your working directory (or keep it under ``config/``; The Farmer searches both locations). Edit it to point to your images:

.. code-block:: python
   :caption: config/config.py (excerpt)

   PATH_DATA = '/path/to/your/data/'
   PATH_BRICKS   = os.path.join(PATH_DATA, 'interim/bricks')
   PATH_CATALOGS = os.path.join(PATH_DATA, 'output/catalogs')
   # ... other paths

   DETECTION = {
       'science': '/path/to/data/external/detection_image.fits',
       'subtract_background': True,
       'backtype': 'flat',
       'backregion': 'mosaic',
   }

   BANDS = {}
   BANDS['hsc_i'] = {
       'science':  '/path/to/data/external/hsc_i.fits',
       'weight':   '/path/to/data/external/hsc_i_weight.fits',
       'psfmodel': '/path/to/data/interim/psfmodels/hsc_i.fits',
       'subtract_background': True,
       'backtype': 'flat',
       'backregion': 'mosaic',
       'zeropoint': 31.4,
   }
   # Add more bands as needed

   N_BRICKS = (2, 4)       # 2 columns × 4 rows = 8 bricks
   MODEL_BANDS = ['hsc_i'] # bands used for morphology fitting

Step 2 — Validate configuration
---------------------------------

Start Python from your working directory (where ``config/`` is visible) and validate all configured bands:

.. code-block:: python

   import farmer

   farmer.validate()

``validate()`` checks that every science image exists, the WCS is readable, and each PSF model can be found. Errors here point to misconfigured paths or missing files.

Step 3 — Build bricks
-----------------------

Build bricks from all configured bands. This cuts postage stamps from every mosaic and saves them to HDF5:

.. code-block:: python

   # Build all bricks (slow for large surveys — add bands one at a time
   # for production runs)
   farmer.build_bricks()

   # Or build a single brick for testing
   brick = farmer.build_bricks(brick_ids=1)

Each brick is saved as ``data/interim/bricks/B{brick_id}.h5``.

Step 4 — Detect sources
-------------------------

Run source detection on the detection band. Sources outside the brick boundary are discarded; nearby sources are grouped for simultaneous fitting:

.. code-block:: python

   farmer.detect_sources(brick_ids=1, write=True)

Step 5 — Determine morphological models
-----------------------------------------

The Farmer iterates through a decision tree of profile types (point source → SimpleGalaxy → Exp → deV → Composite) to find the best-fit model for each source in the detection band(s):

.. code-block:: python

   farmer.generate_models(brick_ids=1)

Results are stored in ``brick.model_catalog`` and written to ``B1.h5``.

Step 6 — Multi-band photometry
--------------------------------

With morphologies fixed, measure fluxes in every configured band:

.. code-block:: python

   farmer.photometer(brick_ids=1)

This writes the final catalog to ``data/output/catalogs/``.

Step 7 — Inspect results
--------------------------

.. code-block:: python

   brick = farmer.load_brick(1)

   # Print brick summary
   brick.summary()

   # Access the model catalog (OrderedDict keyed by source ID)
   cat = brick.model_catalog

   # Inspect a single group interactively
   group = farmer.quick_group(brick_id=1, group_id=42)
   group.summary()

Full Production Example
-----------------------

For a complete survey run processing all bricks in parallel:

.. code-block:: python

   import farmer

   farmer.validate()

   # 1. Build bricks (detection band first, then photometric bands)
   good_bricks = farmer.build_bricks()

   # 2. Detect sources in all bricks
   farmer.detect_sources(brick_ids=good_bricks)

   # 3. Determine models in all bricks
   farmer.generate_models(brick_ids=good_bricks)

   # 4. Measure photometry in all bands
   farmer.photometer(brick_ids=good_bricks)

See ``bin/example_script.py`` in the repository for a worked example with specific file paths.

Next Steps
----------

- **Configuration reference**: See :doc:`configuration` for all tunable parameters.
- **Pipeline deep-dive**: See :doc:`pipeline` for how the decision tree and optimization loop work.
- **API reference**: See :doc:`api/index` for complete class and function documentation.
- **PSF preparation**: The helper script ``bin/prep_psf.py`` clips, normalizes, and optionally resamples a raw PSF image to the format The Farmer expects.
