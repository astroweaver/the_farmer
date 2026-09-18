Usage Guide
===========

This page covers common workflows in detail. For the big-picture pipeline, see :doc:`pipeline`; for individual class methods, see :doc:`api/index`.

Working with Mosaics
--------------------

A :class:`~farmer.mosaic.Mosaic` represents one band's full-field image. You rarely need to interact with it directly — the top-level API functions handle mosaic loading internally — but it is useful for inspection and for preparing PSF coordinate tables.

.. code-block:: python

   import farmer

   # Load the detection mosaic into memory
   mosaic = farmer.get_mosaic('detection', load=True)
   print(mosaic.position)   # SkyCoord of the mosaic center
   print(mosaic.size)       # angular size as an array of astropy Quantities
   print(mosaic.wcs)        # astropy WCS

   # Load a photometric band
   hsc_i = farmer.get_mosaic('hsc_i', load=True)

   # Run source detection on the full mosaic (rare; usually done per brick)
   hsc_i.extract()
   hsc_i.identify_groups()

Passing ``load=False`` validates paths and WCS without reading pixel data, which is what ``farmer.validate()`` uses internally.

Working with Bricks
--------------------

The :class:`~farmer.brick.Brick` class is the main unit of work. A brick is a rectangular cutout of the mosaic grid, defined by its integer ID (1-indexed).

Building bricks
~~~~~~~~~~~~~~~

.. code-block:: python

   # Build brick #1 from all configured bands and save it
   brick = farmer.build_bricks(brick_ids=1)

   # Build several bricks at once (returns list of successful brick IDs)
   good_ids = farmer.build_bricks(brick_ids=[1, 2, 3])

   # Build only specific bands (e.g., just add irac_ch1 to existing bricks)
   farmer.update_bricks(brick_ids=[1, 2, 3], bands=['irac_ch1'])

Loading an existing brick
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   brick = farmer.load_brick(1)
   brick.summary()   # prints a table of image statistics per band

Checking whether a band is present
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Cheap check — reads only HDF5 metadata, does not load pixel data
   if farmer.brick_has_band(brick_id=1, band='irac_ch1'):
       print('Band already present, skipping mosaic load')

Source Detection
-----------------

Detection runs on a single dedicated image (usually a high-resolution chi-mean or stacked image). The result is a source catalog plus a segmentation map and group map stored inside the brick.

.. code-block:: python

   # Full detection pipeline for brick #1
   brick.detect_sources(band='detection', imgtype='science')

   # Equivalent top-level call (loads brick from disk first)
   farmer.detect_sources(brick_ids=1, write=True)

After detection, ``brick.catalogs['detection']['science']`` contains an ``astropy.Table`` with columns ``id``, ``ra``, ``dec``, ``x``, ``y``, ``group_id``, ``group_pop``, and standard SEP photometric columns (``a``, ``b``, ``theta``, ``flux``, ``peak``, ``npix``, etc.).

Group population and size limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sources are grouped via morphological dilation (see :func:`~farmer.utils.dilate_and_group`). The ``DILATION_RADIUS`` config parameter controls how far apart two sources can be and still be placed in the same group. Groups larger than ``GROUP_SIZE_LIMIT`` (default: 5) are skipped.

Modeling
---------

Model determination runs a staged decision tree to find the best-fit profile for each source in the ``MODEL_BANDS``. The tree progresses from a simple point source through increasingly complex galaxy profiles.

.. code-block:: python

   # Determine models for all groups in brick #1
   farmer.generate_models(brick_ids=1)

   # Or run group-level modeling directly
   brick = farmer.load_brick(1)
   brick.process_groups(mode='model')

   # Or model just a few groups
   brick.process_groups(group_ids=[42, 43, 44], mode='model')

Photometry
----------

Once morphological models are determined, fluxes are measured in every configured band by freezing morphology and optimizing only the flux parameters.

.. code-block:: python

   farmer.photometer(brick_ids=1)

   # Lower-level: already have a brick in memory
   brick.process_groups(mode='photometry')

   # All at once (determine + photometry in a single pass)
   brick.process_groups(mode='all')

Model and Residual Images
-------------------------

``generate_models`` and ``photometer`` write brick-level model, residual, and chi
images as they go. Both can also be rebuilt afterwards, without refitting, and the
full field can be reconstructed as one seamless image per band.

.. code-block:: python

   # Full-field images for every configured band, from the brick catalogs
   farmer.rebuild_mosaic()

   # One band, chi map as well, using 8 processes
   farmer.rebuild_mosaic(bands='hsc_i', imgtypes=('model', 'residual', 'chi'), ncpus=8)

   # Brick-level images only, e.g. after a fix, for brick 1
   farmer.rebuild_brick(1)

``rebuild_mosaic`` rebuilds each brick's fitted models from its catalog (see
:func:`~farmer.utils.models_from_catalog`) and renders them on the band's own mosaic
pixel grid, so nothing needs to be refit and no brick HDF5 files are required. It
writes ``M{band}_{imgtype}.fits`` to ``PATH_ANCILLARY``, streaming to disk so that a
survey-sized mosaic never has to fit in memory.

Prefer the full-field images to stitching the per-brick ones. A brick's images stop
at its buffered footprint, so near a brick edge the light of sources in the
neighbouring brick is missing and a large profile is cut off at the buffer. The
full-field render has no such seams. Each source is drawn once, with the same PSF
and the same ``RESIDUAL_*`` cuts the fit used.

The residual is the science image minus the model, and minus the background the fit
subtracted if the band sets ``subtract_background``. Pixels with no data are NaN in
the residual; masked or zero-weight pixels are NaN in chi, so they fall out of
nan-aware statistics. On drizzled or otherwise resampled data the pixel-to-pixel
noise is correlated, so a chi value there overstates the significance of a residual.

Interactive Group Inspection
------------------------------

For debugging or exploring individual detections, use ``quick_group``:

.. code-block:: python

   # Load brick, detect sources, spawn group, model it, and return
   group = farmer.quick_group(brick_id=1, group_id=42)

   group.summary()          # print image statistics
   group.farm()             # determine models + photometry + plots (in one call)

   # Step through manually
   group.determine_models()
   group.force_models()
   group.plot_summary()

Adding Bands After the Fact
----------------------------

If you have already built and modeled bricks but need to add a new band (e.g., a newly reduced Spitzer image), use ``update_bricks``. It checks whether each brick already contains the band and only loads mosaics for bricks that need updating:

.. code-block:: python

   farmer.update_bricks(bands=['irac_ch1'])

   # Force re-adding even if band exists
   farmer.update_bricks(bands=['irac_ch1'], overwrite=True)

After updating, re-run ``photometer`` to measure fluxes in the new band.

Parallel Processing
--------------------

Set ``NCPUS`` in ``config.py`` to the number of cores you want to use. Groups are distributed across workers using a ``pathos.ProcessPool``:

.. code-block:: python
   :caption: config/config.py

   NCPUS = 8   # 0 = serial (recommended for debugging)

Parallel processing uses ``imap`` with ``chunksize=1``, so groups are farmed out one at a time. This keeps memory usage low but adds some overhead per group. For very small groups, serial mode (``NCPUS=0``) is often faster.

.. warning::
   Parallel processing requires that all objects sent to worker processes are
   picklable. The Tractor WCS objects can be tricky; if you see pickling errors,
   fall back to serial mode.

Timeout Protection
~~~~~~~~~~~~~~~~~~~

To prevent a single slow or diverging group from halting a production run, set ``GROUP_TIMEOUT``:

.. code-block:: python
   :caption: config/config.py

   GROUP_TIMEOUT = 120   # seconds; None to disable

Groups that exceed the timeout are flagged and skipped. The failure is logged at ``WARNING`` level.

.. note::
   ``GROUP_TIMEOUT`` relies on ``signal.alarm`` (Unix only) and has no effect on Windows.

Output Files
-------------

Bricks
~~~~~~

Each brick is saved as an HDF5 file under ``PATH_BRICKS``:

- ``B{brick_id}.h5`` — full brick state (images, catalogs, models)

Source catalogs
~~~~~~~~~~~~~~~

FITS tables are written to ``PATH_CATALOGS``:

- ``B{brick_id}_catalog.fits`` — all sources in the brick

Key catalog columns:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``id``
     - Source integer ID (1-indexed within the brick)
   * - ``brick_id``
     - Parent brick integer ID
   * - ``ra``, ``dec``
     - Sky position in degrees, from the modelling stage
   * - ``ra_det``, ``dec_det``
     - Detection-stage centroid, before any fitting
   * - ``group_id``
     - Group integer ID this source belongs to
   * - ``group_pop``
     - Number of sources in the group
   * - ``{band}_flux``
     - Tractor-fitted flux (native image units)
   * - ``{band}_flux_err``
     - 1-sigma flux uncertainty from Hessian
   * - ``{band}_flux_ujy``
     - Flux converted to microjanskys
   * - ``{band}_mag``
     - AB magnitude from zeropoint
   * - ``{band}_mag_err``
     - AB magnitude uncertainty
   * - ``logre``
     - log10 effective radius (arcsec); -99 for point sources
   * - ``ellip``
     - Ellipticity ε = (1 − b/a)
   * - ``theta``
     - Position angle (degrees E of N)
   * - ``reff``
     - Effective radius (arcsec)
   * - ``ba``
     - Axis ratio b/a
   * - ``pa``
     - Position angle (degrees)
   * - ``chisq``
     - Total chi-squared residual
   * - ``rchisq``
     - Reduced chi-squared
   * - ``ndof``
     - Degrees of freedom
   * - ``flag``
     - Quality flag (0 = good)
   * - ``model_ra``, ``model_dec``
     - Centroid the morphology was solved at
   * - ``model_{band}_flux``, ``model_{band}_flux_err``
     - Flux measured with morphology free, for each ``MODEL_BANDS`` band
   * - ``model_total_chisq``, ``model_total_rchisq``, ``model_total_ndof``
     - Goodness of fit of the solved model
   * - ``model_ps_total_chisq``, ``model_ps_total_rchisq``, ``model_ps_total_ndof``
     - The same for the stage-1 PointSource fit, to compare against
   * - ``{band}_chisq_ref``, ``total_chisq_ref``
     - Chi-squared of the PointSource fit made in the photometry bands immediately
       before the forced fit — a third reference, in those bands rather than in
       ``MODEL_BANDS``
   * - ``phot_{param}`` / ``{band}_{param}``
     - Present only where ``PHOT_PRIORS`` thaws a parameter: the value forced
       photometry measured for it (``phot_`` for a multi-band run, ``{band}_``
       for a single-band one). The plain column keeps the modelling solution.

Two stages, two sets of numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plain ``{band}_flux`` and ``chisq``/``rchisq`` columns hold the forced
photometry: fluxes fitted in every band with morphology frozen. That is the
deliverable, but it is not the fit that chose each model. The decision tree
measured its own photometry and chi-squared in ``MODEL_BANDS``, with morphology
free, and forced photometry then remeasures those same bands and would overwrite
them.

So the modelling stage keeps its own columns. The ``model_*`` block above is a
summary of it, always present, including a point-source reference so the chosen
model can be weighed against an unresolved one. Comparing ``model_{band}_flux``
with ``{band}_flux`` for a ``MODEL_BANDS`` band is a direct check on the
morphology; if ``PHOT_PRIORS`` thaws the position, ``model_ra`` and ``phot_ra``
are the two centroids those two fluxes were measured at, and they are not the
same aperture.

The full modelling solution — every parameter, flux and statistic of the solved
model, and of the PointSource fit it was chosen over — is written next to the
catalog as ``B{brick_id}_models.cat``, which joins on ``id``. It is written by
whichever run determined the models; a photometry-only run leaves it alone.

Figures
~~~~~~~

If ``PLOT > 0``, diagnostic images are written to ``PATH_FIGURES``:

- ``B{brick_id}_{band}_{imgtype}.png`` — brick-level science/model/residual images
- ``G{group_id}_B{brick_id}_{band}_{imgtype}.png`` — group-level cutouts
- ``G{group_id}_B{brick_id}_summary.png`` — per-group model summary plot

Ancillary files
~~~~~~~~~~~~~~~

Image products and DS9 region files are written to ``PATH_ANCILLARY``:

- ``B{brick_id}.fits`` — per-brick science/model/residual/chi extensions, one per band
- ``M{band}_model.fits``, ``M{band}_residual.fits``, ``M{band}_chi.fits`` — full-field
  images from ``rebuild_mosaic``
- ``B{brick_id}_detection_science_objects.reg`` — elliptical apertures for all detections
