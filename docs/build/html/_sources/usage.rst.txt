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
     - Sky position in degrees
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

Figures
~~~~~~~

If ``PLOT > 0``, diagnostic images are written to ``PATH_FIGURES``:

- ``B{brick_id}_{band}_{imgtype}.png`` — brick-level science/model/residual images
- ``G{group_id}_B{brick_id}_{band}_{imgtype}.png`` — group-level cutouts
- ``G{group_id}_B{brick_id}_summary.png`` — per-group model summary plot

Ancillary files
~~~~~~~~~~~~~~~

DS9 region files are written to ``PATH_ANCILLARY``:

- ``B{brick_id}_detection_science_objects.reg`` — elliptical apertures for all detections
