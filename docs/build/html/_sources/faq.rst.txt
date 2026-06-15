Frequently Asked Questions
===========================

.. _faq-config:

Setup & Configuration
---------------------

**Q: I get** ``RuntimeError: Cannot find configuration file!`` **when I import farmer.**

The Farmer searches for ``config/config.py`` relative to your current working directory. Make sure you are running Python from the directory that *contains* ``config/`` (not from inside it). You can verify with:

.. code-block:: python

   import os
   print(os.getcwd())        # should end in your project root, not in config/
   print(os.path.exists('config/config.py'))   # should print True

If you want to run from a different directory, add ``sys.path.insert(0, '/path/to/your/config')`` before importing farmer.

----

**Q: How do I add a new band without rebuilding everything from scratch?**

Use :func:`farmer.update_bricks`. It reads only the HDF5 metadata to check whether a brick already has the band, loading full pixel data only for bricks that need updating:

.. code-block:: python

   farmer.update_bricks(bands=['irac_ch1'])

After updating, re-run :func:`farmer.photometer` to measure fluxes in the new band. You do not need to re-run detection or model determination.

----

**Q: Can band names contain dots, slashes, or spaces?**

No. Band names must not contain a ``'.'`` character (this would break HDF5 key paths). Underscores are fine: ``'hsc_i'``, ``'uvista_ks'``. Spaces are fine in the ``'name'`` display label but not in the dict key.

----

**Q: The code cannot find my PSF file. What format is required?**

The Farmer uses PSFEx-format FITS files via ``tractor.psfex.PixelizedPsfEx``. If you have a plain PSF stamp image (not a PSFEx output), you can still use it via ``tractor.psfex.PixelizedPSF``, but you may need to set the appropriate header keywords.

Use ``bin/prep_psf.py`` to:

- Subtract any residual background from the PSF stamp
- Clip the outer low-signal wings to a target radius
- Normalize the PSF to unit sum
- Optionally resample to a target pixel scale

Run it with ``--help`` for all options.

----

**Q: My detection image has a different pixel scale from my photometric bands. Is that supported?**

Yes. The Farmer re-maps the detection segmentation map to each photometric band's pixel grid using WCS-based coordinate transformations. The mapping is handled by :func:`~farmer.utils.map_ids_to_coarse_pixels`. You may need to set ``FORCE_SIMPLE_MAPPING = True`` if the resolution difference is extreme and the WCS transformation is slow.

----

**Q: I have a FITS file with the science image in extension 1 (not 0). How do I tell The Farmer?**

Add ``'extension': 1`` to the band dictionary in ``config.py``:

.. code-block:: python

   BANDS['my_band'] = {
       'science': '/path/to/file.fits',
       'extension': 1,
       ...
   }

----

Detection & Grouping
---------------------

**Q: The Farmer detects far too many (or too few) sources. What should I adjust?**

Detection sensitivity is primarily controlled by ``THRESH``. Lower values detect fainter sources but increase spurious detections. If using a weight image (``USE_DETECTION_WEIGHT = True``), the threshold is in units of local sigma; otherwise it is an absolute image value.

Other parameters to tune:

- ``MINAREA``: increase to suppress noise detections (requires more contiguous pixels).
- ``FILTER_KERNEL``: a broader Gaussian (e.g., ``gauss_3.0_7x7.conv``) reduces noise sensitivity.
- ``DEBLEND_NTHRESH`` / ``DEBLEND_CONT``: increase ``DEBLEND_CONT`` to merge aggressively blended sources.

----

**Q: I have very crowded regions where groups end up with more than 5 sources. Can I increase the limit?**

Yes, via ``GROUP_SIZE_LIMIT``:

.. code-block:: python

   GROUP_SIZE_LIMIT = 10   # allow up to 10 sources per group

Be aware that fitting time scales super-linearly with group size, and the optimizer becomes less stable for large groups. The default of 5 is a practical compromise. Groups exceeding the limit are flagged and skipped.

----

**Q: Some of my sources are missing from the output catalog. Why?**

Common reasons:

1. **Detected in the buffer zone** — sources whose centroids fall in the ``BRICK_BUFFER`` region are removed.
2. **Group size exceeded** — groups with more than ``GROUP_SIZE_LIMIT`` sources are skipped entirely.
3. **Optimizer timeout** — if ``GROUP_TIMEOUT`` is set and a group takes too long, it is skipped.
4. **Rejected group** — a group is marked rejected if its bounding box falls outside the brick boundary after dilation (an edge-case of extreme dilation).
5. **Negative flux** — sources with fitted fluxes below zero are retained in the catalog but may be excluded from some output products depending on ``RESIDUAL_SHOW_NEGATIVE``.

Check the log file for ``WARNING`` messages about skipped groups.

----

**Q: What is the** ``group_id`` **column and why does it start at non-consecutive numbers?**

The ``group_id`` is assigned by the connected-component labeling step after dilation. It is a positive integer but not necessarily contiguous (labels are assigned from the component labeling algorithm, which skips indices for small regions filtered out during cleanup). The ``group_pop`` column gives the number of sources in each group.

----

Modeling & Photometry
----------------------

**Q: The optimizer produces negative fluxes. Is that a problem?**

Negative fluxes arise when the profile model overcorrects for the background or when a source's flux is genuinely consistent with zero (non-detection). The Farmer does not clamp fluxes to positive values — negative values carry statistical information (they indicate the source is not detected at that location). Treat them as upper limits in your science analysis.

----

**Q: How do I know which model type was assigned to a source?**

Check the ``name`` field in ``brick.model_catalog[source_id]``. Values are:

- ``'PointSource'``
- ``'SimpleGalaxy'``
- ``'ExpGalaxy'``
- ``'DevGalaxy'``
- ``'FixedCompositeGalaxy'``

In the output FITS catalog, morphological columns (``logre``, ``reff``, ``ellip``, etc.) are ``-99`` for point sources.

----

**Q: Can I force all sources to be treated as point sources?**

Yes. In your config, set the decision tree to stop after stage 0 by effectively making the transition cost very high:

.. code-block:: python

   SIMPLEGALAXY_PENALTY = 1e10
   SUFFICIENT_THRESH    = 1e10

Alternatively, call ``brick.process_groups(mode='photometry')`` on a brick where all sources are already initialized as PointSource models.

----

**Q: Can I use Ceres Solver instead of the default optimizer?**

Yes, if you have the Ceres Python bindings installed:

.. code-block:: python

   USE_CERES = True

The Ceres Solver is generally faster on large groups but requires the optional ``ceres`` package. If it is not available or encounters a fatal error, The Farmer logs the failure and falls back to the constrained optimizer. Check the log for ``CERES`` messages.

----

**Q: A group is taking forever. Can I set a timeout?**

Yes:

.. code-block:: python

   GROUP_TIMEOUT = 120   # 2 minutes per group

Groups that exceed the timeout are skipped and logged at ``WARNING`` level. Set ``IGNORE_FAILURES = True`` (the default) to continue processing remaining groups.

.. note::
   Timeout protection uses ``signal.alarm``, which is **Unix-only** (Linux and macOS).
   On Windows, ``GROUP_TIMEOUT`` is parsed but ignored — timeouts will not fire.

----

**Q: I want to fit in only one band. How?**

Set ``MODEL_BANDS`` to a list with just that band:

.. code-block:: python

   MODEL_BANDS = ['hsc_i']

All sources will be modeled using only the HSC i-band image. Fluxes in other bands are measured with frozen morphologies during the photometry stage.

----

**Q: How are position uncertainties computed?**

The Farmer extracts parameter covariances from the diagonal of the inverse Hessian of the log-likelihood at the optimum. These are formal 1σ uncertainties assuming Gaussian noise and a well-determined minimum — they do not account for PSF model errors, background estimation errors, or other systematic effects.

----

Performance
-----------

**Q: The run is much slower than expected. What are the bottlenecks?**

The main performance levers are:

1. **Parallelism** — set ``NCPUS > 0`` to use multiple cores. Typical speedup is near-linear up to ~8 cores; communication overhead limits gains beyond that.
2. **Group size** — fitting a group of 5 sources in 3 bands is much slower than a lone point source. Reduce ``GROUP_SIZE_LIMIT`` or ``DILATION_RADIUS`` to create smaller groups.
3. **Number of model bands** — each additional band in ``MODEL_BANDS`` increases the per-step cost roughly linearly.
4. **Optimizer iterations** — reduce ``MAX_STEPS`` or increase ``DLNP_CRIT`` for faster (but potentially less well-converged) results.
5. **IRAC / low-resolution bands** — the WCS reprojection step for coarse-pixel bands can be slow. ``FORCE_SIMPLE_MAPPING = True`` accelerates it.

----

**Q: How much memory does The Farmer use?**

Memory usage scales with:

- Image size (width × height × bytes per pixel × number of bands)
- Number of bricks held in memory simultaneously (normally 1)
- Number of parallel workers × group cutout size

A single brick with 4 bands at 0.15″/pixel and a 10′ × 10′ footprint uses roughly 500 MB. Use ``detect_sources_lite(cleanup=True)`` or call ``brick.cleanup_after_detection()`` / ``brick.cleanup_after_modeling()`` to free memory between stages.

----

Output & Catalogs
------------------

**Q: What magnitude system does The Farmer use?**

AB magnitudes, computed as:

.. math::

   m_{\rm AB} = {\rm zeropoint} - 2.5 \log_{10}(f)

where ``f`` is the fitted flux in the native image units. The zeropoint is set per band via the ``'zeropoint'`` key in the band configuration.

----

**Q: How do I combine catalogs from multiple bricks?**

Stack the FITS tables with ``astropy.table.vstack``:

.. code-block:: python

   from astropy.table import vstack
   from astropy.io.fits import open as fits_open
   import glob

   catalogs = [Table.read(f) for f in sorted(glob.glob('data/output/catalogs/B*.fits'))]
   full_catalog = vstack(catalogs)
   full_catalog.write('full_catalog.fits', overwrite=True)

Source IDs are unique within a brick but may repeat across bricks. Add a composite key ``brick_id * 1000000 + id`` for a globally unique identifier.

----

**Q: I want to use the FITS catalogs in DS9. Is there a region file?**

Yes. After detection, The Farmer writes a DS9 region file to ``PATH_ANCILLARY/B{brick_id}_detection_science_objects.reg``. Load it in DS9 with ``Region → Load Region``.

----

Citing The Farmer
------------------

**Q: What should I cite when using The Farmer?**

Please cite `Weaver et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023arXiv231007757W/abstract>`_ (the algorithm paper) and `Lang et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016ascl.soft04008L/abstract>`_ (The Tractor).

If your work uses COSMOS2020 photometry, also cite `Weaver et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022ApJS..258...11W/abstract>`_.

----

**Q: The GitHub repository has a "TheFarmer_v1" branch. Which version should I use?**

Use the ``master`` branch (this version, v2.x). Version 1 corresponds to the code described in the Weaver et al. 2023 paper and is kept for reproducibility only. Version 2 includes significant performance improvements, better memory management, parallel processing, and expanded model support.

----

Output & Interpretation
-----------------------

**Q: My reduced chi-squared values are much greater than 1. What does that mean?**

Large reduced chi-squared (``rchisq`` ≫ 1) can indicate:

- The fitted profile is a poor match to the source morphology (e.g., a strongly irregular galaxy forced into an exponential profile).
- The PSF model is inaccurate — even a small PSF error dominates for bright compact sources.
- Neighboring sources were inadequately deblended, leaving residual flux in the segment.
- The weight map underestimates the true noise (e.g., correlated noise from drizzling).

``rchisq`` ≲ 2 is generally acceptable. Sources with ``rchisq`` > 10 should be treated with caution.

----

**Q: What does** ``flag = 1`` **mean in the output catalog?**

The flag column records fitting issues. A non-zero value indicates the group was skipped, the optimizer did not converge, or the source was assigned a fallback model. Check ``model_tracker`` for the per-stage chi-squared history to diagnose convergence problems.

----

**Q: Fluxes in one band look systematically wrong. How do I diagnose this?**

1. Check ``PLOT > 1`` to generate group-level residual images and visually inspect the fit.
2. Verify the zeropoint: fluxes are in native image units; check the ``'zeropoint'`` key.
3. Verify the PSF model matches the actual image PSF — a mismatch biases fluxes for compact sources.
4. Check that background subtraction is appropriate: ``backtype = 'flat'`` with ``backregion = 'mosaic'`` works for most ground-based images, but extended emission can require a per-brick background.

----

Getting Help
------------

**Q: Where should I report bugs?**

Open an issue at https://github.com/astroweaver/farmer/issues. Please include the full traceback, your Python and package versions, and a minimal reproducer if possible.

**Q: I want to use The Farmer for my survey. Who should I contact?**

Email john.weaver.astro@gmail.com. We are happy to help you adapt the configuration to your specific data and suggest PSF preparation strategies.
