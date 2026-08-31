Configuration Reference
========================

All configuration lives in ``config/config.py``. The Farmer searches for this file first in the current working directory under ``config/``, then one level up. No environment variables or command-line flags are used — the config file is the single source of truth.

.. note::
   ``config.py`` is a plain Python module, so you can compute paths programmatically,
   import constants from elsewhere, or use ``os.environ`` for secrets.

General Controls
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``CONSOLE_LOGGING_LEVEL``
     - ``'DEBUG'``
     - Verbosity of console output. One of ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``, ``'CRITICAL'``.
   * - ``LOGFILE_LOGGING_LEVEL``
     - ``None``
     - Verbosity of the log file. Same options as above, or ``None`` to disable file logging.
   * - ``PLOT``
     - ``0``
     - Diagnostic plotting level. ``0`` = no plots; ``1`` = brick-level; ``2`` = group-level; ``3`` = verbose group-level; ``4`` = maximum.
   * - ``NCPUS``
     - ``0``
     - Number of CPU cores for parallel group processing. ``0`` = serial (recommended for debugging).
   * - ``OVERWRITE``
     - ``True``
     - If ``True``, existing output files are silently overwritten.
   * - ``OUTPUT``
     - ``True``
     - If ``True``, write output files to disk.
   * - ``AUTOLOAD``
     - ``True``
     - If ``True``, automatically load existing bricks from disk where possible.

Directory Paths
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``PATH_DATA``
     - Root data directory. All other paths are derived from this.
   * - ``PATH_BRICKS``
     - HDF5 brick files (``B{id}.h5``). Default: ``PATH_DATA/interim/bricks/``.
   * - ``PATH_PSFMODELS``
     - PSF model FITS files. Default: ``PATH_DATA/interim/psfmodels/``.
   * - ``PATH_FIGURES``
     - Diagnostic plots. Default: ``PATH_DATA/output/figures/``.
   * - ``PATH_CATALOGS``
     - Output source catalogs. Default: ``PATH_DATA/output/catalogs/``.
   * - ``PATH_ANCILLARY``
     - DS9 region files and other ancillary outputs. Default: ``PATH_DATA/output/ancillary/``.
   * - ``PATH_LOGS``
     - Log files. Default: ``PATH_DATA/interim/logs/``.

Detection Band
--------------

The detection image is configured separately from photometric bands:

.. code-block:: python

   DETECTION = {
       'science': '/path/to/detection_image.fits',
       'weight':  '/path/to/detection_weight.fits',  # optional
       'mask':    '/path/to/detection_mask.fits',     # optional
       'subtract_background': True,
       'backtype': 'flat',       # 'flat' or 'variable'
       'backregion': 'mosaic',   # 'mosaic' or 'brick'
       'name': 'Detection',      # display label
   }

   USE_DETECTION_WEIGHT = False   # use weight image for detection threshold
   USE_DETECTION_MASK   = False   # apply mask before detection
   APPLY_DETECTION_MASK = False   # remove masked sources after detection

Photometric Bands
-----------------

Each photometric band is a dictionary entry in ``BANDS``:

.. code-block:: python

   BANDS = {}
   BANDS['hsc_i'] = {
       'science':  '/path/to/hsc_i.fits',
       'weight':   '/path/to/hsc_i_weight.fits',   # optional but recommended
       'mask':     '/path/to/hsc_i_mask.fits',      # optional
       'psfmodel': '/path/to/psfmodels/hsc_i.fits', # required for photometry
       'subtract_background': True,
       'backtype': 'flat',
       'backregion': 'mosaic',
       'zeropoint': 31.4,   # AB magnitude zeropoint
       'name': r'HSC $i$',  # LaTeX-compatible display label
   }

Band dictionary keys:

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Key
     - Required
     - Description
   * - ``science``
     - Yes
     - Path to science image FITS file.
   * - ``weight``
     - No
     - Path to the weight image. See ``weight_type`` for what it is assumed to contain. If absent, a uniform weight is derived from the clipped image RMS and a warning is logged — uncertainties in that band are then only approximate.
   * - ``weight_type``
     - No
     - What the weight image actually holds: ``'invvar'`` (inverse variance, the default), ``'sigma'``, or ``'variance'``. Everything downstream assumes inverse variance, so declaring this wrongly rescales every uncertainty in the catalog with no other symptom. Converted once, at ingest.
   * - ``mask``
     - No
     - Path to mask image (non-zero = masked). If absent, no pixels are masked.
   * - ``psfmodel``
     - Yes
     - Path to a single PSF FITS file **or** path to a two-column ASCII/FITS table of ``RA``, ``DEC``, ``PSF_PATH`` for position-dependent PSFs.
   * - ``zeropoint``
     - Yes
     - AB magnitude zeropoint. Magnitudes are computed as ``mag = zeropoint − 2.5 × log10(flux)``.
   * - ``subtract_background``
     - No
     - If ``True``, subtract background before fitting. Default: ``False``.
   * - ``backtype``
     - No
     - Background model type: ``'flat'`` (scalar) or ``'variable'`` (2-D map). Default: ``'flat'``.
   * - ``backregion``
     - No
     - Region over which to estimate the background: ``'mosaic'`` (entire image, estimated once) or ``'brick'`` (re-estimated per brick). Default: ``'brick'``.
   * - ``name``
     - No
     - Human-readable label used in plot titles.
   * - ``extension``
     - No
     - FITS extension number (integer). Default: ``0``.

PSF Models
~~~~~~~~~~

Two PSF formats are supported:

1. **Constant PSF** — a single FITS image (PSFEx ``.fits`` format or plain stamp). All sources in the band share this PSF.

   .. code-block:: python

      'psfmodel': '/path/to/psfmodels/hsc_i.fits'

2. **Variable PSF** — an ASCII or FITS table with columns ``ra`` (degrees), ``dec`` (degrees), and ``psf_path`` (path to a per-position FITS file). The Farmer selects the nearest PSF for each source.

   .. code-block:: python

      'psfmodel': '/path/to/psfmodels/hsc_i_psflist.fits'

Use ``bin/prep_psf.py`` to clip, normalize, and optionally resample a raw PSF stamp before using it with The Farmer.

Source Detection Parameters
-----------------------------

These control the SEP source extraction step:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``BACK_BW``
     - ``32``
     - Background mesh box width (pixels).
   * - ``BACK_BH``
     - ``32``
     - Background mesh box height (pixels).
   * - ``BACK_FW``
     - ``2``
     - Background smoothing filter width (mesh cells).
   * - ``BACK_FH``
     - ``2``
     - Background smoothing filter height (mesh cells).
   * - ``USE_DETECTION_WEIGHT``
     - ``False``
     - Hand the weight map to SEP as a per-pixel noise array. This changes what ``THRESH`` means -- see below -- so the two must be set together.
   * - ``USE_DETECTION_MASK``
     - ``False``
     - Hand the mask to SEP, so masked pixels take no part in detection or deblending.
   * - ``APPLY_DETECTION_MASK``
     - ``False``
     - Flag sources whose centroid lands on a masked pixel *after* detection, in the ``masked`` column. Flagged sources are **kept** in the catalog with their full detection block and excluded from grouping, so they are never modelled and carry NaN photometry — see :ref:`masked-sources` below. Independent of ``USE_DETECTION_MASK``: setting this alone leaves detection and deblending untouched.
   * - ``THRESH``
     - ``1.5``
     - Detection threshold. If ``USE_DETECTION_WEIGHT=True``, this is in sigma units (relative). Otherwise, absolute image units.
   * - ``MINAREA``
     - ``5``
     - Minimum number of contiguous pixels above threshold for a valid detection.
   * - ``FILTER_KERNEL``
     - ``'gauss_2.0_5x5.conv'``
     - Convolution kernel filename (from ``config/conv_filters/``). See available kernels below.
   * - ``FILTER_TYPE``
     - ``'matched'``
     - Kernel type passed to SEP (``'matched'`` or ``'conv'``).
   * - ``DEBLEND_NTHRESH``
     - ``256``
     - Number of deblending thresholds (``2**8``).
   * - ``DEBLEND_CONT``
     - ``1e-10``
     - Minimum contrast ratio for deblending.
   * - ``CLEAN``
     - ``False``
     - If ``True``, apply SEP cleaning step.
   * - ``CLEAN_PARAM``
     - ``1.0``
     - Cleaning parameter passed to SEP.
   * - ``PIXSTACK_SIZE``
     - ``1000000``
     - SEP pixel stack size. Increase if detection fails on crowded fields.

Available convolution kernels (``config/conv_filters/``):

- ``block_3x3.conv`` — flat 3×3 top-hat
- ``default.conv``, ``default_3.0_7x7.conv`` — SExtractor default Gaussian
- ``gauss_1.5_3x3.conv``, ``gauss_2.0_5x5.conv``, ``gauss_3.0_5x5.conv``, ``gauss_3.0_7x7.conv``, ``gauss_4.0_7x7.conv``, ``gauss_5.0_9x9.conv`` — Gaussian kernels
- ``mexhat_*.conv`` — Mexican-hat (DoG) kernels at various scales
- ``tophat_*.conv`` — circular top-hat kernels

.. _masked-sources:

Masked sources
~~~~~~~~~~~~~~

``APPLY_DETECTION_MASK`` does not remove anything. A flagged source keeps its row,
its ``id``, its ``ra_det``/``dec_det`` and the whole SEP detection block (``npix``,
``tnpix``, ``flux``, ``cflux``, ``peak``, ``a``, ``b``, ``theta``, …). What it loses
is modelling: flagged sources are dropped before the dilation in
``dilate_and_group``, so they never enter a group, are never fitted, and every
photometric column — model fluxes and apertures alike — is NaN. They are marked
``group_id = 0`` and ``fit_status = 4``.

Dropping them *before* the dilation rather than after it matters: a masked source
that sits between two real ones would otherwise bridge them into a single group.
Excluding it at that point makes the groups, and therefore every fit, identical to
a run in which the source had never been detected.

Earlier versions deleted the rows outright, which removed 22–39 percent of
detections depending on the field, and did so before ``id`` was assigned — so the
survivors were renumbered to a dense 1..N and there was no way to reconcile a
catalog made under one mask with a catalog made under another. Keeping the rows
restores the choice of applying the mask, a different mask, or none, after the fact.

The ``masked`` column is one boolean derived from the combined mask, which is all
The Farmer needs to decide what to strip from groups. It is sampled at
``(round(x), round(y))`` — the same detection pixel ``ra_det``/``dec_det`` are
derived from — so a downstream tool that re-samples the separate star and edge
masks at ``ra_det``/``dec_det`` must reproduce it exactly:

.. code-block:: python

   assert (cat['masked'] == (cat['FLAG_STAR'] | cat['FLAG_EDGE'])).all()

Two independent derivations of the same fact, so a misregistered or stale mask
fails loudly instead of shipping.

Brick Parameters
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``N_BRICKS``
     - ``(2, 4)``
     - Number of bricks along (x, y) — total is the product. Brick IDs run 1 through ``N_BRICKS[0] × N_BRICKS[1]``.
   * - ``BRICK_BUFFER``
     - ``0.1 * u.arcmin``
     - Overlap region added around each brick to avoid edge effects. Sources detected in the buffer are discarded; photometry is measured through it.

Source Grouping
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``DILATION_RADIUS``
     - ``0.2 * u.arcsec``
     - Morphological dilation radius. Segments dilated by this amount are unioned to form groups.
   * - ``GROUP_BUFFER``
     - ``2 * u.arcsec``
     - Padding added around each group's bounding box when cutting out data for fitting.
   * - ``GROUP_SIZE_LIMIT``
     - ``5``
     - Maximum number of sources per group. Groups with more members are skipped.
   * - ``FORCE_SIMPLE_MAPPING``
     - ``False``
     - If ``True``, use a simplified (potentially less accurate) pixel-mapping method for multi-resolution segmentation.

Background Subtraction (Photometry)
--------------------------------------

Background subtraction for photometric band fitting uses a separate mesh:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``SUBTRACT_BW``
     - ``64``
     - Background mesh box width for photometry (pixels).
   * - ``SUBTRACT_BH``
     - ``64``
     - Background mesh box height for photometry (pixels).
   * - ``SUBTRACT_FW``
     - ``3``
     - Background smoothing filter width (mesh cells).
   * - ``SUBTRACT_FH``
     - ``3``
     - Background smoothing filter height (mesh cells).

Modeling and the Decision Tree
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``MODEL_BANDS``
     - ``['hsc_i', 'hsc_z', 'uvista_ks']``
     - Bands used jointly to determine morphological models.
   * - ``SUFFICIENT_THRESH``
     - ``1``
     - Chi-squared improvement threshold for accepting a more complex model.
   * - ``SIMPLEGALAXY_PENALTY``
     - ``0.1``
     - Extra chi-squared cost added when considering a SimpleGalaxy over a PointSource (discourages trivial point-source rejection).
   * - ``SIMPLEGALAXY_REFF``
     - ``0.45``
     - Fixed effective radius of the ``SimpleGalaxy`` model, in arcsec. The 0.45" default comes from the Legacy Survey SIMP model, tuned for roughly 1.2" ground-based seeing, and is too large for space-based data: for Euclid NISP (PSF FWHM 0.32-0.45") it exceeds the resolution limit in most bands, and 82% of fitted ``ExpGalaxy`` radii are smaller (CDFS median 0.296"). Use 0.25-0.30" for NISP and set it per field. Read once at import, so it is a per-run setting rather than a per-band one. It interacts with ``SIMPLEGALAXY_PENALTY`` and ``SUFFICIENT_THRESH``: shrinking it makes ``SimpleGalaxy`` more PSF-like, so sources migrate in *both* directions through the decision tree. Check the model mix and ``total_rchisq`` on one brick before adopting a new value.
   * - ``EXP_DEV_SIMILAR_THRESH``
     - ``0.1``
     - If the chi-squared difference between Exp and deV models is smaller than this, prefer the simpler model.
   * - ``RENORM_PSF``
     - ``None``
     - Rescale every PSF stamp so it sums to this value. ``None`` leaves the stamp as-is (the usual choice if ``prepare_psf`` already normalised it). Setting it to ``1.0`` folds whatever PSF flux lies *outside* the stamp into the fitted fluxes as an implicit aperture correction; that factor is logged once per band, stored on ``BaseImage.psf_aperture_correction``, and written to the output headers. Incompatible with PsfEx (``.psf``) models, which carry no pixel image.

Optimizer Settings
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``MAX_STEPS``
     - ``50``
     - Maximum iterations per optimization call.
   * - ``DAMPING``
     - ``0.1``
     - Levenberg–Marquardt damping factor. Larger values slow convergence but improve stability.
   * - ``DLNP_CRIT``
     - ``1e-3``
     - Convergence criterion: stop when the change in log-likelihood per step falls below this.
   * - ``GROUP_TIMEOUT``
     - ``None``
     - Maximum wall-clock seconds allowed per group. ``None`` disables the timeout.
   * - ``IGNORE_FAILURES``
     - ``True``
     - If ``True``, continue processing remaining groups after a failure. If ``False``, raise an exception.
   * - ``USE_CERES``
     - ``False``
     - If ``True``, use the Ceres Solver (requires the ``ceres`` Python binding). Falls back to The Tractor's built-in ``ConstrainedOptimizer`` otherwise.

.. note::

   A fit can stop moving because it converged, or because the line search ran into
   a parameter bound -- The Tractor's constrained optimizer reports ``dlnp = 0`` for
   both. Every fit therefore records two flags in the catalog: ``total_hit_limit``
   (some step hit a bound) and ``total_at_limit`` (the fit *ended* against one). A
   source with ``total_at_limit = 1`` has not converged in any meaningful sense; its
   shape is sitting on the edge of the range allowed in ``stage_models``. Both flags
   are group-wide, since the optimizer reports that some parameter in the joint fit
   hit a bound rather than which one.

Priors
-------

Priors control how model parameters are constrained during optimization. Two sets are defined: ``MODEL_PRIORS`` (used during model determination) and ``PHOT_PRIORS`` (used during photometry-only fitting).

.. code-block:: python

   MODEL_PRIORS = {
       'pos':     0.1 * u.arcsec,   # Gaussian prior on position offset
       'reff':    'none',            # No prior on effective radius
       'shape':   'none',            # No prior on ellipticity/PA
       'fracDev': 'none',            # No prior on bulge fraction
   }

   PHOT_PRIORS = {
       'pos':     0.001 * u.arcsec,  # Tight position prior during photometry
       'reff':    'freeze',           # Hold effective radius fixed
       'shape':   'freeze',           # Hold ellipticity/PA fixed
       'fracDev': 'freeze',           # Hold bulge fraction fixed
   }

Prior values:

- **``'none'``** — no prior; parameter is free.
- **``'freeze'``** — parameter is held fixed at its current value.
- **``astropy.Quantity`` (angle)** — Gaussian prior with the given standard deviation (angular units, e.g. ``0.1 * u.arcsec``). Only valid for the ``'pos'`` key.

Aperture Photometry
--------------------

Circular aperture photometry, measured on the science frames alongside the model
fits. It is off by default; setting ``DO_APERTURE_PHOT = True`` is the only switch
needed, and when it is ``False`` none of the other settings are read and no
aperture columns are written.

Apertures are measured on the same pixels The Tractor fits (NaN-zeroed, with the
background removed according to each band's ``subtract_background`` property), so
an aperture flux and a model flux for the same source share a zeropoint and are
directly comparable. Positions are the fitted model centroids where a model
exists and the detection centroids otherwise, so a source whose fit failed still
gets aperture measurements.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``DO_APERTURE_PHOT``
     - ``False``
     - Master switch for the whole aperture stage.
   * - ``APER_DIAMETERS``
     - ``[1.0, 2.0] * u.arcsec``
     - Fixed circular apertures, as **diameters** on the sky. May be empty.
   * - ``APER_PSF_FACTORS``
     - ``[2.0,]``
     - Apertures scaled to each band's PSF: diameter = factor x PSF FWHM.
   * - ``APER_REFF_FACTORS``
     - ``[2.0,]``
     - Apertures scaled to each source's fitted size: diameter = factor x reff.
   * - ``APER_IMGTYPES``
     - ``['science',]``
     - Image types to measure. Models and residuals work but cost a full extra pass per band.
   * - ``APER_SUBPIX``
     - ``5``
     - SEP sub-pixel sampling of the aperture edge; ``0`` uses the exact overlap area.

Output file and columns
~~~~~~~~~~~~~~~~~~~~~~~

Aperture measurements are written to their **own** FITS table,
``B{brick}_apertures.cat``, beside the main catalog rather than as extra columns
on it. A FITS binary table is capped at 999 columns (``TFIELDS`` is a three-digit
keyword), and each aperture costs eight columns per band: an 11-band run spends
~326 columns before apertures and runs out at eight of them. Keeping them
separate means a cross-check can never make the science catalog unwritable, and
apertures can be added without doing that arithmetic. The table carries ``id``,
``brick_id`` and the centroid each aperture was placed on (``aper_ra``,
``aper_dec``), so it joins to the main catalog on ``id`` and also stands alone.

Each aperture writes eight columns per band, named ``{band}_{tag}_{quantity}``
where ``quantity`` is one of ``flux``, ``flux_err``, ``flux_ujy``,
``flux_ujy_err``, ``mag``, ``mag_err``, ``diam`` (the aperture diameter actually
used, in arcsec) and ``flag`` (the SEP aperture flag; non-zero means the aperture
was truncated at an image edge or overlapped masked or zero-weight pixels). The
tag encodes the aperture: ``aper1as`` for a fixed 1 arcsec diameter, ``aperpsf2``
for two PSF FWHM, ``aperreff2`` for two effective radii. Column count is eight
per aperture per band, so trim the lists above on wide multi-band catalogs.

The effective radius used by ``APER_REFF_FACTORS`` is ``exp(logre)`` for a single
component model, the fixed ``SIMPLEGALAXY_REFF`` for a ``SimpleGalaxy``, and the
bulge-fraction weighted mean of the two components for a ``FixedCompositeGalaxy``.
A ``PointSource`` has no size, so its reff-scaled columns are ``NaN`` rather than
falling back to some other radius.

.. warning::

   Aperture uncertainties are the quadrature sum of the per-pixel variances
   inside the aperture, read from the band's inverse-variance weight map. On
   drizzled or otherwise resampled data the pixel-to-pixel noise is correlated,
   so these are underestimates; treat an aperture signal-to-noise on such
   products as an upper bound. Aperture fluxes are raw -- no aperture correction
   is applied.

Ancillary Map Controls
-----------------------

These parameters control which sources appear in the residual model images:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``RESIDUAL_BA_MIN``
     - ``0.01``
     - Minimum axis ratio for rendering a source in the residual map.
   * - ``RESIDUAL_REFF_MAX``
     - ``5 * u.arcsec``
     - Maximum effective radius for rendering a source in the residual map.
   * - ``RESIDUAL_SHOW_NEGATIVE``
     - ``False``
     - If ``True``, include sources with negative fitted fluxes in residual maps.
