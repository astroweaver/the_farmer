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
   * - ``MIN_STEPS``
     - ``3``
     - Minimum optimiser steps before ``DLNP_CRIT`` is allowed to end a fit. A model that barely moves on its first step is not necessarily converged -- it may simply have been handed a stationary starting point.
   * - ``MIN_STEPS_COMPOSITE``
     - ``5``
     - As ``MIN_STEPS``, but applied when the group contains a ``FixedCompositeGalaxy``, which carries twice the shape parameters of an exponential or de Vaucouleurs model.
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
