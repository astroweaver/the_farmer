Pipeline Architecture
======================

This page describes the internal logic of The Farmer: how the mosaic is subdivided, how sources are grouped, and how the morphological decision tree works.

Overview
--------

The pipeline has four major stages:

1. **Brick construction** — the survey mosaic is tiled into a regular grid of overlapping bricks.
2. **Source detection** — within each brick, SEP finds sources in the detection image and assigns them to groups.
3. **Model determination** — each group is modeled jointly using a staged decision tree.
4. **Forced photometry** — morphologies are frozen and fluxes are measured in every photometric band.

Brick Grid
-----------

The mosaic is divided into ``N_BRICKS[0] × N_BRICKS[1]`` rectangular bricks. Brick IDs run from 1 to ``N_BRICKS[0] × N_BRICKS[1]`` in row-major order. Each brick has a small overlap region (``BRICK_BUFFER``) on all four sides:

- **Interior region**: the true brick footprint used for catalog output.
- **Buffered region**: a padded cutout used for fitting (sources near the edges see full profiles from their neighbors rather than truncated ones).

Sources detected in the buffer are discarded from the catalog. Their flux is still accounted for during fitting via their segment pixels.

.. code-block:: text

   ┌─────────────────────────────┐
   │  MOSAIC                     │
   │  ┌──────┬──────┬──────┬──┐  │
   │  │ B1   │ B2   │ B3   │B4│  │
   │  ├──────┼──────┼──────┼──┤  │
   │  │ B5   │ B6   │ B7   │B8│  │
   │  └──────┴──────┴──────┴──┘  │
   └─────────────────────────────┘

Source Detection
-----------------

Detection uses ``sep`` (the Python interface to Source Extractor) on the detection image. The steps are:

1. **Background estimation** — a 2-D background model is computed on a ``BACK_BW × BACK_BH`` mesh and smoothed with a ``BACK_FW × BACK_FH`` filter.
2. **Convolution** — the image is convolved with the kernel specified by ``FILTER_KERNEL``.
3. **Thresholding** — pixels exceeding ``THRESH`` (relative to RMS if ``USE_DETECTION_WEIGHT=True``, else absolute) in contiguous areas of at least ``MINAREA`` pixels are marked as detections.
4. **Deblending** — overlapping detections are split using ``DEBLEND_NTHRESH`` thresholds and minimum contrast ``DEBLEND_CONT``.
5. **Buffer masking** — sources whose centroids fall in the brick buffer zone are removed. Pixels belonging to their segments are masked.
6. **World coordinates** — pixel centroids are projected to RA/Dec via the WCS.

The output is an ``astropy.Table`` (the detection catalog) and a segmentation image (the ``segmap``).

Source Grouping
----------------

After detection, nearby sources are grouped using morphological dilation:

1. A binary mask is created from the segmentation map (any pixel belonging to any source = 1).
2. The mask is dilated with a circular kernel of radius ``DILATION_RADIUS`` (converted to pixels using the detection WCS).
3. Optionally, holes in the dilated mask are filled.
4. Connected components of the dilated mask are labeled.
5. Each original source segment is assigned the group ID of its overlapping connected component. If a segment overlaps more than one component (can happen for sources near group boundaries), union-find merges those components.

The result is a ``group_id`` column in the catalog and a ``groupmap`` image.

Modeling Pipeline
------------------

For each group, The Farmer runs a complete optimization pipeline. The group contains all images (detection plus all model bands), and models are fitted simultaneously across all model bands.

Initial Model Setup
~~~~~~~~~~~~~~~~~~~~

Each source in the group is initialized as a :class:`~tractor.PointSource` at its detected position. The detection segmentation map defines which pixels "belong" to each source for chi-squared accounting.

The Tractor Engine
~~~~~~~~~~~~~~~~~~~

The Farmer uses `The Tractor <https://github.com/dstndstn/tractor>`_ as its optimization backend. The Tractor models the observed image as:

.. math::

   I_{\rm model}(x, y) = \sum_k \left[ f_k \cdot (P * S_k)(x, y) \right] + B(x, y)

where :math:`f_k` is the flux of source :math:`k`, :math:`S_k` is its surface brightness profile, :math:`P` is the PSF, and :math:`B` is the background. The likelihood is:

.. math::

   \mathcal{L} = -\frac{1}{2} \sum_{x,y} \frac{\left[ I_{\rm obs}(x,y) - I_{\rm model}(x,y) \right]^2}{\sigma^2(x,y)}

where :math:`\sigma^2` is the inverse weight (variance) image. The optimizer minimizes :math:`-\ln \mathcal{L}` using damped least-squares (Levenberg–Marquardt), or optionally the Ceres Solver.

Decision Tree
~~~~~~~~~~~~~~

The decision tree determines the best-fit profile type for each source. It runs in stages:

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Stage
     - Model Type
     - Description
   * - 0
     - PointSource
     - Delta-function convolved with PSF. No free shape parameters.
   * - 1
     - SimpleGalaxy
     - Fixed exponential profile, effective radius = 0.45″, circular. Tests whether the source is at all resolved.
   * - 2
     - ExpGalaxy
     - Exponential disk with free effective radius, ellipticity, and position angle.
   * - 3
     - DevGalaxy
     - de Vaucouleurs (r^1/4) profile with free shape parameters.
   * - 4
     - FixedCompositeGalaxy (Comp)
     - Bulge + disk composite with a free bulge fraction ``fracDev``. If no model at stages 0–3 gives a significantly better fit than PointSource, the source remains a PointSource.
   * - 10 (final)
     - Winning model
     - Re-optimize the winning model with tighter position prior before photometry.
   * - 11 (phot)
     - Forced photometry
     - Morphology frozen via ``PHOT_PRIORS``; only fluxes (and optionally position within a tight prior) are free.

At each stage, the optimizer runs up to ``MAX_STEPS`` iterations. Convergence is declared when :math:`|\Delta \ln \mathcal{L}| < {\rm DLNP\_CRIT}`. The chi-squared statistic:

.. math::

   \chi^2 = \sum_{x,y} \frac{\left[ I_{\rm obs}(x,y) - I_{\rm model}(x,y) \right]^2}{\sigma^2(x,y)}

is computed for each source using its segmentation pixels.

Decision logic at each step:

- **PointSource → SimpleGalaxy**: if Δχ² > ``SIMPLEGALAXY_PENALTY``, try SimpleGalaxy.
- **SimpleGalaxy → ExpGalaxy**: if Δχ² > ``SUFFICIENT_THRESH``, allow free-shape Exp.
- **ExpGalaxy vs DevGalaxy**: both are tried; the better fit wins (ties go to Exp).
- **Exp/Dev → Composite**: if Δχ² > ``SUFFICIENT_THRESH`` for any source in the group, upgrade.
- **Fallback**: if a more complex model fails to converge or produces unphysical parameters, revert to the simpler one.

Positions are updated with a Gaussian prior (``MODEL_PRIORS['pos']``) to prevent large astrometric shifts. The prior is tightened between the modeling and photometry stages (``PHOT_PRIORS['pos']``).

Forced Photometry
~~~~~~~~~~~~~~~~~~

After model determination, morphological parameters (``reff``, ``ellip``, ``theta``, ``fracDev``) are frozen via the ``PHOT_PRIORS`` configuration. Only fluxes — and optionally positions with a tight Gaussian prior — are free. Each photometric band is then optimized in turn.

Parameter Uncertainty
~~~~~~~~~~~~~~~~~~~~~~

Uncertainties on all free parameters are extracted from the diagonal of the inverse Hessian (Fisher information matrix). This gives the formal 1-sigma error on each parameter assuming Gaussian noise, which is a lower bound on the true uncertainty.

Results Storage
~~~~~~~~~~~~~~~~

After processing, the group's results are "absorbed" back into the parent brick:

- ``brick.model_catalog`` — ``OrderedDict`` mapping source IDs to Tractor model objects.
- ``brick.model_tracker`` — convergence history (chi-squared per stage) for each source.
- ``brick.model_tracker_groups`` — group-level convergence history.

Multi-Resolution Support
--------------------------

The detection and photometric bands may have different pixel scales. When transferring segmentation maps from the detection grid to a photometric-band grid, The Farmer uses ``map_ids_to_coarse_pixels`` (in :mod:`farmer.utils`):

1. For each fine pixel (detection grid), the four corners are projected via WCS to coarse pixels (photometric-band grid).
2. The object ID of the fine pixel is assigned to all coarse pixels that overlap it.
3. The process is vectorized and runs in 10,000-pixel chunks to manage memory.

For very large resolution differences (e.g., detection at 0.15″/pix, IRAC at 0.6″/pix), the ``FORCE_SIMPLE_MAPPING`` option can accelerate this step at some precision cost.

HDF5 Persistence
-----------------

All brick state is serialized to HDF5 via ``h5py``. The hierarchy is:

.. code-block:: text

   B{brick_id}.h5
   ├── attributes               ← brick metadata (position, size, bands, config)
   ├── detection/
   │   ├── science              ← Cutout2D (data array + WCS + position)
   │   ├── weight
   │   ├── mask
   │   ├── background
   │   ├── rms
   │   ├── segmap
   │   ├── groupmap
   │   └── catalogs/
   │       └── science          ← astropy.Table
   ├── {band}/
   │   ├── science
   │   ├── weight
   │   ├── mask
   │   ├── background
   │   ├── psfcoords
   │   ├── psflist
   │   └── catalogs/
   │       └── science
   └── model_catalog/
       └── {source_id}/
           ├── name             ← model type string
           ├── ra, dec          ← sky coordinates
           ├── ra_err, dec_err  ← 1σ uncertainties
           ├── {band}_flux      ← fitted flux
           ├── {band}_flux_err
           ├── logre, reff      ← effective radius
           ├── ellip, theta     ← shape
           └── chisq, rchisq    ← goodness of fit

The serialization handles ``astropy.Table``, ``astropy.nddata.Cutout2D``, numpy arrays, Python scalars, and nested dicts. Custom types (``OrderedDict``, ``SkyCoord``, ``Quantity``) are reconstructed on load.

Memory Management
-----------------

For large surveys, memory is the primary constraint. The Farmer provides several mechanisms to limit footprint:

- **Lazy brick loading** — ``brick_has_band()`` reads only HDF5 metadata to check whether a band exists, without loading pixel arrays.
- **Generator-based group spawning** — groups are spawned one at a time and destroyed after absorption. The full list of groups is never held in memory simultaneously.
- **Cleanup methods** — ``cleanup_after_detection()`` and ``cleanup_after_modeling()`` delete pixel arrays that are no longer needed between stages.
- **Chunk-based WCS mapping** — the multi-resolution ID mapping processes 10,000 pixels at a time.
