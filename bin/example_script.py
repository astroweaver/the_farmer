"""End-to-end example run for The Farmer.

Run this script from the directory that contains your config/ folder:

    cd /path/to/your/project
    python bin/example_script.py

Edit the CONFIGURATION block below to match your survey before running.
"""

import farmer

# ---------------------------------------------------------------------------
# CONFIGURATION
# Edit these to match your N_BRICKS setting in config.py.
# ---------------------------------------------------------------------------

N_BRICKS_X = 2   # conf.N_BRICKS[0]
N_BRICKS_Y = 4   # conf.N_BRICKS[1]

# ---------------------------------------------------------------------------
# STEP 0 — Validate
# Checks that all science images, weight maps, and PSF models listed in
# config.py can be found on disk and have readable WCS headers.
# Fix any reported errors before proceeding.
# ---------------------------------------------------------------------------

farmer.validate()

# ---------------------------------------------------------------------------
# STEP 1 — Build the central brick (for testing)
# Rather than processing the full grid on your first run, build only the
# central brick so you can inspect it quickly.  Brick IDs are assigned in
# row-major order starting from 1.
# ---------------------------------------------------------------------------

central_id = (N_BRICKS_Y // 2) * N_BRICKS_X + (N_BRICKS_X // 2) + 1
print(f"Building central brick (ID {central_id}) ...")
brick = farmer.build_bricks(brick_ids=central_id)

# To build the entire grid instead, just call:
#   good_bricks = farmer.build_bricks()
# This returns a list of brick IDs that had valid detection-band data.

# ---------------------------------------------------------------------------
# STEP 2 — Detect sources
# Runs SEP (SourceExtractor Python) on the detection image inside the brick.
# The segmentation map and source catalog are written into the HDF5 file.
# ---------------------------------------------------------------------------

print("Detecting sources ...")
farmer.detect_sources(brick_ids=central_id, write=True)

# After detection you can inspect the raw catalog:
#   brick = farmer.load_brick(central_id)
#   print(brick.catalogs['detection']['science'])

# ---------------------------------------------------------------------------
# STEP 3 — Determine morphological models
# Runs the staged decision tree (PointSource → SimpleGalaxy → Exp → deV →
# Composite) on the MODEL_BANDS defined in config.py.  Results are stored
# in brick.model_catalog and written to the HDF5 file.
# ---------------------------------------------------------------------------

print("Determining models ...")
farmer.generate_models(brick_ids=central_id)

# ---------------------------------------------------------------------------
# STEP 4 — Forced photometry
# Freezes the morphological parameters found above and optimizes only fluxes
# (and optionally positions within a tight prior) in every configured band.
# The final FITS catalog is written to PATH_CATALOGS.
# ---------------------------------------------------------------------------

print("Running forced photometry ...")
farmer.photometer(brick_ids=central_id)

# ---------------------------------------------------------------------------
# STEP 4b — Alternative photometry at a coarser deblending scale (optional)
# Re-group the sources with a LARGER dilation radius and force photometry on a
# different set of bands -- the classic case is deblending IRAC against the
# morphological priors fitted above, where a group built at the detection
# resolution is far too small for a ~2" PSF.
#
# Everything is passed explicitly, not through config.py, so this block can be
# repeated with different radii, bands, and tags for as many alternative
# photometry sets as needed.
#
# Rules of the road:
#   * Run this only AFTER the normal photometer() call above has written its
#     catalog. The photometry pass rebuilds each model's flux set for just the
#     requested bands, so the in-memory models afterwards carry ONLY the
#     alternative bands -- hence the snapshot below.
#   * Consider excluding the alternative bands from the normal pass with
#     farmer.photometer(brick_ids=..., bands=[...]): a small-group IRAC flux is
#     exactly the contaminated measurement this pass exists to replace.
#   * exclude_unmodelled=True de-groups sources without a fitted model BEFORE
#     dilation (judged by the brick's persisted fit_status records, so it works
#     on a brick loaded from disk). This keeps their segments from seeding,
#     extending, or bridging any group footprint, and keeps groups from failing
#     on members that have no model to fit. It does NOT remove their flux from
#     footprints they overlap -- an unmodelled source close to a modelled one
#     still contributes unfitted light there. In the tagged catalog they carry
#     group_id = 0 and NaN photometry; group_id == 0 is the reliable marker of
#     a de-grouped source.
#   * group_size_limit: the small config default protects the model decision
#     tree, which is the expensive stage. Forced photometry with all-frozen
#     priors fits fluxes only (a linear solve), so re-grouped groups can be
#     large -- pass an effectively unlimited override. The limit actually used
#     is stamped into the catalog header as GRPLIM.
#   * Priors are PINNED into the brick when it is first built, so set
#     brick.phot_priors explicitly here rather than relying on config.py.
#   * identify_groups(overwrite=True) invalidates every band's segmap/groupmap
#     (they hold the old grouping's ids), so transfer_maps(overwrite=True) must
#     run before photometry -- as below.
#   * The tag writes a SEPARATE catalog file (B{id}_irac.cat), which keeps the
#     two deblendings unambiguous and avoids overwriting the fit_status and
#     chi-squared columns of the normal pass ON DISK. The in-memory brick is
#     another matter: the alt pass rewrites its model_tracker, fit_status, and
#     live catalog, which the snapshot below does not cover. After this block,
#     RELOAD the brick before doing anything else with it, and never write_hdf5
#     or write an untagged catalog from the object that ran the alt pass.
# ---------------------------------------------------------------------------

# import copy
# import astropy.units as u
#
# alt_bands = ['irac_ch1', 'irac_ch2']       # must be configured in BANDS
# brick = farmer.load_brick(central_id)
# snapshot = copy.deepcopy(brick.model_catalog)
# brick.phot_priors = {'pos': 'freeze', 'reff': 'freeze',
#                      'shape': 'freeze', 'fracDev': 'freeze'}
#
# brick.identify_groups(radius=3.0 * u.arcsec, overwrite=True,
#                       exclude_unmodelled=True)
# brick.transfer_maps(bands=alt_bands, overwrite=True)
# brick.process_groups(mode='photometry', bands=list(alt_bands),
#                      group_size_limit=10**9)
# brick.write_catalog(tag='irac', allow_update=True)
#
# brick.model_catalog = snapshot             # undo the alt-band-only flux sets
# brick = farmer.load_brick(central_id)      # then start clean: the alt pass
#                                            # also rewrote model_tracker,
#                                            # fit_status and the live catalog

# ---------------------------------------------------------------------------
# STEP 5 — Inspect results
# Load the finished brick and print a summary table of per-band statistics.
# ---------------------------------------------------------------------------

brick = farmer.load_brick(central_id)
brick.summary()

# Access the source catalog as an astropy Table:
#   from astropy.io import fits
#   from astropy.table import Table
#   cat = Table.read(f'data/output/catalogs/B{central_id}_catalog.fits')
#   print(cat['ra', 'dec', 'hsc_i_flux', 'hsc_i_mag', 'rchisq'])

# ---------------------------------------------------------------------------
# STEP 6 — Debug a single group interactively
# quick_group loads the brick, spawns the group, runs detection, and returns
# a ready-to-model Group object.  Set PLOT >= 2 in config.py to generate
# cutout images at each decision-tree stage.
# ---------------------------------------------------------------------------

# Replace 42 with any group_id from the detection catalog.
group = farmer.quick_group(brick_id=central_id, group_id=42)
group.summary()

# Run the full modeling + photometry pipeline on this group and save plots:
#   group.farm()

# Or step through manually for detailed inspection:
#   group.determine_models()   # run decision tree
#   group.force_models()       # forced photometry
#   group.plot_summary()       # save diagnostic plot

# ---------------------------------------------------------------------------
# FULL PRODUCTION RUN
# Once you are happy with the single-brick result, process the full grid:
# ---------------------------------------------------------------------------

# good_bricks = farmer.build_bricks()
# farmer.detect_sources(brick_ids=good_bricks)
# farmer.generate_models(brick_ids=good_bricks)
# farmer.photometer(brick_ids=good_bricks)
