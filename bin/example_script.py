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
