import os
import sys
import types

# Make farmer importable from the repo root
sys.path.insert(0, os.path.abspath('../..'))

# ---------------------------------------------------------------------------
# Minimal config mock
# Must come before any farmer import so function-default expressions resolve
# to real values instead of MagicMock reprs.
# ---------------------------------------------------------------------------
import astropy.units as u

_cfg = types.ModuleType('config')

# General
_cfg.CONSOLE_LOGGING_LEVEL = 'WARNING'
_cfg.LOGFILE_LOGGING_LEVEL = None
_cfg.PLOT = 0
_cfg.NCPUS = 0
_cfg.OVERWRITE = False
_cfg.OUTPUT = False
_cfg.AUTOLOAD = False

# Paths (dummies — never touched during a doc build)
_cfg.PATH_DATA        = '/tmp'
_cfg.PATH_BRICKS      = '/tmp'
_cfg.PATH_CATALOGS    = '/tmp'
_cfg.PATH_FIGURES     = '/tmp'
_cfg.PATH_ANCILLARY   = '/tmp'
_cfg.PATH_PSFMODELS   = '/tmp'
_cfg.PATH_LOGS        = '/tmp'

# Detection & bands
_cfg.DETECTION = {'name': 'Detection', 'subtract_background': False,
                  'backtype': 'flat', 'backregion': 'brick'}
_cfg.BANDS = {}

# Brick / group geometry
_cfg.N_BRICKS         = (1, 1)
_cfg.BRICK_BUFFER     = 0.1 * u.arcmin
_cfg.GROUP_BUFFER     = 2 * u.arcsec
_cfg.DILATION_RADIUS  = 0.2 * u.arcsec
_cfg.GROUP_SIZE_LIMIT = 5
_cfg.FORCE_SIMPLE_MAPPING = False

# Detection parameters
_cfg.THRESH           = 1.5
_cfg.MINAREA          = 5
_cfg.FILTER_KERNEL    = 'gauss_2.0_5x5.conv'
_cfg.FILTER_TYPE      = 'matched'
_cfg.DEBLEND_NTHRESH  = 256
_cfg.DEBLEND_CONT     = 1e-10
_cfg.BACK_BW = _cfg.BACK_BH = 32
_cfg.BACK_FW = _cfg.BACK_FH = 2
_cfg.SUBTRACT_BW = _cfg.SUBTRACT_BH = 64
_cfg.SUBTRACT_FW = _cfg.SUBTRACT_FH = 3
_cfg.PIXSTACK_SIZE    = 1_000_000
_cfg.USE_DETECTION_WEIGHT = False
_cfg.USE_DETECTION_MASK   = False
_cfg.APPLY_DETECTION_MASK = False
_cfg.CLEAN       = False
_cfg.CLEAN_PARAM = 1.0

# Modelling
_cfg.MODEL_BANDS          = []
_cfg.SUFFICIENT_THRESH    = 1
_cfg.SIMPLEGALAXY_PENALTY = 0.1
_cfg.EXP_DEV_SIMILAR_THRESH = 0.1
_cfg.RENORM_PSF           = 1
_cfg.MAX_STEPS            = 50
_cfg.DAMPING              = 0.1
_cfg.DLNP_CRIT            = 1e-3
_cfg.GROUP_TIMEOUT        = None
_cfg.IGNORE_FAILURES      = True
_cfg.USE_CERES            = False
_cfg.TIMEOUT              = 60

# Priors
_cfg.MODEL_PRIORS = {'pos': 0.1 * u.arcsec,
                     'reff': 'none', 'shape': 'none', 'fracDev': 'none'}
_cfg.PHOT_PRIORS  = {'pos': 0.001 * u.arcsec,
                     'reff': 'freeze', 'shape': 'freeze', 'fracDev': 'freeze'}

# Residual maps
_cfg.RESIDUAL_BA_MIN       = 0.01
_cfg.RESIDUAL_REFF_MAX     = 5 * u.arcsec
_cfg.RESIDUAL_SHOW_NEGATIVE = False

sys.modules['config'] = _cfg

# ---------------------------------------------------------------------------
# Mock C extensions / packages not available in the RTD environment.
# Use autodoc_mock_imports (not manual sys.modules) so Sphinx creates
# _MockObject instances that can be used as base classes in `class Foo(Bar)`.
# ---------------------------------------------------------------------------

project   = 'The Farmer'
copyright = '2018-2026, John Weaver'
author    = 'John Weaver'
release   = '2.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
]

autodoc_mock_imports = [
    'tractor', 'tractor.psfex', 'tractor.galaxy', 'tractor.ellipses', 'tractor.wcs',
    'tractor.pointsource', 'tractor.sercore', 'tractor.sersic',
    'astrometry', 'astrometry.util', 'astrometry.util.util',
    'sep', 'pathos', 'pathos.pools',
    'reproject', 'regions', 'h5py',
    'scipy', 'scipy.stats', 'scipy.ndimage',
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.colors',
    'matplotlib.cm', 'matplotlib.patches', 'matplotlib.ticker',
    'matplotlib.backends', 'matplotlib.backends.backend_pdf',
    'tqdm',
]

autodoc_default_options = {
    'members':         True,
    'undoc-members':   False,
    'show-inheritance': True,
}
autodoc_member_order   = 'bysource'
autodoc_preserve_defaults = True   # show conf.X names instead of evaluated reprs

templates_path   = ['_templates']
exclude_patterns = []

html_theme       = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
}

intersphinx_mapping = {
    'python':  ('https://docs.python.org/3/', None),
    'numpy':   ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

napoleon_google_docstring      = True
napoleon_numpy_docstring       = True
napoleon_include_init_with_doc = True
