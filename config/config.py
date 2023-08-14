import os
import astropy.units as u

# General Controls
CONSOLE_LOGGING_LEVEL = 'DEBUG'																# VERBOSE console level ('INFO', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
LOGFILE_LOGGING_LEVEL = None																	# VERBOSE logfile level (same options, but can also be None)
PLOT = 0																	# Plot level (0 to 3)
NCPUS = 0															# Number of threads to run on (0 is serial)
OVERWRITE = True																				# Overwrite existing files without warning?
OUTPUT = True
AUTOLOAD = True

# Directory Structure
PATH_DATA = '/Users/jweaver/Projects/Software/the_farmer/data/'
PATH_BRICKS = os.path.join(PATH_DATA, 'interim/bricks')
PATH_FIGURES = os.path.join(PATH_DATA, 'output/figures')
PATH_PSFMODELS = os.path.join(PATH_DATA, 'interim/psfmodels')
PATH_CATALOGS = os.path.join(PATH_DATA, 'output/catalogs')
PATH_ANCILLARY = os.path.join(PATH_DATA, 'output/ancillary')
PATH_LOGS = os.path.join(PATH_DATA, 'interim/logs')

# Image Configuration
BANDS = {}
BANDS['hsc_i'] = {
    'science': os.path.join(PATH_DATA, 'external/COSMOS_Cgalsim_hsc_i_22626PS.fits'),
    'weight': os.path.join(PATH_DATA, 'external/COSMOS_Cgalsim_hsc_i_noise_WEIGHT.fits'),
    'psfmodel': os.path.join(PATH_PSFMODELS, 'hsc_i.fits'), # can either be a PSF file, or a table of ra/dec/psf_path
    'subtract_background': True, 
    'backtype': 'flat',
    'backregion': 'mosaic',
    'name': r'HSC $i$',
    'zeropoint': 31.4,
}
BANDS['hsc_z'] = {
    'science': os.path.join(PATH_DATA, 'external/COSMOS_Cgalsim_hsc_z_22626PS.fits'),
    'weight': os.path.join(PATH_DATA, 'external/COSMOS_Cgalsim_hsc_z_noise_WEIGHT.fits'),
    'psfmodel': os.path.join(PATH_PSFMODELS, 'hsc_z.fits'),
    'subtract_background': True, 
    'backtype': 'flat',
    'backregion': 'mosaic',
    'name': r'HSC $z$',
    'zeropoint': 31.4,
}
BANDS['uvista_ks'] = {
    'science': os.path.join(PATH_DATA, 'external/COSMOS_Cgalsim_uvista_ks_22626PS.fits'),
    'weight': os.path.join(PATH_DATA, 'external/COSMOS_Cgalsim_uvista_ks_noise_WEIGHT.fits'),
    'psfmodel': os.path.join(PATH_PSFMODELS, 'uvista_ks.fits'),
    'subtract_background': True, 
    'backtype': 'flat',
    'backregion': 'mosaic',
    'name': r'UVISTA $K_s$',
    'zeropoint': 31.4,
}
BANDS['irac_ch1'] = {
    'science': os.path.join(PATH_DATA, 'external/COSMOS_Cgalsim_irac_ch1.fits'),
    'weight': os.path.join(PATH_DATA, 'external/COSMOS_Cgalsim_irac_ch1_noise.fits'),
    'psfmodel': os.path.join(PATH_PSFMODELS, 'irac_ch1.fits'),
    'backtype': 'flat',
    'backregion': 'mosaic',
    'subtract_background': True, 
    'name': 'IRAC ch1',
    'zeropoint': 23.94,
}

# Source Detection
DETECTION = {
    'science': os.path.join(PATH_DATA, 'external/COSMOS_Cgalsim_izKs_CHIMEAN.fits'),
    'backtype': 'flat',
    'backregion': 'mosaic',
    'subtract_background': True, 
}
USE_DETECTION_WEIGHT = False   # uses the weight -- thresh is then relative!
USE_DETECTION_MASK = False     # applies mask before detection
APPLY_DETECTION_MASK = False    # applies mask after detection, removes sources and segments
DETECT_BW = 32																					# Detection mesh box width
DETECT_BH = 32																					# Detection mesh box height
DETECT_FW = 2																					# Detection filter box width
DETECT_FH = 2																					# Detection filter box height
THRESH = 1.5																					# If weight is used, this is a relative threshold in sigma. If not, then absolute.
MINAREA = 5
CLEAN = False
CLEAN_PARAM = 1.0																				# Minumum contiguous area above threshold required for detection
FILTER_KERNEL = 'gauss_2.0_5x5.conv'															# Detection convolution kernel
FILTER_TYPE = 'matched' 																		# Detection convolution kernel type
DEBLEND_NTHRESH = 2**8																		# Debelnding n-threshold
DEBLEND_CONT = 1E-10																		# Deblending continuous threshold
PIXSTACK_SIZE = 1000000	

# Bricks
N_BRICKS = (2, 4) # Number of bricks along x, y
BRICK_BUFFER = 0.1 * u.arcmin

# Background Subtraction for Photometry
SUBTRACT_BW = 64																				# Background mesh box width
SUBTRACT_BH = 64																				# Background mesh box height
SUBTRACT_FW = 3																					# Background filter box width
SUBTRACT_FH = 3	

# Source Groups
GROUP_BUFFER = 2 * u.arcsec
DILATION_RADIUS = 0.2 * u.arcsec
GROUP_SIZE_LIMIT = 5

# Modeling and the Decision Tree
MODEL_BANDS = ['hsc_i', 'hsc_z', 'uvista_ks']
SUFFICIENT_THRESH = 1
SIMPLEGALAXY_PENALTY = 0.1
EXP_DEV_SIMILAR_THRESH = 0.1

# Engine
MAX_STEPS = 50
DAMPING = 1e-1

# Priors and parameters
MODEL_PRIORS = {  'pos':  0.1 * u.arcsec,
                'reff': None,
                'shape': None,
                'fracDev': None,
}
PHOT_PRIORS = { 'pos':   0.001 * u.arcsec,
                'reff': 'freeze',
                'shape': 'freeze',
                'fracDev': 'freeze'
}

# Ancillary Maps: Models, Chi, Residuals, Effective Areas
