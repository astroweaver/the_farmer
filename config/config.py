#### THE FARMER v1.3 #####
#   CONFIGURATION FILE   #
##########################


#### GENERAL FUNCTIONALITY #####
CONSOLE_LOGGING_LEVEL = 'DEBUG'																# VERBOSE console level ('INFO', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
LOGFILE_LOGGING_LEVEL = None																	# VERBOSE logfile level (same options, but can also be None)
PLOT = 0																		# Plot level (0 to 3)
NTHREADS = 8															# Number of threads to run on (0 is serial)
OVERWRITE = True																				# Overwrite existing files without warning?
USE_CERES = False
OUTPUT = True

##### FILE LOCATION #####
WORKING_DIR = '/Users/jweaver/Projects/Current/Farmer_GalSim/'
IMAGE_DIR = WORKING_DIR + 'data/images'										# INPUT Raw images, weights, and masks
PSF_DIR = WORKING_DIR + 'data/intermediate/psfmodels'							# INTERMEDIATE Point-spread function files
BRICK_DIR = WORKING_DIR + 'data/intermediate/bricks'							# INTERMEDIATE Bricks (i.e. subimages)
INTERIM_DIR = WORKING_DIR + 'data/intermediate/interim'						# OUTPUT Other stuff I make 		
PLOT_DIR = WORKING_DIR + 'data/intermediate/plots'							# OUTPUT Figures
CATALOG_DIR = WORKING_DIR + 'data/output/catalogs'					# OUTPUT Catalog
LOGGING_DIR = INTERIM_DIR																			# VERBOSE logfile location. If None, reports directly to command line
SFDMAP_DIR = '/Users/jweaver/Projects/Common/py_tools/sfddata-master'

STARCATALOG_DIR = WORKING_DIR + 'data/external/'
STARCATALOG_FILENAME = 'acs_clean_only.fits'
STARCATALOG_COORDCOLS = ['ALPHA_J2000', 'DELTA_J2000']
STARCATALOG_MATCHRADIUS = 0.6 # arcsec

##### FILE NAME PATTERNS, BANDS, AND ZEROPOINTS #####	
IMAGE_EXT = ''																					# Image file extension
WEIGHT_EXT = '_weight'																			# Weight file extension
MASK_EXT = '_mask'																				# Mask file extension
MASTER_MASK = None

DETECTION_NICKNAME = 'DETECTION'																# Detection image file nickname
DETECTION_FILENAME = 'hsc_I-band_10x10_10000srcs_COSMOS_withPSEXT.fits' 										# Detection image file name (fill IMAGE_EXT with EXT)
DETECTION_ZPT = 31.4																			# Detection image zeropoint

MODELING_NICKNAME = 'MODELING'																	# Modeling image file nickname
MODELING_FILENAME = 'hsc_I-band_10x10_10000srcs_COSMOS_withPSEXT.fits'										# Modeling image file name (fill IMAGE_EXT with EXT)
MODELING_ZPT = 31.4																				# Modeling image zeropoint

MULTIBAND_NICKNAME = 'MULTIBAND'																# Multiwavelength image files nickname
MULTIBAND_FILENAME = 'BANDEXT.fits' # fill with 'BAND' and 'EXT'								# Multiwavelength image files name (fill BANDEXT with BANDS)
BANDS = [
'hsc_i0',																						# Multiwavelength band names
'hsc_i1',
'hsc_i2',
'hsc_i3',
]
MULTIBAND_ZPT = [		
31.4,																		# Multiwavelength band zeropoints
31.4,
31.4,
31.4
]


##### IMAGE PROPERTIES #####
PIXEL_SCALE = 0.15																				# Image pixel scale in arcsec/px^2
MOSAIC_WIDTH = 4000																		# Mosaic image width
MOSAIC_HEIGHT = 4000																			# Mosaic image height


##### POINT-SPREAD FUNCTIONS #####
CONSTANT_PSF = ['MODELING',] + BANDS
PRFMAP_PSF = []											# Which PSFs are created constant AND realized as constant
PRFMAP_GRID_FILENAME = {}
PRFMAP_DIR = {}
PRFMAP_COLUMNS = ('ID_GRIDPT', 'RA', 'Dec') #('col5', 'col1', 'col2') #('ID_GRID', 'RA', 'DEC')																	# ID, RA, DEC
PRFMAP_FILENAME = 'mosaic_gp'
USE_BLOB_IDGRID = False
PRFMAP_MAXSEP = 3000  #80 # arcsec (i.e. 50 arcsec = 333 pixels)
PRFMAP_PIXEL_SCALE_ORIG = 0.150
PRFMAP_FORCE_SIZE = 0
PRFMAP_MASKRAD = 25/2.
PSF_RADIUS = 0 #80.5 #px
PSFVAR_NSNAP = 9 	

PSFGRID = [	]
PSFGRID_OUT_DIR = '/Volumes/WD4/Current/COSMOS2020/data/intermediate/PSFGRID/'   # Where to find the OUT directories
PSFGRID_MAXSEP = 3000

USE_STARCATALOG = True
FLAG_STARCATALOG = ['MU_CLASS==2',]																			# For varying PSFs, how many realizations along an axis?
MOD_REFF_LIMITS = (2.5, 3.2)																	# Model PSF selection in Reff
MOD_VAL_LIMITS = (19.5, 21.0) 																	# Model PSF selection in Magnitude
MULTIBAND_REFF_LIMITS = (																		# Multiwavelength PSF selection in Reff
)
MULTIBAND_VAL_LIMITS = (																		# Multiwavelength PSF selection in Magnitude
																	)																	# Sigma of the mock gaussian PSF to be realized
NORMALIZE_PSF = True 	
NORMALIZATION_THRESH = 1E-2																		# Normalize your PSFs? (Say yes...)
RMBACK_PSF = ['MODELING',] + BANDS 																				# On the fly PSF background subtraction in an annulus
PSF_MASKRAD = 8.5															# Size of background annulus, in arcsec
FORCE_GAUSSIAN_PSF = False																		# Force Farmer to realize mock gaussian PSFs instead
USE_GAUSSIAN_PSF = FORCE_GAUSSIAN_PSF	
USE_MOG_PSF = False #experimental!														# TODO
PSF_SIGMA = 0.8 


##### TRACTOR ENGINE #####
DAMPING = 1e-6
TRACTOR_MAXSTEPS = 100																			# Maximum number of iterations to try
TRACTOR_CONTHRESH = 1E-2
USE_RMS_WEIGHTS = []
SCALE_WEIGHTS = ['hsc_i',]
USE_MASKED_SEP_RMS = False
USE_MASKED_DIRECT_RMS = False
APPLY_SEGMASK = True
ITERATIVE_SUBTRACTION_THRESH = 1E31
CORRAL_SOURCES = True
REFF_MIN = 0.1 * 0.15 # arcsec




##### DEBUG MODE #####
NBLOBS = 0	
TRY_OPTIMIZATION = True

##### MODELING #####
MODELING_BANDS = ['hsc_i1','hsc_i2']	
MODEL_PHOT_MAX_NBLOB = 0																		# Skips modelling of blobs with N sources greater than this value
FORCE_POSITION = False																			# If True, will force the position from an external catalog
FREEZE_POSITION = False																	# If True, model positions will be frozen after the PS-Model stage
FREEZE_FINAL_POSITION = False
USE_MODEL_POSITION_PRIOR = True #True
MODEL_POSITION_PRIOR_SIG = 1 * 0.15 # arcsec
USE_MODEL_SHAPE_PRIOR = True # True
MODEL_REFF_PRIOR_SIG = 2 * 0.15 # arcsec
# MODEL_EE_PRIOR_SIG = 0														# TODO													# Skips photometering blobs with N sources greater than this value
ITERATIVE_SUBTRACTION_THRESH = 1E31
USE_BIC = False		
USE_SEP_INITIAL_FLUX = True	

##### DECISION TREE #####																			# Use BIC instead of Chi2 in decision tree
DECISION_TREE = 1
	
### OPTION 1		
PS_SG_THRESH1 = 0.3																	# Threshold which PS-models must beat to avoid tree progression
EXP_DEV_THRESH = 0.5	
CHISQ_FORCE_EXP_DEV = 1.5
CHISQ_FORCE_COMP = 1.5																		# Threshold which either EXP or DEV must beat to avoid tree progression

### OPTION 2		
PS_SG_THRESH2 = 0.2																	# Threshold which PS-models must beat to avoid tree progression
CHISQ_FORCE_SERSIC = 1.0	
CHISQ_FORCE_SERSICCORE = 1.0
CHISQ_FORCE_COMP = 1.2															# Otherwise will guess inital flux from peak of image and peak of psf

##### FORCED PHOTOMETRY #####
INIT_FLUX_BAND = 'hsc_i1'
FORCED_PHOT_MAX_NBLOB = 0
FREEZE_FORCED_POSITION = True 
USE_FORCE_POSITION_PRIOR = True
FORCE_POSITION_PRIOR_SIG = 0.3 * 0.15 # arcsec   #'AUTO'
FREEZE_FORCED_SHAPE = True
USE_FORCE_SHAPE_PRIOR = False
FORCE_REFF_PRIOR_SIG = 0 #arcsec
# FORCE_EE_PRIOR_SIG = 0 #arcsec


##### BRICKS AND BLOBS #####
BRICK_WIDTH = 2000   																			# Width of bricks (in pixels)
BRICK_HEIGHT = 2000																		# Height of bricks (in pixels)
BRICK_BUFFER = 100																				# Buffer around bricks (in pixels)
BLOB_BUFFER = 5 #px																				# Buffer around blobs (in pixels)
DILATION_RADIUS = 1																			# Dialation structure function radius to expand segemetation maps
# SEGMAP_SNR = 0
SEGMAP_MINAREA = 50

###### SOURCE DETECTION ######
USE_DETECTION_WEIGHT = True																	# If True, the weight associated with the detection map will be used
DETECTION_SUBTRACT_BACKGROUND = True															# If True, the background of the detection map will be subtracted
SUBTRACT_BACKGROUND = BANDS
SUBTRACT_BACKGROUND_WITH_MASK = False
SUBTRACT_BACKGROUND_WITH_DIRECT_MEDIAN = False
MANUAL_BACKGROUND = {}
SAVE_BACKGROUND = False																		# If True, the background of each brick will be subtracted
DETECT_BW = 128																					# Detection mesh box width
DETECT_BH = 128																					# Detection mesh box height
DETECT_FW = 3																					# Detection filter box width
DETECT_FH = 3																					# Detection filter box height
SUBTRACT_BW = 128																				# Background mesh box width
SUBTRACT_BH = 128																				# Background mesh box height
SUBTRACT_FW = 3																					# Background filter box width
SUBTRACT_FH = 3																					# Background filter box height
USE_FLAT = True																		# If True, will use brick-global background level in subtraction
THRESH = 1.5																					# If weight is used, this is a relative threshold in sigma. If not, then absolute.
MINAREA = 2																				# Minumum contiguous area above threshold required for detection
FILTER_KERNEL = 'gauss_2.0_5x5.conv'															# Detection convolution kernel
FILTER_TYPE = 'matched' 																		# Detection convolution kernel type
DEBLEND_NTHRESH = 2**8																		# Debelnding n-threshold
DEBLEND_CONT = 1E-10																		# Deblending continuous threshold
PIXSTACK_SIZE = 1000000																			# Allowed number of pixels in a single detection image



##### APERTURE PHOTOMETRY #####
DO_APPHOT = True																				# If True, each object will be aperture photometered (image, model, and residual)
APER_APPLY_SEGMASK = False																		# if True, values beyond the segmentation will be masked out in aperture photometry
APER_PHOT = [1.0, 2.0, 3.0, 5.0, 10.0] 											# Diameter sizes of apertures (in arcsec)
DO_SEPHOT = True
PHOT_AUTOPARAMS = [2.5, 3.5] # kron factor, min radius in pixels
PHOT_FLUXFRAC =  [0.2, 0.5, 0.8]  # Fraction of FLUX_AUTO defining FLUX_RADIUS


##### RESIDUAL SOURCE DETECTION #####
DO_SEXPHOT = True																				# If True, each blob will be re-detected for residual sources and/or noise
RES_THRESH = 5 																					# Threshold for detection (relative, in sigma)
RES_MINAREA = 5																					# Mininum area required for detection
RES_DEBLEND_NTHRESH = 10																		# Deblending n-threshold
RES_DEBLEND_CONT = 0.0001																		# Deblending continuous threshold


##### MISCELLANEOUS #####
X_COLNAME = 'x'																					# x coordinate column name from previous centroiding
Y_COLNAME = 'y'																					# y coordinate column name from previous centroiding

ESTIMATE_EFF_AREA = True																		# Toggles effective area calculation, written to catalog header.
MAKE_MODEL_IMAGE = False																		# If True, a model image will be made for each brick and band
MAKE_RESIDUAL_IMAGE = True																	# If True, a residual image will be made for each brick and band
RESIDUAL_CHISQ_REJECTION = 1E31
RESIDUAL_NEGFLUX_REJECTION = True  
RESIDUAL_AB_REJECTION = None 																		# Otherwise set to None

SPARSE_SIZE = 1000																				# Threshold of square pixel area above which SPARSE_THRESH Will be applied.
SPARSE_THRESH = 0.85																			# If number of maksed pixels in a blob exceeds this value, it will be skipped