#### THE FARMER v1.3 #####
#   CONFIGURATION FILE   #
##########################


#### GENERAL FUNCTIONALITY #####
VERBOSE = 0																						# Verbosity level (0 to 5)
PLOT = True																					# Plot level (0 to 2)
NTHREADS = 0																					# Number of threads to run on (0 is serial)
OVERWRITE = True																				# Overwrite existing files without warning?


##### FILE LOCATION #####
IMAGE_DIR = '/Volumes/WD4/Current/COSMOS2020/data/images'										# INPUT Raw images, weights, and masks
PSF_DIR = '/Volumes/WD4/Current/COSMOS2020/data/intermediate/psfmodels'							# INTERMEDIATE Point-spread function files
BRICK_DIR = '/Volumes/WD4/Current/COSMOS2020/data/intermediate/bricks'							# INTERMEDIATE Bricks (i.e. subimages)
INTERIM_DIR = '/Volumes/WD4/Current/COSMOS2020/data/intermediate/interim'						# OUTPUT Other stuff I make 		
PLOT_DIR = '/Volumes/WD4/Current/COSMOS2020/data/intermediate/plots'							# OUTPUT Figures
CATALOG_DIR = '/Volumes/WD4/Current/COSMOS2020/data/output/catalogs/testing4'					# OUTPUT Catalog
LOGGING_DIR = None																				# VERBOSE logfile location. If None, reports directly to command line
LOGGING_LEVEL = 'WARNING'																		# VERBOSE logfile level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

##### FILE NAME PATTERNS, BANDS, AND ZEROPOINTS #####	
IMAGE_EXT = ''																					# Image file extension
WEIGHT_EXT = '_weight'																			# Weight file extension
MASK_EXT = '_mask'																				# Mask file extension

DETECTION_NICKNAME = 'DETECTION'																# Detection image file nickname
DETECTION_FILENAME = 'COSMOS2019_YJHKs_chimeanEXT.fits' 										# Detection image file name (fill IMAGE_EXT with EXT)
DETECTION_ZPT = 0.00																			# Detection image zeropoint

MODELING_NICKNAME = 'MODELING'																	# Modeling image file nickname
MODELING_FILENAME = 'HSC_i_SSP_PDR2_19_07_19_v3EXT.fits'										# Modeling image file name (fill IMAGE_EXT with EXT)
MODELING_ZPT = 31.4																				# Modeling image zeropoint

MULTIBAND_NICKNAME = 'MULTIBAND'																# Multiwavelength image files nickname
MULTIBAND_FILENAME = 'BANDEXT.fits' # fill with 'BAND' and 'EXT'								# Multiwavelength image files name (fill BANDEXT with BANDS)
BANDS = [																						# Multiwavelength band names
	'hsc_U',
	'hsc_g',
	'hsc_r',
	'hsc_i',
	'hsc_z',
	'hsc_y',
	'uvista_y',
	'uvista_h',
	'uvista_j',
	'uvista_ks',
	'subaru_IA484',
	'subaru_IA527',
	'subaru_IA624',
	'subaru_IA679',
	'subaru_IA738',
	'subaru_IA767',
	'subaru_IB427',
	'subaru_IB464',
	'subaru_IB505',
	'subaru_IB574',
	'subaru_IB709',
	'subaru_IB827',
	'subaru_NB711',
	'subaru_NB816',
	'irac_ch1',
	'irac_ch2',
	'irac_ch3',
	'irac_ch4'
]
MULTIBAND_ZPT = [																				# Multiwavelength band zeropoints
	23.9,	# hsc_U 
 	31.4, 	# hsc_g
	31.4,	# hsc_r
	31.4,	# hsc_i
	31.4,	# hsc_z
	31.4, 	# hsc_y
	30.0, 	# uvista_Y
	30.0, 	# uvista_H
	30.0, 	# uvista_J
	30.0, 	# uvista_Ks
	31.4, 	# IA484
	31.4, 	# IA527
	31.4, 	# IA624
	31.4, 	# IA679
	31.4, 	# IA738
	31.4, 	# IA767
	31.4, 	# IB427
	31.4, 	# IB464
	31.4, 	# IB505
	31.4, 	# IB574
	31.4, 	# IB709
	31.4, 	# IB827
	31.4,	# NB711
	31.4,	# NB816
	27.0,	# IRAC1
	27.0, 	# IRAC2
	27.0,	# IRAC3
	27.0,	# IRAC4 
	# 23.93	# IRAC1 RESAMP
]


##### IMAGE PROPERTIES #####
PIXEL_SCALE = 0.15																				# Image pixel scale in arcsec/px^2
MOSAIC_WIDTH = 48096																			# Mosaic image width
MOSAIC_HEIGHT = 48096																			# Mosaic image height


##### POINT-SPREAD FUNCTIONS #####
CONSTANT_PSF = ['MODELING',] + BANDS[:-4] 														# Which PSFs are created constant AND realized as constant
PSFVAR_NSNAP = 9 																				# For varying PSFs, how many realizations along an axis?
MOD_REFF_LIMITS = (2.5, 3.2)																	# Model PSF selection in Reff
MOD_VAL_LIMITS = (19.5, 21.0) 																	# Model PSF selection in Magnitude
MULTIBAND_REFF_LIMITS = (																		# Multiwavelength PSF selection in Reff
	(3.2, 4.0),	# hsc_U 
	(3.0, 3.8), 	# hsc_g
	(2.6, 3.2), 	# hsc_r
	(2.5, 3.2), 	# hsc_i
	(2.1, 2.8), 	# hsc_z
	(2.3, 3.3), 	# hsc_y
	(3.3, 4.0), 	# uvista_Y
	(3.0, 3.5), 	# uvista_J
	(3.0, 3.5), 	# uvista_H
	(2.5, 3.5),	# uvista_Ks
	# (2.5, 3.5),	# IA484
	# (2.0, 3.0),	# IA527
	# (2.1, 3.0),	# IA624
	# (3.3, 4.5),	# IA679
	# (2.5, 3.4),	# IA738
	# (3.0, 4.0),	# IA767
	# (2.5, 3.5),	# IB427
	# (3.5, 5.0),	# IB464
	# (2.5, 3.5),	# IB505
	# (3.5, 4.5),	# IB574
	# (2.5, 3.5),	# IB709
	# (3.5, 4.5),	# IB827
	# (1.9, 2.8),	# NB711
	# (2.5, 3.5),	# NB816
	(6.5, 7.2),	# IRAC1
	# (6.5, 7.2),	# IRAC1 RESAMP
)
MULTIBAND_VAL_LIMITS = (																		# Multiwavelength PSF selection in Magnitude
	(18.0, 22.0),	# hsc_U
	(18.0, 22.0), 	# hsc_g
	(19.0, 21.5), 	# hsc_r
	(19.5, 21.0), 	# hsc_i
	(18.0, 21.0), 	# hsc_z
	(18.0, 20.8), 	# hsc_y
	(16.0, 21.0), 	# uvista_Y
	(16.0, 20.0), 	# uvista_H
	(16.0, 20.0), 	# uvista_J
	(16.0, 19.0),	# uvista_Ks
	# (18.0, 22.0),	# IA484
	# (20.0, 22.0),	# IA527
	# (19.5, 22.0),	# IA624
	# (19.0, 21.0),	# IA679
	# (19.0, 21.0),	# IA738
	# (19.0, 21.0),	# IA767
	# (19.0, 22.5),	# IB427
	# (18.0, 22.0),	# IB464
	# (19.0, 22.0),	# IB505
	# (19.0, 22.0),	# IB574
	# (19.5, 21.5),	# IB709
	# (19.5, 20.5),	# IB827
	# (18.0, 21.0),	# NB711
	# (19.5, 20.5),	# NB816
	(16.1, 18.5), 	# IRAC1
	# (16.1, 18.5),	# IRAC1 RESAMP
)
NORMALIZE_PSF = True 																			# Normalize your PSFs? (Say yes...)
RMBACK_PSF = True																				# On the fly PSF background subtraction in an annulus
PSF_MASKRAD = 8.0																				# Size of background annulus, in arcsec
FORCE_GAUSSIAN_PSF = False																		# Force Farmer to realize mock gaussian PSFs instead
USE_GAUSSIAN_PSF = FORCE_GAUSSIAN_PSF															# TODO
PSF_SIGMA = 0.8 																				# Sigma of the mock gaussian PSF to be realized


##### MODELING #####
NBLOBS = 0																						# Number of blobs to model
FORCE_POSITION = False																			# If True, will force the position from an external catalog
FREEZE_POSITION = True																			# If True, model positions will be frozen after the PS-Model stage
APPLY_SEGMASK = True																			# TODO
MODEL_PHOT_MAX_NBLOB = 1																		# Skips modelling of blobs with N sources greater than this value
FORCED_PHOT_MAX_NBLOB = 1 																		# Skips photometering blobs with N sources greater than this value
TRACTOR_MAXSTEPS = 100																			# Maximum number of iterations to try
TRACTOR_CONTHRESH = 1E-6																		# Convergence threshold
USE_BIC = True																					# Use BIC instead of Chi2 in decision tree
USE_REDUCEDCHISQ = True																			# Use reduced chisq in decision tree
PS_SG_THRESH = 6.0																				# Threshold which PS-models must beat to avoid tree progression
EXP_DEV_THRESH = 6.0																			# Threshold which either EXP or DEV must beat to avoid tree progression


##### BRICKS AND BLOBS #####
BRICK_WIDTH = 2004    																			# Width of bricks (in pixels)
BRICK_HEIGHT = 2004   																			# Height of bricks (in pixels)
BRICK_BUFFER = 100																				# Buffer around bricks (in pixels)
BLOB_BUFFER = 10																				# Buffer around blobs (in pixels)
DILATION_RADIUS = 4																				# Dialation structure function radius to expand segemetation maps


###### SOURCE DETECTION ######
USE_DETECTION_WEIGHT = False																	# If True, the weight associated with the detection map will be used
DETECTION_SUBTRACT_BACKGROUND = False															# If True, the background of the detection map will be subtracted
SUBTRACT_BACKGROUND = True																		# If True, the background of each brick will be subtracted
DETECT_BW = 3																					# Detection mesh box width
DETECT_BH = 3																					# Detection mesh box height
SUBTRACT_BW = 126																				# Background mesh box width
SUBTRACT_BH = 126																				# Background mesh box height
USE_FLAT = True																					# If True, will use brick-global background level in subtraction
THRESH = 1.5 																					# If weight is used, this is a relative threshold in sigma. If not, then absolute.
MINAREA = 5																						# Minumum contiguous area above threshold required for detection
FILTER_KERNEL = 'gauss_3.0_5x5.conv'															# Detection convolution kernel
FILTER_TYPE = 'matched' 																		# Detection convolution kernel type
DEBLEND_NTHRESH = 32																			# Debelnding n-threshold
DEBLEND_CONT = 0.00001																			# Deblending continuous threshold
PIXSTACK_SIZE = 1000000																			# Allowed number of pixels in a single detection image


##### APERTURE PHOTOMETRY #####
DO_APPHOT = True																				# If True, each object will be aperture photometered (image, model, and residual)
APER_APPLY_SEGMASK = False																		# if True, values beyond the segmentation will be masked out in aperture photometry
APER_PHOT = [0.5, 1.0, 1.5, 1.9, 2.0, 2.5, 3.0, 3.5, 3.8, 4.0, 4.5, 5.0, 10.0 ] 				# Diameter sizes of apertures (in arcsec)


##### RESIDUAL SOURCE DETECTION #####
DO_SEXPHOT = False																				# If True, each blob will be re-detected for residual sources and/or noise
RES_THRESH = 5 																					# Threshold for detection (relative, in sigma)
RES_MINAREA = 5																					# Mininum area required for detection
RES_DEBLEND_NTHRESH = 10																		# Deblending n-threshold
RES_DEBLEND_CONT = 0.0001																		# Deblending continuous threshold


##### MISCELLANEOUS #####
X_COLNAME = 'X_IMAGE'																			# TODO
Y_COLNAME = 'Y_IMAGE'																			# TODO

MAKE_MODEL_IMAGE = True																			# If True, a model image will be made for each brick and band
MAKE_RESIDUAL_IMAGE = True																		# If True, a residual image will be made for each brick and band

SPARSE_SIZE = 1000																				# Threshold of square pixel area above which SPARSE_THRESH Will be applied.
SPARSE_THRESH = 0.85																			# If number of maksed pixels in a blob exceeds this value, it will be skipped
