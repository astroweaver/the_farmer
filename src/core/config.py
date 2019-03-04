##### CONFIGURATION FILE #####

# SOURCE DETECTION WITH SEXTRACTOR
BW = 16
BH = 16
THRESH = 5 # <sigma>
MINAREA = 5
DEBLEND_NTHRESH = 10
DEBLEND_CONT = 0.0001

# BUFFERS AND SEGMENT DILATION
BRICK_BUFFER = 75
BLOB_BUFFER = 10
DILATION_RADIUS = 12

# TRACTOR OPTIMIZATION
TRACTOR_MAXSTEPS = 100
TRACTOR_CONTHRESH = 1E-6
EXP_DEV_THRESH = 25
