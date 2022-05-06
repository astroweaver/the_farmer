# need to fake a config
from fileinput import filename
import numpy as np
from astropy.io import fits


class foo_config:
    def __init__(self):
        self.BRICK_WIDTH = 2004    																			# Width of bricks (in pixels)
        self.BRICK_HEIGHT = 2004   																			# Height of bricks (in pixels)
        self.BRICK_BUFFER = 100																				# Buffer around bricks (in pixels)
        self.BLOB_BUFFER = 5 #px																				# Buffer around blobs (in pixels)
        self.DILATION_RADIUS = 5
        self.PIXEL_SCALE = 0.15																				# Image pixel scale in arcsec/px^2
        self.MOSAIC_WIDTH = 48096																			# Mosaic image width
        self.MOSAIC_HEIGHT = 48096	
        self.dims = (2204, 2204)
        INPUT_DIR = '/Volumes/LaCie12/Projects_offsite/COSMOS2020_offsite/'
        self.IN_BRICK_DIR = INPUT_DIR+'/data/intermediate/bricks'
        self.IN_INTERIM_DIR = INPUT_DIR+'segments/'
        self.IN_CATALOG_DIR = INPUT_DIR+'releases/R1_v2.0/R1_v2.1/COSMOS2020_FARMER_R1_v2.1.fits'					# OUTPUT Catalog

        WORKING_DIR = '/n07data/weaver/COSMOS2020/data_iracout'
        # self.IMAGE_DIR = WORKING_DIR+'/data/images'										# INPUT Raw images, weights, and masks
        # self.PSF_DIR = INPUT_DIR+'/data/intermediate/psfmodels'							# INTERMEDIATE Point-spread function files
        self.BRICK_DIR = WORKING_DIR+'/intermediate/bricks'
        self.INTERIM_DIR = WORKING_DIR+'/intermediate/interim'
        self.CATALOG_DIR = WORKING_DIR+'/output/catalogs'						# INTERMEDIATE Bricks (i.e. subimages)
        # BRICK_DIR = '/Volumes/LaCie12/Data/COSMOS/mosaics/brick_tests'
        						# OUTPUT Other stuff I make 		
        self.PLOT_DIR = WORKING_DIR+'/intermediate/plots'							# OUTPUT Figures
conf = foo_config()


TYPE = 'SEG' # IMG (image, mask, weight) or SEG (segment) or BLOB (blobmap)
BAND = 'hsc_z'
if TYPE == 'IMG':
    pass
elif TYPE == 'SEG':
    filenames = glob.glob(conf.IN_INTERIM_DIR+'*SEGMAPS.fits')

# make big emtpy thing
imap = np.zeros((conf.MOSAIC_WIDTH, conf.MOSAIC_HEIGHT))

# loop over files
for i, fn in enumerate(filenames):
    print(i, fn)