# Take a large catalog and optimally add in missing information from a secondary catalog
import sys
from astropy.table import Table


FN_MAIN_CAT = sys.argv[1]
FN_EXT_CAT = sys.argv[2]


