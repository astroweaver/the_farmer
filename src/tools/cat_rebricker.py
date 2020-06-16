# Turn catalog back into bricks
import os
import numpy as np
from astropy.table import Table

# THIS SCRIPT WILL NOT PRESERVE THE HDUL EXTENSIONS! 

CATALOG = '/Volumes/WD4/Current/COSMOS2020/data/output/catalogs/candide/cosmos_chimeanmodel4/master_forced_photometry.fits'
SUFFIX = '_MULTIBAND'
OUTPUT_DIR = '/Volumes/WD4/Current/COSMOS2020/data/output/catalogs/candide/cosmos_chimeanmodel4/rebrick/'


tab = Table.read(CATALOG)
all_bricks = np.unique(tab['brick_id'])

for bid in all_bricks:
    subcat = tab[tab['brick_id'] == bid]
    print(f'Bricking up {bid} (N={len(subcat)})')
    subcat.write(os.path.join(OUTPUT_DIR, f'B{bid}{SUFFIX}.cat'), format='fits')
print('*** DONE.')