# Take a large catalog and optimally add in missing information from a secondary catalog
import sys
from astropy.table import Table
import numpy as np

# Find bands
def find_bandnames(tab):
    bandnames = []
    for coln in tab.colnames:
        if 'FLUX_' in coln:
            if 'MODELING' in coln:
                continue
            if 'DIRECT' in coln:
                continue
            if 'ERR' in coln:
                continue
            if 'RAW' in coln:
                continue
            else:
                bn = coln[len('FLUX_'):]
                bandnames.append(bn)
    return np.array(bandnames)


# User input
FN_MAIN_CAT = sys.argv[1]
FN_EXT_CAT = sys.argv[2]


# Read in files & tell me stuff
print(f'Loading primary catalog from {FN_MAIN_CAT}')
MAIN_CAT = Table.read(FN_MAIN_CAT)
print(f'*** loaded')
n_main = len(MAIN_CAT)
uniq_main = np.unique(MAIN_CAT['uniq_id'])
n_uniq_main = len(uniq_main)
cols_main = np.array(MAIN_CAT.colnames)
print(f'*** has {n_main} entires, of which {n_uniq_main} are unique ({n_uniq_main / n_main*100})%')
bricks_main = np.unique(MAIN_CAT['brick_id'])
n_bricks_main = len(bricks_main)
print(f'*** has {n_bricks_main} bricks')
bands_main = find_bandnames(MAIN_CAT)
n_bands_main = len(bands_main)
print(f'*** has {n_bands_main} bands')
print(bands_main)

print()

print(f'Loading secondary catalog from {FN_EXT_CAT}')
EXT_CAT = Table.read(FN_EXT_CAT)
print(f'*** loaded')
n_ext = len(EXT_CAT)
uniq_ext = np.unique(EXT_CAT['uniq_id'])
n_uniq_ext = len(uniq_ext)
cols_ext = np.array(EXT_CAT.colnames)
print(f'*** has {n_ext} entires, of which {n_uniq_ext} are unique ({n_uniq_ext / n_ext*100})%')
bricks_ext = np.unique(EXT_CAT['brick_id'])
n_bricks_ext = len(bricks_ext)
print(f'*** has {n_bricks_ext} bricks')
bands_ext = find_bandnames(EXT_CAT)
n_bands_ext = len(bands_ext)
print(f'*** has {n_bands_ext} bands')
print(bands_ext)

print()

# compare them please

print('Calculating unique columns...')
uniq_cols_main = cols_main[~np.in1d(cols_main, cols_ext)]
n_uniq_cols_main = len(uniq_cols_main)
uniq_cols_ext = cols_ext[~np.in1d(cols_ext, cols_main)]
n_uniq_cols_ext = len(uniq_cols_ext)
print(f'Primary catalog has {n_uniq_cols_main} unique columns')
print(f'Secondary catalog has {n_uniq_cols_ext} unique columns')


print('Calculating unique bricks...')
uniq_bricks_main = bricks_main[~np.in1d(bricks_main, bricks_ext)]
n_uniq_bricks_main = len(uniq_bricks_main)
uniq_bricks_ext = bricks_ext[~np.in1d(bricks_ext, bricks_main)]
n_uniq_bricks_ext = len(uniq_bricks_ext)
print(f'Primary catalog has {n_uniq_bricks_main} unique bricks')
print(np.array(uniq_bricks_main))
print(f'Secondary catalog has {n_uniq_bricks_ext} unique bricks')


print('Calculating unique bands...')
uniq_bands_main = bands_main[~np.in1d(bands_main, bands_ext)]
n_uniq_bands_main = len(uniq_bands_main)
uniq_bands_ext = bands_ext[~np.in1d(bands_ext, bands_main)]
n_uniq_bands_ext = len(uniq_bands_ext)
print(f'Primary catalog has {n_uniq_bands_main} unique bands')
print(np.array(uniq_bands_main))
print(f'Secondary catalog has {n_uniq_bands_ext} unique bands')
print(np.array(uniq_bands_ext))




# Now loop + add in, with lots of checking!
# for bid in 
