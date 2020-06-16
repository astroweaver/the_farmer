# User inputs relevant directories and a model catalog
# System loops over the directories and intelligently populates catalog

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column, join

WORKING_DIR = '/Volumes/WD4/Current/COSMOS2020/data/output/catalogs/candide/cosmos_chimeanmodel4/'
CATALOG_FNAME = 'master_models.fits'
FPHOT_FNAMES = ('master_uhscuvista.fits', 'master_uhscuvista_missing_28_05.fits',
                'master_subarumb.fits', 'master_subarumb_missing_01_06.fits',
                'master_IRAC12_nobacksub.fits',
                # 'master_ubvistanb_subarubb.fits'
                )
SAVE = True
OVERWRITE = True
OUTPUT_FNAME = 'master_forced_photometry.fits'
OUTPUT_DIR = WORKING_DIR


# Start with the models
print('Opening the master model catalog...')
tab_master = Table.read(os.path.join(WORKING_DIR, CATALOG_FNAME))
uniq_col = Column([f'{i}_{j}' for i,j in zip(tab_master['brick_id'], tab_master['source_id'])], name='uniq_id')
tab_master.add_column(uniq_col, index=0)
print('Done.')

# Loop over the fphots
for fname in FPHOT_FNAMES:
    print()
    print('Opening '+fname)
    tab = Table.read(os.path.join(WORKING_DIR, fname))

    # QUICK FIX
    if fname == FPHOT_FNAMES[0]:
        for i in tab.colnames:
            if ('irac_ch1' in i) | ('irac_ch2' in i):
                tab.remove_column(i)

    # Make unique id column
    uniq_col = Column([f'{i}_{j}' for i,j in zip(tab['brick_id'], tab['source_id'])], name='uniq_id')
    tab.add_column(uniq_col, index=0)
    # Find out the band names
    print('Loading bands...')
    band_names = {}
    for col in tab.colnames:
        if col.startswith('CHISQ_') & ~col.startswith('CHISQ_MODELING_'):  # MAG gets used in apertures too..
            bn = col[len('CHISQ_'):]
            print('*** ' + bn)
            colnames = []
            for col in tab.colnames:
                if (bn in col) & (not 'MODELING' in col) & ('MAG' in col):
                    tab[col].mask |= np.isnan(tab[col])
                    tab[col] = tab[col].filled(-99)
                    colnames.append(col)
            band_names[bn] = colnames
    nbands = len(band_names)
    print(f'Found {nbands} bands.')

    # For each band, add the relevant columns
    for i, band in enumerate(band_names.keys()):

        # What if the column already exists?
        if band_names[band][0] in tab_master.colnames:
            print(f'[{i+1}/{nbands}] Columns exist for {band} -- adding what you are missing.')

            # Go brick by brick
            for bid in np.unique(tab['brick_id']):
                selection = tab_master['brick_id'] == bid
                selection2 = tab['brick_id'] == bid
                if (tab_master[selection][f'MAG_{band}'] <= 0).all():
                    continue
                if (tab[selection2][f'MAG_{band}'] <= 0).all():
                    print(f'*** No valid sources found for brick #{bid} -- MISSING!')
                    continue
                # if there is nothing there, then:
                print(f'*** Adding sources for brick #{bid}')
                idx = np.where(np.in1d(tab_master['uniq_id'], tab['uniq_id']) & selection)[0]
                idx2 = np.where(np.in1d(tab['uniq_id'], tab_master['uniq_id']) & selection2)[0]
                for col in band_names[band]:
                    print(band, bid, np.sum(tab_master[col][idx]), np.median(tab_master[col][idx]))
                    tab_master[col][idx] = tab[col][idx2]
                    assert((tab_master[col][idx] == tab[col][idx2]).all())
                    print('***** ', np.sum(tab[col][idx2]), np.median(tab[col][idx2]))
                    print('***** ', np.sum(tab_master[col][idx]), np.median(tab_master[col][idx]))
                    print()

        
        else:
            print(f'[{i+1}/{nbands}] Appending columns for {band}')
            join_cat = tab[['uniq_id',]+list(band_names[band]) ]
            tab_master = join(tab_master, join_cat, keys='uniq_id', join_type='left')

print()
print('Finished!')

for i in tab_master.colnames:
    if 'MAG_APER' in i:
        tab_master.remove_column(i)

if SAVE:
    print('Saving...')
    tab_master.write(os.path.join(OUTPUT_DIR, OUTPUT_FNAME), format='fits', overwrite=OVERWRITE)
else:
    print('NO SAVE.')
