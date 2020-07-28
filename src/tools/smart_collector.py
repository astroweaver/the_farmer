# User inputs relevant directories and a model catalog
# System loops over the directories and intelligently populates catalog

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column, join, hstack

WORKING_DIR = '/Volumes/WD4/Current/COSMOS2020/data/output/catalogs/candide/cosmos_chimeanmodel4/'
CATALOG_FNAME = 'master_models.fits'
FPHOT_FNAMES = (
                # 'master_forced_photometry_checkstyle_draft.fits',
                'master_uhscuvista.fits', 'master_uhscuvista_missing_28_05.fits',
                'master_subaruMB_orig.fits', 'master_subarumb_missing_22_06.fits', 'master_missing_subaruMB_06_07.fits',
                'master_IRAC12_nobacksub.fits', 
                'master_irac34.fits',
                'master_uvista_misc_missing_08_07.fits',
                'master_irac_ch2_missing_09_07.fits',
                'master_uvistanb_subarubb.fits',
                'master_subaru_redone_09_07.fits',
                'master_missing_misc_13_07.fits',
                'master_misc_missing_06_07.fits',
                'master_misc_09_07.fits',
                'master_missing_subarubb_16_07.fits'
                            
                )
SAVE = True
OVERWRITE = True
OUTPUT_FNAME = 'master_forced_photometry_checkstyle_draft3.fits'
OUTPUT_DIR = WORKING_DIR

SAVE_TINY = True
OUTPUT_TINY_FNAME = 'master_forced_photometry_minimal_checkstyle_draft3.fits'


# Start with the models
print('Opening the master model catalog...')
tab_master = Table.read(os.path.join(WORKING_DIR, CATALOG_FNAME), memmap=True)
uniq_col = Column([f'{i}_{j}' for i,j in zip(tab_master['brick_id'], tab_master['source_id'])], name='uniq_id')
tab_master.add_column(uniq_col, index=0)
tab_master = tab_master.filled(-99)
print(len(tab_master), len(tab_master.colnames))
print('Done.')

# Loop over the fphots
for fname in FPHOT_FNAMES:
    print()
    print('Opening '+fname)
    tab = Table.read(os.path.join(WORKING_DIR, fname))

    # Make unique id column
    uniq_col = Column([f'{i}_{j}' for i,j in zip(tab['brick_id'], tab['source_id'])], name='uniq_id')
    try:
        tab.add_column(uniq_col, index=0)
    except:
        pass

    for i in tab.colnames:
        if 'MAG_APER' in i:
            tab.remove_column(i)
        if 'NORM' in i:
            tab.remove_column(i)
        if 'SNR' in i:
            tab.remove_column(i)

    # get rid of the subaru stuff -- it's bad!
    if fname ==  'master_uvistanb_subarubb.fits':
        for col in tab.colnames:
            if 'subaru' in col:
                tab.remove_column(col)
    

    # How long is it? Are there duplicates?!
    olen = len(tab)
    tab = tab[np.unique(uniq_col, return_index=True)[1]]
    print(f'Chopped off {olen-len(tab)} rows')

    # Find out the band names
    print('Loading bands...')
    band_names = {}
    for col in tab.colnames:
        if col.startswith('CHISQ_') & ~col.startswith('CHISQ_MODELING_'):  # MAG gets used in apertures too..
            bn = col[len('CHISQ_'):]
            print('*** ' + bn)
            colnames = []
            for col in tab.colnames:
                if (bn in col) & (not 'MODELING' in col): #& ('MAG' in col):
                    try:
                        tab[col].mask |= np.isnan(tab[col])
                        tab[col] = tab[col].filled(-99)
                    except:
                        pass
                    if 'SNR' in col:
                        continue
                    if 'NORM' in col:
                        continue
                    colnames.append(col)
            band_names[bn] = colnames
    nbands = len(band_names)
    print(f'Found {nbands} bands.')

    # For each band, add the relevant columns
    for i, band in enumerate(band_names.keys()):

        # # What if the column already exists?
        if band_names[band][0] in tab_master.colnames:
            print(f'[{i+1}/{nbands}] Columns exist for {band} -- adding what you are missing.')
            n_valid = len(np.unique(tab_master['brick_id'][(tab_master[f'MAG_{band}'] > 0) & (tab_master[f'MAG_{band}'] < 100)]))
            n_total = len(np.unique(tab_master['brick_id']))
            print(f'{n_valid}/{n_total}')
            o_n_valid = n_valid

            # Go brick by brick
            for bid in np.unique(tab['brick_id']):
                # print(f'... {bid}')
                selection = tab_master['brick_id'] == bid
                selection2 = tab['brick_id'] == bid

                # The current data is NOT OK
                if (tab[selection2][f'MAG_{band}'] == 0).all():
                    # print(f'*** WARNING: All zeros for brick #{bid}.')
                    continue
                if (tab[selection2][f'MAG_{band}'] == -99).all():
                    # print(f'*** WARNING: All -99 for brick #{bid}.')
                    continue

                # Are the existing data better? 
                if np.sum((tab_master[selection][f'MAG_{band}'] > 10) & (tab_master[selection][f'MAG_{band}'] < 40)) \
                        > np.sum((tab[selection2][f'MAG_{band}'] > 10) & (tab[selection2][f'MAG_{band}'] < 40)):
                    print(f'*** WARNING: Existing data is BETTER. Skipping on #{bid}.')
                    continue

                # if there is nothing there, then:
                # print(f'*** Adding sources for brick #{bid}')
                idx =[np.nonzero(np.isin(tab_master['uniq_id'], tab['uniq_id']) & selection)][0][0]
                idx = idx[np.argsort(tab_master['uniq_id'][idx])]
                tab2 = tab[np.isin(tab['uniq_id'], tab_master['uniq_id']) & selection2]
                tab_insert = tab2[np.argsort(tab2['uniq_id'])]
                assert((tab_master['uniq_id'][idx] == tab_insert['uniq_id']).all())
                for col in band_names[band]:
                    # print(band, bid, np.sum(tab_master[col][idx]), np.median(tab_master[col][idx]))
                    tab_master[col][idx] = tab_insert[col]
                    
                    # print('***** ', np.sum(tab[col][idx2]), np.median(tab[col][idx2]))
                    # print('***** ', np.sum(tab_master[col][idx]), np.median(tab_master[col][idx]))
                    # print()
                    # input('...')
            n_valid = len(np.unique(tab_master['brick_id'][(tab_master[f'MAG_{band}'] > 0) & (tab_master[f'MAG_{band}'] < 100)]))
            n_total = len(np.unique(tab_master['brick_id']))
            print(f'--> {n_valid}/{n_total}')
            print(f'...gained {n_valid - o_n_valid} bricks.')
            print()

        
        else:
            print(f'[{i+1}/{nbands}] Appending columns for {band}')
            join_cat = tab[['brick_id', 'source_id']+list(band_names[band]) ]
            # print(len(tab), len(np.unique(tab['uniq_id'])))
            # print(len(tab_master), len(tab_master.colnames))
            # print(len(join_cat), len(join_cat.colnames))
            tab_master = join(tab_master, join_cat, keys=['brick_id', 'source_id'], join_type='left')
            # tab_master = hstack(tab_master, join_cat, uniq_col_name='uniq_id')
            # print('--> ', len(tab_master), len(tab_master.colnames))
            n_valid = len(np.unique(join_cat['brick_id'][(join_cat[f'MAG_{band}'] > 0) & (join_cat[f'MAG_{band}'] < 100)]))
            n_total = len(np.unique(join_cat['brick_id']))
            print(f'{n_valid}/{n_total}')
            n_valid = len(np.unique(tab_master['brick_id'][(tab_master[f'MAG_{band}'] > 0) & (tab_master[f'MAG_{band}'] < 100)]))
            n_total = len(np.unique(tab_master['brick_id']))
            print(f'{n_valid}/{n_total}')
            print()
            # input('...')

        bad = tab_master[f'FLUX_{band}'] == 0
        tab_master[f'MAG_{band}'][bad] = -99
        tab_master[f'MAGERR_{band}'][bad] = -99
        tab_master[f'FLUX_{band}'][bad] = -99
        tab_master[f'FLUXERR_{band}'][bad] = -99

        bad = tab_master[f'MAG_{band}'] > 40
        tab_master[f'MAG_{band}'][bad] = -99
        tab_master[f'MAGERR_{band}'][bad] = -99
        tab_master[f'FLUX_{band}'][bad] = -99
        tab_master[f'FLUXERR_{band}'][bad] = -99

print()
print('Finished!')

for i in tab_master.colnames:
    if 'MAG_APER' in i:
        tab_master.remove_column(i)
    if 'NORM' in i:
        tab_master.remove_column(i)
    if 'SNR' in i:
        tab_master.remove_column(i)

# review
# band_names = {}
# for col in tab.colnames:
#     if col.startswith('CHISQ_') & ~col.startswith('CHISQ_MODELING_'):  # MAG gets used in apertures too..
#         bn = col[len('CHISQ_'):]
#         print('*** ' + bn)
#         colnames = []
#         for col in tab.colnames:
#             if (bn in col) & (not 'MODELING' in col): #& ('MAG' in col):
#                 try:
#                     tab[col].mask |= np.isnan(tab[col])
#                     tab[col] = tab[col].filled(-99)
#                 except:
#                     pass
#                 if 'SNR' in col:
#                     continue
#                 if 'NORM' in col:
#                     continue
#                 colnames.append(col)
#         band_names[bn] = colnames

# for i, band in enumerate(band_names.keys()):
#     ubid = np.unique(tab_master['brick_id'])
#     revbid = np.zeros((len(ubid), 2))
#     revbid[:,0] = ubid
#     for j, bid in enumerate(ubid):
#         selection = tab_master['brick_id'] == bid
#         revbid[j, 1] = np.sum((tab_master[selection][f'MAG_{band}'] > 10) & (tab_master[selection][f'MAG_{band}'] < 40)) / np.sum(selection)
#     print(band, revbid[revbid[:, 1] < 0.5, 0])

if SAVE:
    print('Saving...')
    tab_master.write(os.path.join(OUTPUT_DIR, OUTPUT_FNAME), format='fits', overwrite=OVERWRITE)
else:
    print('NO SAVE.')

if SAVE_TINY:
    for i in tab_master.colnames:
        if 'APER' in i:
            tab_master.remove_column(i)

    print('Saving tiny version...')
    tab_master.write(os.path.join(OUTPUT_DIR, OUTPUT_TINY_FNAME), format='fits', overwrite=OVERWRITE)
