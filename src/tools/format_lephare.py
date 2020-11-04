import numpy as np 
import sys
import os
from astropy.table import Table, Column
sys.path.insert(0, os.path.join('/Volumes/WD4/Current/COSMOS2020/config'))
import config as conf

fn = '/Volumes/WD4/Current/COSMOS2020/data/output/catalogs/candide/cosmos_chimeanmodel4/master_join_v1.6/master_forced_photometry_allfixed_291020b.fits'
fnout = '/Volumes/WD4/Current/COSMOS2020/data/output/catalogs/candide/cosmos_chimeanmodel4/master_join_v1.6/master_forced_photometry_allfixed_291020b_lephare.fits'

OVERWRITE = True

def farmer_to_lephare(tab, sfddir=conf.SFDMAP_DIR, idx=0):
    
   

    map_dict = {
            'ustar_1':'U1',
            'ustar_2':'U2',
            'hsc_g': 'g',
            'hsc_r': 'r',
            'hsc_i': 'i',
            'hsc_z': 'z',
            'hsc_y': 'y',
            'uvista_y':'Y',
            'uvista_j':'J',
            'uvista_h':'H',
            'uvista_ks':'Ks',
            'uvista_NB118':'NB118',
            'subaru_IA484':'IA484',
            'subaru_IA527':'IA527',
            'subaru_IA624':'IA624',
            'subaru_IA679':'IA679',
            'subaru_IA738':'IA738',
            'subaru_IA767':'IA767',
            'subaru_IB427':'IB427',
            'subaru_IB464':'IB464',
            'subaru_IB505':'IB505',
            'subaru_IB574':'IB574',
            'subaru_IB709':'IB709',
            'subaru_IB827':'IB827',
            'subaru_NB711':'NB711',
            'subaru_NB816':'NB816',
            'irac_ch1': 'IRAC_1',
            'irac_ch2': 'IRAC_2',
            'irac_ch3': 'IRAC_3',
            'irac_ch4': 'IRAC_4',
            'subaru_B': 'B_SC',
            'subaru_Gp': 'gp_SC',
            'subaru_Ip': 'ip_SC',
            'subaru_Rp': 'r_SC',
            'subaru_V': 'V_SC',
            'subaru_Zp': 'zp_SC',
            'subaru_Zq': 'zpp_SC'
    }

    for colname in tab.colnames:
        print(colname)
    #     print(colname)
        breakit = False
        for i in ('MAG_APER_', 'MAGERR_APER_', 
                'FLUX_APER_', 'FLUXERR_APER_',
                'MAG_MODELING_', 'MAGERR_MODELING_',
                'FLUX_MODELING_', 'FLUXERR_MODELING',
                 'RAWFLUX_', 'RAWFLUXERR_', 'RAW_DIRECTFLUXERR_',
                 'DIRECTFLUX_',
                 'BIC_', 'SNR_', 'NORM_'):
            if colname.startswith(i):
                tab.remove_column(colname)
                breakit = True
                print('...Removed!')
                break
        if breakit:
            continue

        if colname == 'VALID_SOURCE_MODELING':
            continue

        # if ('MODELING' in colname) & (colname != 'VALID_SOURCE_MODELING'):
        #     tab.remove_column(colname)
        #     print('...Removed!')
        #     continue
            
        if colname.startswith('MAG_'):
            name = colname[4:]
            newname = map_dict[name] + '_MAG'
            print(f'{colname}...{newname}')
            tab[colname].name =  newname
            
        if colname.startswith('MAGERR_'):
            name = colname[7:]
            newname = map_dict[name] + '_MAGERR'
            print(f'{colname}...{newname}')
            tab[colname].name =  newname 

        if colname.startswith('FLUX_'):
            # tab[colname] *= 1E-29     # TODO: convert to cgs units too...
            name = colname[5:]
            newname = map_dict[name] + '_FLUX'
            print(f'{colname}...{newname}')
            tab[colname].name =  newname   

        if colname.startswith('DIRECTFLUXERR_'):
            # tab[colname] *= 1E-29
            name = colname[len('DIRECTFLUXERR_'):]
            tab.remove_column('FLUXERR_' + name)
            newname = map_dict[name] + '_FLUXERR'
            print(f'{colname}...{newname}')
            tab[colname].name =  newname 
            # tab.remove_column(colname) # KEEP THIS HERE TO REMIND YOU -- ITS ALREADY RENAMED!

        for i in ('CHISQ_', 'CHI_MU_', 'CHI_SIG_', 'CHI_K2_', 'X_MODEL_', 'Y_MODEL_',
                'XERR_MODEL_', 'YERR_MODEL_', 'RA_', 'DEC_'
                ):
            if colname.startswith(i):
                name = colname[len(i):]
                if name in map_dict.keys():
                    newname = map_dict[name] + '_' + i[:-1]
                    
                    tab[colname].name =  newname
                    print(f'...{newname}')
                    breakit = True
                    break
        if breakit:
            continue
        

            

        if colname == 'RA':
            tab[colname].name = 'ALPHA_J2000'
        if colname == 'DEC':
            tab[colname].name = 'DELTA_J2000'

        


    tab.add_column(Column(1+np.arange(len(tab)), name='ID'), index=0)        

    from sfdmap import SFDMap

    m = SFDMap(sfddir, scaling=0.86)
    ebv = m.ebv(tab['ALPHA_J2000'], tab['DELTA_J2000'], frame='icrs')
    tab.add_column(Column(ebv, name='EBV'))

    return tab
    

def crossmatch_with_aux(tab):

    print('Crossmatching with auxillary catalogs...')

    AUX_DIR = '/Volumes/WD4/Current/COSMOS2020/data/external/ancillary/'

    fname_galex = ('COSMOS_GALEX_emphot_v3.dat', \
        {'NUMBER': 'ID_GALEX',
        'FLUX_NUV': 'GALEX_NUV_FLUX',
        'FLUXERR_NUV' : 'GALEX_NUV_FLUXERR',
        'MAG_NUV': 'GALEX_NUV_MAG',
        'MERR_NUV' : 'GALEX_NUV_MAGERR',
        'FLUX_FUV': 'GALEX_FUV_FLUX',
        'FLUXERR_FUV': 'GALEX_FUV_FLUXERR',
        'MAG_FUV': 'GALEX_FUV_MAG',
        'MERR_FUV': 'GALEX_FUV_MAGERR'
        },
        'ascii.fixed_width'
    )
    fname_fir = ('COSMOS_Super_Deblended_FIRmm_Catalog_20180719.fits', \
        {   'ID' : 'ID_FIR',
            'F24': 'FIR_24_FLUX',
            'DF24': 'FIR_24_FLUXERR',
            'F100': 'FIR_100_FLUX',
            'DF100': 'FIR_100_FLUXERR',
            'F160': 'FIR_160_FLUX',
            'DF160': 'FIR_160_FLUXERR',
            'F250': 'FIR_250_FLUX',
            'DF250': 'FIR_250_FLUXERR',
            'F350': 'FIR_350_FLUX',
            'DF350': 'FIR_350_FLUXERR',
            'F500': 'FIR_500_FLUX',
            'DF500': 'FIR_500_FLUXERR',
            'F850': 'FIR_850_FLUX',
            'DF850': 'FIR_850_FLUXERR',
            'F1100': 'FIR_1100_FLUX',
            'DF1100': 'FIR_1100_FLUXERR',
            'F1200': 'FIR_1200_FLUX',
            'DF1200': 'FIR_1200_FLUXERR',
            'F10CM': 'FIR_10CM_FLUX',
            'DF10CM': 'FIR_10CM_FLUXERR',
            'F20CM': 'FIR_20CM_FLUX',
            'DF20CM': 'FIR_20CM_FLUXERR',},
        'fits'
    )
    fname_xray = ('Chandra_COSMOS_Legacy_20151120_4d.fits', \
        {'id_x': 'ID_CHANDRA',
        'flux_F': 'XF_FLUX',
        'flux_F_err': 'XF_FLUXERR',
        'flux_S': 'XS_FLUX',
        'flux_S_err': 'XS_FLUXERR',
        'flux_H': 'XH_FLUX',
        'flux_H_err': 'XH_FLUXERR',
        },
        'fits'
    )
    fname_acs = ( 'COSMOS_ACS_catalog/acs_clean.fits', \
        {
        'NUMBER': 'ID_ACS',
        'MAG_AUTO': 'F184W_MAG',
        'MAGERR_AUTO': 'F814W_MAGERR',
        'FLUXERR_AUTO': 'F814W_FLUX',
        'FLUX_AUTO': 'F814W_FLUXERR',
        'A_WORLD': 'ACS_A_WORLD',
        'B_WORLD': 'ACS_B_WORLD',
        'THETA_WORLD': 'ACS_THETA_WORLD',
        'FWHM_WORLD': 'ACS_FWHM_WORLD',
        'MU_MAX': 'ACS_MU_MAX',
        'MU_CLASS': 'ACS_MU_CLASS',
        },
        'fits'
    )

    fname_laigle = ('/Volumes/WD4/Current/COSMOS2020/data/external/COSMOS2015_Laigle+_v1.1.fits', \
        {'NUMBER': 'ID_COSMOS2015',
        'SPLASH_1_FLUX': 'SPLASH_1_FLUX',
        'SPLASH_1_FLUX_ERR': 'SPLASH_1_FLUXERR',
        'SPLASH_1_MAG': 'SPLASH_1_MAG',
        'SPLASH_1_MAGERR': 'SPLASH_1_MAGERR',
        'SPLASH_2_FLUX': 'SPLASH_2_FLUX',
        'SPLASH_2_FLUX_ERR': 'SPLASH_2_FLUXERR',
        'SPLASH_2_MAG': 'SPLASH_2_MAG',
        'SPLASH_2_MAGERR': 'SPLASH_2_MAGERR',
        'SPLASH_3_FLUX': 'SPLASH_3_FLUX',
        'SPLASH_3_FLUX_ERR': 'SPLASH_3_FLUXERR',
        'SPLASH_3_MAG': 'SPLASH_3_MAG',
        'SPLASH_3_MAGERR': 'SPLASH_3_MAGERR',
        'SPLASH_4_FLUX': 'SPLASH_4_FLUX',
        'SPLASH_4_FLUX_ERR': 'SPLASH_4_FLUXERR',
        'SPLASH_4_MAG': 'SPLASH_4_MAG',
        'SPLASH_4_MAGERR': 'SPLASH_4_MAGERR',
        },
        'fits'
    )

    # loop over + do 2step xmatch + add column

    from catalog_tools import crossmatch
    import astropy.units as u
    inputs = [fname_galex, fname_acs,]
    # inputs = [fname_galex, fname_fir, fname_xray, fname_acs, fname_laigle,]

    # fname_galex, fname_fir, f
    for (fname, cols, fmt) in inputs:

        if fname.startswith('COSMOS_GALEX'):
            aux = Table.read(os.path.join(AUX_DIR, fname), format=fmt, data_start=4)
            # modify
            # print(aux.colnames)
            aux['FLUX_NUV'] = 10**(-0.4 * (aux['MAG_NUV'] - 23.9))
            aux['FLUX_FUV'] = 10**(-0.4 * (aux['MAG_FUV'] - 23.9))
            aux['FLUXERR_NUV'] = aux['FLUX_NUV'] * aux['MERR_NUV'] / 1.089
            aux['FLUXERR_FUV'] = aux['FLUX_FUV'] * aux['MERR_FUV'] / 1.089
            aux['FLUX_NUV'][aux['FLUXERR_NUV'] < 0] = -99
            aux['FLUXERR_NUV'][aux['FLUXERR_NUV'] < 0] = -99
            aux['FLUX_FUV'][aux['FLUXERR_FUV'] < 0] = -99
            aux['FLUXERR_FUV'][aux['FLUXERR_FUV'] < 0] = -99
        elif fname.startswith('/Volumes/'):
            aux = Table.read(fname, format=fmt)
        elif 'FIR' in fname:
            aux = Table.read(os.path.join(AUX_DIR, fname), format=fmt)
            aux = aux[aux['goodArea']==1]
        elif 'ACS' in fname:
            aux = Table.read(os.path.join(AUX_DIR, fname), format=fmt)
            clean = aux['CLEAN']
            aux = aux[clean==1]
            mag = aux['MAG_AUTO'] 
            idx = mag != 99.
            aux['MAG_AUTO'][~idx] = -99.
            aux['MAGERR_AUTO'][~idx] = -99.
            aux['FLUX_AUTO'] = -99 * np.ones(len(aux))
            aux['FLUXERR_AUTO'] = -99 * np.ones(len(aux))
            aux['FLUX_AUTO'][idx] = 10**(-0.4*(aux['MAG_AUTO'][idx] - 23.9))
            aux['FLUXERR_AUTO'][idx] =  aux['FLUX_AUTO'][idx] * aux['MAGERR_AUTO'][idx] / 1.089

            idx = np.isnan(aux['B_WORLD'])
            aux['B_WORLD'][idx] = aux['A_WORLD'][idx] / aux['ELONGATION'][idx]
            
        else:
            aux = Table.read(os.path.join(AUX_DIR, fname), format=fmt)
        print(f'{fname}')

        if fname.startswith('Chandra'):
            aux['RA_x'].name = 'RA'
            aux['DEC_x'].name = 'DEC'

        # print(aux.colnames)

        if fname.startswith('Chandra'):
            aux['RA'] = aux['RA'].astype(float)
            aux['DEC'] = aux['DEC'].astype(float)

       
        # crossmatch(tab, aux, thresh=[1*u.arcsec, 0.6*u.arcsec], plot=False, return_idx=True)

        mcat_aux, mcat_farmer, idx, sel_thresh = crossmatch(aux, tab, thresh=[1*u.arcsec, 0.6*u.arcsec], plot=False, return_idx=True)

        for coln in cols.keys():
            # print(coln+' ...')
            col_aux = -99 * np.ones(len(tab))
            # print(col_aux[idx[sel_thresh]])
            # # print(mcat_aux[coln][sel_thresh])
            # try:
            #     col_aux[idx[sel_thresh]] = aux[coln][sel_thresh]
            # except:
            #     col_aux = np.ones(len(tab), dtype=object) # this is overkill, but OK.
            #     col_aux[idx[sel_thresh]] = aux[coln][sel_thresh]

            # col_aux[idx[sel_thresh]] = mcat_aux[coln][idx]
            try:
                col_aux = -99.0 * col_aux
                for idx, val in zip(mcat_farmer['ID'], mcat_aux[coln]):
                    col_aux[tab['ID']==idx] = val
            except:
                print('column cannot be coverted to float...')
                col_aux = np.ones(len(tab), dtype=object) # this is overkill, but OK.
                for idx, val in zip(mcat_farmer['ID'], mcat_aux[coln]):
                    col_aux[tab['ID']==idx] = val
            
            print(f'*** {coln} --> {cols[coln]}')
            tab[cols[coln]] = col_aux

    print('*** DONE.')
    return tab

def add_zspec():

    print('Matching to specz...')

    from astropy.io import ascii
    import astropy.units as u
    from catalog_tools import crossmatch
    spectro = ascii.read('/Volumes/WD4/Current/COSMOS2020/data/external/spectro_full.dat')

    mcat_spectro, mcat_farmer = crossmatch(spectro, tab, thresh=[1*u.arcsec, 0.6*u.arcsec], plot=False)

    col_zspec = -99.*np.ones_like(tab['ID'])
    for idx, zspec in zip(mcat_farmer['ID'], mcat_spectro['ZSPEC']):
        col_zspec[tab['ID']==idx] = zspec
    tab.add_column(Column(col_zspec, name='ZSPEC'))

    print('*** DONE.')


tab = Table.read(fn, 1)

farmer_to_lephare(tab)

crossmatch_with_aux(tab)

# add_zspec()

tab['VALID_SOURCE'] = tab['VALID_SOURCE_MODELING'] #& (tab['i_CHISQ'] < 100) # sanity
tab.remove_column('VALID_SOURCE_MODELING')

tab.write(fnout, format='fits', overwrite=OVERWRITE)

# Then push it to candide and run prepare_input_trac.py