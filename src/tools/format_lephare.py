import numpy as np 
import sys
import os
from astropy.table import Table, Column
sys.path.insert(0, os.path.join('/Volumes/WD4/Current/COSMOS2020/config'))
import config as conf

fn = '/Volumes/WD4/Current/COSMOS2020/data/output/catalogs/candide/cosmos_chimeanmodel4/master_forced_photometry_minimal_checkstyle_draft3.fits'
fnout = '/Volumes/WD4/Current/COSMOS2020/data/output/catalogs/candide/cosmos_chimeanmodel4/master_forced_photometry_minimal_checkstyle_draft3_userfriendly.fits'

OVERWRITE = True

def farmer_to_lephare(sfddir=conf.SFDMAP_DIR, idx=0):
    
   

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
                 'RAWFLUX_', 'RAWFLUXERR_',
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

        if ('MODELING' in colname) & (colname != 'VALID_SOURCE_MODELING'):
            tab.remove_column(colname)
            print('...Removed!')
            continue
            
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

        if colname.startswith('FLUXERR_'):
            # tab[colname] *= 1E-29
            name = colname[8:]
            newname = map_dict[name] + '_FLUXERR'
            print(f'{colname}...{newname}')
            tab[colname].name =  newname 

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
    

def crossmatch_with_aux():

    print('Crossmatching with auxillary catalogs...')

    AUX_DIR = '/Volumes/WD4/Current/COSMOS2020/data/external/ancillary/'

    fname_galex = ('COSMOS_GALEX_emphot_v3.dat', \
        {'NUMBER': 'ID_GALEX',
        'FLUX_NUV': 'NUV_FLUX',
        'FLUXERR_NUV' : 'NUV_FLUXERR',
        'FLUX_FUV': 'FUV_FLUX',
        'FLUXERR_FUV': 'FUV_FLUXERR'
        },
        'ascii.fixed_width'
    )
    fname_fir = ('COSMOS_Super_Deblended_FIRmm_Catalog_20180719.fits', \
        {   'ID' : 'ID_FIR',
            'F24': 'FIR24_FLUX',
            'DF24': 'FIR24_FLUXERR',
            'F100': 'FIR100_FLUX',
            'DF100': 'FIR100_FLUXERR',
            'F160': 'FIR160_FLUX',
            'DF160': 'FIR160_FLUXERR',
            'F250': 'FIR250_FLUX',
            'DF250': 'FIR250_FLUXERR',
            'F350': 'FIR350_FLUX',
            'DF350': 'FIR350_FLUXERR',
            'F500': 'FIR500_FLUX',
            'DF500': 'FIR500_FLUXERR',
            'F850': 'FIR850_FLUX',
            'DF850': 'FIR850_FLUXERR',
            'F1100': 'FIR1100_FLUX',
            'DF1100': 'FIR1100_FLUXERR',
            'F1200': 'FIR1200_FLUX',
            'DF1200': 'FIR1200_FLUXERR',
            'F10CM': 'FIR10CM_FLUX',
            'DF10CM': 'FIR10CM_FLUXERR',},
        'fits'
    )
    # fname_xray = ('Chandra_COSMOS_Legacy_20151120_4d.fits', \
    #     {'id_x': 'ID_CHANDRA',
    #     'flux_F': 'XF_FLUX',
    #     'flux_F_err': 'XF_FLUXERR',
    #     'flux_S': 'XS_FLUX',
    #     'flux_S_err': 'XS_FLUXERR',
    #     'flux_H': 'XH_FLUX',
    #     'flux_H_err': 'XH_FLUXERR',
    #     },
    #     'fits'
    # )
    # fname_acs = ( , \
    #     {}
    # )

    # loop over + do 2step xmatch + add column

    from catalog_tools import crossmatch
    import astropy.units as u

    for (fname, cols, fmt) in (fname_galex,):

        if fname.startswith('COSMOS_GALEX'):
            aux = Table.read(os.path.join(AUX_DIR, fname), format=fmt, data_start=4)
        else:
            aux = Table.read(os.path.join(AUX_DIR, fname), format=fmt)
        print(f'{fname}')

        if fname.startswith('Chandra'):
            aux['RA_x'].name = 'RA'
            aux['DEC_x'].name = 'DEC'

        print(aux.colnames)

        if fname.startswith('Chandra'):
            aux['RA'] = aux['RA'].astype(float)
            aux['DEC'] = aux['DEC'].astype(float)

       

        mcat_aux, mcat_farmer = crossmatch(aux, tab, thresh=[1*u.arcsec, 0.6*u.arcsec], plot=False)

        for coln in cols.keys():
            col_aux = -99.*np.ones(len(tab))
            for idx, val in zip(mcat_farmer['ID'], mcat_aux[coln]):
                col_aux[tab['ID']==idx] = val
            print(f'*** {coln} --> {cols[coln]}')
            tab.add_column(Column(col_aux, name=cols[coln]))

    print('*** DONE.')

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

farmer_to_lephare()

crossmatch_with_aux()

add_zspec()

tab['VALID_SOURCE'] = tab['VALID_SOURCE_MODELING'] & (tab['i_CHISQ'] < 100) # sanity
tab.remove_column('VALID_SOURCE_MODELING')

tab.write(fnout, format='fits', overwrite=OVERWRITE)

# Then push it to candide and run prepare_input_trac.py