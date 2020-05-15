import numpy as np 
import sys
import os
from astropy.table import Table, Column
sys.path.insert(0, os.path.join('/Volumes/WD4/Current/COSMOS2020/config'))
import config as conf

fn = sys.argv[1]
fnout = sys.argv[2]

OVERWRITE = True

def farmer_to_lephare(fn, fnout, sfddir=conf.SFDMAP_DIR, idx=0):
    
    tab = Table.read(fn, idx)

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
            'irac_ch1': 'SPLASH_1',
            'irac_ch2': 'SPLASH_2',

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

        if 'MODELING' in colname:
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
            tab[colname] *= 1E-29     # TODO: convert to cgs units too...
            name = colname[5:]
            newname = map_dict[name] + '_FLUX'
            print(f'{colname}...{newname}')
            tab[colname].name =  newname   

        if colname.startswith('FLUXERR_'):
            tab[colname] *= 1E-29
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

        


    tab.add_column(Column(1+np.arange(len(tab)), name='NUMBER'), index=0)        

    from sfdmap import SFDMap

    m = SFDMap(sfddir, scaling=1) #, scaling=0.86)
    ebv = m.ebv(tab['ALPHA_J2000'], tab['DELTA_J2000'], frame='icrs')
    tab.add_column(Column(ebv, name='EBV'))
    
    tab.write(fnout, format='fits', overwrite=OVERWRITE)


farmer_to_lephare(fn, fnout)