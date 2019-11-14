import numpy as np 
from astropy.table import Table, Column

import config as conf

def farmer_to_lephare(fn, fnout, sfddir=conf.SFDMAP_DIR, idx=0):
    
    tab = Table.read(fn, idx)

    for colname in tab.colnames:
    #     print(colname)

        if colname.startswith('MAG_'):
            tab[colname].name = colname[4:] + '_MAG'
        if colname.startswith('MAGERR_'):
            tab[colname].name = colname[7:] + '_MAGERR'
        if colname.startswith('FLUX_'):
            tab[colname].name = colname[5:] + '_FLUX'
        if colname.startswith('FLUXERR_'):
            tab[colname].name = colname[9:] + '_FLUXERR'

        if colname == 'RA':
            tab[colname].name = 'ALPHA_J2000'
        if colname == 'DEC':
            tab[colname].name = 'DELTA_J2000'


    tab.add_column(Column(1+np.arange(len(tab)), name='NUMBER'), index=0)        

    from sfdmap import SFDMap

    m = SFDMap(sfddir, scaling=1) #, scaling=0.86)
    ebv = m.ebv(tab['ALPHA_J2000'], tab['DELTA_J2000'], frame='icrs')
    tab.add_column(Column(ebv, name='EBV'))
    
    tab.write(fnout, format='fits')