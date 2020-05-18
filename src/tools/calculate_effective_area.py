import numpy as np 
import sys
import os
from astropy.table import Table, Column
sys.path.insert(0, os.path.join('/Volumes/WD4/Current/COSMOS2020/config'))
import config as conf

from src.core import interface

def walk_through_files(path, file_prefix = 'B', file_extension='.fits'):
    """
    Generate a list of filenames at given path
    """
    output = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.startswith(file_prefix) & filename.endswith(file_extension):
                output.append(os.path.join(dirpath, filename))
    return output


MODELING = False
if MODELING:
    bands = conf.MODELING_BANDS
    MULTIBAND=''
else:
    bands = conf.BANDS
    MULTIBAND = 'MULTIBAND'

catalog = Table.read(sys.argv[1])
brick_id = np.unique(catalog['brick_id'])
# brick_fnames = walk_through_files(conf.CATALOG_DIR, file_extension=f'{MULTIBAND}.cat')
# brick_id = []
# for brick_fname in brick_fnames:    
#     fname = brick_fname.split('/')[-1]
#     print(fname)
#     if MODELING:
#         if 'MULTIBAND' in fname:
#             continue
#     int(fname.split('_')[0][1:len(f'{MULTIBAND}.cat')]) 

if len(brick_id)==0:
    raise RuntimeError('No bricks found!')

good_area = dict(zip(bands, [np.zeros(len(brick_id)) for i in np.arange(len(bands))]))
masked_area = dict(zip(bands, [np.zeros(len(brick_id)) for i in np.arange(len(bands))]))


for band in bands:
    if MODELING:
        band = f'{conf.MODELING_NICKNAME}_{band}'
    for i, bid in enumerate(brick_id):
        good_area[band][i], masked_area[band][i] = interface.estimate_effective_area(brick_id=bid,  band=band)

    good_area_deg = good_area[band].sum() * (conf.PIXEL_SCALE/3600)**2
    masked_area_deg = masked_area[band].sum() * (conf.PIXEL_SCALE/3600)**2
    print(f'Total area for {band} over {len(brick_id)} bricks: {good_area_deg:4.4f} deg2')