import sys
from pathlib import Path

from .version import __version__

# General imports
import os
import sys

if os.path.exists(os.path.join(os.getcwd(), 'config')): # You're 1 up from config?
    sys.path.insert(0, os.path.join(os.getcwd(), 'config'))
else: # You're working from a directory parallel with config?
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../config')))

# Miscellaneous science imports
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathos.pools import ProcessPool


# Local imports
try:
    import config as conf
except:
    raise RuntimeError('Cannot find configuration file!')
if 'name' not in conf.DETECTION:
    conf.DETECTION['name'] = 'Detection'
for band in conf.BANDS:
    if 'name' not in conf.BANDS[band].keys():
        conf.BANDS[band]['name'] = band.replace('_', ' ')
from .mosaic import Mosaic
from .brick import Brick


# Make sure no interactive plotting is going on.
plt.ioff()
import warnings
warnings.filterwarnings("ignore")# General imports
if os.path.exists(os.path.join(os.getcwd(), 'config')): # You're 1 up from config?
    sys.path.insert(0, os.path.join(os.getcwd(), 'config'))
else: # You're working from a directory parallel with config?
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../config')))

print(
f"""
====================================================================
T H E
 ________    _       _______     ____    ____  ________  _______        
|_   __  |  / \     |_   __ \   |_   \  /   _||_   __  ||_   __ \    
  | |_ \_| / _ \      | |__) |    |   \/   |    | |_ \_|  | |__) |   
  |  _|   / ___ \     |  __ /     | |\  /| |    |  _| _   |  __ /    
 _| |_  _/ /   \ \_  _| |  \ \_  _| |_\/_| |_  _| |__/ | _| |  \ \_ 
|_____||____| |____||____| |___||_____||_____||________||____| |___|
                                                                    
--------------------------------------------------------------------
 M O D E L   P H O T O M E T R Y   W I T H   T H E   T R A C T O R   
--------------------------------------------------------------------
    Version {__version__}                               
    (C) 2018-2023 -- J. Weaver (DAWN, UMASS)          
====================================================================

CONSOLE_LOGGING_LEVEL ..... {conf.CONSOLE_LOGGING_LEVEL}			
LOGFILE_LOGGING_LEVEL ..... {conf.LOGFILE_LOGGING_LEVEL}												
PLOT ...................... {conf.PLOT}																		
NCPUS ..................... {conf.NCPUS}																			
OVERWRITE ................. {conf.OVERWRITE} 
"""	
)

# Load logger
from .utils import start_logger
logger = start_logger()

print('You should start by running farmer.validate()!')

# General imports
import numpy as np
from tqdm import tqdm


# Look at mosaics and check they exist
def validate():
    logger.info('Validate bands...')
    Mosaic('detection', load=False)
    for band in conf.BANDS.keys():
        Mosaic(band, load=False)
    logger.info('All bands validated successfully.')


def get_mosaic(band, load=True):
    return Mosaic(band, load=load)

def build_bricks(brick_ids=None, include_detection=True, bands=None, write=False):
    if bands is not None: # some kind of manual job
        if np.isscalar(bands):
            bands = [bands,]
    elif bands == 'detection':
        bands = [bands,]
        include_detection = False
    else:
        bands = list(conf.BANDS.keys())

    # Check first
    for band in bands:
        if band == 'detection':
            break
        if band not in conf.BANDS.keys():
            raise RuntimeError(f'Cannot find {band} -- check your configuration file!')

    if include_detection:
        bands = ['detection'] + bands

    # Generate brick_ids
    n_bricks = 1
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)

    # Build bricks
    if np.isscalar(brick_ids) | (n_bricks == 1): # single brick built in memory and saved
        for band in bands:
            mosaic = get_mosaic(band, load=True)
            try:
                mosaic.add_to_brick(brick)
            except:
                brick = mosaic.spawn_brick(brick_ids)
            del mosaic
        if write: 
            brick.write(allow_update=False, filetype='hdf5')
        if conf.PLOT > 2:
            brick.plot_image(show_catalog=False, show_groups=False)
        return brick
    else: # If brick_ids is none, then we're in production. Load in mosaics, make bricks, update files.
        for band in bands:
            mosaic = get_mosaic(band, load=True)
            arr = brick_ids
            if conf.CONSOLE_LOGGING_LEVEL != 'DEBUG':
                arr = tqdm(brick_ids)
            logger.info('Spawning or updating bricks...')
            for brick_id in arr:
                try:
                    mosaic.add_to_brick(brick)
                except:
                    brick = mosaic.spawn_brick(brick_id)
                brick.write(allow_update=True, filetype='hdf5')
                if conf.PLOT > 2:
                    brick.plot_image(show_catalog=False, show_groups=False)
            del mosaic
        return

def load_brick(brick_id):
    return Brick(brick_id, load=True)

def update_bricks(brick_ids=None, bands=None):
    if bands is not None: # some kind of manual job
        if np.isscalar(bands):
            bands = [bands,]
    else:
        bands = list(conf.BANDS.keys())
        
    # get bricks with 'brick_ids' for 'bands'
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)

    # Update bricks where needed
    
    if np.isscalar(brick_ids): # single brick built in memory and saved
        brick = load_brick(brick_ids)
        for band in bands:
            if band not in brick.bands:
                logger.warning(f'{band} not found in brick #{brick_ids}! Updating...')
                mosaic = get_mosaic(band, load=True)
                mosaic.add_to_brick(brick)
                del mosaic
                brick.write(allow_update=False, filetype='hdf5')
        if conf.PLOT > 2:
            brick.plot_image(show_catalog=False, show_groups=False)
        return brick

    else: # If brick_ids is none, then we're in production. Load in mosaics, make bricks, update files.
        for band in bands:
            mosaic = get_mosaic(band, load=True)
            arr = brick_ids
            if conf.CONSOLE_LOGGING_LEVEL != 'DEBUG':
                arr = tqdm(brick_ids)
            logger.info('Spawning or updating bricks...')
            for brick_id in arr:
                brick = load_brick(brick_id)
                for band in bands:
                    if band not in brick.bands:
                        logger.warning(f'{band} not found in brick #{brick_id}! Updating...')
                        mosaic = get_mosaic(band, load=True)
                        mosaic.add_to_brick(brick)
                        del mosaic
                        brick.write(allow_update=False, filetype='hdf5')
                if conf.PLOT > 2:
                    brick.plot_image(show_catalog=False, show_groups=False)
            del mosaic

def detect_sources(brick_ids=None, band='detection', imgtype='science', brick=None, write=False):

    if brick_ids is None and brick is not None:
        # run the brick given directly
        # This can also be run by brick.detect_sources, but we also write it out if asked for!
        brick.detect_sources(band=band, imgtype=imgtype)

        if write:
            brick.write(allow_update=True)

        return brick

    elif brick_ids is not None and brick is None:
        # run the brick(s) asked for

        # Generate brick_ids
        if brick_ids is None:
            n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
            brick_ids = 1 + np.arange(n_bricks)
        elif np.isscalar(brick_ids):
            brick_ids = [brick_ids,]

        # Loop over bricks
        for brick_id in brick_ids:
            
            # does the brick exist? load it.
            try:
                brick = load_brick(brick_id)
            except:
                brick = build_bricks(brick_id,  bands='detection')

            # detection
            brick.detect_sources(band=band, imgtype=imgtype)

            if write:
                brick.write(allow_update=True)

            return brick

    else:
        raise RuntimeError('Arguments are overspecified! Either provide brick_id(s) or a brick directly, not both.')

def generate_models(brick_ids=None, group_ids=None, bands=conf.MODEL_BANDS, imgtype='science'):
    # get bricks with 'brick_ids' for 'bands'
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)
    elif np.isscalar(brick_ids):
        brick_ids = [brick_ids,]

    # Loop over bricks (or just one!)
    for brick_id in brick_ids:
        # does the brick exist? load it.
        try:
            brick = load_brick(brick_id)
        except:
            brick = build_bricks(brick_id,  bands=bands)

        # check that detection exists
        assert 'detection' in brick.bands, f'No detection information contained in brick #{brick.brick_id}!'

        #TODO make sure background is dealt with

        # detect sources
        if imgtype not in brick.catalogs['detection']:
            brick.detect_sources()
            if len(brick_ids) > 1:
                brick.write_hdf5(allow_update=True)

        # process the groups
        brick.process_groups(group_ids=group_ids, imgtype=imgtype, mode='model')

        # write brick
        brick.write_hdf5(allow_update=True)
        brick.write_catalog(allow_update=True)

        # ancillary stuff (e.g., residual brick)
        brick.build_all_images()
        brick.write_fits(allow_update=True)

    if np.isscalar(brick_ids):
        return brick

def photometer(brick_ids=None, group_ids=None, bands=None, imgtype='science'):
    # get bricks with 'brick_ids' for 'bands'
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)

    if np.isscalar(brick_ids):
        brick_ids = [brick_ids,]

    # Loop over bricks (or just one!)
    for brick_id in brick_ids:
        # does the brick exist? load it.
        try:
            brick = load_brick(brick_id)
            update_bricks(brick_id, bands)
        except:
            brick = build_bricks(brick_id)

        # detect sources
        if imgtype not in brick.catalogs['detection']:
            brick.detect_sources()
            if len(brick_ids) > 1:
                brick.write_hdf5(allow_update=True)

        # if models aren't prepared, then determine them and run phot
        if len(brick.model_catalog) == 0:        # TODO make this ironclad!     
            brick.process_groups(group_ids=group_ids, imgtype=imgtype, mode='all')
        else: # just run phot
            brick.process_groups(group_ids=group_ids, imgtype=imgtype, mode='photometry')

        # write brick
        brick.write_hdf5(allow_update=True)
        brick.write_catalog(allow_update=True)

        # ancillary stuff (e.g., residual brick)
        brick.build_all_images()
        brick.write_fits(allow_update=True)
    
    if np.isscalar(brick_ids):
        return brick

def quick_group(brick_id=1, group_id=524, brick=None):
    if not ((brick is not None) & isinstance(brick, Brick)):
        try:
            brick = load_brick(brick_id)
        except:
            brick = build_bricks(brick_id)
    brick.detect_sources()
    group = brick.spawn_group(group_id)
    group.determine_models()
    group.force_models()
    group.write_catalog(overwrite=True)
    return group

def rebuild_mosaic(brick_ids=None, bands=None, imgtype='science'):

    # build a fresh mosaic...

    raise RuntimeError('Not implelented yet!')