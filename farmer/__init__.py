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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
try:
    import config as conf
except ImportError as e:
    raise RuntimeError(f'Cannot find configuration file! Error: {e}')
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
    (C) 2018-2026 -- J. Weaver (DAWN, MIT)          
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


def validate():
    """Validate that all configured mosaics exist and are properly configured.
    
    Checks the detection mosaic and all configured photometric bands to ensure
    they can be loaded and have valid WCS, PSF models, and data paths.
    
    Raises:
        RuntimeError: If any mosaic fails validation
    """
    logger.info('Validate bands...')
    Mosaic('detection', load=False)
    for band in conf.BANDS.keys():
        Mosaic(band, load=False)
    logger.info('All bands validated successfully.')


def get_mosaic(band, load=True):
    """Get a Mosaic object for the specified band.
    
    Args:
        band: Band name (e.g., 'detection', 'g', 'r', 'i')
        load: If True, load image data into memory. If False, only load metadata.
        
    Returns:
        Mosaic object for the specified band
    """
    return Mosaic(band, load=load)

def build_bricks(brick_ids=None, include_detection=True, bands=None, write=True):
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
            continue
        if band not in conf.BANDS.keys():
            raise RuntimeError(f'Cannot find {band} -- check your configuration file!')

    if include_detection:
        bands = ['detection'] + bands

    # Generate brick_ids
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)
    if np.isscalar(brick_ids):
        n_bricks = 1
    else:
        n_bricks = len(brick_ids)

    # Build bricks
    if np.isscalar(brick_ids) | (n_bricks == 1): # single brick built in memory and saved
        for band in bands:
            mosaic = get_mosaic(band, load=True)
            if band == 'detection':
                brick = mosaic.spawn_brick(brick_ids)
            else:
                mosaic.add_to_brick(brick)
            del mosaic
        if write: 
            brick.write(allow_update=False, filetype='hdf5')
        if conf.PLOT > 2:
            brick.plot_image(show_catalog=False, show_groups=False)
        return brick
    else: # If brick_ids is none, then we're in production. Load in mosaics, make bricks, update files.
        skiplist = []
        for band in bands:
            mosaic = get_mosaic(band, load=True)
            arr = brick_ids
            if conf.CONSOLE_LOGGING_LEVEL != 'DEBUG':
                arr = tqdm(brick_ids)
            logger.info(f'Spawning or updating bricks for band {band}...')
            for brick_id in arr:
                if brick_id in skiplist:
                    logger.debug(f'Brick {brick_id} has been skipped due to no detection information! Skipping...')
                    continue
                if band == 'detection':
                    brick = mosaic.spawn_brick(brick_id, silent=(conf.CONSOLE_LOGGING_LEVEL != 'DEBUG'))
                    if 'detection' not in brick.data:
                        logger.debug(f'Brick {brick_id} has no detection information! Skipping...')
                        skiplist.append(brick_id)
                        continue
                    if np.nansum(brick.data['detection']['science'].data>0) == 0:
                        logger.debug(f'Brick {brick_id} has no detection information! Skipping...')
                        skiplist.append(brick_id)
                        continue
                else:
                    # Lazy load: check if band already exists before loading full brick
                    if brick_has_band(brick_id, band, silent=(conf.CONSOLE_LOGGING_LEVEL != 'DEBUG')):
                        logger.debug(f'Brick {brick_id} already has band {band}, skipping...')
                        continue
                    
                    brick = load_brick(brick_id, silent=(conf.CONSOLE_LOGGING_LEVEL != 'DEBUG'))
                    mosaic.add_to_brick(brick)
                brick.write(allow_update=True, filetype='hdf5')
                if conf.PLOT > 2:
                    brick.plot_image(show_catalog=False, show_groups=False)
                del brick
            del mosaic
        return [bid for bid in brick_ids if bid not in skiplist] # return the useful brick numbers

def brick_has_band(brick_id, band, tag=None, silent=True):
    """Check if a brick has a specific band without loading full image data.
    
    Reads only the metadata from the HDF5 file to check if the brick already
    contains the specified band. This is much faster than loading the full brick.
    
    Args:
        brick_id: Brick identifier (integer)
        band: Band name to check for
        tag: Optional tag for alternate brick versions
        silent: If True, suppress logger output
        
    Returns:
        bool: True if brick has the band, False if not or brick doesn't exist
    """
    import h5py
    from .utils import recursively_load_dict_contents_from_group
    
    stag = f'_{tag}' if tag is not None else ''
    filename = f'B{brick_id}{stag}.h5'
    path = os.path.join(conf.PATH_BRICKS, filename)
    
    try:
        with h5py.File(path, 'r') as hf:
            attr = recursively_load_dict_contents_from_group(hf)
            if 'bands' in attr:
                return band in attr['bands']
        return False
    except (IOError, FileNotFoundError, Exception):
        return False

def load_brick(brick_id, silent=False, tag=None):
    """Load an existing brick from disk.
    
    Args:
        brick_id: Brick identifier (integer)
        silent: If True, suppress logger output
        tag: Optional tag for alternate brick versions
        
    Returns:
        Brick object loaded from disk
        
    Raises:
        IOError: If brick file cannot be found or loaded
    """
    return Brick(brick_id, load=True, silent=silent, tag=tag)

def update_bricks(brick_ids=None, bands=None, overwrite=False):
    """Update existing bricks with missing bands.
    
    Uses lazy loading to check if a brick already has a band before loading
    the full brick into memory. Only loads and modifies bricks that need updates.
    
    Args:
        brick_ids: Brick ID(s) to update. If None, update all bricks.
        bands: Bands to add/update. If None, use all configured bands.
        overwrite: If True, re-add all bands even if they exist.
        
    Returns:
        For single brick: returns the brick object
        For multiple: returns list of updated brick IDs
    """
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
            if overwrite or band not in brick.bands:
                logger.warning(f'{band} not found in brick #{brick_ids}! Updating...')
                mosaic = get_mosaic(band, load=True)
                mosaic.add_to_brick(brick)
                del mosaic
                brick.write(allow_update=True, filetype='hdf5')
        if conf.PLOT > 2:
            brick.plot_image(show_catalog=False, show_groups=False)
        return brick

    else: # Multiple bricks - use lazy loading to check before full load
        updated_bricks = []
        for band in bands:
            mosaic = get_mosaic(band, load=True)
            arr = brick_ids
            if conf.CONSOLE_LOGGING_LEVEL != 'DEBUG':
                arr = tqdm(brick_ids, desc=f'Updating bricks with {band}')
            logger.info(f'Updating bricks for band {band}...')
            
            for brick_id in arr:
                # Lazy load: check if band already exists before loading full brick
                if not overwrite and brick_has_band(brick_id, band, silent=(conf.CONSOLE_LOGGING_LEVEL != 'DEBUG')):
                    logger.debug(f'Brick {brick_id} already has band {band}, skipping...')
                    continue
                
                # Only load brick if it needs updating
                brick = load_brick(brick_id, silent=(conf.CONSOLE_LOGGING_LEVEL != 'DEBUG'))
                mosaic.add_to_brick(brick)
                brick.write(allow_update=True, filetype='hdf5')
                updated_bricks.append((brick_id, band))
                
                if conf.PLOT > 2:
                    brick.plot_image(show_catalog=False, show_groups=False)
                del brick
            del mosaic
        
        return updated_bricks

def detect_sources_lite(brick_ids=None, band='detection', imgtype='science', 
                       write_catalog=True, cleanup=True):
    """Hit-and-run source detection with minimal memory footprint.
    
    Extract source catalogs from bricks without keeping large image arrays
    or model tracking data in memory. Ideal for batch processing many bricks
    when you only need the catalogs.
    
    Args:
        brick_ids: Brick ID(s) to process. If None, process all bricks.
        band: Detection band (default: 'detection')
        imgtype: Image type to process (default: 'science')
        write_catalog: If True, write catalog to disk immediately
        cleanup: If True, aggressively clean up temporary data after detection
        
    Returns:
        For single brick: returns the brick object
        For multiple: returns list of successful brick IDs
        
    Example:
        # Process all bricks, write catalogs only
        detect_sources_lite(write_catalog=True)
        
        # Process specific bricks
        detect_sources_lite(brick_ids=[1, 2, 3])
    """
    from .utils import log_memory_usage
    
    logger.info(f'Running detect_sources_lite (cleanup={cleanup})')
    
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)
    elif np.isscalar(brick_ids):
        brick_ids = [brick_ids,]
    
    successful_bricks = []
    last_brick = None
    
    for brick_id in tqdm(brick_ids, desc='Detecting sources (lite mode)'):
        try:
            # Load brick
            try:
                brick = load_brick(brick_id, silent=True)
            except (IOError, FileNotFoundError) as e:
                logger.debug(f'Building brick #{brick_id} from mosaics...')
                brick = build_bricks(brick_id, bands='detection')
            
            # Log initial memory
            log_memory_usage(logger, f'Brick {brick_id} start', verbose=False)
            
            # Detection
            brick.detect_sources(band=band, imgtype=imgtype)
            brick.transfer_maps()
            
            # Write catalog immediately
            if write_catalog:
                brick.write_catalog(allow_update=True)
                logger.debug(f'Wrote catalog for brick {brick_id}')
            
            # Aggressive cleanup
            if cleanup:
                brick.cleanup_after_detection(keep_segmap=False, keep_groupmap=False)
                brick.cleanup_headers(keep_wcs_only=True)
                
                # Delete image data if not needed for anything else
                for band_name in brick.data:
                    brick.data[band_name] = {}
                
                logger.debug(f'Memory cleanup for brick {brick_id}')
            
            log_memory_usage(logger, f'Brick {brick_id} end', verbose=False)
            successful_bricks.append(brick_id)
            last_brick = brick
            
            del brick
            
        except Exception as e:
            logger.error(f'Error processing brick {brick_id}: {e}')
    
    if len(successful_bricks) == 1 and len(brick_ids) == 1:
        return last_brick
    else:
        return successful_bricks

def detect_sources(brick_ids=None, band='detection', imgtype='science', brick=None, 
                   write=False, lite_mode=False, cleanup=False):
    """Detect sources in one or more bricks.
    
    Args:
        brick_ids: Brick ID(s) to process
        band: Detection band (default: 'detection')
        imgtype: Image type (default: 'science')
        brick: Single brick object to process directly
        write: If True, write brick to disk after detection
        lite_mode: If True, use minimal memory mode (experimental)
        cleanup: If True, clean up temporary data after detection
    """

    if lite_mode:
        logger.info('Using lite_mode for detect_sources')
        return detect_sources_lite(brick_ids=brick_ids, band=band, imgtype=imgtype,
                                   write_catalog=write, cleanup=True)

    if brick_ids is None and brick is not None:
        # run the brick given directly
        # This can also be run by brick.detect_sources, but we also write it out if asked for!
        brick.detect_sources(band=band, imgtype=imgtype)
        brick.transfer_maps()

        if write:
            brick.write(allow_update=True)

        if cleanup:
            brick.cleanup_after_detection()

        return brick

    if brick_ids is not None and brick is None:
        if np.isscalar(brick_ids):
            # run the brick given directly
            # This can also be run by brick.detect_sources, but we also write it out if asked for!
            brick.detect_sources(band=band, imgtype=imgtype)
            brick.transfer_maps()

            if write:
                brick.write(allow_update=True)

            if cleanup:
                brick.cleanup_after_detection()

            return brick
        else:
            # have multiple
            pass

    elif brick_ids is None and brick is None:
        # Generate brick_ids
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)

    else:
        raise RuntimeError('Arguments are overspecified! Either provide brick_id(s) or a brick directly, not both.')

    # Loop over bricks
    for brick_id in brick_ids:
        
        # does the brick exist? load it.
        try:
            brick = load_brick(brick_id)
        except (IOError, FileNotFoundError) as e:
            logger.warning(f'Could not load brick {brick_id} ({e}). Building a new brick from mosaics...')
            brick = build_bricks(brick_id, bands='detection')

        # detection
        brick.detect_sources(band=band, imgtype=imgtype)

        if write:
            brick.write(allow_update=True)

        if cleanup:
            brick.cleanup_after_detection()

def generate_models(brick_ids=None, group_ids=None, bands=conf.MODEL_BANDS, imgtype='science'):
    # get bricks with 'brick_ids' for 'bands'
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)
    elif np.isscalar(brick_ids):
        brick_ids = [brick_ids,]

    # Loop over bricks (or just one!)
    for brick_id in brick_ids:
        # Attempt to load existing brick; if it doesn't exist, build it from scratch
        try:
            brick = load_brick(brick_id)
        except (IOError, FileNotFoundError):
            logger.info(f'Brick #{brick_id} not found, building from mosaics...')
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
        except (IOError, FileNotFoundError) as e:
            logger.critical(f'Could not load brick {brick_id} ({e}). Building from scratch.')
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
    """Convenience function to quickly process a single group."""
    if not ((brick is not None) & isinstance(brick, Brick)):
        # Load existing brick or build if not found
        try:
            brick = load_brick(brick_id)
        except (IOError, FileNotFoundError):
            logger.info(f'Brick #{brick_id} not found, building...')
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