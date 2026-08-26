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
from astropy.io import fits
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

# Bring the forkserver up NOW, while this process is still small.
#
# run_group() forks a child per group to enforce GROUP_TIMEOUT (a subprocess is
# unavoidable: the work is in C extensions, where signal.alarm cannot interrupt and
# threads cannot be killed). Under 'forkserver' that child is forked from a small
# pristine server instead of from a pool worker holding a multi-GB brick.
#
# The server is otherwise started lazily on first use -- which would be inside a
# loaded worker, losing the entire saving. Placement is load-bearing in both
# directions: below the sys.path insert above, so the server can import `config`;
# and above the mosaic/brick imports below, so it forks from a small process.
#
# Three gotchas this block exists to handle:
#  1. Creating a Queue does NOT spawn the server; only an actual Process does.
#  2. Without set_forkserver_preload, every child re-imports farmer at unpickle time.
#  3. The server preloads 'farmer', so it re-executes THIS FILE. Without the
#     environment guard below that re-runs the probe, which starts another server,
#     which preloads farmer again -- an unbounded chain of processes that hangs the
#     import. Reproduced on CPython 3.9: the guard cuts it to exactly two imports
#     (this process, plus the server). The variable is inherited by the server
#     because it is spawned with the parent's environment.
if getattr(conf, 'GROUP_TIMEOUT', None) is not None and not os.environ.get('_FARMER_FORKSERVER'):
    import multiprocessing as _mp
    try:
        os.environ['_FARMER_FORKSERVER'] = '1'   # set BEFORE start(), so the server sees it
        _mp.set_forkserver_preload(['farmer', 'config'])
        _probe = _mp.get_context('forkserver').Process(target=int)
        _probe.start()
        _probe.join()
    except Exception as _e:
        import warnings as _w
        _w.warn(f'Could not start forkserver ({_e}); timeout children will use fork')

from .mosaic import Mosaic
from .brick import Brick

# Make sure no interactive plotting is going on.
plt.ioff()
import warnings
warnings.filterwarnings("ignore")# General imports

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


def validate(strict=True):
    """Check the configuration before committing hours of compute to it.

    Verifies, without loading any pixel data:

    * every configured output directory exists and is writable (creating it if
      necessary -- nothing else in the package ever did, so a missing
      ``PATH_FIGURES`` used to surface as every group failing at PLOT > 0);
    * each band's science file exists, its WCS parses, and its PSF loads;
    * weight and mask arrays have the same dimensions as their science image,
      read from the FITS headers rather than by loading the arrays;
    * each photometric band declares a ``zeropoint`` -- otherwise the missing
      key surfaces as a KeyError in ``get_params`` after the brick has been fit;
    * each band declares a recognised ``weight_type``.

    Args:
        strict: If True, raise on any problem. If False, log them and return
            the list instead, so a caller can decide.

    Returns:
        list: The problems found. Empty when the configuration is sound.

    Raises:
        RuntimeError: If any problem is found and ``strict`` is True.
    """
    logger.info('Validating configuration...')
    problems = []

    # 1. Output paths. Create them now rather than failing at hour three.
    for name in ('PATH_BRICKS', 'PATH_FIGURES', 'PATH_PSFMODELS',
                 'PATH_CATALOGS', 'PATH_ANCILLARY', 'PATH_LOGS'):
        path = getattr(conf, name, None)
        if path is None:
            problems.append(f'{name} is not set in the configuration')
            continue
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            problems.append(f'{name}: cannot create {path} ({e})')
            continue
        if not os.access(path, os.W_OK):
            problems.append(f'{name}: {path} is not writable')
    logger.info(f'  Output paths ... {"OK" if not problems else "PROBLEMS"}')

    # 2. Per-band checks.
    all_bands = [('detection', conf.DETECTION)] + list(conf.BANDS.items())
    for band, props in all_bands:
        ext = props.get('extension', 0)

        if 'science' not in props:
            problems.append(f'{band}: no science image configured')
            continue
        if not os.path.exists(props['science']):
            problems.append(f'{band}: science image not found at {props["science"]}')
            continue

        # array dimensions must agree, or Cutout2D silently pairs the wrong
        # inverse variance with the science pixels
        ref_shape = None
        for imgtype in ('science', 'weight', 'mask'):
            if imgtype not in props:
                continue
            try:
                hdr = fits.getheader(props[imgtype], ext=ext)
                shape = (hdr['NAXIS2'], hdr['NAXIS1'])
            except (OSError, KeyError) as e:
                problems.append(f'{band}: cannot read {imgtype} header ({e})')
                continue
            if ref_shape is None:
                ref_shape = shape
            elif shape != ref_shape:
                problems.append(f'{band}: {imgtype} is {shape} but science is {ref_shape}')

        if band != 'detection':
            if 'zeropoint' not in props:
                problems.append(f'{band}: no zeropoint configured')
            wtype = props.get('weight_type', 'invvar')
            if wtype not in ('invvar', 'sigma', 'variance'):
                problems.append(f"{band}: weight_type must be 'invvar', 'sigma' or "
                                f"'variance', not {wtype!r}")
            elif 'weight' in props and 'weight_type' not in props:
                logger.warning(f'{band}: weight_type not declared, assuming inverse '
                               f'variance. Set it explicitly in config.BANDS.')

        # this also probe-loads the PSF
        try:
            Mosaic(band, load=False)
        except Exception as e:
            problems.append(f'{band}: {e}')

    for band in conf.MODEL_BANDS:
        if band not in conf.BANDS:
            problems.append(f'MODEL_BANDS lists {band!r}, which is not a configured band')

    if problems:
        msg = 'Configuration problems:\n  ' + '\n  '.join(problems)
        if strict:
            raise RuntimeError(msg)
        logger.error(msg)
        return problems

    logger.info('All bands validated successfully.')
    return problems


def get_mosaic(band, load=True):
    """Return a ``Mosaic`` object for the specified band.

    Args:
        band: Band name (e.g. ``'detection'``, ``'g'``, ``'r'``).
        load: If True, load all configured image arrays into memory.
            If False, only validate paths and read the WCS.
            Defaults to True.

    Returns:
        Mosaic: Initialised mosaic for the requested band.
    """
    return Mosaic(band, load=load)

def build_bricks(brick_ids=None, include_detection=True, bands=None, write=True):
    """Build or update HDF5 brick files from full-field mosaics.

    For each brick ID, loads the required mosaics one at a time and cuts out
    the brick sub-image.  When processing a single brick the populated
    ``Brick`` object is returned directly.  When processing multiple bricks
    the files are written incrementally and a list of successfully built brick
    IDs is returned.  Bricks with no detection-band flux are skipped and
    excluded from the return value.

    Args:
        brick_ids: Integer or array of brick IDs to build.  ``None`` builds
            all bricks defined by ``conf.N_BRICKS``. Defaults to None.
        include_detection: If True, prepend ``'detection'`` to the band list
            even when ``bands`` does not include it. Defaults to True.
        bands: Band name(s) to include.  ``None`` uses all bands in
            ``conf.BANDS``.  Pass a single string or list of strings.
            Defaults to None.
        write: If True, write each brick to an HDF5 file after building.
            Defaults to True.

    Returns:
        Brick or list: The populated ``Brick`` (single-brick mode) or a list
            of successfully built brick IDs (multi-brick mode). Bricks with no
            detection-band flux are excluded from the returned list.

    Raises:
        RuntimeError: If a requested band is not in the configuration.
    """
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
    if np.isscalar(brick_ids) or (n_bricks == 1): # single brick built in memory and saved
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
    """Check whether a brick HDF5 file already contains a specific band.

    Reads only the top-level ``bands`` dataset. This used to call
    ``recursively_load_dict_contents_from_group``, which materialises every image
    array, catalog and model in the file -- the same work ``read_hdf5`` does --
    so the "much faster than loading the full brick" claim was the opposite of
    the truth, and every hit was paid for twice.

    Args:
        brick_id: Integer brick identifier.
        band: Band name to look for.
        tag: Optional tag string appended to the filename as ``_{tag}``.
            Defaults to None.
        silent: If True, suppress log output. Defaults to True.

    Returns:
        bool: True if the band is present; False if not, or if the brick file
            does not exist or has no ``bands`` dataset.

    Raises:
        OSError: If the file exists but cannot be read (e.g. it is corrupt).
            Previously swallowed, which turned a corrupt brick into a silent
            and expensive rebuild.
    """
    import h5py

    stag = f'_{tag}' if tag is not None else ''
    filename = f'B{brick_id}{stag}.h5'
    path = os.path.join(conf.PATH_BRICKS, filename)

    if not os.path.exists(path):
        return False
    with h5py.File(path, 'r') as hf:
        if 'bands' not in hf:
            if not silent:
                logger.debug(f'Brick #{brick_id} has no "bands" dataset.')
            return False
        return band in np.asarray(hf['bands'][...]).astype(str).tolist()

def load_brick(brick_id, silent=False, tag=None):
    """Load an existing brick from its HDF5 file on disk.

    Args:
        brick_id: Integer brick identifier.
        silent: If True, suppress informational log messages.
            Defaults to False.
        tag: Optional tag string appended to the filename as ``_{tag}``.
            Defaults to None.

    Returns:
        Brick: Fully loaded brick object.

    Raises:
        FileNotFoundError: If the brick file cannot be found.
    """
    return Brick(brick_id, load=True, silent=silent, tag=tag)


def _load_or_build_brick(brick_id, bands=None, silent=False):
    """Load a brick from disk, building it from the mosaics if it is not there yet.

    Single implementation of an idiom that had been copy-pasted to six call sites,
    five of which caught an exception ``read_hdf5`` does not raise, leaving the
    build-from-mosaics fallback unreachable.

    Args:
        brick_id: Integer brick identifier.
        bands: Band name(s) to include when the brick has to be built.
            ``None`` builds every configured band.
        silent: If True, suppress informational log messages while loading.

    Returns:
        Brick: The loaded or newly built brick.
    """
    try:
        return load_brick(brick_id, silent=silent)
    except (FileNotFoundError, IOError) as e:
        logger.warning(f'Brick #{brick_id} is not on disk ({e}). Building it from the mosaics...')
        return build_bricks(brick_id, bands=bands)


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
                try:
                    brick = load_brick(brick_id, silent=(conf.CONSOLE_LOGGING_LEVEL != 'DEBUG'))
                except (FileNotFoundError, IOError) as e:
                    # this one skips rather than builds: update_bricks only updates
                    logger.warning(f'Skipping brick #{brick_id}: {e}')
                    continue
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
            brick = _load_or_build_brick(brick_id, bands='detection', silent=True)

            # Log initial memory
            log_memory_usage(logger, f'Brick {brick_id} start', verbose=False)
            
            # Detection
            brick.detect_sources(band=band, imgtype=imgtype)
            if not getattr(brick, 'is_empty', False):
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
        if not getattr(brick, 'is_empty', False):
            brick.transfer_maps()

        if write:
            brick.write(allow_update=True)

        if cleanup:
            brick.cleanup_after_detection()

        return brick

    if brick_ids is not None and brick is None:
        if np.isscalar(brick_ids):
            brick = load_brick(brick_ids)
            brick.detect_sources(band=band, imgtype=imgtype)
            if not getattr(brick, 'is_empty', False):
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
        brick = _load_or_build_brick(brick_id, bands='detection')

        # detection
        brick.detect_sources(band=band, imgtype=imgtype)

        if write:
            brick.write(allow_update=True)

        if cleanup:
            brick.cleanup_after_detection()

def generate_models(brick_ids=None, group_ids=None, bands=conf.MODEL_BANDS, imgtype='science'):
    """Determine the best-fit morphological model for every source in one or more bricks.

    Loads (or builds) each brick, runs source detection if not already done,
    processes all groups through the model-selection decision tree, writes the
    updated brick HDF5, writes the source catalog, and reconstructs
    model/residual images.

    Args:
        brick_ids: Brick ID(s) to process. If ``None``, processes all bricks
            defined by ``conf.N_BRICKS``.
        group_ids: Specific group ID(s) to model. If ``None``, models all
            groups in each brick.
        bands: Band identifiers used jointly for model determination.
            Defaults to ``conf.MODEL_BANDS``.
        imgtype: Image type key for the detection catalog lookup.
            Defaults to ``'science'``.

    Returns:
        Brick: The processed brick when ``brick_ids`` is scalar; ``None``
            when processing multiple bricks.

    Raises:
        AssertionError: If the brick does not contain detection data.
    """
    # get bricks with 'brick_ids' for 'bands'
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)
    elif np.isscalar(brick_ids):
        brick_ids = [brick_ids,]

    # Loop over bricks (or just one!)
    for brick_id in brick_ids:
        # Attempt to load existing brick; if it doesn't exist, build it from scratch
        brick = _load_or_build_brick(brick_id, bands=bands)

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
    """Measure forced photometry in all configured bands for one or more bricks.

    Loads (or builds) each brick, updates it with any missing bands, runs
    detection if needed, and then either runs the full pipeline
    (model determination + photometry) if no models exist, or runs
    photometry-only if models are already present. Writes the HDF5 brick,
    source catalog, and model/residual images.

    Args:
        brick_ids: Brick ID(s) to process. If ``None``, processes all bricks.
        group_ids: Specific group ID(s). If ``None``, processes all groups.
        bands: Bands to measure photometry in. If ``None``, uses all
            configured bands from ``conf.BANDS``.
        imgtype: Image type key for catalog lookup. Defaults to ``'science'``.

    Returns:
        Brick: The processed brick when ``brick_ids`` is scalar; ``None``
            when processing multiple bricks.
    """
    # get bricks with 'brick_ids' for 'bands'
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)

    if np.isscalar(brick_ids):
        brick_ids = [brick_ids,]

    # Loop over bricks (or just one!)
    for brick_id in brick_ids:
        # does the brick exist? load it.
        brick = _load_or_build_brick(brick_id, bands=bands)
        update_bricks(brick_id, bands)

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

        # aperture photometry -- a no-op unless conf.DO_APERTURE_PHOT is set
        brick.measure_apertures()

        # write brick
        brick.write_hdf5(allow_update=True)
        brick.write_catalog(allow_update=True)
        # Apertures go to their own file. Eight columns per aperture per band would
        # otherwise push a wide run past the 999-column FITS ceiling, and a
        # cross-check should not be able to make the main catalog unwritable.
        brick.write_aperture_catalog(allow_update=True)

        # ancillary stuff (e.g., residual brick)
        brick.build_all_images()
        brick.write_fits(allow_update=True)
    
    if np.isscalar(brick_ids):
        return brick

def quick_group(brick_id=1, group_id=524, brick=None):
    """Convenience function to quickly process a single group."""
    if not ((brick is not None) & isinstance(brick, Brick)):
        # Load existing brick or build if not found
        brick = _load_or_build_brick(brick_id)
    brick.detect_sources()
    group = brick.spawn_group(group_id)
    group.determine_models()
    group.force_models()
    group.write_catalog(overwrite=True)
    return group

def rebuild_mosaic(brick_ids=None, bands=None, imgtype='science'):
    """Reconstruct a full-field mosaic from processed bricks.

    Not yet implemented.

    Raises:
        NotImplementedError: Always — this function is a placeholder. Use the
            per-brick FITS products in ``conf.PATH_ANCILLARY`` and mosaic them
            with an external tool (e.g. ``reproject.mosaicking``) in the meantime.
    """
    raise NotImplementedError(
        'rebuild_mosaic is a placeholder. Mosaic the per-brick FITS products in '
        f'{conf.PATH_ANCILLARY} with an external tool for now.')