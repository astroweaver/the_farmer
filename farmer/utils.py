"""Utility functions for The Farmer astronomical source detection pipeline.

This module provides core utilities for:
- Loading and processing brick position data with WCS transformations
- Source detection and grouping using morphological operations
- Optimized coordinate transformations between multi-resolution grids
- Tractor model parameter extraction and caching
- HDF5 serialization/deserialization for complex nested structures
- Segmentation map manipulation and analysis

Optimized for memory efficiency and performance on large astronomical images.
"""

import config as conf
import os
import logging

import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord
from scipy.ndimage import label, binary_dilation, binary_erosion, binary_fill_holes
import astropy.units as u
from astropy.wcs import WCS
from astropy.nddata import utils
from astropy.nddata import Cutout2D
from tractor.ellipses import EllipseESoft

from tractor.psfex import PixelizedPsfEx, PixelizedPSF #PsfExModel
# from tractor.psf import HybridPixelizedPSF
from tractor.galaxy import ExpGalaxy, FracDev, SoftenedFracDev
from tractor import PointSource, DevGalaxy, EllipseE, FixedCompositeGalaxy, Fluxes
from astrometry.util.util import Tan
from tractor import ConstantFitsWcs

import time
from collections import OrderedDict
from reproject import reproject_interp
from tqdm import tqdm
import h5py
from astropy.table import meta
from tractor.wcs import RaDecPos

# from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool


# Module constants for dtype selection
# Integer type ranges
INT16_MIN = -32768
INT16_MAX = 32767
INT32_MIN = -2147483648
INT32_MAX = 2147483647
UINT16_MAX = 65535
UINT32_MAX = 4294967295

# Model type names
MODEL_TYPES = {
    'POINT': 'PointSource',
    'SIMPLE': 'SimpleGalaxy',
    'EXP': 'ExpGalaxy',
    'DEV': 'DevGalaxy',
    'COMP': 'FixedCompositeGalaxy',
}

# HDF5 string encoding
HDF5_STRING_DTYPE = '|S99'

# PSF-related constants
DEFAULT_FWHM_MIN = 1.0  # Minimum FWHM cap
PSF_SIGMA_FACTOR = 2.5  # Factor for resolution calculation


def start_logger():
    print('Starting up logging system...')

    # Start the logging
    import logging.config
    logger = logging.getLogger('farmer')

    if not len(logger.handlers):
        if conf.LOGFILE_LOGGING_LEVEL is not None:
            logging_level = logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL)
        else:
            logging_level = logging.DEBUG
        logger.setLevel(logging_level)
        logger.propagate = False
        formatter = logging.Formatter('[%(asctime)s] %(name)s :: %(levelname)s - %(message)s', '%H:%M:%S')

        # Logging to the console at logging level
        ch = logging.StreamHandler()
        ch.setLevel(logging.getLevelName(conf.CONSOLE_LOGGING_LEVEL))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if (conf.LOGFILE_LOGGING_LEVEL is None) | (not os.path.exists(conf.PATH_LOGS)):
            print('Logging information wills stream only to console.\n')
            
        else:
            # create file handler which logs even debug messages
            logging_path = os.path.join(conf.PATH_LOGS, 'logfile.log')
            print(f'Logging information will stream to console and {logging_path}\n')
            # If overwrite is on, remove old logger
            if conf.OVERWRITE & os.path.exists(logging_path):
                print('WARNING -- Existing logfile will be overwritten.')
                os.remove(logging_path)

            fh = logging.FileHandler(logging_path)
            fh.setLevel(logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger

def read_wcs(wcs, scl=1):
    t = Tan()
    t.set_crpix(wcs.wcs.crpix[0] * scl, wcs.wcs.crpix[1] * scl)
    t.set_crval(wcs.wcs.crval[0], wcs.wcs.crval[1])
    # Try PC matrix first, fall back to CD matrix if not available
    try:
        cd = wcs.wcs.pc / scl
    except AttributeError:
        cd = wcs.wcs.cd / scl
    # Assumes images have no rotation (off-diagonal terms are zero)
    t.set_cd(cd[0,0], cd[0,1], cd[1,0], cd[1,1])
    t.set_imagesize(wcs.array_shape[0] * scl, wcs.array_shape[1] * scl)
    wcs = ConstantFitsWcs(t)
    return wcs

# def load_brick_position(brick_id):
#     logger = logging.getLogger('farmer.load_brick_position')
#     # Do this relative to the detection image
#     ext = None
#     if 'extension' in conf.DETECTION:
#         ext = conf.DETECTION['extension']
#     wcs = WCS(fits.getheader(conf.DETECTION['science'], ext=ext))
#     nx, ny = wcs.array_shape
#     brick_width = nx / conf.N_BRICKS[0]
#     brick_height = ny / conf.N_BRICKS[1]
#     if brick_id <= 0:
#         raise RuntimeError(f'Cannot request brick #{brick_id} (<=0)!')
#     if brick_id > (nx * ny):
#         raise RuntimeError(f'Cannot request brick #{brick_id} on grid {nx} X {ny}!')
#     logger.debug(f'Using bricks of size ({brick_width:2.2f}, {brick_height:2.2f}) px, in grid {nx} X {ny} px')
#     xc = 0.5 * brick_width + int(((brick_id - 1) * brick_height) / nx) * brick_width
#     yc = 0.5 * brick_height + int(((brick_id - 1) * brick_height) % ny)
#     logger.debug(f'Brick #{brick_id} found at ({xc:2.2f}, {yc:2.2f}) px with size {brick_width:2.2f} X {brick_height:2.2f} px')
#     position = wcs.pixel_to_world(xc, yc)
#     upper = wcs.pixel_to_world(xc+brick_width/2., yc+brick_height/2.)
#     lower = wcs.pixel_to_world(xc-brick_width/2., yc-brick_height/2.)
#     size = abs(lower.ra - upper.ra) * np.cos(np.deg2rad(position.dec.to(u.degree).value)), abs(upper.dec - lower.dec)

#     logger.debug(f'Brick #{brick_id} found at ({position.ra:2.1f}, {position.dec:2.1f}) with size {size[0]:2.1f} X {size[1]:2.1f}')
#     return position, size

def load_brick_position(brick_id):
    """Calculate brick position and size from brick ID.
    
    Args:
        brick_id: Integer brick identifier (1-indexed)
        
    Returns:
        tuple: (center, size, buffsize) where:
            - center: SkyCoord of brick center
            - size: Tuple of (dec_height, ra_width) in degrees
            - buffsize: Tuple of buffered (dec_height, ra_width) in degrees
    """
    logger = logging.getLogger('farmer.load_brick_position')
    
    ext = None
    if 'extension' in conf.DETECTION:
        ext = conf.DETECTION['extension']
    wcs = WCS(fits.getheader(conf.DETECTION['science'], ext=ext))
    ny, nx = wcs.array_shape
    
    # Number of bricks in x and y directions
    num_bricks_x = conf.N_BRICKS[0]
    num_bricks_y = conf.N_BRICKS[1]
    
    # Calculate the exact width and height of each brick
    brick_width = nx // num_bricks_x
    brick_height = ny // num_bricks_y
    
    if brick_id <= 0:
        raise RuntimeError(f'Cannot request brick #{brick_id} (<=0)!')
    if brick_id > (num_bricks_x * num_bricks_y):
        raise RuntimeError(f'Cannot request brick #{brick_id} on grid {num_bricks_x} X {num_bricks_y}!')
    
    # Calculate the row and column for this brick_id (0-indexed)
    row = (brick_id - 1) // num_bricks_x
    column = (brick_id - 1) % num_bricks_x
    
    # Calculate the center of the brick in pixel coordinates
    xc = (column * brick_width) + (brick_width / 2)
    yc = (row * brick_height) + (brick_height / 2)
    
    logger.debug(f'Brick #{brick_id} found at ({xc:2.2f}, {yc:2.2f}) px with size {brick_width} X {brick_height} px')
    
    # Calculate the center of the region in sky coordinates
    center = wcs.pixel_to_world(xc, yc)

    # Helper function to calculate sky size from pixel dimensions
    def _calculate_sky_size(width, height):
        """Calculate sky size from pixel dimensions centered at (xc, yc)."""
        corners_x = np.array([xc - width / 2., xc + width / 2.,
                              xc + width / 2., xc - width / 2.])
        corners_y = np.array([yc - height / 2., yc - height / 2.,
                              yc + height / 2., yc + height / 2.])
        sky_corners = wcs.pixel_to_world(corners_x, corners_y)
        ra_width = sky_corners[0].separation(sky_corners[1])
        dec_height = sky_corners[1].separation(sky_corners[2])
        return (dec_height.to(u.deg), ra_width.to(u.deg))

    # Calculate base brick size in sky coordinates
    size = _calculate_sky_size(brick_width, brick_height)

    # Calculate buffered size
    pixel_scale = wcs.proj_plane_pixel_scales()[0]
    buff_size = conf.BRICK_BUFFER.to(u.deg) / pixel_scale.to(u.deg)
    brick_buffwidth = brick_width + 2 * buff_size
    brick_buffheight = brick_height + 2 * buff_size
    buffsize = _calculate_sky_size(brick_buffwidth, brick_buffheight)
    
    logger.debug(f'Brick #{brick_id} found at ({center.ra:2.1f}, {center.dec:2.1f}) with size {size[0]:2.1f} X {size[1]:2.1f}')

    return center, size, buffsize



def clean_catalog(catalog, mask, segmap=None):
    """Remove masked sources from catalog and renumber segmentation map.
    
    Args:
        catalog: Source catalog (astropy Table)
        mask: Boolean mask indicating pixels to exclude
        segmap: Segmentation map (optional). If provided, will be updated in-place.
        
    Returns:
        tuple: (cleaned_catalog, segmap) if segmap provided, else just cleaned_catalog
        
    Note:
        Segmentation map is modified in-place for efficiency with large images.
    """
    logger = logging.getLogger('farmer.clean_catalog')
    if segmap is not None:
        assert mask.shape == segmap.shape, f'Mask {mask.shape} is not the same shape as the segmentation map {segmap.shape}!'
    zero_seg = np.sum(segmap==0)
    logger.debug('Cleaning catalog...')
    tstart = time.time()

    # map the pixel coordinates to the map
    x, y = np.round(catalog['x']).astype(int), np.round(catalog['y']).astype(int)
    keep = ~mask[y, x]
    cleancat = catalog[keep]
    
    # Create a mapping array for vectorized relabeling
    # Original segment IDs are 1-indexed (1 to len(catalog))
    # Make mapping large enough to handle max segment ID in the map
    max_seg_id = max(len(catalog), np.max(segmap))
    mapping = np.zeros(max_seg_id + 1, dtype=segmap.dtype)
    
    # Set removed segments to 0, kept segments to new sequential IDs
    kept_indices = np.where(keep)[0]
    new_ids = np.arange(1, len(kept_indices) + 1, dtype=segmap.dtype)
    mapping[kept_indices + 1] = new_ids
    
    # Apply mapping in one vectorized operation
    segmap[:] = mapping[segmap]

    pc = (np.sum(segmap==0) - zero_seg) / np.size(segmap)
    logger.info(f'Cleaned {np.sum(~keep)} sources ({pc*100:2.2f}% by area), {np.sum(keep)} remain. ({time.time()-tstart:2.2f}s)')
    if segmap is not None:
        return cleancat, segmap
    else:
        return cleancat

def dilate_and_group(catalog, segmap, radius=0, fill_holes=False):
    """Dilate segmentation map and group nearby sources together.
    
    Uses morphological dilation to merge nearby sources into groups, then applies
    optimized union-find algorithm to handle sources split across multiple groups.
    
    Args:
        catalog: Source catalog with x, y positions
        segmap: Segmentation map with source IDs
        radius: Dilation radius in pixels (default: 0, no dilation)
        fill_holes: If True, fill holes in dilated regions (default: False)
        
    Returns:
        tuple: (group_ids, group_populations, groupmap)
            - group_ids: Array of group IDs for each source
            - group_populations: Array of population counts for each source's group
            - groupmap: 2D array with group IDs for each pixel
            
    Note:
        Radius is assumed to be in pixels, not physical units.
    """
    logger = logging.getLogger('farmer.identify_groups')

    # Create binary mask
    segmask = (segmap > 0).astype(np.uint8)

    # dilation
    if (radius is not None) & (radius > 0):
        logger.debug(f'Dilating segments with radius of {radius:2.2f} px')
        struct2 = create_circular_mask(2*radius, 2*radius, radius=radius)
        segmask = binary_dilation(segmask, structure=struct2).astype(np.uint8)

    if fill_holes:
        logger.debug(f'Filling holes...')
        segmask = binary_fill_holes(segmask).astype(np.uint8)

    # relabel connected components
    groupmap, n_groups = label(segmask)

    # Need to check for segments split across multiple groups
    # Build mapping of segment ID -> list of group IDs it appears in
    logger.debug('Checking for split segments...')
    
    # Vectorized approach: get all (segment, group) pairs
    seg_mask = segmap > 0
    if not np.any(seg_mask):
        # No segments - return empty results
        return np.array([]), np.array([]), groupmap
        
    seg_vals = segmap[seg_mask]
    grp_vals = groupmap[seg_mask]
    
    # Find segments appearing in multiple groups
    # Use numpy's efficient group-by functionality
    unique_segs = np.unique(seg_vals)
    seg_to_groups = {}
    
    for seg_id in unique_segs:
        seg_pixels = (seg_vals == seg_id)
        groups_for_seg = np.unique(grp_vals[seg_pixels])
        if len(groups_for_seg) > 1:
            seg_to_groups[seg_id] = groups_for_seg
    
    if seg_to_groups:
        logger.debug(f'Found {len(seg_to_groups)} segments split across multiple groups')
        
        # Union-find with path compression
        group_mapping = np.arange(n_groups + 1, dtype=np.int32)
        
        for seg_id, groups in seg_to_groups.items():
            # Merge all groups to the smallest one
            primary_group = groups[0]
            for bad_group in groups[1:]:
                group_mapping[bad_group] = primary_group
                logger.debug(f'  * merging group {bad_group} into {primary_group}')
        
        # Vectorized path compression using find-root operation
        # This replaces the slow Python loop
        def find_root(mapping):
            """Find root for all nodes with path compression."""
            changed = True
            while changed:
                new_mapping = mapping[mapping]
                changed = not np.array_equal(mapping, new_mapping)
                mapping = new_mapping
            return mapping
        
        group_mapping = find_root(group_mapping)
        
        # Apply mapping to groupmap
        groupmap = group_mapping[groupmap]
    
    # Renumber groups to be sequential starting from 1
    unique_groups = np.unique(groupmap[groupmap > 0])
    n_groups = len(unique_groups)
    final_mapping = np.zeros(groupmap.max() + 1, dtype=groupmap.dtype)
    final_mapping[unique_groups] = np.arange(1, n_groups + 1, dtype=groupmap.dtype)
    groupmap = final_mapping[groupmap]

    # report back
    logger.debug(f'Found {np.max(groupmap)} groups for {np.max(segmap)} sources.')
    segid, idx = np.unique(segmap.flatten(), return_index=True)
    group_ids = groupmap.flatten()[idx[segid>0]]

    # Vectorized group population counting
    unique_gids, gid_counts = np.unique(group_ids, return_counts=True)
    gid_to_count = dict(zip(unique_gids, gid_counts))
    group_pops = np.array([gid_to_count[gid] for gid in group_ids], dtype=np.int16)
    
    __, idx_first = np.unique(group_ids, return_index=True)
    ugroup_pops, ngroup_pops = np.unique(group_pops[idx_first], return_counts=True)

    for i in np.arange(1, 5):
        if np.sum(ugroup_pops == i) == 0:
            ngroup = 0
        else:
            ngroup = ngroup_pops[ugroup_pops==i][0]
        pc = ngroup / n_groups
        logger.debug(f'... N  = {i}: {ngroup} ({pc*100:2.2f}%) ')
    ngroup = np.sum(ngroup_pops[ugroup_pops>=5])
    pc = ngroup / n_groups
    logger.debug(f'... N >= {5}: {ngroup} ({pc*100:2.2f}%) ')

    return group_ids, group_pops, groupmap

def get_fwhm(img):
    # Simple FWHM estimation: find pixels above half-maximum
    dx, dy = np.nonzero(img > np.nanmax(img)/2.)
    try:
        fwhm = np.mean([dx[-1] - dx[0], dy[-1] - dy[0]])
    except (IndexError, ValueError):
        # Empty array or single pixel - cannot compute FWHM
        fwhm = np.nan
    return np.nanmin([DEFAULT_FWHM_MIN, fwhm])  # Cap to prevent unrealistic values

def get_resolution(img, sig=3.):
    fwhm = get_fwhm(img)
    return np.pi * (sig / (2 * PSF_SIGMA_FACTOR) * fwhm)**2

def validate_psfmodel(band, return_psftype=False):
    logger = logging.getLogger('farmer.validate_psfmodel')
    psfmodel_path = conf.BANDS[band]['psfmodel']

    if not os.path.exists(psfmodel_path):
        raise RuntimeError(f'PSF path for {band} does not exist!\n({psfmodel_path})')

    # maybe it's a table of ra, dec, and path_ending?
    try:
        psfgrid = Table.read(psfmodel_path)
        ra_try = ('ra', 'RA', 'ra_deg')
        dec_try = ('dec', 'DEC', 'dec_try')

        if np.any([ra in psfgrid.colnames for ra in ra_try]) & np.any([dec in psfgrid.colnames for dec in dec_try]):
            ra_matches = [ra for ra in ra_try if ra in psfgrid.colnames]
            dec_matches = [dec for dec in dec_try if dec in psfgrid.colnames]
            if not ra_matches or not dec_matches:
                raise RuntimeError(f'Could not find ra, dec columns in {psfmodel_path}!')
            ra_col = ra_matches[0]
            dec_col = dec_matches[0]

            psfgrid_ra = psfgrid[ra_col]
            psfgrid_dec = psfgrid[dec_col]
        
        else:
            raise RuntimeError(f'Could not find ra, dec columns in {psfmodel_path}!')
            
        psfcoords = SkyCoord(ra=psfgrid_ra*u.degree, dec=psfgrid_dec*u.degree)
        # I'm expecting that all of these psfnames are based in PATH_PSFMODELS
        psflist = np.array(psfgrid['filename'])
        psftype = 'variable'

    except (IOError, OSError, KeyError, ValueError, RuntimeError):
        # Not a valid table or missing required columns - treat as single file
        psfcoords = 'none'
        psflist = os.path.join(conf.PATH_PSFMODELS, psfmodel_path)
        psftype = 'constant'

    psfmodel = (psfcoords, psflist)

    # try out the first one
    if psftype == 'constant':
        fname = str(psfmodel[1][0])

        if fname.endswith('.psf'):
            # Try loading as PsfEx format first
            try:
                PixelizedPsfEx(fn=fname)
                logger.debug(f'PSF model for {band} identified as {psftype} PixelizedPsfEx.')

            except (ValueError, RuntimeError):
                # Fall back to generic pixelized PSF format
                img = fits.open(fname)[0].data
                img = img.astype('float32')
                PixelizedPSF(img)
                logger.debug(f'PSF model for {band} identified as {psftype} PixelizedPSF.')
            
        elif fname.endswith('.fits'):
            img = fits.open(fname)[0].data
            img = img.astype('float32')
            PixelizedPSF(img)
            logger.debug(f'PSF model for {band} identified as {psftype} PixelizedPSF.')

    if return_psftype:
        return psfmodel, psftype
    else:
        return psfmodel


def header_from_dict(params):
    logger = logging.getLogger('farmer.header_from_dict')
    """ Take in dictionary and churn out a header. Never forget configs again. """
    hdr = fits.Header()
    total_public_entries = np.sum([ not k.startswith('__') for k in params.keys()])
    logger.debug(f'header_from_dict :: Dictionary has {total_public_entries} entires')
    tstart = time()
    for i, attr in enumerate(params.keys()):
        if not attr.startswith('__'):
            logger.debug(f'header_from_dict ::   {attr}')
            value = params[attr]
            if type(value) == str:
                # store normally
                hdr.set(f'CONF{i+1}', value, attr)
            if type(value) in (float, int):
                # store normally
                hdr.set(f'CONF{i+1}', value, attr)
            if type(value) in (list, tuple):
                # freak out.
                for j, val in enumerate(value):
                    hdr.set(f'CONF{i+1}_{j+1}', str(val), f'{attr}_{j+1}')
            
    logger.debug(f'header_from_dict :: Completed writing header ({time() - tstart:2.3f}s)')
    return hdr


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = np.zeros((h, w), dtype=int)
    mask[dist_from_center <= radius] = 1
    return mask


def _select_optimal_dtype(coordinates):
    """Select optimal numpy dtype based on coordinate range.
    
    Chooses the smallest dtype that can represent the coordinate values,
    preferring unsigned types for non-negative coordinates.
    
    Args:
        coordinates: Array-like or iterable of coordinate values
        
    Returns:
        numpy dtype: Optimal dtype (uint16, uint32, uint64, int16, int32, or int64)
        
    Examples:
        >>> _select_optimal_dtype([0, 100, 1000])
        dtype('uint16')
        >>> _select_optimal_dtype([-100, 100])
        dtype('int16')
    """
    if not hasattr(coordinates, '__len__') or len(coordinates) == 0:
        return np.uint16  # Default for empty
    
    coords_array = np.asarray(coordinates)
    if coords_array.size == 0:
        return np.uint16
        
    max_coord = coords_array.max()
    min_coord = coords_array.min()
    
    # Select appropriate dtype based on data range
    if min_coord < 0:
        # Need signed integer
        if max_coord < INT16_MAX and min_coord >= INT16_MIN:
            return np.int16
        elif max_coord < INT32_MAX and min_coord >= INT32_MIN:
            return np.int32
        else:
            return np.int64
    else:
        # Can use unsigned integer (more range for same bytes)
        if max_coord < UINT16_MAX:
            return np.uint16  # 2 bytes, range: 0-65,535
        elif max_coord < UINT32_MAX:
            return np.uint32  # 4 bytes, range: 0-4,294,967,295
        else:
            return np.uint64  # 8 bytes
    

class SimpleGalaxy(ExpGalaxy):
    '''This defines the 'SIMP' galaxy profile -- an exponential profile
    with a fixed shape of a 0.45 arcsec effective radius and spherical
    shape.  It is used to detect marginally-resolved galaxies.
    '''
    shape = EllipseE(0.45, 0., 0.)

    def __init__(self, *args):
        super(SimpleGalaxy, self).__init__(*args)

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) + ')')

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1)

    def getName(self):
        return 'SimpleGalaxy'

    ### HACK -- for Galaxy.getParamDerivatives()
    def isParamFrozen(self, pname):
        if pname == 'shape':
            return True
        return super(SimpleGalaxy, self).isParamFrozen(pname) 


def map_discontinuous(input, out_wcs, out_shape, force_simple=False):
    # print(input[0].shape)
    # print(input[1])
    # print(out_wcs)
    # print(out_shape)
    # for some resolution regimes, you cannot make a new, complete segmap with the new pixel scale!
    array, in_wcs = input
    logger = logging.getLogger('farmer.map_discontinuous')
    logger.info(f'Mapping {len(np.unique(array)[1:])} regions to new resolution...')

    scl_in = np.array([val.value for val in in_wcs.proj_plane_pixel_scales()])
    scl_out = np.array([val.value for val in out_wcs.proj_plane_pixel_scales()])

    if (array.shape == out_shape) & (np.abs(scl_in - scl_out).max() < 0.001):
        logger.debug('No reprojection needed -- same shape and pixel scale')
        segs = np.unique(array.flatten()).astype(int)
        segs = segs[segs!=0]

        outdict = {}
        logger.info('Building a mapping dictionary...')
        # Get all unique segment values and their indices at once
        y, x = np.nonzero(array)
        all_segments = array[y, x]
        
        # Determine optimal dtype based on coordinate range
        if len(y) > 0 and len(x) > 0:
            max_coord = max(y.max(), x.max())
            min_coord = min(y.min(), x.min())
            
            # Select appropriate dtype
            if min_coord < 0:
                dtype = np.int16 if (max_coord < 32767 and min_coord >= -32768) else np.int32
            else:
                if max_coord < 65535:
                    dtype = np.uint16
                elif max_coord < 4294967295:
                    dtype = np.uint32
                else:
                    dtype = np.uint64
        else:
            dtype = np.uint16  # Default for empty arrays

        # Use np.unique to split the coordinates by segment
        unique_segs, inverse = np.unique(all_segments, return_inverse=True)
        for idx, seg in enumerate(unique_segs):
            seg_indices = np.where(inverse == idx)
            # Store as numpy arrays with efficient dtype
            outdict[seg] = (y[seg_indices].astype(dtype), x[seg_indices].astype(dtype))

    elif force_simple:
        logger.warning(f'Simple mapping has been forced! Small regions may be cannibalized... ')
        segs = np.unique(array.flatten()).astype(int)
        segs = segs[segs!=0]

        outdict = {}
        logger.info('Building a mapping dictionary...')
        # Get all unique segment values and their indices at once
        y, x = np.nonzero(array)
        all_segments = array[y, x]
        
        # Determine optimal dtype based on coordinate range
        dtype = _select_optimal_dtype(np.concatenate([y, x]) if len(y) > 0 else [])

        # Use np.unique to split the coordinates by segment
        unique_segs, inverse = np.unique(all_segments, return_inverse=True)
        for idx, seg in enumerate(unique_segs):
            seg_indices = np.where(inverse == idx)
            # Store as numpy arrays with efficient dtype
            outdict[seg] = (y[seg_indices].astype(dtype), x[seg_indices].astype(dtype))

    # Determine processing strategy based on configuration
    # NOTE: Multiprocessing is disabled by default (NCPUS=0) due to high memory overhead
    # when copying WCS objects across processes. The vectorized single-core version
    # (map_ids_to_coarse_pixels) is typically faster for most use cases.
    if conf.NCPUS == 0:
        logger.info('Mapping to different resolution using single-core vectorized reprojection')
        outdict = map_ids_to_coarse_pixels(array, out_wcs, in_wcs)
    else:
        logger.info(f'Mapping to different resolution using multiprocessing (NCPU = {conf.NCPUS})')
        logger.warning('Multiprocessing may consume significant memory due to WCS object copying')
        outdict = parallel_process(array, out_wcs, in_wcs, n_processes=conf.NCPUS)

    return outdict



def map_ids_to_coarse_pixels(fine_pixel_data, coarse_wcs, fine_wcs, offset=0):
    """Map object IDs from fine grid to coarse grid pixels.
    
    Vectorized implementation for performance - transforms all pixels in batches
    rather than looping over individual pixels.
    
    Args:
        fine_pixel_data: 2D array with object IDs
        coarse_wcs: WCS for coarse (output) grid
        fine_wcs: WCS for fine (input) grid
        offset: Y-offset for parallel processing chunks (default: 0)
        
    Returns:
        dict: Mapping of object_id -> (y_coords, x_coords) as numpy arrays
        
    Note:
        Uses sets to eliminate duplicate coordinates, then converts to
        memory-efficient numpy arrays with optimal dtype.
    """
    logger = logging.getLogger('farmer.map_ids_to_coarse_pixels')
    
    # Get non-zero pixels (vectorized)
    y_fine, x_fine = np.nonzero(fine_pixel_data)
    obj_ids = fine_pixel_data[y_fine, x_fine]
    
    if len(obj_ids) == 0:
        logger.warning('No objects found in fine pixel data')
        return {}
    
    logger.debug(f'Processing {len(obj_ids)} non-zero pixels from {len(np.unique(obj_ids))} objects')
    
    # Process in chunks to avoid memory issues with very large images
    chunk_size = 10000  # Adjust based on available memory
    id_to_coarse_pixel_map = {}
    
    for chunk_start in range(0, len(obj_ids), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(obj_ids))
        y_chunk = y_fine[chunk_start:chunk_end]
        x_chunk = x_fine[chunk_start:chunk_end]
        ids_chunk = obj_ids[chunk_start:chunk_end]
        
        # Create all 4 corners for all pixels in this chunk (vectorized)
        # Shape: (n_pixels, 4) for each coordinate
        x_corners = np.column_stack([
            x_chunk,        # bottom-left
            x_chunk + 1,    # bottom-right
            x_chunk,        # top-left
            x_chunk + 1     # top-right
        ])
        y_corners = np.column_stack([
            y_chunk + offset,      # bottom-left
            y_chunk + offset,      # bottom-right
            y_chunk + offset + 1,  # top-left
            y_chunk + offset + 1   # top-right
        ])
        
        # Flatten to transform all corners at once
        x_all = x_corners.flatten()
        y_all = y_corners.flatten()
        
        # Batch WCS transformation (much faster than individual transforms)
        try:
            world_coords = fine_wcs.pixel_to_world_values(x_all, y_all)
            coarse_coords = coarse_wcs.world_to_pixel_values(world_coords[0], world_coords[1])
        except Exception as e:
            logger.error(f'WCS transformation failed: {e}')
            # Fall back to pixel-by-pixel for this chunk
            logger.warning(f'Falling back to slow per-pixel processing for chunk {chunk_start}-{chunk_end}')
            for i, (y, x, obj_id) in enumerate(zip(y_chunk, x_chunk, ids_chunk)):
                if obj_id == 0:
                    continue
                pixel_corners = [(x, y + offset), (x + 1, y + offset), 
                                (x, y + offset + 1), (x + 1, y + offset + 1)]
                try:
                    world_corners = fine_wcs.pixel_to_world_values(
                        [c[0] for c in pixel_corners], [c[1] for c in pixel_corners])
                    coarse_pixel_coords = coarse_wcs.world_to_pixel_values(
                        world_corners[0], world_corners[1])
                    
                    min_x = int(np.floor(min(coarse_pixel_coords[0])))
                    max_x = int(np.ceil(max(coarse_pixel_coords[0])))
                    min_y = int(np.floor(min(coarse_pixel_coords[1])))
                    max_y = int(np.ceil(max(coarse_pixel_coords[1])))
                    
                    if obj_id not in id_to_coarse_pixel_map:
                        id_to_coarse_pixel_map[obj_id] = set()
                    for cx in range(min_x, max_x):
                        for cy in range(min_y, max_y):
                            id_to_coarse_pixel_map[obj_id].add((cy, cx))
                except Exception as e2:
                    logger.warning(f'Failed to process pixel ({x}, {y}): {e2}')
                    continue
            continue
        
        # Reshape back to (n_pixels, 4, 2) - 4 corners per pixel
        coarse_x = coarse_coords[0].reshape(-1, 4)
        coarse_y = coarse_coords[1].reshape(-1, 4)
        
        # For each pixel, find bounding box in coarse grid
        min_x = np.floor(coarse_x.min(axis=1)).astype(int)
        max_x = np.ceil(coarse_x.max(axis=1)).astype(int)
        min_y = np.floor(coarse_y.min(axis=1)).astype(int)
        max_y = np.ceil(coarse_y.max(axis=1)).astype(int)
        
        # Accumulate coordinates for each object
        for i, obj_id in enumerate(ids_chunk):
            if obj_id == 0:
                continue
                
            if obj_id not in id_to_coarse_pixel_map:
                id_to_coarse_pixel_map[obj_id] = set()
            
            # Add all pixels in the bounding box
            for cx in range(min_x[i], max_x[i]):
                for cy in range(min_y[i], max_y[i]):
                    id_to_coarse_pixel_map[obj_id].add((cy, cx))
    
    logger.debug(f'Mapped {len(id_to_coarse_pixel_map)} objects to coarse grid')
    
    # Convert sets to numpy arrays for efficient storage and usage
    # Determine optimal dtype based on coordinate range
    all_coords = [c for coords_set in id_to_coarse_pixel_map.values() for c in coords_set]
    dtype = _select_optimal_dtype([c[0] for c in all_coords] + [c[1] for c in all_coords] if all_coords else [])
    
    for obj_id in id_to_coarse_pixel_map:
        coords_set = id_to_coarse_pixel_map[obj_id]
        if len(coords_set) > 0:
            coords = np.array(list(coords_set), dtype=dtype)
            # Keep as numpy arrays - much more memory efficient than Python lists
            id_to_coarse_pixel_map[obj_id] = (coords[:, 0], coords[:, 1])
        else:
            id_to_coarse_pixel_map[obj_id] = (np.array([], dtype=dtype), np.array([], dtype=dtype))

    return id_to_coarse_pixel_map

def parallel_process(fine_pixel_data, coarse_wcs, fine_wcs, n_processes=1):
    """
    Parallelize the processing of fine grid chunks.
    """
    if n_processes < 1:
        n_processes = 1

    # Split the fine grid into chunks
    chunks = np.array_split(fine_pixel_data, n_processes)
    arrs = np.array_split(np.arange(len(fine_pixel_data)), n_processes)
    offsets = [0]
    for i, arr in enumerate(arrs[:-1]):
        offsets.append(offsets[i] + len(arr))
    # print(offsets)
    
    # # Use multiprocessing to process chunks in parallel
    with Pool(n_processes) as pool:
        results = pool.starmap(map_ids_to_coarse_pixels, [(chunk, coarse_wcs, fine_wcs, offset) for (chunk, offset) in zip(chunks, offsets)])

    # Combine results from all processes
    combined_results = {}
    for result in results:
        for obj_id, pixels in result.items():
            if obj_id in combined_results:
                # Concatenate numpy arrays efficiently
                combined_results[obj_id] = (
                    np.concatenate([combined_results[obj_id][0], pixels[0]]),
                    np.concatenate([combined_results[obj_id][1], pixels[1]])
                )
            else:
                combined_results[obj_id] = pixels

    # Sort the combined results by object ID and return as an OrderedDict
    sorted_combined_results = OrderedDict(sorted(combined_results.items()))
    
    return sorted_combined_results

def recursively_save_dict_contents_to_group(h5file, dic, path='/'):
    """Recursively save dictionary to HDF5 group.
    
    Optimized version with type checking ordered by frequency of occurrence
    and pre-compiled type tuples for faster isinstance() checks.
    
    Args:
        h5file: Open h5py file handle
        dic: Dictionary to save
        path: HDF5 group path (default: '/')
        
    Raises:
        ValueError: If arguments are invalid types
    """
    logger = logging.getLogger('farmer.hdf5')

    # Argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")        
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    
    # Pre-compiled type tuples (faster than repeating in each isinstance call)
    numeric_types = (np.int32, np.int64, np.float64, str, float, np.float32, int)
    tractor_model_types = (PointSource, SimpleGalaxy, ExpGalaxy, DevGalaxy, FixedCompositeGalaxy)
    
    # Ensure path exists
    if path not in h5file.keys():
        h5file.create_group(path)
    
    # Save items to the hdf5 file
    for key, item in dic.items():
        logger.debug(f'  ... {path}{key} ({type(item)})')
        key = str(key)
        if key == 'logger':
            continue
            
        # Convert lists/tuples to numpy arrays early
        if isinstance(item, (list, tuple)):
            try:
                item = np.array(item)
            except (ValueError, TypeError):
                # Items are Quantity objects - extract values in degrees
                item = np.array([i.to(u.deg).value for i in item])
        
        # Most common types first for faster short-circuiting
        # 1. Numpy arrays (most common in scientific code)
        if isinstance(item, np.ndarray):
            if key == 'bands':
                # Convert band names to byte strings for HDF5 compatibility
                item = np.array(item).astype(HDF5_STRING_DTYPE)
                try:
                    h5file[path].create_dataset(key, data=item)
                except (ValueError, RuntimeError):
                    # Dataset already exists - delete and recreate
                    item = np.array(item)
                    del h5file[path][key]
                    h5file[path].create_dataset(key, data=item)
            else:
                # Try creating new dataset, update if exists, convert to bytes as last resort
                try:
                    try:
                        h5file[path].create_dataset(key, data=item)
                    except (ValueError, RuntimeError):
                        # Dataset exists - update in place
                        h5file[path][key][...] = item
                except (TypeError, ValueError):
                    # Type incompatible - try as byte string
                    try:
                        item = np.array(item).astype(HDF5_STRING_DTYPE)
                        h5file[path].create_dataset(key, data=item)
                    except (ValueError, RuntimeError):
                        # Update existing dataset
                        h5file[path][key][...] = item
        
        # 2. Dictionaries (also very common)
        elif isinstance(item, dict):
            if len(item) == 0:
                try:
                    h5file[path][key]
                except KeyError:
                    h5file[path].create_group(key)  # empty group
            else:
                recursively_save_dict_contents_to_group(h5file, item, path + key + '/')
        
        # 3. Simple numeric types (attributes)
        elif isinstance(item, numeric_types):
            h5file[path].attrs[key] = item
        
        # 4. Boolean (needs conversion)
        elif isinstance(item, np.bool_):
            h5file[path].attrs[key] = int(item)
        
        # 5. Tractor model types (recursive)
        elif isinstance(item, tractor_model_types):
            if key == 'variance':
                continue
            item.unfreezeParams()
            item.variance.unfreezeParams()
            model_params = dict(zip(item.getParamNames(), item.getParams()))
            model_params['name'] = item.name
            model_params['variance'] = dict(zip(item.variance.getParamNames(), item.variance.getParams()))
            model_params['variance']['name'] = item.name
            recursively_save_dict_contents_to_group(h5file, model_params, path + key + '/')
        
        # 6. Astropy Table
        elif isinstance(item, Table):
            # NOTE: Default astropy.table handler creates a new file - we already have one open!
            if any(col.info.dtype.kind == "U" for col in item.itercols()):
                item = item.copy(copy_data=False)
                item.convert_unicode_to_bytestring()
            
            header_yaml = meta.get_yaml_from_table(item)
            header_encoded = np.array([h.encode("utf-8") for h in header_yaml])
            if key in h5file[path]:
                del h5file[path][key]
                del h5file[path][f"{key}.__table_column_meta__"]
            h5file[path].create_dataset(key, data=item.as_array())
            h5file[path].create_dataset(f"{key}.__table_column_meta__", data=header_encoded)
        
        # 7. Astropy Quantity
        elif isinstance(item, u.quantity.Quantity):
            h5file[path].attrs[key] = item.value
            h5file[path].attrs[key + '_unit'] = item.unit.to_string()
        
        # 8. Astropy SkyCoord
        elif isinstance(item, SkyCoord):
            h5file[path].attrs[key] = item.to_string(precision=10)
        
        # 9. WCS
        elif isinstance(item, WCS):
            h5file[path].attrs[key] = item.to_header_string()
        
        # 10. FITS Header
        elif isinstance(item, fits.header.Header):
            h5file[path].attrs[key] = item.tostring()
        
        # 11. Cutout2D (recursive)
        elif isinstance(item, utils.Cutout2D):
            recursively_save_dict_contents_to_group(h5file, item.__dict__, path + key + '/')
        
        # Other types cannot be saved
        else:
            logger.debug(f'Cannot save {key} of type {type(item)}')

def recursively_load_dict_contents_from_group(h5file, path='/', ans=None): 

    logger = logging.getLogger('farmer.hdf5')

    # return h5file[path]
    if ans is None:
        ans = {}
    if path == '/':
        for attr in h5file.attrs:
            value = h5file.attrs[attr]
            if attr == 'position':
                ans[attr] = SkyCoord(value, unit=u.deg)
            else:
                ans[attr] = value
    for key, item in h5file[path].items():
        try:
            key = int(key)
        except (ValueError, TypeError):
            pass  # Key is not an integer, keep as string
        logger.debug(f'  ... {item.name} ({type(item)})')
        if isinstance(item, h5py._hl.dataset.Dataset):
            if '.__table_column_meta__' in item.name:
                continue
            if item.shape is None:
                ans[key] = {}
            elif item[...].dtype in (HDF5_STRING_DTYPE, '|S9'):
                if key == 'bands':
                    ans[key] = item[...].astype(str).tolist()
                else:
                    ans[key] = item[...].astype(str)
            elif ('pixel_scales' in item.name) | (key in ('size', 'buffsize')):
                dx, dy = item[...]
                ans[key] = (dx*u.deg, dy*u.deg)
            elif ('catalogs' in item.name):
                ans[key] = Table.read(h5file.filename, item.name)
                ans[key]['ra'].unit = u.deg
                ans[key]['dec'].unit = u.deg
            else:
                ans[key] = item[...] 
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = {}
            if key == 'model_catalog':
                ans[key] = OrderedDict()
            if ('data' in item.name) & (key in ('science', 'weight', 'mask', 'segmap', 'groupmap', 'background', 'rms', 'model', 'residual', 'chi')):
                if 'data' in item.keys():
                    awcs = WCS(fits.header.Header.fromstring(item.attrs['wcs']))
                    ppix = item['input_position_cutout'][...]
                    pos = awcs.pixel_to_world(ppix[0], ppix[1])
                    ans[key] = Cutout2D(item['data'][...], pos, np.shape(item['data'][...]), wcs=awcs)
                    logger.debug(f'  ...... building cutout')
                    for key2 in item:
                        logger.debug(f'          * {key2}')
                        ans[key].__dict__[key2] = item[key2][...]
                    for key2 in item.attrs.keys():
                        logger.debug(f'          * {key2}')
                        if key2 != 'wcs':
                            ans[key].__dict__[key2] = item.attrs[key2]
                elif (('detection') not in path) & (key in ('segmap', 'groupmap')):
                    ans[key] = recursively_load_dict_contents_from_group(h5file, path + str(key) + '/', ans[key])

            elif 'variance' in item.name:
                # TODO
                continue

            elif ('model' in item.name) & ('name' in item.attrs):
                is_variance = False
                for item in (item, item['variance']):
                    name = item.attrs['name']
                    pos = RaDecPos(item.attrs['pos.ra'], item.attrs['pos.dec'])
                    fluxes = {}
                    for param in item.attrs:
                        if param.startswith('brightness'):
                            fluxes[param.split('.')[-1]] = item.attrs[param]
                    flux = Fluxes(**fluxes)
        
                    if name == 'PointSource':
                        model = PointSource(pos, flux)
                        model.name = name
                    elif name == 'SimpleGalaxy':
                        model = SimpleGalaxy(pos, flux)
                    elif name == 'ExpGalaxy':
                        shape = EllipseESoft(item.attrs['shape.logre'], item.attrs['shape.ee1'], item.attrs['shape.ee2'])
                        model = ExpGalaxy(pos, flux, shape)
                    elif name == 'DevGalaxy':
                        shape = EllipseESoft(item.attrs['shape.logre'], item.attrs['shape.ee1'], item.attrs['shape.ee2'])
                        model = DevGalaxy(pos, flux, shape)
                    elif name == 'FixedCompositeGalaxy':
                        shape_exp = EllipseESoft(item.attrs['shapeExp.logre'], item.attrs['shapeExp.ee1'], item.attrs['shapeExp.ee2'])
                        shape_dev = EllipseESoft(item.attrs['shapeDev.logre'], item.attrs['shapeDev.ee1'], item.attrs['shapeDev.ee2'])
                        model = FixedCompositeGalaxy(pos, flux, SoftenedFracDev(item.attrs['fracDev.SoftenedFracDev']), shape_exp, shape_dev)
                    
                    if not is_variance:
                        ans[key] = model
                        is_variance = True
                    else:
                        ans[key].variance = model

            else:
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + str(key) + '/', ans[key])
                for key2 in item.attrs:
                    logger.debug(f'  ...... attribute: {key2}')
                    if '_unit' in key2: continue
                    value = item.attrs[key2]
                    if key2 == 'psfcoords':
                        if np.any(value == 'none'):
                            ans[key][key2] = value
                        else:
                            if np.isscalar(value):
                                value = np.array([value])
                            ra, dec = np.array([val.split() for val in value]).astype(np.float64).T
                            ans[key][key2] = SkyCoord(ra*u.deg, dec*u.deg)
                    elif 'headers' in item.name:
                        ans[key][key2] = fits.header.Header.fromstring(value)
                    elif 'wcs' in item.name:
                        ans[key][key2] = WCS(fits.header.Header.fromstring(value))
                    elif 'position' in item.name:
                        ans[key][key2] = SkyCoord(value, unit=u.deg)
                    elif (key2+'_unit' in item.attrs):
                        ans[key][key2] = value * u.Unit(item.attrs[key2+'_unit'])
                    else:
                        ans[key][key2] = value
    return ans   
     
def dcoord_to_offset(coord1, coord2, offset='arcsec', pixel_scale=None):
    """Calculate offset between two sky coordinates.
    
    Computes the angular or pixel offset between two SkyCoord objects,
    accounting for spherical geometry with cosine declination correction.
    
    Args:
        coord1: First astropy SkyCoord
        coord2: Second astropy SkyCoord (reference position)
        offset: Output unit - 'arcsec' or 'pixel' (default: 'arcsec')
        pixel_scale: Tuple of (RA, Dec) pixel scales with units (required if offset='pixel')
        
    Returns:
        tuple: (dra, ddec) offset in specified units
            - dra: RA offset with cosine declination correction (negative for westward)
            - ddec: Declination offset
            
    Note:
        Uses midpoint declination for cosine correction to handle large separations.
    """
    if offset == 'arcsec':
        corr = np.cos(0.5*(coord1.dec.to(u.rad).value + coord2.dec.to(u.rad).value))
        dra = (corr * (coord1.ra - coord2.ra)).to(u.arcsec).value
        ddec = (coord1.dec - coord2.dec).to(u.arcsec).value
    elif offset == 'pixel':
        corr = np.cos(0.5*(coord1.dec.to(u.rad).value + coord2.dec.to(u.rad).value))
        dra = ((coord1.ra - coord2.ra) * corr / pixel_scale[0]).value
        ddec = ((coord1.dec - coord2.dec) / pixel_scale[1]).value
    return -dra, ddec

def cumulative(x):
    """Compute cumulative distribution function from data.
    
    Args:
        x: Array of numeric values (may contain NaN)
        
    Returns:
        tuple: (sorted_values, cumulative_probabilities)
            - sorted_values: x sorted in ascending order (NaN removed)
            - cumulative_probabilities: Normalized cumulative distribution [0, 1]
            
    Example:
        >>> x = np.array([3, 1, 2, np.nan, 4])
        >>> vals, cdf = cumulative(x)
        >>> vals
        array([1, 2, 3, 4])
        >>> cdf
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    x = x[~np.isnan(x)]
    N = len(x)
    return np.sort(x), np.array(np.linspace(0, N, N)) / float(N)

def get_params(model):
    """Extract source parameters from a Tractor model.
    
    Args:
        model: Tractor model object (PointSource, ExpGalaxy, DevGalaxy, etc.)
        
    Returns:
        OrderedDict: Dictionary of source parameters and errors
    """
    source = OrderedDict()

    if isinstance(model, PointSource):
        name = MODEL_TYPES['POINT']
    else:
        name = model.name
    source['name'] = name
    
    # Cache frequently accessed attributes
    brightness = model.getBrightness()
    variance_brightness = model.variance.getBrightness()
    source['_bands'] = np.array(list(brightness.getParamNames()))

    # position
    pos = model.pos
    var_pos = model.variance.pos
    source['ra'] = pos.ra * u.deg
    source['ra_err'] = np.sqrt(var_pos.ra) * u.deg
    source['dec'] = pos.dec * u.deg
    source['dec_err'] = np.sqrt(var_pos.dec) * u.deg

    # total statistics
    for stat in model.statistics:
        if (stat not in source['_bands']) & (stat not in ('model', 'variance')):
            source[f'total_{stat}'] = model.statistics[stat]

    # Helper function for shape parameters (reduces duplication)
    def _extract_shape_params(shape, variance_shape, suffix=''):
        """Extract shape parameters with optional suffix for composite models."""
        logre = shape.logre
        e = shape.e
        theta = shape.theta
        
        source[f'logre{suffix}'] = logre
        source[f'logre{suffix}_err'] = np.sqrt(variance_shape.logre)
        source[f'ellip{suffix}'] = e
        source[f'ellip{suffix}_err'] = np.sqrt(variance_shape.e)
        source[f'ee1{suffix}'] = shape.ee1
        source[f'ee1{suffix}_err'] = np.sqrt(variance_shape.ee1)
        source[f'ee2{suffix}'] = shape.ee2
        source[f'ee2{suffix}_err'] = np.sqrt(variance_shape.ee2)

        theta_deg = np.rad2deg(theta)
        source[f'theta{suffix}'] = theta_deg * u.deg
        source[f'theta{suffix}_err'] = np.sqrt(np.rad2deg(variance_shape.theta)) * u.deg

        reff = np.exp(logre) * u.arcsec
        source[f'reff{suffix}'] = reff
        source[f'reff{suffix}_err'] = np.sqrt(variance_shape.logre) * reff * np.log(10)

        abs_e = np.abs(e)
        boa = (1. - abs_e) / (1. + abs_e)
        if e == 1:
            boa_sig = np.inf
        else:
            sqrt_var_e = np.sqrt(variance_shape.e)
            boa_sig = boa * sqrt_var_e * np.sqrt((1/(1.-e))**2 + (1/(1.+e))**2)
        source[f'ba{suffix}'] = boa
        source[f'ba{suffix}_err'] = boa_sig
        
        source[f'pa{suffix}'] = 90. * u.deg + theta_deg * u.deg
        source[f'pa{suffix}_err'] = np.rad2deg(variance_shape.theta) * u.deg

    # shape
    if model.name == 'SimpleGalaxy':
        pass  # SimpleGalaxy has fixed shape
    elif isinstance(model, (ExpGalaxy, DevGalaxy)):
        _extract_shape_params(model.shape, model.variance.shape)
    elif isinstance(model, FixedCompositeGalaxy):
        frac_dev = model.fracDev
        var_frac_dev = model.variance.fracDev
        source['softfracdev'] = frac_dev.getValue()
        source['fracdev'] = frac_dev.clipped()
        source['softfracdev_err'] = np.sqrt(var_frac_dev.getValue())
        source['fracdev_err'] = np.sqrt(var_frac_dev.clipped())
        
        _extract_shape_params(model.shapeExp, model.variance.shapeExp, '_exp')
        _extract_shape_params(model.shapeDev, model.variance.shapeDev, '_dev')

    # Photometry - cache constants
    log10_e = np.log10(np.e)
    
    for band in source['_bands']:
        # photometry - cache method calls
        flux = brightness.getFlux(band)
        var_flux = variance_brightness.getFlux(band)
        flux_err = np.sqrt(var_flux)
        mask = ((flux_err > 0) & np.isfinite(flux_err)).astype(np.int8)
        
        source[f'{band}_flux'] = flux * mask
        source[f'{band}_flux_err'] = flux_err * mask
        
        zpt = conf.BANDS[band]['zeropoint']
        source[f'_{band}_zpt'] = zpt
        
        # Pre-calculate conversion factor
        flux_to_ujy = 10**(-0.4 * (zpt - 23.9))
        source[f'{band}_flux_ujy'] = flux * flux_to_ujy * u.microjansky * mask
        source[f'{band}_flux_ujy_err'] = flux_err * flux_to_ujy * u.microjansky * mask

        source[f'{band}_mag'] = (-2.5 * np.log10(flux) * u.mag + zpt * u.mag) * mask
        source[f'{band}_mag_err'] = (2.5 * log10_e / (flux / flux_err)) * mask

        # statistics
        if band in model.statistics:
            for stat in model.statistics[band]:
                source[f'{band}_{stat}'] = model.statistics[band][stat]

    return source

def set_priors(model, priors):
    logger = logging.getLogger('farmer.priors')

    if priors is None:
        logger.warning('I was asked to set priors but I have none!')
        return model
        
    params = model.getNamedParams()
    for name in params:
        idx = params[name]
        if name == 'pos':
            if 'pos' in priors:
                if priors['pos'] in ('fix', 'freeze'):
                    model[idx].freezeAllParams()
                    logger.debug('Froze position')
                elif priors['pos'] != 'none':
                    sigma = priors['pos'].to(u.deg).value
                    psigma = priors['pos'].to(u.arcsec)
                    model[idx].addGaussianPrior('ra', mu=model[idx][0], sigma=sigma)
                    model[idx].addGaussianPrior('dec', mu=model[idx][1], sigma=sigma)
                    logger.debug(f'Set positon prior +/- {psigma}')
                else:
                    logger.debug('Position is free to vary')

        elif name == 'fracDev':
            if 'fracDev' in priors:
                if priors['fracDev'] in ('fix', 'freeze'):
                    model.freezeParam(idx)
                    logger.debug(f'Froze {name}')
                    # params = model[idx].getParamNames()
                    # for i, param in enumerate(params):
                    #     model[idx].freezeParam(i)
                    #     logger.debug(f'Froze {param}')
                else:
                    logger.debug('fracDev is free to vary') 

        elif name in ('shape', 'shapeDev', 'shapeExp'):
            if 'shape' in priors:
                if priors['shape'] in ('fix', 'freeze'):
                    sparams = model[idx].getParamNames()
                    for i, param in enumerate(sparams):
                        if i != 0: # leave reff alone
                            model[idx].freezeParam(i)
                            logger.debug(f'Froze {param}')
                else:
                    logger.debug(f'{name} is free to vary')   

            if 'reff' in priors:
                if priors['reff'] in ('fix', 'freeze'):
                    model[idx].freezeParam(0)
                    logger.debug(f'Froze {name} radius')
                elif priors['reff'] != 'none':
                    sigma = np.log(priors['reff'].to(u.arcsec).value)
                    psigma = priors['reff'].to(u.arcsec)
                    model[idx].addGaussianPrior('logre', mu=model[idx][0], sigma=sigma)    
                    logger.debug(f'Set {name} radius prior +/- {psigma}')   
                else:
                    logger.debug(f'{name} radius is free to vary')           
    return model

def get_detection_kernel(filter_kernel):
    kernel_kwargs = {}
    # if string, grab from config
    if isinstance(filter_kernel, str):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../config/conv_filters/'+conf.FILTER_KERNEL)
        if os.path.exists(filename):
            convfilt = np.array(np.array(ascii.read(filename, data_start=1)).tolist())
        else:
            raise FileExistsError(f"Convolution file at {filename} does not exist!")
        return convfilt

    elif np.isscalar(filter_kernel):
        from astropy.convolution import Gaussian2DKernel
        # else, assume FWHM in pixels
        kernel_kwargs['x_stddev'] = filter_kernel/2.35
        kernel_kwargs['factor']=1
        convfilt = np.array(Gaussian2DKernel(**kernel_kwargs))
        return convfilt
    
    else:
        raise RuntimeError(f'Requested kernel {filter_kernel} not understood!')

def build_regions(catalog, pixel_scale, outpath='objects.reg', scale_factor=2.0):
    from regions import EllipseSkyRegion, Regions
    detcoords = SkyCoord(catalog['ra'], catalog['dec'])
    regs = []
    for coord, obj in tqdm(zip(detcoords, catalog), total=len(catalog)):
        width = scale_factor * 2 * obj['a'] * pixel_scale
        height = scale_factor * 2 * obj['b'] * pixel_scale
        angle = np.rad2deg(obj['theta']) * u.deg
        objid = str(obj['id'])
        regs.append(EllipseSkyRegion(coord, width, height, angle, meta={'text':objid}))
    bigreg = Regions(regs)
    bigreg.write(outpath, overwrite=True, format='ds9')


def _clear_h5():
    """Close any open HDF5 file handles to prevent file locking issues."""
    import gc
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except (ValueError, RuntimeError):
                pass  # File already closed or invalid


def prepare_psf(filename, outfilename=None, pixel_scale=None, mask_radius=None, clip_radius=None, norm=None, norm_radius=None, target_pixel_scale=None, ext=0):
    # NOTE: THESE INPUTS NEED ASTROPY UNITS!

    hdul = fits.open(filename)
    psfmodel = hdul[ext].data

    if pixel_scale is None:
        w = WCS(hdul[ext].header)
        pixel_scale = w.proj_plane_pixel_scales[0]
        hdul[ext].header.update(w.wcs.to_header())

    if target_pixel_scale is not None:
        from scipy.ndimage import zoom
        zoom_factor = pixel_scale / target_pixel_scale
        psfmodel = zoom(psfmodel, zoom_factor.value, order=1)
        print(f'Resampled image from {pixel_scale}/pixel to {target_pixel_scale}/pixel')

    # estimate plateau
    if mask_radius is not None:
        pw, ph = np.shape(psfmodel)
        cmask = create_circular_mask(pw, ph, radius=mask_radius / pixel_scale)
        bcmask = ~cmask.astype(bool) & (psfmodel > 0)
        back_level = np.nanpercentile(psfmodel[bcmask], q=95)
        print(f'Subtracted back level of {back_level} based on {np.sum(bcmask)} pixels outside {mask_radius}')
        psfmodel -= back_level
        psfmodel[(psfmodel < 0) | np.isnan(psfmodel)] = 1e-31

    if clip_radius is not None:
        pw, ph = np.shape(psfmodel)
        psf_rad_pix = int(clip_radius / pixel_scale)
        if psf_rad_pix%2 == 0:
            psf_rad_pix += 0.5
        print(f'Clipping PSF ({psf_rad_pix}px radius)')
        psfmodel = psfmodel[int(pw/2.-psf_rad_pix):int(pw/2+psf_rad_pix), int(ph/2.-psf_rad_pix):int(ph/2+psf_rad_pix)]
        print(f'New shape: {np.shape(psfmodel)}')
        
    if norm is not None:
        print(f'Normalizing PSF to {norm:4.4f} within {norm_radius} radius circle')
        norm_radpix = norm_radius / pixel_scale
        pw, ph = np.shape(psfmodel)
        cmask = create_circular_mask(pw, ph, radius=norm_radpix).astype(bool)
        psfmodel *= norm / np.sum(psfmodel[cmask])

    if outfilename is None:
        outfilename = filename

    hdul[ext].data = psfmodel
    hdul.writeto(outfilename)
    print(f'Wrote updated PSF to {outfilename}')

    return psfmodel

def spawn_and_run_group(brick, group_id, imgtype='science', bands=None, mode='all'):
    """Spawn a group from brick and process it, returning only essential results.
    
    This function is designed for parallel processing to avoid holding all groups in memory.
    It spawns the group, processes it, extracts results, and deletes the group object.
    """
    import time
    tstart = time.time()
    
    # Spawn the group
    group = brick.spawn_group(group_id, imgtype=imgtype, bands=bands, silent=True)
    
    if not group.rejected:
        if mode == 'all':
            status = group.determine_models()
            if status:
                status = group.force_models()
        elif mode == 'model':
            status = group.determine_models()
        elif mode == 'photometry':
            status = group.force_models()
        elif mode == 'pass':
            status = False
    
    # Record group processing time in model_tracker for all sources
    elapsed = time.time() - tstart
    for source_id in group.source_ids:
        if source_id in group.model_tracker:
            if len(group.model_tracker[source_id]) > 0:
                final_stage = max(group.model_tracker[source_id].keys())
                if final_stage not in group.model_tracker[source_id]:
                    group.model_tracker[source_id][final_stage] = {}
                group.model_tracker[source_id][final_stage]['group_time'] = elapsed
    
    # Extract only what's needed and delete the group
    output = group.group_id, group.model_catalog.copy(), group.model_tracker.copy()
    del group
    return output


def run_group(group, mode='all'):
    """Process an already-spawned group object (for serial processing)."""
    import time
    tstart = time.time()

    if not group.rejected:
    
        if mode == 'all':
            status = group.determine_models()
            if status:
                status = group.force_models()
    
        elif mode == 'model':
            status = group.determine_models()

        elif mode == 'photometry':
            status = group.force_models()

        elif mode == 'pass':
            status = False

        # if not status:
        #     group.rejected = True

    # else:
    #     self.logger.warning(f'Group {group.group_id} has been rejected!')
    
    # Record group processing time in model_tracker for all sources
    elapsed = time.time() - tstart
    for source_id in group.source_ids:
        if source_id in group.model_tracker:
            # Find the final stage that was run for this source
            if len(group.model_tracker[source_id]) > 0:
                final_stage = max(group.model_tracker[source_id].keys())
                if final_stage not in group.model_tracker[source_id]:
                    group.model_tracker[source_id][final_stage] = {}
                group.model_tracker[source_id][final_stage]['group_time'] = elapsed
    
    # Log completion
    group.logger.info(f'Group #{group.group_id} processing completed ({elapsed:.2f}s)')
    
    output = group.group_id, group.model_catalog.copy(), group.model_tracker.copy()
    del group
    return output

    # return group
