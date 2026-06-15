import config as conf
from .utils import validate_psfmodel, dilate_and_group
from .brick import Brick
from .image import BaseImage

import logging
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
import astropy.units as u
import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales

default_properties = {}
default_properties['subtract_background'] = False
default_properties['backtype'] = 'flat'
default_properties['backregion'] = 'brick'
default_properties['zeropoint'] = -99

class Mosaic(BaseImage):
    """A full-field survey image for a single photometric or detection band."""

    def __init__(self, band, load=False) -> None:
        """Initialise a Mosaic for the specified band.

        Validates the band against ``conf.BANDS`` (or ``conf.DETECTION`` for
        the detection image), verifies that the science FITS file exists and
        has a readable WCS, and optionally loads the full image data into
        memory.

        Args:
            band: Band name (key in ``conf.BANDS``) or ``'detection'``.
                Band names must not contain a ``'.'`` character.
            load: If True, read all configured image arrays (science, weight,
                mask) and PSF model into memory.  If False, only validate
                paths and load the WCS. Defaults to False.

        Raises:
            ValueError: If ``band`` contains a ``'.'`` character.
            RuntimeError: If the science FITS header does not contain a
                valid WCS.
        """
        if '.' in band:
            raise ValueError(f'Band name {band} cannot contain a "."! Rename without one please.')

        # Housekeeping
        self.band = band
        self.is_loaded = load
        self.type = 'mosaic'
        self.brick_ids = 1 + np.arange(conf.N_BRICKS[0] * conf.N_BRICKS[1])

        # Load the logger
        self.logger = logging.getLogger(f'farmer.mosaic_{band}')

        self.filename = f'M{band}.h5'

        # Check data status
        if (band != 'detection') & (band not in conf.BANDS.keys()):
            self.logger.critical(f'{band} is not a configured band!')
            return None
        else:
            if band == 'detection':
                self.properties = conf.DETECTION
            else:
                self.properties = conf.BANDS[band]
            for key in self.properties:
                if isinstance(self.properties[key], bool):
                    self.properties[key] = int(self.properties[key]) # turn Trues/Falses into 1/0
            for key in default_properties:
                if key not in self.properties:
                    self.properties[key] = default_properties[key]
            if 'name' not in self.properties:
                self.properties['name'] = band
            good = '✓'
            bad = 'X'
            # verify the band
            self.paths = {}
            # Verify required data products
            good = '✓'
            bad = 'X'
            data_status = bad
            data_provided = []
            
            # Check for required image types efficiently
            required_types = ['science', 'weight', 'mask', 'psfmodel']
            for imgtype in required_types:
                if imgtype not in self.properties.keys():
                    if imgtype.startswith('psf') and band == 'detection':
                        continue
                    self.logger.warning(f'{imgtype} is not configured for {band}!')
                    if imgtype == 'science':
                        self.logger.critical(f'{imgtype} must be configured!')
                        return None
                else:
                    self.paths[imgtype] = self.properties[imgtype]
                    data_provided.append(imgtype)
                data_status = good

            # verify the psf model
            psf_status = bad
            if band == 'detection':
                psf_status = 'no'
                psftype = ''
            else:
                __, psftype = validate_psfmodel(band, return_psftype=True)
                psftype = ' '+psftype
                psf_status = good

            # verify the WCS
            wcs_status = bad
            ext = 0
            if 'extension' in self.properties:
                ext = self.properties['extension']
            try:
                self.wcs = WCS(fits.getheader(self.properties['science'], ext=ext))
                self.pixel_scale = proj_plane_pixel_scales(self.wcs) * u.deg
                wcs_status = good
            except Exception as e:
                raise RuntimeError(f'The World Coordinate System for {band} cannot be understood! Error: {e}')

            arr_shape = self.wcs.array_shape
            self.position = self.wcs.pixel_to_world(arr_shape[0]/2., arr_shape[1]/2.)
            # upper = self.wcs.pixel_to_world(arr_shape[0], arr_shape[1])
            # lower = self.wcs.pixel_to_world(0, 0)
            self.size = arr_shape * self.pixel_scale
            
            self.logger.debug(f'Mosaic {band} is centered at {self.position.ra:2.1f}, {self.position.dec:2.1f}')
            self.logger.debug(f'Mosaic {band} has size at {self.size[0]:2.1f}, {self.size[1]:2.1f}')

            self.logger.info(f'{band:10}: {data_status} Data {tuple(data_provided)} {psf_status}{psftype} PSF {wcs_status} WCS ({self.position.ra:2.1f}, {self.position.dec:2.1f})')

        # Now load in the data (memory use!) -- and only what is necessary!
        if load:
            self.data = {}
            self.headers = {}
            self.catalogs = {}
            self.n_sources = {}
            self.logger.info(f'Loading {list(self.paths.keys())} for {band}')
            for attr in self.paths.keys():
                if attr == 'psfmodel':
                    self.data['psfcoords'], self.data['psflist'] = validate_psfmodel(band)
                else:
                    ext = 0
                    if 'extension' in self.properties:
                        ext = self.properties['extension']
                    self.data[attr] = fits.getdata(self.paths[attr], ext=ext)
                    self.headers[attr] = fits.getheader(self.paths[attr], ext=ext)
                # if attr in ('science', 'weight'):
                #     self.estimate_properties(band=band, imgtype=attr)
            if band in conf.BANDS:
                if 'backregion' in conf.BANDS[band]:
                    if conf.BANDS[band]['backregion'] == 'mosaic':
                        self.estimate_background(band=band, imgtype='science')
            elif band == 'detection':
                if 'backregion' in conf.DETECTION:
                    if conf.DETECTION['backregion'] == 'mosaic':
                            self.estimate_background(band=band, imgtype='science')

    def get_bands(self):
        """Return the band(s) associated with this mosaic.

        Returns:
            numpy.ndarray: Single-element array containing ``self.band``.
        """
        return np.array([self.band])

    def get_figprefix(self, imgtype, band=None):
        """Generate a filename prefix for output figures of this mosaic.

        Args:
            imgtype: Image type string (e.g. ``'science'``, ``'model'``).
            band: Ignored for mosaics; ``self.band`` is always used.

        Returns:
            str: Prefix in the format ``'{band}_{imgtype}'``.
        """
        return f'{self.band}_{imgtype}'

    def add_to_brick(self, brick):
        """Cut out this mosaic's data at the brick's footprint and attach it.

        Delegates to :meth:`~farmer.brick.Brick.add_band`, which extracts
        ``Cutout2D`` sub-images centred on the brick position (including
        buffer) for every image type present in this mosaic.

        Args:
            brick: ``Brick`` object to receive the cut-out data.

        Returns:
            Brick: The same brick object, now containing this mosaic's band.
        """
        # Cut up science, weight, and mask, if available
        brick.add_band(self)

        # Return it
        return brick

    def spawn_brick(self, brick_id=None, position=None, size=None, silent=False):
        """Create a new blank brick and populate it from this mosaic.

        Instantiates a fresh ``Brick`` (without loading from disk) and
        calls :meth:`add_to_brick` to cut out the relevant sub-image.  The
        brick position is derived either from ``brick_id`` (looked up on the
        detection WCS grid) or from explicit ``position``/``size`` arguments.

        Args:
            brick_id: Integer brick identifier (1-indexed).  If provided,
                ``position`` and ``size`` must be None.
            position: ``astropy.coordinates.SkyCoord`` of the brick centre.
                Used only when ``brick_id`` is None.
            size: ``(dec_height, ra_width)`` tuple of angular
                ``astropy.units.Quantity`` values.  Used only when
                ``brick_id`` is None.
            silent: If True, suppress informational log messages.
                Defaults to False.

        Returns:
            Brick: Newly created brick populated with this mosaic's band.
        """
        # Instantiate brick
        if brick_id is None:
            brick = Brick(position, size, load=False, silent=silent)
        else:
            brick = Brick(brick_id, load=False, silent=silent)
        
        # Cut up science, weight, and mask, if available
        brick.add_band(self)

        # Return it
        return brick

    def extract(self, background=None):
        """Detect sources across the full mosaic and build a catalog.

        Calls the base-class ``_extract`` method on the full mosaic array,
        stores the resulting catalog in ``self.catalogs['science']`` and the
        segmentation map in ``self.data['segmap']``, then appends sequential
        ``id``, ``ra``, and ``dec`` columns.

        Args:
            background: Pre-computed background array to subtract before
                detection.  When None no subtraction is performed.
                Defaults to None.

        Returns:
            None. Populates ``self.catalogs['science']``,
            ``self.data['segmap']``, and ``self.n_sources['science']``.
        """
        catalog, segmap = self._extract(band=None, background=background)

        self.catalogs['science'] = catalog
        self.data['segmap'] = segmap
        self.n_sources['science'] = len(catalog)

        # add ids
        colname = 'id'
        self.catalogs['science'].add_column(1+np.arange(self.n_sources['science']), name=colname, index=0)

        # add world positions
        skycoords = self.wcs.all_pix2world(catalog['x'], catalog['y'], 0)
        self.catalogs['science'].add_column(skycoords[0], name=f'ra')
        self.catalogs['science'].add_column(skycoords[1], name=f'dec')


    def identify_groups(self, radius=conf.DILATION_RADIUS, overwrite=False):
        """Group nearby sources by morphologically dilating the segmentation map.

        Converts ``radius`` from arcsec to pixels using the mosaic WCS,
        calls :func:`~farmer.utils.dilate_and_group`, and stores
        ``group_id`` and ``group_pop`` columns in
        ``self.catalogs['science']`` and the group map in
        ``self.data['groupmap']``.

        Args:
            radius: Dilation radius as an ``astropy.units.Quantity`` angle.
                Defaults to ``conf.DILATION_RADIUS``.
            overwrite: If True, replace existing ``group_id``/``group_pop``
                columns; if False, add them as new columns.
                Defaults to False.
        """

        catalog = self.catalogs['science']
        segmap = self.data['segmap']
        radius = radius.to(u.arcsec)
        radius_px = radius / (self.wcs.pixel_scale_matrix[-1,-1] * u.deg).to(u.arcsec) # this won't be so great for non-aligned images...
        radius_rpx = round(radius_px.value)
        self.logger.debug(f'Dilation radius of {radius} or {radius_px:2.2f} px rounded to {radius_rpx} px')

        group_ids, group_pops, groupmap = dilate_and_group(catalog, segmap, radius=radius_rpx, fill_holes=True)

        if overwrite:
            self.catalogs['science']['group_id'] = group_ids
            self.catalogs['science']['group_pop'] = group_pops
        else:
            self.catalogs['science'].add_column(group_ids, name='group_id')
            self.catalogs['science'].add_column(group_pops, name='group_pop', index=4)
        self.data['groupmap'] = groupmap
