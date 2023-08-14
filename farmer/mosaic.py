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
default_properties['backregion'] = 'mosaic'
default_properties['zeropoint'] = -99

class Mosaic(BaseImage):
    def __init__(self, band, load=False) -> None:

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
            for key in default_properties:
                if key not in self.properties:
                    self.properties[key] = default_properties[key]
            if 'name' not in self.properties:
                self.properties['name'] = band
            good = '✓'
            bad = 'X'
            # verify the band
            self.paths = {}
            data_status = bad
            data_provided = []
            for imgtype in ['science', 'weight', 'mask', 'psfmodel']:
                if imgtype not in self.properties.keys():
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
                psf_status = '--'
            else:
                psf_status = good

            # verify the WCS
            wcs_status = bad
            try:
                self.wcs = WCS(fits.getheader(self.properties['science']))
                self.pixel_scale = proj_plane_pixel_scales(self.wcs) * u.deg
                wcs_status = good
            except:
                raise RuntimeError(f'The World Coordinate System for {band} cannot be understood!')

            arr_shape = self.wcs.array_shape
            self.position = self.wcs.pixel_to_world(arr_shape[0]/2., arr_shape[1]/2.)
            # upper = self.wcs.pixel_to_world(arr_shape[0], arr_shape[1])
            # lower = self.wcs.pixel_to_world(0, 0)
            self.size = arr_shape * self.pixel_scale
            
            self.logger.debug(f'Mosaic {band} is centered at {self.position.ra:2.1f}, {self.position.dec:2.1f}')
            self.logger.debug(f'Mosaic {band} has size at {self.size[0]:2.1f}, {self.size[1]:2.1f}')

            self.logger.info(f'{band:10}: {data_status} Data {tuple(data_provided)} {psf_status} PSF {wcs_status} WCS ({self.position.ra:2.1f}, {self.position.dec:2.1f})')

        # Now load in the data (memory use!) -- and only what is necessary!
        if load:
            self.data = {}
            self.headers = {}
            self.catalogs = {}
            self.n_sources = {}
            self.logger.info(f'Loading {list(self.paths.keys())} for {band}')
            for attr in self.paths.keys():
                if attr == 'psfmodel':
                    self.data[attr] = validate_psfmodel(band)
                else:
                    self.data[attr] = fits.getdata(self.paths[attr])
                    self.headers[attr] = fits.getheader(self.paths[attr])
                if attr in ('science', 'weight'):
                    self.estimate_properties(band=band, imgtype=attr)
            if band in conf.BANDS:
                if 'backregion' in conf.BANDS[band]:
                    if conf.BANDS[band]['backregion'] == 'mosaic':
                        self.estimate_background(band=band, imgtype='science')
            elif band == 'detection':
                if 'backregion' in conf.DETECTION:
                    if conf.DETECTION['backregion'] == 'mosaic':
                            self.estimate_background(band=band, imgtype='science')
            

    def get_bands(self):
        return np.array([self.band])

    def get_figprefix(self, imgtype, band=None):
        return f'{self.band}_{imgtype}'

    def add_to_brick(self, brick):
        # Cut up science, weight, and mask, if available
        brick.add_band(self)

        # Return it
        return brick

    def spawn_brick(self, brick_id=None, position=None, size=None):
        # Instantiate brick
        if brick_id is None:
            brick = Brick(position, size, load=False)
        else:
            brick = Brick(brick_id, load=False)
        
        # Cut up science, weight, and mask, if available
        brick.add_band(self)

        # Return it
        return brick

    def extract(self, background=None):
        catalog, segmap = self._extract(band=None, background=background)

        self.catalogs['science'] = catalog
        self.data['segmap'] = segmap
        self.n_sources['science'] = len(catalog)

        # add ids
        colname = 'ID'
        self.catalogs['science'].add_column(1+np.arange(self.n_sources['science']), name=colname, index=0)

        # add world positions
        skycoords = self.wcs.all_pix2world(catalog['x'], catalog['y'], 0)
        self.catalogs['science'].add_column(skycoords[0], name=f'ra')
        self.catalogs['science'].add_column(skycoords[1], name=f'dec')


    def identify_groups(self, radius=conf.DILATION_RADIUS):
        """Takes the catalog and segmap 
        """

        catalog = self.catalogs['science']
        segmap = self.data['segmap']
        radius = radius.to(u.arcsec)
        radius_px = radius / (self.wcs.pixel_scale_matrix[-1,-1] * u.deg).to(u.arcsec) # this won't be so great for non-aligned images...
        radius_rpx = round(radius_px.value)
        self.logger.debug(f'Dilation radius of {radius} or {radius_px:2.2f} px rounded to {radius_rpx} px')

        group_ids, group_pops, groupmap = dilate_and_group(catalog, segmap, radius=radius_rpx, fill_holes=True)

        self.catalogs['science'].add_column(group_ids, name='group_id')
        self.catalogs['science'].add_column(group_pops, name='group_pop', index=3)
        self.data['groupmap'] = groupmap
