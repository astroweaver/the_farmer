from collections import OrderedDict
import config as conf
from .image import BaseImage
from .utils import load_brick_position, dilate_and_group, clean_catalog, build_regions
from .group import Group

import logging
import os
from functools import partial
from astropy.nddata import Cutout2D
import astropy.units as u
import numpy as np
from pathos.pools import ProcessPool
from copy import copy
from astropy.wcs.utils import proj_plane_pixel_scales


class Brick(BaseImage):
    def __init__(self, brick_id=None, position=None, size=None, load=True) -> None:


        if not np.isscalar(brick_id):
            if len(brick_id) == 1:
                brick_id = brick_id[0]
        
        self.filename = f'B{brick_id}.h5'
        self.logger = logging.getLogger(f'farmer.brick_{brick_id}')

        if load:
            self.logger.info(f'Trying to load brick from {self.filename}...')
            attributes = self.read_hdf5()
            for key in attributes:
                self.__dict__[key] = attributes[key]
                self.logger.debug(f'  ... {key}')
        
        else:

            # Housekeeping
            self.brick_id = brick_id
            self.bands = []
            self.wcs = {}
            self.pixel_scales = {}
            self.data = {} 
            self.headers = {}
            self.properties = {}
            self.catalogs = {}
            self.type = 'brick'
            self.n_sources = {}
            self.group_ids = {}
            self.group_pops = {}
            self.model_catalog = OrderedDict()
            self.model_tracker = OrderedDict()
            self.model_tracker_groups = OrderedDict()
            self.catalog_band='detection'
            self.catalog_imgtype='science'
            self.priors = None
            

            # Position
            if (brick_id is not None) & ((position is not None) | (size is not None)):
                raise RuntimeError('Cannot create brick from BOTH brick_id AND position/size!')
            if brick_id is not None:
                self.position, self.size = load_brick_position(brick_id)
            else:
                self.position, self.size = position, size
            self.buffsize = (self.size[0]+2*conf.BRICK_BUFFER, self.size[1]+2*conf.BRICK_BUFFER)

        self.logger.info(f'Spawned brick #{self.brick_id} at ({self.position.ra:2.1f}, {self.position.dec:2.1f}) with size {self.size[0].to(u.arcmin):2.1f} X {self.size[1].to(u.arcmin):2.1f}')



    def get_figprefix(self, imgtype, band):
        return f'B{self.brick_id}_{band}_{imgtype}'

    def get_bands(self):
        return np.array(self.bands)

    def summary(self):
        print(f'Summary of brick {self.brick_id}')
        print(f'Located at ({self.position.ra:2.2f}, {self.position.dec:2.2f}) with size {self.size[0]:2.2f} x {self.size[1]:2.2f}')
        print(f'   (w/ buffer: {self.buffsize[0]:2.2f} x {self.buffsize[1]:2.2f})')
        print(f'   (w/ buffer: {self.buffsize[0]:2.2f} x {self.buffsize[1]:2.2f})')
        print(f'Has {len(self.bands)} bands: {self.bands}')
        for band in self.bands:
            print(f' --- Data {band} ---')
            for imgtype in self.data[band].keys():
                if imgtype == 'psfmodel': continue
                img = self.data[band][imgtype].data
                tsum, mean, med, std = np.nansum(img), np.nanmean(img), np.nanmedian(img), np.nanstd(img)
                print(f'  {imgtype} ... {np.shape(img)} ( {tsum:2.2f} / {mean:2.2f} / {med:2.2f} / {std:2.2f})')
            # print(f'--- Properties {band} ---')
            for attr in self.properties[band].keys():
                print(f'  {attr} ... {self.properties[band][attr]}')

    def add_band(self, mosaic, overwrite=False):

        if (~overwrite) & (mosaic.band in self.bands):
            raise RuntimeError('{mosaic.band} already exists in brick #{self.brick_id}!')

        # Add band information
        self.data[mosaic.band] = {}
        self.properties[mosaic.band] = {}
        self.headers[mosaic.band] = {}
        self.n_sources[mosaic.band] = {}
        self.catalogs[mosaic.band] = {}
        self.group_ids[mosaic.band] = {}
        self.group_pops[mosaic.band] = {}
        self.bands.append(mosaic.band)

        # Loop over properties
        for attr in mosaic.properties.keys():
            self.properties[mosaic.band][attr] = mosaic.properties[attr]
            self.logger.debug(f'... property \"{attr}\" adopted from mosaic')

        # Loop over provided data
        for imgtype in mosaic.data.keys():
            if imgtype in ('science', 'weight', 'mask', 'segmap', 'groupmap', 'background', 'rms', 'model', 'residual', 'chi'):
                fill_value = np.nan
                if imgtype == 'mask':
                    fill_value = True
                cutout = Cutout2D(mosaic.data[imgtype], self.position, self.buffsize[::-1], wcs=mosaic.wcs,
                                 copy=True, mode='partial', fill_value = fill_value)
                self.logger.debug(f'... data \"{imgtype}\" subimage cut from {mosaic.band} at {cutout.input_position_original}')
                self.data[mosaic.band][imgtype] = cutout
                if imgtype in ('science', 'weight', 'mask'):
                    self.headers[mosaic.band][imgtype] = mosaic.headers[imgtype] #TODO update WCS!
                if imgtype == 'science':
                    self.wcs[mosaic.band] = cutout.wcs
                    self.pixel_scales[mosaic.band] = proj_plane_pixel_scales(cutout.wcs) * u.deg
                    self.estimate_properties(band=mosaic.band, imgtype=imgtype)
            else:
                self.data[mosaic.band][imgtype] = mosaic.data[imgtype]
                self.logger.debug(f'... data \"{imgtype}\" adopted from mosaic')

        # if weights or masks dont exist, make them as dummy arrays
        if 'weight' not in self.data[mosaic.band]:
            weight = np.ones_like(mosaic.data['science']) # big, but OK...
            cutout = Cutout2D(weight, self.position, self.buffsize[::-1], wcs=mosaic.wcs, mode='partial', fill_value = np.nan)
            self.logger.debug(f'... data \"weight\" subimage generated as ones at {cutout.input_position_original}')
            self.data[mosaic.band]['weight'] = cutout
            self.headers[mosaic.band]['weight'] = self.headers[mosaic.band]['science']

        if 'mask' not in self.data[mosaic.band]:
            mask = np.zeros_like(mosaic.data['science']).astype(bool) # big, but OK...
            cutout = Cutout2D(mask, self.position, self.buffsize[::-1], wcs=mosaic.wcs, mode='partial', fill_value = True)
            self.logger.debug(f'... data \"mask\" subimage generated as ones at {cutout.input_position_original}')
            self.data[mosaic.band]['mask'] = cutout
            self.headers[mosaic.band]['mask'] = self.headers[mosaic.band]['science']

        if 'background' not in self.data[mosaic.band]:
            background = np.zeros_like(mosaic.data['science']) # big, but OK...
            cutout = Cutout2D(background, self.position, self.buffsize[::-1], wcs=mosaic.wcs, mode='partial', fill_value = np.nan)
            self.logger.debug(f'... data \"background\" subimage generated as ones at {cutout.input_position_original}')
            self.data[mosaic.band]['background'] = cutout
            self.headers[mosaic.band]['background'] = self.headers[mosaic.band]['science']

        if 'rms' not in self.data[mosaic.band]:
            rms = np.zeros_like(mosaic.data['science']) # big, but OK...
            cutout = Cutout2D(rms, self.position, self.buffsize[::-1], wcs=mosaic.wcs, mode='partial', fill_value = np.nan)
            self.logger.debug(f'... data \"rms\" subimage generated as ones at {cutout.input_position_original}')
            self.data[mosaic.band]['rms'] = cutout
            self.headers[mosaic.band]['rms'] = self.headers[mosaic.band]['science']


        # get background info if backregion is 'brick' -- WILL overwrite inhereted info if it exists...
        if 'backregion' in self.properties[mosaic.band]:
            if self.properties[mosaic.band]['backregion'] == 'brick':
                self.estimate_background(band=mosaic.band, imgtype='science')
        
        
        # TODO -- should be able to INHERET catalogs from the parent mosaic, if they exist!

    def extract(self, band='detection', imgtype='science', background=None):
        if self.properties[band]['subtract_background']:
            background = self.get_background(band)
        catalog, segmap = self._extract(band, imgtype='science', background=background)

        # clean out buffer -- these are bricks!
        self.logger.debug('Removing sources detected in brick buffer...')
        cutout = Cutout2D(self.data[band][imgtype].data, self.position, self.size[::-1], wcs=self.data[band][imgtype].wcs)
        mask = Cutout2D(np.zeros(cutout.data.shape), self.position, self.buffsize[::-1], wcs=cutout.wcs, fill_value=1, mode='partial').data.astype(bool)
        segmap = Cutout2D(segmap, self.position, self.buffsize[::-1], self.wcs[band], fill_value=0, mode='partial')
        # do I actually need to do this?
        if np.any(mask):
            catalog, segmap.data = clean_catalog(catalog, mask, segmap=segmap.data)
            mask[mask & (segmap.data>0)] = False

        # save stuff
        self.catalogs[band][imgtype] = catalog
        self.data[band]['segmap'] = segmap
        self.headers[band]['segmap'] = self.headers[band]['science']
        self.data[band]['weight'].data[mask] = 0 #removes buffer but keeps segment pixels
        self.data[band]['mask'].data[mask] = True # adds buffer to existing mask
        self.n_sources[band][imgtype] = len(catalog)

        # add ids
        self.catalogs[band][imgtype].add_column(self.brick_id * np.ones(self.n_sources[band][imgtype], dtype=np.int32), name='brick_id', index=0)
        self.catalogs[band][imgtype].add_column(1+np.arange(self.n_sources[band][imgtype]), name='ID', index=0)

        # add world positions
        skycoords = self.data[band][imgtype].wcs.all_pix2world(catalog['x'], catalog['y'], 0)
        self.catalogs[band][imgtype].add_column(skycoords[0]*u.deg, name=f'ra', index=1, )
        self.catalogs[band][imgtype].add_column(skycoords[1]*u.deg, name=f'dec', index=2)

        # generate regions file
        build_regions(self.catalogs[band][imgtype], self.pixel_scales[band][0], # you better have square pixels!
                      outpath = os.path.join(conf.PATH_ANCILLARY, f'B{self.brick_id}_{band}_{imgtype}_objects.reg'))


    def identify_groups(self, band='detection', imgtype='science', radius=conf.DILATION_RADIUS):
        """Takes the catalog and segmap 
        """
        catalog = self.catalogs[band][imgtype]
        segmap = self.data[band]['segmap'].data
        radius = radius.to(u.arcsec)
        radius_px = radius / (self.wcs['detection'].pixel_scale_matrix[-1,-1] * u.deg).to(u.arcsec) # this won't be so great for non-aligned images...
        radius_rpx = round(radius_px.value)
        self.logger.debug(f'Dilation radius of {radius} or {radius_px:2.2f} px rounded to {radius_rpx} px')

        group_ids, group_pops, groupmap = dilate_and_group(catalog, segmap, radius=radius_rpx, fill_holes=True)

        self.catalogs[band][imgtype].add_column(group_ids, name='group_id', index=3)
        self.catalogs[band][imgtype].add_column(group_pops, name='group_pop', index=3)
        self.data[band]['groupmap'] = Cutout2D(groupmap, self.position, self.buffsize[::-1], self.wcs[band], mode='partial', fill_value = 0)
        self.group_ids[band][imgtype] = np.unique(group_ids)
        # self.group_pops[band][imgtype] = dict(zip(group_ids, group_pops))
        self.headers[band]['groupmap'] = self.headers[band]['science']

    def spawn_group(self, group_id=None, imgtype='science', bands=None):
        # Instantiate brick
        self.logger.info(f'Spawning Group #{group_id} from Brick #{self.brick_id}...')
        group = Group(group_id, self, imgtype=imgtype)
        
        # Cut up science, weight, and mask, if available
        group.add_bands(self, bands=bands)
        
        nsrcs = group.n_sources['detection'][imgtype]
        source_ids = np.array(group.get_catalog()['ID'])
        group.source_ids = source_ids
        self.logger.debug(f'Group #{group_id} has {nsrcs} source: {source_ids}')
        if nsrcs > conf.GROUP_SIZE_LIMIT:
            self.logger.warning(f'Group #{group_id} has {nsrcs} sources, but the limit is set to {conf.GROUP_SIZE_LIMIT}!')
            group.rejected = True
            return group
        
        group.priors = self.priors
        
        # Loop over model catalog
        for source_id, model in self.model_catalog.items():
            group.model_catalog[source_id] = model
            
            group.model_tracker[source_id] = {}
            for stage, stats in self.model_tracker[source_id].items():
                group.model_tracker[source_id][stage] = stats

        group.model_tracker['group'] = {}
        if group_id in self.model_tracker_groups:
            for stage, stats in self.model_tracker_groups[group_id].items():
                group.model_tracker['group'][stage] = stats

        # transfer maps
        group.transfer_maps()

        # Return it
        return group

    def detect_sources(self, band='detection', imgtype='science'):

        # detection
        self.extract(band=band, imgtype=imgtype)

        # grouping
        self.identify_groups(band=band, imgtype=imgtype)

        if conf.PLOT > 1:
            self.plot_image(imgtype='science')

        return self.catalogs[band][imgtype]

    def process_groups(self, group_ids=None, imgtype='science', mode='all'):

        if group_ids is None:
            group_ids = self.group_ids['detection'][imgtype]
        elif np.isscalar(group_ids):
            group_ids = [group_ids,]

        groups = (self.spawn_group(group_id) for group_id in group_ids)

        # loop or parallel groups
        if (conf.NCPUS == 0) | (len(group_ids) == 1):
            for group in groups:

                group = self.run_group(group, mode=mode)

                # cleanup and hand back to brick
                self.absorb(group)
        else:
            pool = ProcessPool(ncpus=conf.NCPUS)
            result = pool.uimap(partial(self.run_group, mode=mode), groups)
            [self.absorb(group) for group in result]

    def run_group(self, group, mode='all'):

        if not group.rejected:
        
            if mode == 'all':
                group.determine_models()
                group.force_models()
        
            elif mode == 'model':
                group.determine_models()

            elif mode == 'photometry':
                group.force_models()

        else:
            self.logger.warning(f'Group {group.group_id} has been rejected!')

        return group

            
    def absorb(self, group): # eventually allow mosaic to do this too! absorb bricks + make a huge model catalog!

        # check ownership
        assert self.brick_id == group.brick_id, 'Group does not belong to this brick!'

        # # rebuild maps NOTE don't do this with cutouts. Realize it for the brick itself.
        # for band in self.data:
        #     if band == 'detection': # doesn't need new seg/group, and has no models
        #         continue
        #     for imgtype in group.data[band]:
        #         if imgtype in ('model', 'residual', 'chi'):
        #             (ymin, ymax), (xmin, xmax) = group.data[band][imgtype].bbox_original
        #             (yminc, ymaxc), (xminc, xmaxc) = group.data[band][imgtype].bbox_cutout
        #             if imgtype not in self.data[band].keys():
        #                 self.data[band][imgtype] = copy(self.data[band]['science'])
        #             self.data[band][imgtype].data[ymin:ymax, xmin:xmax] \
        #                     += group.data[band][imgtype].data[yminc:ymaxc, xminc:xmaxc]

        # model catalog
        for source in group.model_catalog:
            self.model_catalog[source] = group.model_catalog[source]

        # model tracker
        for source in group.model_tracker:
            if source == 'group':
                self.model_tracker_groups[group.group_id] = group.model_tracker[source]
            else:
                self.model_tracker[source] = group.model_tracker[source]

        self.logger.debug(f'Group {group.group_id} has been absorbed')
        del group