from collections import OrderedDict
import config as conf
from .image import BaseImage
from .utils import load_brick_position, dilate_and_group, clean_catalog, build_regions, run_group
from .group import Group

import logging
import os, time, gc
from functools import partial
from astropy.nddata import Cutout2D
import astropy.units as u
import numpy as np
from pathos.pools import ProcessPool
from copy import copy
from astropy.wcs.utils import proj_plane_pixel_scales
import tqdm
                


class Brick(BaseImage):
    def __init__(self, brick_id=None, position=None, size=None, load=True, silent=False, tag=None) -> None:


        if not np.isscalar(brick_id):
            if len(brick_id) == 1:
                brick_id = brick_id[0]
        stag = ''
        if tag is not None:
            stag = f'_{tag}'
        self.filename = f'B{brick_id}{stag}.h5'
        self.logger = logging.getLogger(f'farmer.brick_{brick_id}')
        # if silent:
        #     self.logger.setLevel(logging.ERROR)

        if load:
            self.logger.info(f'Trying to load brick from {self.filename}...')
            attributes = self.read_hdf5()
            for key in attributes:
                self.__dict__[key] = attributes[key]
                self.logger.debug(f'  ... {key}')

            # TODO cross-check with config
        
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
            # self.group_pops = {}
            self.model_catalog = OrderedDict()
            self.model_tracker = OrderedDict()
            self.model_tracker_groups = OrderedDict()
            self.catalog_band='detection'
            self.catalog_imgtype='science'
            self.phot_priors = conf.PHOT_PRIORS
            self.model_priors = conf.MODEL_PRIORS
            self.config = {}
            for key in conf.__dict__:
                if not key.startswith('_'):
                    self.config[key] = conf.__dict__[key]
            
            # Position
            if (brick_id is not None) and ((position is not None) or (size is not None)):
                raise RuntimeError('Cannot create brick from BOTH brick_id AND position/size!')
            if brick_id is not None:
                self.position, self.size, self.buffsize = load_brick_position(brick_id)
            else:
                self.position, self.size = position, size
                self.buffsize = (self.size[0]+2*conf.BRICK_BUFFER, self.size[1]+2*conf.BRICK_BUFFER)

        self.logger.info(f'Spawned brick #{self.brick_id} at ({self.position.ra:2.1f}, {self.position.dec:2.1f}) with size {self.size[0].to(u.arcmin):2.3f} X {self.size[1].to(u.arcmin):2.3f}')



    def get_figprefix(self, imgtype, band):
        """Generate filename prefix for output figures.
        
        Args:
            imgtype: Type of image (e.g., 'science', 'model', 'residual')
            band: Band name
            
        Returns:
            str: Filename prefix in format 'B{brick_id}_{band}_{imgtype}'
        """
        return f'B{self.brick_id}_{band}_{imgtype}'

    def get_bands(self):
        """Get list of bands available in this brick.
        
        Returns:
            numpy array of band names
        """
        return np.array(self.bands)

    def summary(self):
        print(f'Summary of brick {self.brick_id}')
        print(f'Located at ({self.position.ra:2.2f}, {self.position.dec:2.2f}) with size {self.size[0]:2.2f} x {self.size[1]:2.2f}')
        print(f'   (w/ buffer: {self.buffsize[0]:2.2f} x {self.buffsize[1]:2.2f})')
        print(f'Has {len(self.bands)} bands: {self.bands}')
        for band in self.bands:
            print(f' --- Data {band} ---')
            for imgtype in self.data[band].keys():
                if imgtype.startswith('psf'): 
                    continue
                if isinstance(self.data[band][imgtype], dict): 
                    continue
                img = self.data[band][imgtype].data
                tsum, mean, med, std = np.nansum(img), np.nanmean(img), np.nanmedian(img), np.nanstd(img)
                print(f'  {imgtype} ... {np.shape(img)} ( {tsum:2.2f} / {mean:2.2f} / {med:2.2f} / {std:2.2f})')
            # print(f'--- Properties {band} ---')
            for attr in self.properties[band].keys():
                print(f'  {attr} ... {self.properties[band][attr]}')

    def add_band(self, mosaic, overwrite=False):

        if (not overwrite) and (mosaic.band in self.bands):
            self.logger.warning(f'{mosaic.band} already exists in brick #{self.brick_id}!')
            return

        # Loop over provided data
        for imgtype in mosaic.data.keys():
            if imgtype in ('science', 'weight', 'mask', 'background', 'rms', 'model', 'residual', 'chi'):
                fill_value = np.nan
                if imgtype == 'mask':
                    fill_value = True
                try:
                    cutout = Cutout2D(mosaic.data[imgtype], self.position, self.buffsize, wcs=mosaic.wcs,
                                    copy=True, mode='partial', fill_value = fill_value)
                    if imgtype == 'science':
                        # Add band information
                        self.data[mosaic.band] = {}
                        self.properties[mosaic.band] = {}
                        self.headers[mosaic.band] = {}
                        self.n_sources[mosaic.band] = {}
                        self.catalogs[mosaic.band] = {}
                        self.group_ids[mosaic.band] = {}
                        # self.group_pops[mosaic.band] = {}
                        self.bands.append(mosaic.band)
                except (ValueError, IndexError) as e:
                    self.logger.debug(f'{mosaic.band} mosaic has no overlap with detection footprint ({e})! Skipping band.')
                    return

                self.logger.debug(f'... data \"{imgtype}\" subimage cut from {mosaic.band} at {cutout.input_position_original}')
                self.data[mosaic.band][imgtype] = cutout
                if imgtype in ('science', 'weight', 'mask'):
                    self.headers[mosaic.band][imgtype] = mosaic.headers[imgtype] #TODO update WCS!
                if imgtype == 'science':
                    self.wcs[mosaic.band] = cutout.wcs
                    self.pixel_scales[mosaic.band] = proj_plane_pixel_scales(cutout.wcs) * u.deg
                    self.estimate_properties(band=mosaic.band, imgtype=imgtype)
            elif imgtype in ('segmap', 'groupmap'):
                self.transfer_maps()
            elif imgtype in ('psfcoords', 'psflist'):
                if imgtype == 'psflist': continue # do these together!
                if mosaic.data['psfcoords'] != 'none':
                    # within_brick = np.array([coord.contained_by(self.wcs[mosaic.band]) for coord in mosaic.data['psfcoords']])
                    within_brick = mosaic.data['psfcoords'].contained_by(self.wcs[mosaic.band])
                    if np.sum(within_brick) == 0:
                        # separation = np.array([coord.separation(self.position).to(u.arcsec).value for coord in mosaic.data['psfcoords']])
                        separation = mosaic.data['psfcoords'].separation(self.position).to(u.arcsec).value
                        nearest = np.argmin(separation)
                        self.logger.warning(f'No PSF coords within brick for {mosaic.band}! Adopting nearest at {mosaic.data["psfcoords"][nearest]}')
                        self.data[mosaic.band]['psfcoords'] = mosaic.data['psfcoords'][nearest]
                        self.data[mosaic.band]['psflist'] = [mosaic.data['psflist'][nearest]]
                    else:
                        self.data[mosaic.band]['psfcoords'] = mosaic.data['psfcoords'][within_brick]
                        self.data[mosaic.band]['psflist'] = mosaic.data['psflist'][within_brick]
                else:
                    self.data[mosaic.band]['psfcoords'] = mosaic.data['psfcoords']
                    self.data[mosaic.band]['psflist'] = mosaic.data['psflist']
                for imgtype in ('psfcoords', 'psflist'):
                    self.logger.debug(f'... data \"{imgtype}\" adopted from mosaic')
            else:
                self.data[mosaic.band][imgtype] = mosaic.data[imgtype]
                self.logger.debug(f'... data \"{imgtype}\" adopted from mosaic')

        # Loop over properties
        for attr in mosaic.properties.keys():
            self.properties[mosaic.band][attr] = mosaic.properties[attr]
            self.logger.debug(f'... property \"{attr}\" adopted from mosaic')

        # make a big filler
        # filler = np.zeros_like(mosaic.data['science']) # big, but OK...
        subheader = self.headers[mosaic.band]['science'].copy()
        subheader.update(cutout.wcs.to_header())

        # if weights or masks dont exist, make them as dummy arrays
        if 'weight' not in self.data[mosaic.band]:
            self.logger.debug(f'... data \"weight\" subimage generated as ones at {cutout.input_position_original}')
            cutout = Cutout2D(mosaic.data['science'], self.position, self.buffsize, wcs=mosaic.wcs, mode='partial', fill_value = np.nan, copy=True)
            cutout.data *= 0.
            self.data[mosaic.band]['weight'] = cutout
            self.headers[mosaic.band]['weight'] = subheader

        if 'mask' not in self.data[mosaic.band]:
            self.logger.debug(f'... data \"mask\" subimage generated as ones at {cutout.input_position_original}')
            cutout = Cutout2D(mosaic.data['science'], self.position, self.buffsize, wcs=mosaic.wcs, mode='partial', fill_value = np.nan, copy=True)
            cutout.data *= 0.
            self.data[mosaic.band]['mask'] = cutout
            self.headers[mosaic.band]['mask'] = subheader

        if 'background' not in self.data[mosaic.band]:
            self.logger.debug(f'... data \"background\" subimage generated as ones at {cutout.input_position_original}')
            cutout = Cutout2D(mosaic.data['science'], self.position, self.buffsize, wcs=mosaic.wcs, mode='partial', fill_value = np.nan, copy=True)
            cutout.data *= 0.
            self.data[mosaic.band]['background'] = cutout
            self.headers[mosaic.band]['background'] = subheader

        if 'rms' not in self.data[mosaic.band]:
            self.logger.debug(f'... data \"rms\" subimage generated as ones at {cutout.input_position_original}')
            cutout = Cutout2D(mosaic.data['science'], self.position, self.buffsize, wcs=mosaic.wcs, mode='partial', fill_value = np.nan, copy=True)
            cutout.data *= 0.
            self.data[mosaic.band]['rms'] = cutout
            self.headers[mosaic.band]['rms'] = subheader

        # get background info if backregion is 'brick' -- WILL overwrite inhereted info if it exists...
        if 'backregion' in self.properties[mosaic.band]:
            if self.properties[mosaic.band]['backregion'] == 'brick':
                self.estimate_background(band=mosaic.band, imgtype='science')

        self.logger.info(f'Added {mosaic.band} to brick #{self.brick_id}')
        
        
        # TODO -- should be able to INHERET catalogs from the parent mosaic, if they exist!

    def extract(self, band='detection', imgtype='science', background=None):
        if self.properties[band]['subtract_background']:
            background = self.get_background(band)
        catalog, segmap = self._extract(band, imgtype='science', background=background)

        # clean out buffer -- these are bricks!
        self.logger.info('Removing sources detected in brick buffer...')
        cutout = Cutout2D(self.data[band][imgtype].data, self.position, self.size, wcs=self.data[band][imgtype].wcs)
        mask = Cutout2D(np.zeros(cutout.data.shape), self.position, self.buffsize, wcs=cutout.wcs, fill_value=1, mode='partial').data.astype(bool)
        segmap = Cutout2D(segmap, self.position, self.buffsize, self.wcs[band], fill_value=0, mode='partial')
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
        self.catalogs[band][imgtype].add_column(1+np.arange(self.n_sources[band][imgtype]), name='id', index=0)

        # add world positions
        skycoords = self.data[band][imgtype].wcs.all_pix2world(catalog['x'], catalog['y'], 0)
        self.catalogs[band][imgtype].add_column(skycoords[0]*u.deg, name=f'ra', index=1, )
        self.catalogs[band][imgtype].add_column(skycoords[1]*u.deg, name=f'dec', index=2)

        # generate regions file
        build_regions(self.catalogs[band][imgtype], self.pixel_scales[band][0], # you better have square pixels!
                      outpath = os.path.join(conf.PATH_ANCILLARY, f'B{self.brick_id}_{band}_{imgtype}_objects.reg'))


    def identify_groups(self, band='detection', imgtype='science', radius=conf.DILATION_RADIUS, overwrite=False):
        """Identify groups of nearby sources using morphological dilation.
        
        Args:
            band: Band to process (default: 'detection')
            imgtype: Image type to process (default: 'science')
            radius: Dilation radius (default: from config)
            overwrite: If True, overwrite existing columns; if False, add new columns
        """
        catalog = self.catalogs[band][imgtype]
        segmap = self.data[band]['segmap'].data
        radius = radius.to(u.arcsec)
        radius_px = radius / (self.wcs['detection'].pixel_scale_matrix[-1,-1] * u.deg).to(u.arcsec) # this won't be so great for non-aligned images...
        radius_rpx = round(radius_px.value)
        self.logger.debug(f'Dilation radius of {radius} or {radius_px:2.2f} px rounded to {radius_rpx} px')

        group_ids, group_pops, groupmap = dilate_and_group(catalog, segmap, radius=radius_rpx, fill_holes=True)

        if overwrite:
            self.catalogs[band][imgtype]['group_id'] = group_ids
            self.catalogs[band][imgtype]['group_pop'] = group_pops
        else:
            self.catalogs[band][imgtype].add_column(group_ids, name='group_id', index=3)
            self.catalogs[band][imgtype].add_column(group_pops, name='group_pop', index=4)
        self.data[band]['groupmap'] = Cutout2D(groupmap, self.position, self.buffsize, self.wcs[band], mode='partial', fill_value = 0)
        self.group_ids[band][imgtype] = np.unique(group_ids)
        # self.group_pops[band][imgtype] = dict(zip(group_ids, group_pops))
        self.headers[band]['groupmap'] = self.headers[band]['science']

    def spawn_group(self, group_id=None, imgtype='science', bands=None, silent=False):
        # Instantiate group
        if not silent:
            n_sources = np.sum(self.catalogs['detection'][imgtype]['group_id'] == group_id)
            self.logger.info(f'Spawning Group #{group_id} with {n_sources} sources...')
        group = Group(group_id, self, imgtype=imgtype, silent=silent)
        if group.rejected:
            self.logger.warning(f'Group #{group_id} cannot be created!')
            return group

        # Cut up science, weight, and mask, if available
        group.add_bands(self, bands=bands)

        nsrcs = group.n_sources[group.catalog_band][imgtype]
        source_ids = np.array(group.get_catalog()['id'])
        group.source_ids = source_ids
        
        # Validate segmap - skip group if coordinates are out of bounds
        for band_name in group.bands:
            if band_name not in group.data or 'segmap' not in group.data[band_name]:
                continue
            segmap_dict = group.data[band_name]['segmap']
            if isinstance(segmap_dict, dict) and 'science' in group.data[band_name]:
                # Get image shape to check bounds
                img_shape = group.data[band_name]['science'].data.shape
                
                # Check each source's coordinates are within bounds
                for source_id, (seg_y, seg_x) in segmap_dict.items():
                    out_of_bounds = (seg_y < 0) | (seg_y >= img_shape[0]) | (seg_x < 0) | (seg_x >= img_shape[1])
                    if np.any(out_of_bounds):
                        if not silent:
                            self.logger.warning(
                                f'Group #{group_id} has source {source_id} with {np.sum(out_of_bounds)}/{len(seg_y)} '
                                f'pixels out of bounds in {band_name} ({img_shape}). Edge case from dilation. Skipping group.'
                            )
                        group.rejected = True
                        return group
        
        if not silent:
            self.logger.debug(f'Group #{group_id} has {nsrcs} sources: {source_ids}')
        if nsrcs > conf.GROUP_SIZE_LIMIT:
            if not silent:
                self.logger.debug(f'Group #{group_id} has {nsrcs} sources, but the limit is set to {conf.GROUP_SIZE_LIMIT}! Skipping...')
            group.rejected = True
            return group
        
        group.model_priors = self.model_priors
        group.phot_priors = self.phot_priors
        
        # Loop over model catalog
        group.model_catalog = {}
        for source_id, model in self.model_catalog.items():
            if source_id not in source_ids: continue
            group.model_catalog[source_id] = model
            
            group.model_tracker[source_id] = {}
            for stage, stats in self.model_tracker[source_id].items():
                group.model_tracker[source_id][stage] = stats

        group.model_tracker['group'] = {}
        if group_id in self.model_tracker_groups:
            for stage, stats in self.model_tracker_groups[group_id].items():
                group.model_tracker['group'][stage] = stats

        # # transfer maps
        # group.transfer_maps(group_id=group_id)

        # Return it
        return group

    def detect_sources(self, band='detection', imgtype='science'):

        # detection
        self.extract(band=band, imgtype=imgtype)

        # # grouping
        self.identify_groups(band=band, imgtype=imgtype)

        # transfer maps
        self.transfer_maps()

        if conf.PLOT > 1:
            self.plot_image(imgtype='science')

        return self.catalogs[band][imgtype]

    def process_groups(self, group_ids=None, imgtype='science', bands=None, mode='all'):
        self.logger.info(f'Processing groups for brick {self.brick_id}...')

        tstart = time.time()

        if group_ids is None:
            group_ids = self.group_ids['detection'][imgtype]
        elif np.isscalar(group_ids):
            group_ids = [group_ids,]

        if mode == 'pass':
            bands = ['detection',]

        # Serial processing: use generator to spawn groups one at a time
        if (conf.NCPUS == 0) | (len(group_ids) == 1):
            for group_id in group_ids:
                group = self.spawn_group(group_id, bands=bands, silent=False)
                group = run_group(group, mode=mode)
                self.absorb(group)
        else:
            # Parallel processing: use generator with imap to spawn groups in small chunks
            # This avoids both: (1) materializing all groups at once, and (2) copying brick to workers
            groups_gen = (self.spawn_group(group_id, bands=bands, silent=True) for group_id in group_ids)
            with ProcessPool(ncpus=conf.NCPUS) as pool:
                pool.restart()

                # Stream results directly - imap yields as workers complete
                for result in tqdm.tqdm(pool.imap(partial(run_group, mode=mode), groups_gen, chunksize=1), total=len(group_ids)):
                    self.absorb(result)

            self.logger.info('All results absorbed.')
            # with ProcessPool(ncpus=conf.NCPUS) as pool:
            #     pool.restart()
            #     import tqdm
            #     # imap consumes generator lazily in chunks - only spawns ~ncpus groups at a time
            #     # Collect results to avoid OrderedDict mutation during iteration
            #     results = list(tqdm.tqdm(pool.imap(partial(run_group, mode=mode), groups_gen, chunksize=1), total=len(group_ids)))
            
            # # Absorb results after pool is closed to avoid concurrent modification
            # self.logger.info('Absorbing results from parallel processing...')
            # ttstart = time.time()
            # for result in results:
            #     self.absorb(result)
            # self.logger.info(f'All results absorbed. ({time.time() - ttstart:2.2f}s)')

        self.logger.info(f'Brick {self.brick_id} has processed {len(group_ids)} groups ({time.time() - tstart:2.2f}s)')


    def absorb(self, group): # eventually allow mosaic to do this too! absorb bricks + make a huge model catalog!

        group_id, model_catalog, model_tracker = group

        # check ownership
        # assert self.brick_id == brick_id, 'Group does not belong to this brick!'

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
        for source in list(model_catalog.keys()):
            self.model_catalog[source] = model_catalog[source]

        # model tracker
        for source in list(model_tracker.keys()):
            if source == 'group':
                self.model_tracker_groups[group_id] = model_tracker[source]
            else:
                self.model_tracker[source] = model_tracker[source]

        self.logger.debug(f'Group {group_id} has been absorbed')
        del group