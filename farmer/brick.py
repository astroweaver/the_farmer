from collections import OrderedDict
import config as conf
from .image import BaseImage, MASK_FLAG_COLUMN
from .utils import load_brick_position, dilate_and_group, clean_catalog, build_regions, run_group
from .group import Group

import logging
import os, time, gc
from functools import partial
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
import astropy.units as u
import numpy as np
from scipy.ndimage import find_objects
from pathos.pools import ProcessPool
from copy import copy
from astropy.wcs.utils import proj_plane_pixel_scales
import tqdm
                


class Brick(BaseImage):
    """A spatial tile of the survey mosaic used for parallel source detection and photometry."""

    def __init__(self, brick_id=None, position=None, size=None, load=True, silent=False, tag=None) -> None:
        """Initialise a Brick, either by loading from disk or creating fresh.

        A brick is a spatial sub-region of the full mosaic used for parallel
        processing.  When ``load`` is True the object is populated from an
        existing HDF5 file named ``B{brick_id}[_{tag}].h5`` in
        ``conf.PATH_BRICKS``; otherwise a blank brick is built from the given
        position/size or looked up from the detection WCS via ``brick_id``.

        Args:
            brick_id: Integer brick identifier (1-indexed on the
                ``conf.N_BRICKS`` grid).  Mutually exclusive with providing
                both ``position`` and ``size``.
            position: ``astropy.coordinates.SkyCoord`` of the brick centre.
                Used only when ``brick_id`` is None.
            size: ``(dec_height, ra_width)`` tuple of
                ``astropy.units.Quantity`` angles giving the brick size.
                Used only when ``brick_id`` is None.
            load: If True, load an existing HDF5 brick file from disk.
                Defaults to True.
            silent: If True, suppress informational log messages.
                Defaults to False.
            tag: Optional string appended to the filename as ``_{tag}``.
                Defaults to None.

        Raises:
            RuntimeError: If both ``brick_id`` and ``position``/``size``
                are supplied simultaneously.
        """
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

            self._condition_all_bands()

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
            self.is_empty = False       # set by extract() when nothing is detected
            self.fit_status = {}        # source_id -> FIT_* code, see image.py
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


    def _condition_band_data(self, band):
        """Sanitise a single band's image arrays at ingest or load time.

        Enforces three invariants required by the rest of the pipeline:

        1. The science array contains no NaN or Inf values (replaced with 0).
        2. The mask array is strictly boolean (non-zero → True).
        3. Weight values are non-negative and finite; pixels that are bad in
           science or masked are zeroed in the weight array. Zero-weight pixels
           are folded back into the mask, unless the weight is entirely zero --
           the signature of the dummy weight synthesised for a band with none
           configured, which would otherwise mask the whole image.

        Args:
            band: Band name (key in ``self.data``) to condition in place.
                Returns immediately if the band or its science array is
                absent.
        """
        if band not in self.data:
            return
        if 'science' not in self.data[band]:
            return

        science = self.data[band]['science'].data
        bad_science = ~np.isfinite(science)
        if np.any(bad_science):
            science[bad_science] = 0

        mask_bool = np.zeros(science.shape, dtype=bool)
        bad_mask = np.zeros(science.shape, dtype=bool)
        if 'mask' in self.data[band]:
            mask = self.data[band]['mask'].data
            # Check dtype without converting to array
            if mask.dtype == bool:
                mask_bool = mask.copy()
            else:
                bad_mask = ~np.isfinite(mask)
                mask_bool = (mask != 0)
                if np.any(bad_mask):
                    mask_bool[bad_mask] = True
            self.data[band]['mask'].data = mask_bool

        if 'weight' in self.data[band]:
            weight = self.data[band]['weight'].data
            bad_weight = ~np.isfinite(weight) | (weight < 0)
            if np.any(bad_weight):
                weight[bad_weight] = 0

            # Everything downstream -- _extract's err=1/sqrt(w), build_chi_image's
            # residual*sqrt(w), and tractor's Image(invvar=w) -- assumes inverse
            # variance. Convert here if the band declares otherwise, rather than
            # silently mis-scaling every uncertainty in the catalog.
            # NOTE this must be idempotent: _condition_all_bands re-runs it on every
            # load from HDF5, so the conversion stamps weight_type back to 'invvar'
            # rather than converting a second time.
            wtype = self.properties.get(band, {}).get('weight_type', 'invvar')
            if wtype == 'sigma':
                weight = np.where(weight > 0, 1. / weight**2, 0.)
                self.logger.info(f'Converted {band} weight map from sigma to inverse variance')
            elif wtype == 'variance':
                weight = np.where(weight > 0, 1. / weight, 0.)
                self.logger.info(f'Converted {band} weight map from variance to inverse variance')
            elif wtype != 'invvar':
                raise RuntimeError(
                    f"{band}: weight_type must be 'invvar', 'sigma' or 'variance', "
                    f"not {wtype!r}")
            if band in self.properties:
                self.properties[band]['weight_type'] = 'invvar'

            weight[bad_science | mask_bool] = 0
            self.data[band]['weight'].data = weight
            if 'mask' in self.data[band]:
                # Bands with no weight configured (e.g. detection) get a dummy
                # all-zero weight from add_band. Folding that in would mark every
                # pixel masked, so only union a weight that carries information.
                if np.any(weight > 0):
                    mask_bool = mask_bool | (weight <= 0)
                self.data[band]['mask'].data = mask_bool

        self.logger.debug(
            f'Conditioned band {band}: bad_sci={np.sum(bad_science)}, '
            f'bad_mask={np.sum(bad_mask)}, masked={np.sum(mask_bool)}'
        )


    def _condition_all_bands(self):
        """Sanitise every band currently registered on this brick.

        Calls :meth:`_condition_band_data` for each entry in ``self.bands``.
        Used after loading a brick from HDF5 to ensure data integrity.
        """
        for band in self.bands:
            self._condition_band_data(band)



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
        """Print a human-readable summary of this brick to stdout.

        Reports the brick's sky position, angular size (with and without
        buffer), the list of loaded bands, and per-band image statistics
        (sum, mean, median, std) and properties.
        """
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
        """Cut out and attach a band from a full-field mosaic to this brick.

        Takes a ``Mosaic`` object, extracts a ``Cutout2D`` for each image
        type (science, weight, mask, background, rms) centred on this brick
        including its buffer, and registers the PSF model list.  Dummy
        zero-valued arrays are created for any image type not present in the
        mosaic.  After ingestion, :meth:`_condition_band_data` is called to
        sanitise the arrays.

        Args:
            mosaic: ``Mosaic`` object with pre-loaded image data for a
                single band.
            overwrite: If True, re-add the band even if it already exists
                on the brick.  Defaults to False.

        Returns:
            None. Returns early with a warning if the mosaic has no overlap
            with the brick footprint or if the band already exists and
            ``overwrite`` is False.
        """
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
                    self.logger.warning(f'{mosaic.band} mosaic has no overlap with detection footprint ({e})! Skipping band.')
                    return

                self.logger.info(f'... data \"{imgtype}\" subimage cut from {mosaic.band} at {cutout.input_position_original}')
                self.data[mosaic.band][imgtype] = cutout
                if imgtype in ('science', 'weight', 'mask'):
                    self.headers[mosaic.band][imgtype] = mosaic.headers[imgtype] #TODO update WCS!
                if imgtype == 'science':
                    self.wcs[mosaic.band] = cutout.wcs
                    self.pixel_scales[mosaic.band] = proj_plane_pixel_scales(cutout.wcs) * u.deg
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
            cutout = Cutout2D(mosaic.data['science'], self.position, self.buffsize, wcs=mosaic.wcs, mode='partial', fill_value = np.nan, copy=True)
            # No weight map was supplied. A zero-filled weight would mark every pixel
            # masked in _condition_band_data and silently drop the band from the fit,
            # so derive a uniform inverse variance from the clipped science RMS instead.
            sci = cutout.data
            good = np.isfinite(sci) & (sci != 0)
            rms = 0.
            if good.any():
                __, __, rms = sigma_clipped_stats(sci[good])
            if rms > 0:
                cutout.data = np.full(sci.shape, 1. / rms**2, dtype=np.float32)
                # what we just built IS inverse variance; without this stamp a band
                # declaring weight_type 'sigma'/'variance' would have it converted
                # a second time, reducing the weight to rms**4
                self.properties[mosaic.band]['weight_type'] = 'invvar'
                self.logger.warning(
                    f'{mosaic.band} has no weight map. Inverse variance derived from the '
                    f'clipped image RMS ({rms:.4g}); uncertainties in this band are approximate.')
            else:
                cutout.data = np.zeros(sci.shape, dtype=np.float32)
                self.properties[mosaic.band]['weight_type'] = 'invvar'
                self.logger.warning(
                    f'{mosaic.band} has no weight map and its science cutout has no usable '
                    f'pixels (clipped RMS = {rms}). This band will be EXCLUDED from fitting.')
            self.data[mosaic.band]['weight'] = cutout
            self.headers[mosaic.band]['weight'] = subheader

        if 'mask' not in self.data[mosaic.band]:
            # zeros are correct here: nothing is masked
            self.logger.debug(f'... data \"mask\" subimage generated as zeros (nothing masked)')
            cutout = Cutout2D(mosaic.data['science'], self.position, self.buffsize, wcs=mosaic.wcs, mode='partial', fill_value = np.nan, copy=True)
            cutout.data = np.zeros(cutout.data.shape, dtype=bool)
            self.data[mosaic.band]['mask'] = cutout
            self.headers[mosaic.band]['mask'] = subheader

        self._condition_band_data(mosaic.band)
        self.estimate_properties(band=mosaic.band, imgtype='science')

        if 'background' not in self.data[mosaic.band]:
            # zeros are correct here: no background to subtract
            self.logger.debug(f'... data \"background\" subimage generated as zeros')
            cutout = Cutout2D(mosaic.data['science'], self.position, self.buffsize, wcs=mosaic.wcs, mode='partial', fill_value = np.nan, copy=True)
            cutout.data = np.zeros(cutout.data.shape, dtype=np.float32)
            self.data[mosaic.band]['background'] = cutout
            self.headers[mosaic.band]['background'] = subheader

        if 'rms' not in self.data[mosaic.band]:
            # zeros are correct here: no per-pixel rms map available
            self.logger.debug(f'... data \"rms\" subimage generated as zeros')
            cutout = Cutout2D(mosaic.data['science'], self.position, self.buffsize, wcs=mosaic.wcs, mode='partial', fill_value = np.nan, copy=True)
            cutout.data = np.zeros(cutout.data.shape, dtype=np.float32)
            self.data[mosaic.band]['rms'] = cutout
            self.headers[mosaic.band]['rms'] = subheader

        # get background info if backregion is 'brick' -- WILL overwrite inhereted info if it exists...
        if 'backregion' in self.properties[mosaic.band]:
            if self.properties[mosaic.band]['backregion'] == 'brick':
                self.estimate_background(band=mosaic.band, imgtype='science')

        self.logger.info(f'Added {mosaic.band} to brick #{self.brick_id}')
        
        
        # TODO -- should be able to INHERET catalogs from the parent mosaic, if they exist!

    def extract(self, band='detection', imgtype='science', background=None):
        """Run source extraction on a band and store the resulting catalog.

        Calls the base-class ``_extract`` method to detect sources, then
        masks sources that fall within the brick buffer zone (those
        detections that would be double-counted by adjacent bricks), adds
        ``id``, ``brick_id``, ``ra``, and ``dec`` columns to the catalog,
        saves the segmentation map, and writes a DS9 regions file.

        Args:
            band: Band name to detect sources in. Defaults to
                ``'detection'``.
            imgtype: Image type to run extraction on. Defaults to
                ``'science'``.
            background: Pre-computed background array to subtract before
                detection.  When None and the band's ``subtract_background``
                property is True, the background is fetched automatically.
                Defaults to None.

        Returns:
            None. Populates ``self.catalogs[band][imgtype]``,
            ``self.data[band]['segmap']``, and ``self.n_sources[band][imgtype]``.
        """
        # Reset first: the flag is persisted to HDF5, so without this a brick that was
        # empty once would short-circuit grouping forever, even after a re-run that
        # does find sources.
        self.is_empty = False
        if self.properties[band]['subtract_background']:
            background = self.get_background(band)
        catalog, segmap = self._extract(band, imgtype='science', background=background)

        if len(catalog) == 0:
            # Nothing detected. Record an empty result and stop, rather than letting
            # the buffer-cleaning below run on a zero-row table -- but give the empty
            # catalog the SAME COLUMNS a populated one would have, so downstream code
            # (write_catalog reads catalog['id'], identify_groups reads 'group_id')
            # sees a consistent schema instead of a KeyError.
            self.logger.warning(f'Brick #{self.brick_id} has no detections in {band}. '
                                f'It will be skipped by downstream processing.')
            self.is_empty = True
            for name, dtype, unit in (('id', np.int32, None), ('brick_id', np.int32, None),
                                      ('ra', float, u.deg), ('dec', float, u.deg),
                                      ('ra_det', float, u.deg), ('dec_det', float, u.deg),
                                      ('group_id', np.int32, None), ('group_pop', np.int16, None),
                                      (MASK_FLAG_COLUMN, np.int8, None)):
                if name not in catalog.colnames:
                    catalog[name] = np.array([], dtype=dtype)
                    if unit is not None:
                        catalog[name].unit = unit
            self.catalogs[band][imgtype] = catalog
            self.data[band]['segmap'] = Cutout2D(segmap, self.position, self.buffsize,
                                                 self.wcs[band], fill_value=0, mode='partial')
            self.headers[band]['segmap'] = self.headers[band]['science']
            self.n_sources[band][imgtype] = 0
            self.group_ids.setdefault(band, {})[imgtype] = np.array([], dtype=int)
            return

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
        # ra/dec get overwritten by fitted model positions downstream; keep the
        # detection-time values so they stay recoverable.
        self.catalogs[band][imgtype].add_column(skycoords[0]*u.deg, name='ra_det', index=3)
        self.catalogs[band][imgtype].add_column(skycoords[1]*u.deg, name='dec_det', index=4)

        # generate regions file
        build_regions(self.catalogs[band][imgtype], self.pixel_scales[band][0], # you better have square pixels!
                      outpath = os.path.join(conf.PATH_ANCILLARY, f'B{self.brick_id}_{band}_{imgtype}_objects.reg'))


    def identify_groups(self, band='detection', imgtype='science', radius=conf.DILATION_RADIUS, overwrite=False):
        """Group nearby sources by morphologically dilating the segmentation map.

        Converts ``radius`` from arcsec to pixels using the detection WCS,
        calls :func:`~farmer.utils.dilate_and_group`, and stores the resulting
        ``group_id`` and ``group_pop`` columns in the catalog as well as the
        2-D group map as a ``Cutout2D``.

        Args:
            band: Band whose catalog and segmap to process.
                Defaults to ``'detection'``.
            imgtype: Image type key in the catalog dictionary.
                Defaults to ``'science'``.
            radius: Dilation radius as an ``astropy.units.Quantity`` angle.
                Defaults to ``conf.DILATION_RADIUS``.
            overwrite: If True, replace existing ``group_id``/``group_pop``
                columns; if False, add them as new columns.
                Defaults to False.
        """
        catalog = self.catalogs[band][imgtype]
        segmap = self.data[band]['segmap'].data
        radius = radius.to(u.arcsec)
        radius_px = radius / (self.wcs['detection'].pixel_scale_matrix[-1,-1] * u.deg).to(u.arcsec) # this won't be so great for non-aligned images...
        radius_rpx = round(radius_px.value)
        self.logger.debug(f'Dilation radius of {radius} or {radius_px:2.2f} px rounded to {radius_rpx} px')

        # Detection-masked sources are excluded from grouping entirely -- their
        # pixels are dropped before dilation, so they cannot bridge two groups, and
        # they come back with group_id 0. They keep their catalog row; they just
        # never enter a group and so are never modelled.
        exclude = None
        if MASK_FLAG_COLUMN in catalog.colnames:
            exclude = np.asarray(catalog[MASK_FLAG_COLUMN]).astype(bool)

        group_ids, group_pops, groupmap = dilate_and_group(catalog, segmap, radius=radius_rpx,
                                                           fill_holes=True, exclude=exclude)

        if overwrite:
            self.catalogs[band][imgtype]['group_id'] = group_ids
            self.catalogs[band][imgtype]['group_pop'] = group_pops
        else:
            self.catalogs[band][imgtype].add_column(group_ids, name='group_id', index=3)
            self.catalogs[band][imgtype].add_column(group_pops, name='group_pop', index=4)
        self.data[band]['groupmap'] = Cutout2D(groupmap, self.position, self.buffsize, self.wcs[band], mode='partial', fill_value = 0)
        # group_id 0 is the ungrouped bucket that detection-masked sources land in.
        # It is not a group and must never be spawned or processed.
        self.group_ids[band][imgtype] = np.unique(group_ids[group_ids > 0])
        n_ungrouped = int(np.sum(np.asarray(group_ids) == 0))
        if n_ungrouped:
            self.logger.info(f'{n_ungrouped} masked sources kept in the catalog but not grouped.')
        # self.group_pops[band][imgtype] = dict(zip(group_ids, group_pops))
        self.headers[band]['groupmap'] = self.headers[band]['science']

        # Precompute every group's bounding box and pixel count in ONE pass over the
        # brick-sized groupmap. Group.__init__ used to rescan the whole map for each
        # group it spawned, which is O(N_groups x N_brick_pixels) and, because
        # process_groups feeds pool.imap from a generator, ran serially in the parent
        # process even with NCPUS > 1.
        self._compute_group_bboxes(band=band)

    def _compute_group_bboxes(self, band='detection'):
        """Cache each group's bounding box and pixel count from the brick groupmap.

        ``dilate_and_group`` renumbers groups to a contiguous 1..N, so a single
        ``find_objects`` pass gives every box.

        Args:
            band: Band whose groupmap to scan. Defaults to ``'detection'``.

        Returns:
            dict: ``{group_id: (ylo, yhi, xlo, xhi, npix)}`` with yhi/xhi
                EXCLUSIVE, as numpy slice bounds are. Also stored on
                ``self.group_bboxes``.
        """
        gmap = self.data[band]['groupmap'].data
        slices = find_objects(gmap)
        counts = np.bincount(gmap.ravel())
        self.group_bboxes = {}
        for i, sl in enumerate(slices):
            gid = i + 1                     # find_objects is 0-indexed over labels 1..N
            if sl is None:
                continue
            npix = int(counts[gid]) if gid < len(counts) else 0
            if npix == 0:
                continue
            self.group_bboxes[gid] = (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, npix)
        self.logger.debug(f'Cached bounding boxes for {len(self.group_bboxes)} groups')
        return self.group_bboxes

    def get_group_bbox(self, group_id, band='detection'):
        """Return one group's bounding box, computing the cache on first use.

        The cache is deliberately not serialised to HDF5 -- it is one cheap pass
        over the groupmap, and writing thousands of tiny datasets would cost more
        than recomputing it.

        Args:
            group_id: Integer group identifier.
            band: Band whose groupmap to use. Defaults to ``'detection'``.

        Returns:
            tuple or None: ``(ylo, yhi, xlo, xhi, npix)`` with yhi/xhi exclusive,
                or None if the group has no pixels.
        """
        if not getattr(self, 'group_bboxes', None):
            if band not in self.data or 'groupmap' not in self.data[band]:
                return None
            self._compute_group_bboxes(band=band)
        return self.group_bboxes.get(int(group_id))

    def spawn_group(self, group_id=None, imgtype='science', bands=None, silent=False):
        """Instantiate a ``Group`` from this brick for a single source group.

        Creates a ``Group`` object centred on the group's bounding box,
        cuts out all requested bands, validates that segment pixel
        coordinates are in bounds, enforces the ``conf.GROUP_SIZE_LIMIT``,
        transfers model catalog entries and tracking history from the brick,
        and returns the ready-to-process group.

        Args:
            group_id: Integer identifier of the group to spawn.
            imgtype: Image type used to look up catalog group membership.
                Defaults to ``'science'``.
            bands: List of band names to include in the group.  ``None``
                includes all bands available on the brick.
                Defaults to None.
            silent: If True, suppress informational log messages.
                Defaults to False.

        Returns:
            Group: Populated group object.  ``group.rejected`` is True if
                the group has no pixels, exceeds the size limit, or has
                out-of-bounds segment pixels.
        """
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
        for source_id, model in list(self.model_catalog.items()):
            if source_id not in source_ids: continue
            group.model_catalog[source_id] = model
            
            group.model_tracker[source_id] = {}
            for stage, stats in list(self.model_tracker[source_id].items()):
                group.model_tracker[source_id][stage] = stats

        group.model_tracker['group'] = {}
        if group_id in self.model_tracker_groups:
            for stage, stats in list(self.model_tracker_groups[group_id].items()):
                group.model_tracker['group'][stage] = stats

        # # transfer maps
        # group.transfer_maps(group_id=group_id)

        # Return it
        return group

    def detect_sources(self, band='detection', imgtype='science'):
        """Run the full source detection pipeline on this brick.

        Calls :meth:`extract` to detect sources and build the segmentation
        map, :meth:`identify_groups` to assign group memberships, and
        :meth:`transfer_maps` to propagate the maps to photometric bands.
        Optionally plots the science image when ``conf.PLOT > 1``.

        Args:
            band: Band to run detection on. Defaults to ``'detection'``.
            imgtype: Image type to process. Defaults to ``'science'``.

        Returns:
            astropy.table.Table: The detection catalog for this brick. Empty if
                nothing was detected, in which case grouping and map transfer are
                skipped and ``self.is_empty`` is True.
        """
        # detection
        self.extract(band=band, imgtype=imgtype)

        if getattr(self, 'is_empty', False):
            self.group_ids[band][imgtype] = np.array([], dtype=int)
            return self.catalogs[band][imgtype]

        # # grouping
        self.identify_groups(band=band, imgtype=imgtype)

        # transfer maps
        self.transfer_maps()

        if conf.PLOT > 1:
            self.plot_image(imgtype='science')

        return self.catalogs[band][imgtype]

    def process_groups(self, group_ids=None, imgtype='science', bands=None, mode='all'):
        """Model and/or measure photometry for all (or specified) groups.

        Spawns each group in turn and passes it to :func:`~farmer.utils.run_group`.
        When ``conf.NCPUS > 1`` and more than one group is requested, groups
        are processed in parallel via ``pathos.pools.ProcessPool`` with a
        lazy generator to keep peak memory usage low.

        Args:
            group_ids: Integer group ID(s) to process.  ``None`` processes
                all groups in the detection catalog. Defaults to None.
            imgtype: Image type used to look up catalog group membership.
                Defaults to ``'science'``.
            bands: Band names to include in each group.  ``None`` uses all
                bands on the brick.  Overridden to ``['detection']`` when
                ``mode='pass'``. Defaults to None.
            mode: Processing mode forwarded to :func:`~farmer.utils.run_group`
                — one of ``'all'``, ``'model'``, ``'photometry'``, or
                ``'pass'``. Defaults to ``'all'``.
        """
        self.logger.info(f'Processing groups for brick {self.brick_id}...')

        tstart = time.time()

        if group_ids is None:
            group_ids = self.group_ids['detection'][imgtype]
        elif np.isscalar(group_ids):
            group_ids = [group_ids,]

        if mode == 'pass':
            bands = ['detection',]

        if len(group_ids) == 0:
            self.logger.warning(f'Brick {self.brick_id} has no groups to process.')
            return

        self.fit_status = getattr(self, 'fit_status', {})

        # Serial processing: use generator to spawn groups one at a time
        if (conf.NCPUS == 0) | (len(group_ids) == 1):
            # tqdm on the serial path too -- it is the shipped default (NCPUS = 0)
            # and the loop that dominates wall time.
            iterator = tqdm.tqdm(group_ids, total=len(group_ids),
                                 desc=f'Brick {self.brick_id}: {len(group_ids)} groups',
                                 dynamic_ncols=True, smoothing=0.1)
            for i, group_id in enumerate(iterator):
                group = self.spawn_group(group_id, bands=bands, silent=False)
                was_rejected = getattr(group, 'rejected', False)
                result = run_group(group, mode=mode)
                self.absorb(result)
                self._record_fit_status(result, was_rejected, imgtype=imgtype)
        else:
            # Parallel processing: use generator with imap to spawn groups in small chunks
            # This avoids both: (1) materializing all groups at once, and (2) copying brick to workers
            #
            # KNOWN COST, not yet fixed: the pool is created HERE, after the brick is
            # already in memory, so every worker forks off a multi-GB parent and
            # copy-on-write copies it as CPython touches refcounts. Measured on
            # CDFS B56 (120 groups, peak RSS across the process tree): memory grows
            # linearly at ~3.4 GB per worker -- roughly 1.8 GB in the pool worker plus
            # 1.6 GB in the run_group timeout child. NCPUS is therefore bounded by RAM,
            # not by scaling: the pool itself is healthy, at 3.7x on 4 workers and
            # 3.6x on the 16 largest groups.
            # Fixing it properly means creating the pool BEFORE the brick is loaded, or
            # holding band pixel data in shared memory -- both are restructures rather
            # than local edits, so they are deliberately out of scope here.
            groups_gen = (self.spawn_group(group_id, bands=bands, silent=True) for group_id in group_ids)
            with ProcessPool(ncpus=conf.NCPUS) as pool:
                pool.restart()

                # Stream results directly - imap yields as workers complete
                # Use position=0 and dynamic_ncols to keep progress bar visible despite logging
                pbar = tqdm.tqdm(
                    pool.imap(partial(run_group, mode=mode), groups_gen, chunksize=1),
                    total=len(group_ids),
                    desc=f'Processing {len(group_ids)} groups',
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                    smoothing=0.1
                )
                pbar.refresh()
                for result in pbar:
                    self.absorb(result)
                    # spawn_group ran in the parent, so a rejected group is already
                    # reflected in the empty model_catalog the worker returned -- but
                    # we no longer have the Group object to ask. Look the rejection up
                    # by group size, so the status matches what the serial path records.
                    gid = result[0]
                    cat = self.catalogs[self.catalog_band][imgtype]
                    n_in_group = int(np.sum(np.asarray(cat['group_id']) == gid))
                    was_rejected = n_in_group > conf.GROUP_SIZE_LIMIT or n_in_group == 0
                    self._record_fit_status(result, was_rejected, imgtype=imgtype)
                pbar.close()

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

        # Report what actually happened. Without this a brick could lose an arbitrary
        # fraction of its groups to IGNORE_FAILURES and still look like a clean run.
        n_ok = sum(1 for v in self.fit_status.values() if v == 0)
        n_bad = sum(1 for v in self.fit_status.values() if v != 0)
        elapsed = time.time() - tstart
        summary = (f'Brick {self.brick_id}: processed {len(group_ids)} groups in '
                   f'{elapsed:2.2f}s -- {n_ok} sources fitted, {n_bad} not')
        if n_bad:
            self.logger.warning(summary)
        else:
            self.logger.info(summary)


    def _record_fit_status(self, result, was_rejected, imgtype='science'):
        """Record, per source, whether its group produced a model.

        Args:
            result: The ``(group_id, model_catalog, model_tracker)`` tuple from
                :func:`~farmer.utils.run_group`.
            was_rejected: True if ``spawn_group`` rejected the group outright.
            imgtype: Image type key for the detection catalog lookup.
        """
        from .image import FIT_OK, FIT_GROUP_REJECTED, FIT_FAILED
        group_id, model_catalog, __ = result
        try:
            catalog = self.catalogs[self.catalog_band][imgtype]
            members = np.asarray(catalog['id'])[np.asarray(catalog['group_id']) == group_id]
        except (KeyError, IndexError):
            return
        for sid in members:
            if int(sid) in model_catalog:
                self.fit_status[int(sid)] = FIT_OK
            else:
                self.fit_status[int(sid)] = FIT_GROUP_REJECTED if was_rejected else FIT_FAILED


    def absorb(self, group):
        """Merge a processed group's results back into this brick.

        Accepts the ``(group_id, model_catalog, model_tracker)`` tuple
        returned by :func:`~farmer.utils.run_group` and copies each source's
        model and tracking entry into the brick-level dictionaries.
        Group-level tracker entries (keyed ``'group'``) are stored separately
        in ``self.model_tracker_groups``.

        Args:
            group: Tuple ``(group_id, model_catalog, model_tracker)`` as
                returned by :func:`~farmer.utils.run_group`.
        """
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