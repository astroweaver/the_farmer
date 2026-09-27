import config as conf
from .utils import validate_psfmodel, dilate_and_group, load_brick_position, read_wcs, group_centres
from .utils import models_from_catalog, read_catalog_columns, create_fits_memmap, provenance_header
from .brick import Brick
from .image import BaseImage, FIT_OK, select_reconstruction_models, render_model_image, sep_background

import logging
import os
import copy
from collections import OrderedDict
from functools import partial
from astropy.wcs import WCS, Sip
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales
from pathos.pools import ProcessPool
from tractor.pointsource import PointSource
from tqdm import tqdm

default_properties = {}
default_properties['subtract_background'] = False
default_properties['backtype'] = 'flat'
default_properties['backregion'] = 'brick'
default_properties['zeropoint'] = -99

# Catalog columns that define a model's position and shape (see models_from_catalog).
_MODEL_COLUMNS = ('ra', 'dec', 'logre', 'ee1', 'ee2', 'softfracdev',
                  'logre_exp', 'ee1_exp', 'ee2_exp', 'logre_dev', 'ee1_dev', 'ee2_dev')
_RENDER_COLUMNS = ('id', 'name', 'fit_status', 'group_id') + _MODEL_COLUMNS

# Pixels per row stripe when streaming mosaic-sized arrays (~200 MB per float32 stripe).
_STRIPE_PIXELS = 50_000_000


def _stripes(ny, nx):
    """Yield ``(r0, r1)`` row ranges covering a ``(ny, nx)`` image in bounded chunks."""
    step = max(1, _STRIPE_PIXELS // max(nx, 1))
    for r0 in range(0, ny, step):
        yield r0, min(ny, r0 + step)

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
            # Copy, do not alias: the loops below rewrite bools as ints and add
            # defaults, and BaseImage.set_property writes runtime state (rms,
            # background) straight into self.properties. Aliasing meant a mosaic
            # load permanently mutated the user's config module.
            if band == 'detection':
                self.properties = dict(conf.DETECTION)
            else:
                self.properties = dict(conf.BANDS[band])
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

            # array_shape is (ny, nx) but pixel_to_world takes (x, y), and
            # proj_plane_pixel_scales returns [scale_x, scale_y] -- keep the two
            # orderings straight or the centre lands elsewhere on a non-square mosaic.
            ny, nx = self.wcs.array_shape
            self.position = self.wcs.pixel_to_world(nx / 2., ny / 2.)
            # (dec_height, ra_width), matching Brick.size and Cutout2D's (ny, nx)
            self.size = (ny * self.pixel_scale[1], nx * self.pixel_scale[0])
            
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

    def rebuild_images(self, brick_ids=None, imgtypes=('model', 'residual'), reconstruct=True,
                       max_extent=1*u.arcmin, ncpus=None, overwrite=None, directory=conf.PATH_ANCILLARY):
        """Reconstruct full-field model, residual, and chi images for this band.

        Each brick's fitted models are rebuilt from its catalog
        (``PATH_CATALOGS/B{id}.cat``, see ``models_from_catalog``), rendered onto
        a window of this mosaic's pixel grid, and summed into one full-field
        model. Each source belongs to exactly one brick (detections in a brick's
        buffer are dropped), so the sum counts it once. Unlike the per-brick
        images, the result has no seams. A profile is not cut off at its brick's
        buffer, and light from sources in neighbouring bricks is included.

        Rendering matches the fit. It uses the same Tractor WCS conversion, the
        same ``RESIDUAL_*`` cuts, and the same PSF. For a spatially varying PSF,
        that is the grid point nearest each group, chosen from the points the
        brick held. One approximation: here a group's centre is the mean position
        of its members, while the fit used the centre of the group's bounding
        box. The chosen PSF can differ only for groups right on a PSF-cell
        boundary.

        The residual is science minus model, and minus the background the fit
        removed when ``subtract_background`` is set. With ``backregion =
        'mosaic'``, that background is re-estimated on the full image. With
        ``'brick'``, it is re-estimated on each brick's buffered cutout and applied
        over that brick's tile. Tiles follow the detection-image brick grid, so
        they neither overlap nor leave gaps. Chi is residual x sqrt(inverse
        variance). Pixels with no data (non-finite science) are NaN in the
        residual. Masked or zero-weight pixels are NaN in chi, not the 0 used in
        brick products, so they drop out of nan-aware statistics. On drizzled or
        resampled data the noise is correlated, so chi overstates significance.

        Outputs are single-HDU float32 FITS files with the science header's WCS,
        named ``M{band}_{imgtype}.fits`` and written in ``directory``. They are
        streamed to disk. Peak memory is a few brick windows, plus the full
        science image when a mosaic-level background must be re-estimated.

        Args:
            brick_ids: Bricks to include. ``None`` uses every brick on the grid.
                Bricks without a catalog, or without this band measured, are
                skipped and reported.
            imgtypes: Any of ``'model'``, ``'residual'``, ``'chi'``. The model is
                always written, and chi also writes the residual it derives from.
            reconstruct: Apply the ``RESIDUAL_*`` cuts. Models with non-finite
                parameters are dropped regardless.
            max_extent: Largest radius, as an angle, by which one source may grow
                its brick's window. Tractor sizes a patch to 4 r_e (exp) or 8 r_e
                (deV) plus the PSF radius. Sources needing more are counted and
                may be truncated at the window edge.
            ncpus: Worker processes for rendering. ``None`` uses ``conf.NCPUS``.
            overwrite: Replace existing outputs. ``None`` uses ``conf.OVERWRITE``.
            directory: Output directory. Defaults to ``conf.PATH_ANCILLARY``.

        Returns:
            dict: ``imgtype -> path`` for each file written.

        Raises:
            ValueError: For the detection band, which has no models, or an
                unknown image type.
            RuntimeError: If an output exists and ``overwrite`` is False, or if no
                brick contributed a single model.
        """
        band = self.band
        if band == 'detection':
            raise ValueError('The detection band has no models to rebuild.')
        imgtypes = [imgtypes,] if isinstance(imgtypes, str) else list(imgtypes)
        unknown = set(imgtypes) - {'model', 'residual', 'chi'}
        if unknown:
            raise ValueError(f'Cannot rebuild {sorted(unknown)}; choose from model, residual, chi.')
        if ('chi' in imgtypes) and ('weight' not in self.paths):
            self.logger.warning(f'{band} has no weight map configured, so no chi image can be built.')
            imgtypes.remove('chi')
        imgtypes = ['model'] + [t for t in ('residual', 'chi')
                                if (t in imgtypes) or (t == 'residual' and 'chi' in imgtypes)]

        if brick_ids is None:
            brick_ids = self.brick_ids
        elif np.isscalar(brick_ids):
            brick_ids = [brick_ids,]
        brick_ids = [int(bid) for bid in brick_ids]
        ncpus = conf.NCPUS if ncpus is None else ncpus
        overwrite = conf.OVERWRITE if overwrite is None else overwrite
        max_extent = u.Quantity(max_extent, u.arcsec)

        paths = {imgtype: os.path.join(directory, self.filename.replace('.h5', f'_{imgtype}.fits'))
                 for imgtype in imgtypes}
        existing = [path for path in paths.values() if os.path.exists(path)]
        if existing and not overwrite:
            raise RuntimeError(f'Rebuilt images already exist (overwrite = False): {existing}')

        # Everything that can fail cheaply fails here, before hours of rendering.
        if not hasattr(self, 'data'):
            self.data = {}
        if 'psflist' not in self.data:
            self.data['psfcoords'], self.data['psflist'] = validate_psfmodel(band)
        psfcoords = self.data['psfcoords']
        self.get_psfmodel(band, None if np.any(psfcoords == 'none') else psfcoords[0])
        background_mode = self._fit_background_mode()
        tiles = self._brick_tiles(brick_ids) if background_mode == 'brick' else None

        ext = self.properties.get('extension', 0)
        ny, nx = self.wcs.array_shape
        pixscale = float(np.min(self.pixel_scale.to_value(u.arcsec)))      # arcsec/pix
        science_header = fits.getheader(self.paths['science'], ext=ext)

        def product_header(imgtype):
            hdr = science_header.copy()
            if imgtype == 'chi':
                hdr.remove('BUNIT', ignore_missing=True)        # chi is dimensionless
            hdr['FARMIMG'] = (imgtype, 'Farmer full-field product')
            hdr['FARMBAND'] = (band, 'band rendered')
            hdr['MODELSRC'] = ('catalog', 'models rebuilt from PATH_CATALOGS/B{id}.cat')
            hdr['RECONCUT'] = (int(bool(reconstruct)), 'RESIDUAL_* cuts applied')
            hdr['MAXEXT'] = (float(max_extent.to_value(u.arcsec)), '[arcsec] max_extent')
            hdr['BKGMODE'] = (str(background_mode), 'background subtracted from residual')
            hdr['NBRICKS'] = (len(brick_ids), 'bricks requested')
            hdr.extend(provenance_header(extra=getattr(self, 'psf_aperture_correction', None)),
                       update=True)
            return hdr

        # ---- model ----
        # Workers get a slim copy: a loaded mosaic would otherwise pickle its full
        # science, weight and mask arrays to every process.
        worker = copy.copy(self)
        worker.data = {'psfcoords': self.data['psfcoords'], 'psflist': self.data['psflist']}
        worker.catalogs = {}
        render = partial(worker._render_brick_models, reconstruct=reconstruct,
                         max_extent_px=max_extent.to_value(u.arcsec) / pixscale)

        model = create_fits_memmap(paths['model'], (ny, nx), product_header('model'), overwrite=overwrite)
        status = OrderedDict()
        counts = {}
        tally = dict(n_rendered=0, n_capped=0)
        ras, decs, owners = [], [], []

        def absorb(result):
            status.setdefault(result['status'], []).append(result['brick_id'])
            for reason, n in result['counts'].items():
                counts[reason] = counts.get(reason, 0) + n
            if result['window'] is None:
                return
            y0, x0 = result['origin']
            h, w = result['window'].shape
            model[y0:y0 + h, x0:x0 + w] += result['window']
            tally['n_rendered'] += result['n_rendered']
            tally['n_capped'] += result['n_capped']
            ras.append(result['ra'])
            decs.append(result['dec'])
            owners.append(np.full(len(result['ra']), result['brick_id']))

        desc = f'Rendering {band} models'
        if (ncpus > 1) and (len(brick_ids) > 1):
            with ProcessPool(ncpus=ncpus) as pool:
                pool.restart()
                for result in tqdm(pool.imap(render, brick_ids), total=len(brick_ids), desc=desc):
                    absorb(result)
        else:
            for brick_id in tqdm(brick_ids, desc=desc):
                absorb(render(brick_id))
        model.flush()

        why = dict(no_catalog='have no catalog', band_not_measured=f'have no {band}_flux column',
                   no_models='have no usable models', outside='fall outside this mosaic')
        for key, reason in why.items():
            if key in status:
                shown = status[key][:10]
                more = '...' if len(status[key]) > 10 else ''
                self.logger.warning(f'{band}: {len(status[key])} bricks {reason}: {shown}{more}')
        if tally['n_rendered'] == 0:
            del model
            os.remove(paths['model'])
            raise RuntimeError(f'{band}: no brick contributed a model; nothing was written.')
        self.logger.info(f'{band}: rendered {tally["n_rendered"]} models from '
                         f'{len(status.get("rendered", []))} of {len(brick_ids)} bricks')
        dropped = {reason: n for reason, n in counts.items() if n}
        if dropped:
            self.logger.info(f'{band}: models dropped or zeroed before rendering: {dropped}')
        if tally['n_capped']:
            self.logger.warning(f'{band}: {tally["n_capped"]} models extend past max_extent = '
                                f'{max_extent} and may be truncated at their window edge.')
        self._check_duplicates(np.concatenate(ras), np.concatenate(decs), np.concatenate(owners),
                               tolerance=0.5 * pixscale)

        # ---- residual ----
        if 'residual' in imgtypes:
            residual = create_fits_memmap(paths['residual'], (ny, nx), product_header('residual'),
                                          overwrite=overwrite)
            with fits.open(self.paths['science'], memmap=True) as hdul:
                science = hdul[ext].data
                background = None
                if background_mode == 'mosaic':
                    self.logger.info(f'{band}: re-estimating the mosaic-level background...')
                    back = sep_background(np.asarray(science), band)
                    background = back.globalback if self.properties['backtype'] == 'flat' else back.back()
                    del back
                for r0, r1 in _stripes(ny, nx):
                    stripe = np.asarray(science[r0:r1], dtype=np.float32) - model[r0:r1]
                    if background is not None:
                        stripe -= background if np.isscalar(background) else background[r0:r1]
                    stripe[~np.isfinite(stripe)] = np.nan
                    residual[r0:r1] = stripe
                del background
                if background_mode == 'brick':
                    self._subtract_brick_backgrounds(science, residual, tiles)
                del science
            residual.flush()

        # ---- chi ----
        if 'chi' in imgtypes:
            chi = create_fits_memmap(paths['chi'], (ny, nx), product_header('chi'), overwrite=overwrite)
            wtype = self.properties.get('weight_type', 'invvar')
            whdul = fits.open(self.paths['weight'], memmap=True)
            mhdul = fits.open(self.paths['mask'], memmap=True) if 'mask' in self.paths else None
            try:
                weight = whdul[ext].data
                mask = None if mhdul is None else mhdul[ext].data
                for r0, r1 in _stripes(ny, nx):
                    wgt = np.asarray(weight[r0:r1], dtype=np.float64)
                    good = np.isfinite(wgt) & (wgt > 0)
                    wgt = np.where(good, wgt, 1.)
                    # the same conversions Brick._condition_band_data applies
                    if wtype == 'sigma':
                        wgt = 1. / wgt**2
                    elif wtype == 'variance':
                        wgt = 1. / wgt
                    if mask is not None:
                        good &= (np.asarray(mask[r0:r1]) == 0)      # NaN mask pixels count as masked
                    stripe = (residual[r0:r1] * np.sqrt(wgt)).astype(np.float32)
                    stripe[~good] = np.nan
                    chi[r0:r1] = stripe
                del weight, mask
            finally:
                whdul.close()
                if mhdul is not None:
                    mhdul.close()
            chi.flush()
            del chi

        for imgtype, path in paths.items():
            self.logger.info(f'{band}: wrote {imgtype} to {path}')
        return paths

    def _render_brick_models(self, brick_id, reconstruct=True, max_extent_px=np.inf):
        """Render one brick's fitted models onto a window of this mosaic's pixel grid.

        The window is the bounding box of every source's patch, clipped to the
        mosaic. Unlike in the per-brick images, no model is cut off at the brick's
        buffer.

        Args:
            brick_id: Brick whose ``PATH_CATALOGS/B{id}.cat`` supplies the models.
            reconstruct: Apply the ``RESIDUAL_*`` cuts.
            max_extent_px: Cap on the radius by which any one source grows the
                window, pixels.

        Returns:
            dict: ``brick_id``; ``status`` (``'rendered'``, ``'no_catalog'``,
                ``'band_not_measured'``, ``'no_models'`` or ``'outside'``);
                ``counts`` of models dropped per reason; ``window`` (float32, or
                None) and its ``origin`` ``(y0, x0)`` on the mosaic, pixels;
                ``n_rendered``; ``n_capped``; and the rendered models' ``ra`` and
                ``dec`` (deg), for the cross-brick duplicate check.
        """
        band = self.band
        out = dict(brick_id=int(brick_id), status='rendered', counts={}, window=None, origin=None,
                   n_rendered=0, n_capped=0, ra=np.empty(0), dec=np.empty(0))
        path = os.path.join(conf.PATH_CATALOGS, f'B{brick_id}.cat')
        if not os.path.exists(path):
            out['status'] = 'no_catalog'
            return out

        flux_col = f'{band}_flux'
        prefixed = tuple(f'{prefix}_{name}' for prefix in (band, 'phot') for name in _MODEL_COLUMNS)
        tab = read_catalog_columns(path, _RENDER_COLUMNS + (flux_col,) + prefixed)
        if flux_col not in tab.colnames:
            out['status'] = 'band_not_measured'
            return out
        if 'name' not in tab.colnames:
            out['status'] = 'no_models'
            return out
        # A photometry run with unfrozen priors writes the positions and shapes it
        # measured under a band prefix (single-band run) or 'phot' (multi-band run),
        # and those are what this band's fluxes belong to. Take them where present.
        for name in _MODEL_COLUMNS:
            for prefix in (band, 'phot'):
                if f'{prefix}_{name}' in tab.colnames:
                    measured = np.asarray(tab[f'{prefix}_{name}'], dtype=float)
                    base = (np.asarray(tab[name], dtype=float) if name in tab.colnames
                            else np.full(len(tab), np.nan))
                    tab[name] = np.where(np.isfinite(measured), measured, base)
                    break

        models, fit_status = models_from_catalog(tab, bands=[band], with_variance=False)
        models = OrderedDict((sid, m) for sid, m in models.items() if fit_status[sid] == FIT_OK)
        selected, out['counts'] = select_reconstruction_models(models, band, reconstruct=reconstruct)
        if not selected:
            out['status'] = 'no_models'
            return out

        ids = np.fromiter(selected.keys(), dtype=int, count=len(selected))
        srcs = list(selected.values())
        ra = np.array([src.pos.ra for src in srcs])         # deg
        dec = np.array([src.pos.dec for src in srcs])       # deg

        # wcs_world2pix, not all_world2pix: read_wcs hands Tractor the TAN core only.
        # A position too far off this mosaic's projection comes back as NaN, which would
        # otherwise make an integer window bound (or a NaN patch) out of a single bad row.
        x, y = self.wcs.wcs_world2pix(ra, dec, 0)
        on_sky = np.isfinite(x) & np.isfinite(y)
        out['counts']['off_projection'] = int(np.sum(~on_sky))
        if not np.any(on_sky):
            out['status'] = 'outside'
            return out
        if not np.all(on_sky):
            ids, ra, dec, x, y = ids[on_sky], ra[on_sky], dec[on_sky], x[on_sky], y[on_sky]
            srcs = [src for src, keep in zip(srcs, on_sky) if keep]

        buckets = self._psf_buckets(brick_id, tab, ids, ra, dec)

        # Tractor's patch half-size (galaxy.py, _getUnitFluxPatchSize), plus a pixel
        # each for its rounding and the Lanczos shift of the PSF.
        pixscale = float(np.min(self.pixel_scale.to_value(u.arcsec)))      # arcsec/pix
        radius = np.array([0. if isinstance(src, PointSource) else src.getRadius()
                           for src in srcs])                                # arcsec
        psf_radius = np.zeros(len(srcs))                                    # pix
        for psfmodel, idx in buckets:
            # getRadius is in the stamp's own pixels; sampling converts to image pixels
            psf_radius[idx] = psfmodel.getRadius() * float(getattr(psfmodel, 'sampling', 1.) or 1.)
        extent = np.ceil(np.maximum(1., radius / pixscale) + psf_radius) + 2.     # pix
        capped = extent > max_extent_px
        extent = np.minimum(extent, max_extent_px)

        ny, nx = self.wcs.array_shape
        x0 = max(int(np.floor(np.min(x - extent))), 0)
        x1 = min(int(np.ceil(np.max(x + extent))) + 1, nx)
        y0 = max(int(np.floor(np.min(y - extent))), 0)
        y1 = min(int(np.ceil(np.max(y + extent))) + 1, ny)
        if (x1 <= x0) or (y1 <= y0):
            out['status'] = 'outside'
            return out

        # The window's WCS is this mosaic's shifted by an integer origin, exactly as
        # Cutout2D builds a brick's, so each source lands on the pixels it was fit to.
        wcs = self.wcs.deepcopy()
        wcs.wcs.crpix -= (x0, y0)
        wcs.array_shape = (y1 - y0, x1 - x0)
        if wcs.sip is not None:
            wcs.sip = Sip(wcs.sip.a, wcs.sip.b, wcs.sip.ap, wcs.sip.bp, wcs.sip.crpix - (x0, y0))
        pairs = [(psfmodel, [srcs[i] for i in idx]) for psfmodel, idx in buckets]
        out['window'] = render_model_image(wcs.array_shape, read_wcs(wcs), band, pairs)
        out['origin'] = (y0, x0)
        out['n_rendered'] = len(srcs)
        out['n_capped'] = int(np.sum(capped))
        out['ra'], out['dec'] = ra, dec
        return out

    def _psf_buckets(self, brick_id, catalog, ids, ra, dec):
        """Group a brick's sources by the PSF each was fit with.

        Args:
            brick_id: Brick the sources belong to.
            catalog: The brick's catalog, with ``id`` and ``group_id``.
            ids: Source ids to assign, aligned with ``ra``/``dec``.
            ra, dec: Source positions, deg.

        Returns:
            list: ``(psfmodel, indices)`` pairs, ``indices`` into ``ids``.
        """
        band = self.band
        psfcoords = self.data['psfcoords']
        varies = self.psf_varies(band)
        if np.any(psfcoords == 'none') and not varies:
            return [(self.get_psfmodel(band), np.arange(len(ids)))]

        group_of = dict(zip(np.asarray(catalog['id']).astype(int).tolist(),
                            np.asarray(catalog['group_id']).astype(int).tolist())) \
            if 'group_id' in catalog.colnames else {}
        inverse, centres = group_centres([group_of.get(int(sid), 0) for sid in ids], ra, dec)
        if varies:
            # a varying whole-field PsfEx model: each group's own PSF, as it was fit
            members = np.split(np.argsort(inverse, kind='stable'),
                               np.cumsum(np.bincount(inverse, minlength=len(centres)))[:-1])
            return [(self.get_psfmodel(band, centres[k]), idx) for k, idx in enumerate(members)]

        # The brick held the grid points inside its buffered footprint, or else the
        # single nearest one (Brick.add_band); each group then took the nearest of those.
        position, __, buffsize = load_brick_position(brick_id)
        try:
            footprint = Cutout2D(np.broadcast_to(np.float32(0), self.wcs.array_shape), position,
                                 buffsize, wcs=self.wcs, mode='partial', fill_value=0, copy=False)
            within = np.atleast_1d(psfcoords.contained_by(footprint.wcs))
        except ValueError:                                  # no overlap with this mosaic
            within = np.zeros(len(psfcoords), dtype=bool)
        if np.any(within):
            candidates = psfcoords[within]
        else:
            candidates = psfcoords[[int(np.argmin(psfcoords.separation(position)))]]

        if len(candidates) > 1:
            nearest = np.atleast_1d(centres.match_to_catalog_sky(candidates)[0])
        else:
            nearest = np.zeros(len(centres), dtype=int)
        per_source = nearest[inverse]
        return [(self.get_psfmodel(band, candidates[int(k)]), np.flatnonzero(per_source == k))
                for k in np.unique(per_source)]

    def _fit_background_mode(self):
        """How the fit removed this band's background: ``None``, ``'mosaic'`` or ``'brick'``.

        Mirrors ``BaseImage._staged_data``, where a band subtracts only if
        ``subtract_background`` is set and ``backtype`` is ``'flat'`` or ``'variable'``.
        """
        if not self.properties.get('subtract_background', False):
            return None
        if self.properties.get('backtype') not in ('flat', 'variable'):
            return None
        region = self.properties.get('backregion', 'brick')
        if region not in ('mosaic', 'brick'):
            self.logger.warning(f'{self.band}: backregion {region!r} is not understood; '
                                f'the residual will include the background.')
            return None
        return region

    def _brick_tiles(self, brick_ids):
        """Each brick's tile of this mosaic's pixel grid: ``brick_id -> (slice_y, slice_x)``.

        Tile edges are the detection-image brick grid (see ``load_brick_position``)
        mapped through the WCS, and the outermost tiles run to the image border.
        Neighbouring tiles share edges, so every pixel belongs to exactly one tile.
        Like ``read_wcs``, this assumes the band and detection grids are aligned.

        Raises:
            RuntimeError: If the mapped edges are not increasing (rotated or flipped grids).
        """
        ext = conf.DETECTION.get('extension', None)
        det_wcs = WCS(fits.getheader(conf.DETECTION['science'], ext=ext))
        dny, dnx = det_wcs.array_shape
        nbx, nby = conf.N_BRICKS
        ny, nx = self.wcs.array_shape
        # pixel i spans i - 0.5 .. i + 0.5, so brick k starts at the edge k * width - 0.5
        xs = np.arange(nbx + 1) * (dnx // nbx) - 0.5
        ys = np.arange(nby + 1) * (dny // nby) - 0.5
        xedge = self.wcs.world_to_pixel(det_wcs.pixel_to_world(xs, np.full(len(xs), dny / 2.)))[0]
        yedge = self.wcs.world_to_pixel(det_wcs.pixel_to_world(np.full(len(ys), dnx / 2.), ys))[1]
        xedge = np.clip(np.round(np.asarray(xedge) + 0.5).astype(int), 0, nx)
        yedge = np.clip(np.round(np.asarray(yedge) + 0.5).astype(int), 0, ny)
        xedge[0], xedge[-1], yedge[0], yedge[-1] = 0, nx, 0, ny
        if np.any(np.diff(xedge) < 0) or np.any(np.diff(yedge) < 0):
            raise RuntimeError(f'{self.band}: the brick grid does not map monotonically onto this '
                               f'mosaic (rotated or flipped relative to the detection image?), '
                               f'so brick-level backgrounds cannot be tiled.')
        tiles = OrderedDict()
        for brick_id in brick_ids:
            row, col = (brick_id - 1) // nbx, (brick_id - 1) % nbx
            tiles[brick_id] = (slice(yedge[row], yedge[row + 1]), slice(xedge[col], xedge[col + 1]))
        return tiles

    def _subtract_brick_backgrounds(self, science, residual, tiles):
        """Subtract, over each brick's tile, the background that brick's fit removed.

        The background is re-estimated exactly as ``Brick.add_band`` did it. The
        estimate uses the buffered cutout with non-finite pixels set to 0
        (``_condition_band_data``) and the photometry mesh. Pixels of a tile
        outside its brick's buffered cutout have no such background; they are
        counted and left as they are.

        Args:
            science: The full science array (a memory map is fine).
            residual: Writable residual array, modified in place.
            tiles: ``brick_id -> (slice_y, slice_x)`` from ``_brick_tiles``.
        """
        flat = self.properties['backtype'] == 'flat'
        n_uncovered = 0
        for brick_id, (sy, sx) in tqdm(tiles.items(), desc=f'{self.band} brick backgrounds'):
            area = (sy.stop - sy.start) * (sx.stop - sx.start)
            if area <= 0:
                continue
            position, __, buffsize = load_brick_position(brick_id)
            try:
                cutout = Cutout2D(science, position, buffsize, wcs=self.wcs, mode='partial',
                                  fill_value=np.nan, copy=True)
            except ValueError:                              # no overlap with this mosaic
                n_uncovered += area
                continue
            data = cutout.data
            data[~np.isfinite(data)] = 0
            back = sep_background(data, self.band)
            # original -> cutout offset. The overlap slices carry the partial-mode fill
            # offset, which Cutout2D.to_cutout_position does not.
            dy = cutout.slices_cutout[0].start - cutout.slices_original[0].start
            dx = cutout.slices_cutout[1].start - cutout.slices_original[1].start
            cy0, cy1 = max(sy.start + dy, 0), min(sy.stop + dy, data.shape[0])
            cx0, cx1 = max(sx.start + dx, 0), min(sx.stop + dx, data.shape[1])
            if (cy1 <= cy0) or (cx1 <= cx0):
                n_uncovered += area
                continue
            n_uncovered += area - (cy1 - cy0) * (cx1 - cx0)
            value = back.globalback if flat else back.back()[cy0:cy1, cx0:cx1]
            residual[cy0 - dy:cy1 - dy, cx0 - dx:cx1 - dx] -= value
        if n_uncovered:
            self.logger.warning(f'{self.band}: {n_uncovered} residual pixels lie outside their '
                                f"brick's buffered cutout and have no background subtracted.")

    def _check_duplicates(self, ra, dec, owner, tolerance):
        """Warn about models rendered twice, once from each of two bricks.

        A source whose centroid falls on a brick seam could survive both bricks'
        buffer cleaning. It would then be fit and drawn twice, leaving a negative
        ghost in the residual.

        Args:
            ra, dec: Rendered model positions, deg.
            owner: Brick id of each model.
            tolerance: Separation below which two models from different bricks
                count as one source, arcsec.
        """
        if len(ra) < 2:
            return
        coords = SkyCoord(ra, dec, unit='deg')
        idx, sep2d, __ = coords.match_to_catalog_sky(coords, nthneighbor=2)
        duplicated = (sep2d.to_value(u.arcsec) < tolerance) & (owner != owner[idx])
        if np.any(duplicated):
            self.logger.warning(f'{self.band}: {int(duplicated.sum())} models sit within '
                                f'{tolerance:.3g} arcsec of a model from another brick. They are '
                                f'likely one source drawn twice (bricks: '
                                f'{sorted(set(owner[duplicated].tolist()))[:10]}).')
