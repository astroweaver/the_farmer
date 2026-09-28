"""Brick cores must tile the detection grid: every pixel owned by exactly one brick.

Loads farmer/tiling.py by path, so neither a config nor tractor is needed.
Run with ``python -m pytest tests/test_brick_tiling.py``.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import astropy.units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

_spec = importlib.util.spec_from_file_location(
    'farmer_tiling', Path(__file__).resolve().parents[1] / 'farmer' / 'tiling.py')
tiling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiling)

BUFFER = 120  # pix, BRICK_BUFFER = 12" at 0.1"/pix as deployed


def tan_wcs(nx, ny, pixscl, crpix=None, crval=(150., -40.)):
    """TAN WCS on an nx x ny grid; pixscl in arcsec/pix, crpix FITS 1-based (default: centre)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix) if crpix is not None else [nx / 2 + 0.5, ny / 2 + 0.5]
    w.wcs.cdelt = [-pixscl / 3600, pixscl / 3600]
    w.array_shape = (ny, nx)
    return w


def brick_cutout(det_wcs, n_bricks, brick_id, target_wcs=None):
    """A brick's buffered cutout of ``target_wcs``'s grid, placed as Brick.add_band places it.

    The WCS then goes through a header round trip, as a brick reloaded from HDF5 does.
    """
    target_wcs = det_wcs if target_wcs is None else target_wcs
    dny, dnx = det_wcs.array_shape
    bw, bh = dnx // n_bricks[0], dny // n_bricks[1]
    row, col = tiling.brick_row_col(brick_id, n_bricks)
    position = det_wcs.pixel_to_world(col * bw + bw / 2, row * bh + bh / 2)
    scale = (proj_scale(det_wcs) / proj_scale(target_wcs))
    size = (int(np.ceil((bh + 2 * BUFFER) * scale)), int(np.ceil((bw + 2 * BUFFER) * scale)))  # pix
    cut = Cutout2D(np.broadcast_to(np.int8(0), target_wcs.array_shape), position, size,
                   wcs=target_wcs, mode='partial', fill_value=0, copy=False)
    return cut, WCS(cut.wcs.to_header())


def proj_scale(wcs):
    return np.abs(wcs.wcs.cdelt[0])  # deg / pix


def owned_ranges(det_wcs, n_bricks, target_wcs=None):
    """Each brick's owned (y0, y1, x0, x1) in the target mosaic's pixel frame."""
    out = {}
    for brick_id in range(1, n_bricks[0] * n_bricks[1] + 1):
        cut, wcs = brick_cutout(det_wcs, n_bricks, brick_id, target_wcs)
        xedge, yedge = tiling.edges_in(wcs, det_wcs, n_bricks)
        row, col = tiling.brick_row_col(brick_id, n_bricks)
        ox = cut.slices_original[1].start - cut.slices_cutout[1].start  # cutout -> mosaic
        oy = cut.slices_original[0].start - cut.slices_cutout[0].start
        ny, nx = cut.shape
        assert 0 <= xedge[col] and xedge[col + 1] <= nx, 'core runs outside the buffered cutout'
        assert 0 <= yedge[row] and yedge[row + 1] <= ny, 'core runs outside the buffered cutout'
        out[brick_id] = (yedge[row] + oy, yedge[row + 1] + oy, xedge[col] + ox, xedge[col + 1] + ox)
    return out


def assert_tiles(ranges, n_bricks, shape):
    """Cores partition [0, ny) x [0, nx): x-ranges depend on column only, y-ranges on row only,
    and each axis's ranges abut with no gap or overlap -- which makes the 2D tiling exact."""
    nbx, nby = n_bricks
    ny, nx = shape
    for axis, n_along, n_pix, key in ((1, nbx, nx, lambda r, c: c), (0, nby, ny, lambda r, c: r)):
        per_index = {}
        for brick_id, (y0, y1, x0, x1) in ranges.items():
            row, col = (brick_id - 1) // nbx, (brick_id - 1) % nbx
            rng = (x0, x1) if axis == 1 else (y0, y1)
            per_index.setdefault(key(row, col), set()).add(rng)
        for k, rngs in per_index.items():
            assert len(rngs) == 1, f'bricks sharing index {k} disagree on axis {axis}: {rngs}'
        bounds = [per_index[k].pop() for k in range(n_along)]
        assert bounds[0][0] == 0, f'axis {axis}: first core starts at {bounds[0][0]}, not 0'
        assert bounds[-1][1] == n_pix, f'axis {axis}: last core ends at {bounds[-1][1]}, not {n_pix}'
        seams = [bounds[k + 1][0] - bounds[k][1] for k in range(n_along - 1)]
        assert all(s == 0 for s in seams), f'axis {axis}: seams (+gap/-overlap px) {seams}'


@pytest.mark.parametrize('npix, nb', [(36000, 10), (36007, 10), (216007, 50), (100, 1), (99, 7), (7, 7)])
def test_grid_edges_partition(npix, nb):
    edges = tiling.grid_edges(npix, nb)
    assert len(edges) == nb + 1 and edges[0] == 0 and edges[-1] == npix
    widths = np.diff(edges)
    assert np.all(widths[:-1] == npix // nb)
    assert widths[-1] == npix // nb + npix % nb  # last brick takes the floor remainder


# ~1 deg (CDFS/COSMOS-like) and ~6 deg (EDF-like) fields. The 6 deg field uses 1"/pix so
# its cutouts stay small; the TAN distortion that broke the old angular cores is a
# fraction of the brick width, so it is still several px per seam here. The last case
# puts the tangent point in a corner, the worst case for distance across the field.
GEOMETRIES = [
    dict(nx=36007, ny=36003, pixscl=0.1, n_bricks=(10, 10)),
    dict(nx=21611, ny=21605, pixscl=1.0, n_bricks=(10, 9)),
    dict(nx=21611, ny=21605, pixscl=1.0, n_bricks=(10, 9), crpix=(1., 1.)),
]


@pytest.mark.parametrize('geom', GEOMETRIES)
def test_cores_tile_detection_grid(geom):
    det = tan_wcs(geom['nx'], geom['ny'], geom['pixscl'], geom.get('crpix'))
    ranges = owned_ranges(det, geom['n_bricks'])
    assert_tiles(ranges, geom['n_bricks'], det.array_shape)


def test_cores_tile_other_band_grid():
    """A band on a 2x coarser grid (the NIRCam SW/LW situation): the mapped cores tile it too."""
    det = tan_wcs(21611, 21605, 1.0)
    band = tan_wcs(10806, 10803, 2.0, crpix=(5403.25, 5402.25))
    ranges = owned_ranges(det, (10, 9), target_wcs=band)
    # the outermost cores stop at the detection image's border mapped into the band; only
    # the interior seams are shared, so pin the outer edges to the band border before checking
    ny, nx = band.array_shape
    nbx, nby = 10, 9
    for brick_id, (y0, y1, x0, x1) in list(ranges.items()):
        row, col = (brick_id - 1) // nbx, (brick_id - 1) % nbx
        ranges[brick_id] = (0 if row == 0 else y0, ny if row == nby - 1 else y1,
                            0 if col == 0 else x0, nx if col == nbx - 1 else x1)
    assert_tiles(ranges, (nbx, nby), band.array_shape)


def test_angular_cores_do_not_tile():
    """Guard that the 6 deg geometry actually exercises the old bug: cores derived from the
    angular size (farmer/utils.py load_brick_position + Cutout2D) leave gaps there. If this
    starts failing, the geometry tests above are no longer sensitive to the regression."""
    det = tan_wcs(21611, 21605, 1.0)
    nbx, nby = 10, 9
    dny, dnx = det.array_shape
    bw, bh = dnx // nbx, dny // nby
    row = nby // 2
    ends = []
    for col in range(nbx):
        xc, yc = col * bw + bw / 2, row * bh + bh / 2
        cx = np.array([xc - bw / 2, xc + bw / 2, xc + bw / 2])
        cy = np.array([yc - bh / 2, yc - bh / 2, yc + bh / 2])
        sc = det.pixel_to_world(cx, cy)
        size = (sc[1].separation(sc[2]).to(u.deg), sc[0].separation(sc[1]).to(u.deg))
        core = Cutout2D(np.broadcast_to(np.int8(0), det.array_shape), det.pixel_to_world(xc, yc), size,
                        wcs=det, mode='partial', fill_value=0, copy=False)
        x0 = core.slices_original[1].start - core.slices_cutout[1].start
        ends.append((x0, x0 + core.shape[1]))
    seams = [ends[k + 1][0] - ends[k][1] for k in range(nbx - 1)]
    assert max(seams) >= 3, f'angular cores tile unexpectedly well: seams {seams}'
