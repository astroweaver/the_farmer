"""Brick-grid geometry on the detection image.

Which brick owns a source is decided here, in integer pixels on the detection grid,
so that neighbouring bricks tile it exactly. It must never be derived from a brick's
angular size: turning a pixel width into a corner-to-corner separation and back at
the reference-pixel scale is not the identity on a TAN projection away from the
tangent point, which made cores overlap (duplicate sources) or leave gaps (lost
sources) at brick boundaries.

Deliberately free of ``config`` and tractor imports, so tests/test_brick_tiling.py
can load it on its own.
"""
import numpy as np


def grid_edges(npix, nbricks):
    """Integer edges of the brick grid along one axis of the detection image.

    Brick ``k`` owns pixels ``[edges[k], edges[k+1])``. Every brick is
    ``npix // nbricks`` wide except the last, which also takes the remainder the
    floor leaves over, so the cores cover ``[0, npix)`` with no gaps or overlaps.

    Returns:
        ndarray: ``nbricks + 1`` increasing integers from 0 to ``npix``.
    """
    edges = np.arange(nbricks + 1) * (npix // nbricks)
    edges[-1] = npix
    return edges


def brick_row_col(brick_id, n_bricks):
    """Zero-indexed ``(row, col)`` of a 1-indexed brick on an ``(nx, ny)`` brick grid."""
    nbx, nby = n_bricks
    if not 1 <= brick_id <= nbx * nby:
        raise RuntimeError(f'Cannot request brick #{brick_id} on grid {nbx} X {nby}!')
    return (brick_id - 1) // nbx, (brick_id - 1) % nbx


def _same_grid_offset(wcs, det_wcs, tol=1e-6):
    """``(dx, dy)`` such that pixel ``p`` of ``wcs`` is pixel ``p + (dx, dy)`` of
    ``det_wcs``, when ``wcs`` is an integer shift of the detection grid (as every
    detection-band brick cutout is); otherwise None."""
    a, b = wcs.wcs, det_wcs.wcs
    if list(a.ctype) != list(b.ctype) or not np.allclose(a.crval, b.crval, rtol=0, atol=1e-12):
        return None
    if not np.allclose(wcs.pixel_scale_matrix, det_wcs.pixel_scale_matrix, rtol=1e-9, atol=0):
        return None
    # SIP is deliberately not compared: a header round trip can drop it from a cutout
    # (to_header() omits SIP by default) without changing which pixels it holds
    offset = b.crpix - a.crpix
    if np.any(np.abs(offset - np.round(offset)) > tol):
        return None
    return np.round(offset).astype(int)


def edges_in(wcs, det_wcs, n_bricks):
    """The brick grid's edges in the pixel frame of ``wcs``.

    Brick ``(row, col)`` covers ``[yedge[row], yedge[row+1]) x [xedge[col], xedge[col+1])``
    of ``wcs``'s array; edges may fall outside that array. When ``wcs`` is an
    integer shift of the detection grid the edges are shifted exactly. Otherwise
    each edge is mapped point by point through the sky (x edges along the grid's
    middle row, y edges along its middle column, which assumes the two frames are
    not rotated relative to each other) and rounded to the pixel boundary. Every
    image cut from one mosaic shares that mosaic's grid, so its bricks still tile it.

    Args:
        wcs: Target ``astropy.wcs.WCS`` (a brick cutout, or a whole band mosaic).
        det_wcs: WCS of the detection image, with ``array_shape`` set.
        n_bricks: ``(nx, ny)`` brick grid, i.e. ``conf.N_BRICKS``.

    Returns:
        tuple: ``(xedge, yedge)`` integer arrays of length ``nx + 1`` and ``ny + 1``.
    """
    dny, dnx = det_wcs.array_shape
    xedge = grid_edges(dnx, n_bricks[0])
    yedge = grid_edges(dny, n_bricks[1])
    offset = _same_grid_offset(wcs, det_wcs)
    if offset is not None:
        return xedge - offset[0], yedge - offset[1]
    # pixel i spans i - 0.5 .. i + 0.5, so edge e sits at e - 0.5
    xs, ys = xedge - 0.5, yedge - 0.5
    xw = wcs.world_to_pixel(det_wcs.pixel_to_world(xs, np.full(len(xs), dny / 2.)))[0]
    yw = wcs.world_to_pixel(det_wcs.pixel_to_world(np.full(len(ys), dnx / 2.), ys))[1]
    return _first_pixel_after(xw), _first_pixel_after(yw)


def _first_pixel_after(boundary, tol=1e-6):
    """Index of the first pixel whose centre is at or beyond ``boundary`` (pix).

    A boundary through a pixel centre is common, e.g. an odd-width brick seen on a
    2x coarser grid. There ``np.round(b + 0.5)`` is a tie decided by float noise,
    which differs between bricks (their cutout WCSs differ by an integer shift), so
    neighbours put the same seam a pixel apart. ``ceil(b - tol)`` decides every such
    tie the same way: the pixel goes to the upper brick.
    """
    return np.ceil(np.asarray(boundary) - tol).astype(int)
