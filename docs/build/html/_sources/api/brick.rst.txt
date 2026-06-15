``farmer.brick`` — Brick
=========================

.. module:: farmer.brick

A :class:`Brick` is a rectangular cutout of the survey mosaic at a fixed sky
position. It aggregates data from multiple bands, runs source detection,
manages source groups, and absorbs fitting results. Each brick is persisted to
an HDF5 file under ``PATH_BRICKS``.

.. autoclass:: farmer.brick.Brick
   :members:
   :private-members: _condition_band_data, _condition_all_bands
   :special-members: __init__
   :show-inheritance:
