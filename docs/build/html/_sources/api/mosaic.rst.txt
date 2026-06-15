``farmer.mosaic`` — Mosaic
==========================

.. module:: farmer.mosaic

A :class:`Mosaic` represents a single band's full-field survey image. It
handles path validation, WCS reading, pixel loading, background estimation,
and brick creation. You typically do not instantiate it directly — the
top-level :mod:`farmer` functions do this internally.

.. autoclass:: farmer.mosaic.Mosaic
   :members:
   :special-members: __init__
   :show-inheritance:
