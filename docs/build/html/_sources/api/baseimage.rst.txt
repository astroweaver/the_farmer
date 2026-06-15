``farmer.image`` — BaseImage
=============================

.. module:: farmer.image

:class:`BaseImage` is the abstract base class shared by
:class:`~farmer.mosaic.Mosaic`, :class:`~farmer.brick.Brick`, and
:class:`~farmer.group.Group`. It contains all detection, optimization,
plotting, and I/O logic. Subclasses override only the small number of
methods that differ by context (``get_bands``, ``get_figprefix``).

Do not instantiate :class:`BaseImage` directly.

.. autoclass:: farmer.image.BaseImage
   :members:
   :private-members: _extract
   :special-members: __init__
   :show-inheritance:
