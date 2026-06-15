API Reference
=============

The Farmer exposes a small high-level API through the ``farmer`` package (see :doc:`farmer`), plus five internal modules that contain the core logic.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Contents
   * - :mod:`farmer` (:doc:`farmer`)
     - Top-level functions: ``validate``, ``build_bricks``, ``detect_sources``, ``generate_models``, ``photometer``, ``quick_group``, ``load_brick``, ``update_bricks``, ``get_mosaic``, ``brick_has_band``.
   * - :mod:`farmer.mosaic` (:doc:`mosaic`)
     - :class:`~farmer.mosaic.Mosaic` — full-field image wrapper. Validates paths, loads pixel data, spawns bricks.
   * - :mod:`farmer.brick` (:doc:`brick`)
     - :class:`~farmer.brick.Brick` — single tile of the survey mosaic. Runs detection, groups sources, processes groups, absorbs results.
   * - :mod:`farmer.group` (:doc:`group`)
     - :class:`~farmer.group.Group` — a set of one to five nearby sources fitted simultaneously.
   * - :mod:`farmer.image` (:doc:`baseimage`)
     - :class:`~farmer.image.BaseImage` — abstract base class shared by Mosaic, Brick, and Group. Contains all detection, optimization, and I/O logic.
   * - :mod:`farmer.utils` (:doc:`utils`)
     - Utility functions: WCS conversion, source grouping, HDF5 I/O, PSF validation, parameter extraction, region file writing.

Class Hierarchy
---------------

.. code-block:: text

   BaseImage (farmer.image)
   ├── Mosaic  (farmer.mosaic)
   ├── Brick   (farmer.brick)
   └── Group   (farmer.group)

``Mosaic``, ``Brick``, and ``Group`` all inherit from ``BaseImage`` and override only the methods that differ in their context. The vast majority of detection, optimization, and I/O logic lives in ``BaseImage``.

.. toctree::
   :hidden:

   farmer
   mosaic
   brick
   group
   baseimage
   utils
