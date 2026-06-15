``farmer.utils`` — Utilities
==============================

.. module:: farmer.utils

Core utility functions for The Farmer. Covers WCS conversion, source
grouping, HDF5 serialization, PSF validation, model parameter extraction,
multi-resolution pixel mapping, DS9 region file generation, and logging
initialisation.

Custom Tractor Models
---------------------

.. autoclass:: farmer.utils.SimpleGalaxy
   :members:
   :show-inheritance:

Functions
---------

.. autofunction:: farmer.utils.start_logger
.. autofunction:: farmer.utils.read_wcs
.. autofunction:: farmer.utils.load_brick_position
.. autofunction:: farmer.utils.clean_catalog
.. autofunction:: farmer.utils.dilate_and_group
.. autofunction:: farmer.utils.get_detection_kernel
.. autofunction:: farmer.utils.get_fwhm
.. autofunction:: farmer.utils.get_resolution
.. autofunction:: farmer.utils.validate_psfmodel
.. autofunction:: farmer.utils.prepare_psf
.. autofunction:: farmer.utils.get_params
.. autofunction:: farmer.utils.set_priors
.. autofunction:: farmer.utils.map_ids_to_coarse_pixels
.. autofunction:: farmer.utils.map_discontinuous
.. autofunction:: farmer.utils.parallel_process
.. autofunction:: farmer.utils.recursively_save_dict_contents_to_group
.. autofunction:: farmer.utils.recursively_load_dict_contents_from_group
.. autofunction:: farmer.utils.run_group
.. autofunction:: farmer.utils.build_regions
.. autofunction:: farmer.utils.cumulative
