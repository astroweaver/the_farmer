The Farmer Documentation
=========================

**The Farmer** is a Python package for precision multi-wavelength photometry in deep galaxy surveys. Using parametric surface brightness profile fitting via `The Tractor <https://github.com/dstndstn/tractor>`_ engine, it measures accurate fluxes and morphologies even in heavily blended fields — provided a high-resolution detection image to define where sources are.

The code has been used to build several major survey catalogs:

- **COSMOS2020** (`Weaver et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022ApJS..258...11W/abstract>`_)
- **SHELA** (`Leung et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023arXiv230100908L/abstract>`_)
- **H20** (Zalesky, in preparation)

When using The Farmer in published work, please cite `Weaver et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023arXiv231007757W/abstract>`_.

.. note::
   If you plan to use The Farmer for your own survey, we strongly encourage you to
   get in touch at john.weaver.astro@gmail.com before diving in. We are happy to
   walk you through the setup for your specific data.

----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   configuration
   pipeline

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/farmer
   api/mosaic
   api/brick
   api/group
   api/baseimage
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Help & Reference

   faq

----

Overview
--------

The Farmer divides a survey mosaic into a regular grid of rectangular **bricks**. Within each brick, sources detected in a high-resolution image are collected into small **groups** of one to five neighbors. Each group is modeled simultaneously using The Tractor, which fits parametric profiles (point sources, exponential disks, de Vaucouleurs bulges, composite models) by minimizing a Poisson likelihood. Fitted morphologies are then held fixed and the optimizer measures fluxes across every photometric band.

Key capabilities:

- **Deblended photometry** — simultaneous profile fitting prevents flux from leaking between blended neighbors
- **Multi-resolution support** — detection image can have a different pixel scale from photometric bands
- **Variable PSFs** — accepts position-dependent PSF models via a coordinate table
- **Parallel processing** — groups are farmed out across CPU cores via ``pathos``
- **HDF5 persistence** — brick state (images, catalogs, models) serialized to disk at every stage
- **Flexible model priors** — Gaussian position/shape priors, frozen parameters

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
