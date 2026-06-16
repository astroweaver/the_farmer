Installation
============

Requirements
------------

The Farmer requires **Python 3.9 or later**. It depends on several astronomical and scientific Python libraries, some of which must be installed from source.

**Core dependencies:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Package
     - Version
     - Purpose
   * - ``astropy``
     - ≥ 5.0
     - FITS I/O, WCS, sky coordinates, units
   * - ``sep``
     - 1.2.1
     - Source Extractor Python bindings (detection)
   * - ``h5py``
     - ≥ 3.9.0
     - HDF5 brick serialization
   * - ``pathos``
     - 0.3.0
     - Multiprocessing pool for parallel group processing
   * - ``reproject``
     - 0.11.0
     - Image reprojection between pixel grids
   * - ``regions``
     - 0.7
     - DS9 region file output
   * - ``tqdm``
     - 4.65.0
     - Progress bars
   * - ``matplotlib``
     - ≥ 3.7
     - Diagnostic plots
   * - ``tractor``
     - git HEAD
     - Profile fitting engine (install from source)
   * - ``astrometry.net``
     - git HEAD
     - WCS utilities required by Tractor (install from source)

Installing from Source
----------------------

1. **Clone the repository**

   .. code-block:: bash

      git clone https://github.com/astroweaver/farmer.git
      cd farmer

2. **Create a virtual environment** (recommended)

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # Linux/macOS
      # .venv\Scripts\activate   # Windows

3. **Install Tractor and astrometry.net from source**

   These two packages are not on PyPI and must be installed directly from GitHub.

   .. code-block:: bash

      pip install git+https://github.com/dstndstn/astrometry.net
      pip install git+https://github.com/dstndstn/tractor.git

   .. note::
      Building ``astrometry.net`` requires a C compiler and ``numpy`` headers.
      On macOS, install Xcode Command Line Tools (``xcode-select --install``) first.
      On Linux, ``gcc`` and ``python3-dev`` (or equivalent) are required.

4. **Install The Farmer and remaining dependencies**

   .. code-block:: bash

      pip install -e .

   The ``-e`` flag installs in editable mode, so changes to the source are reflected immediately.

   Alternatively, install dependencies individually and then add the package:

   .. code-block:: bash

      pip install "astropy>=5.0" "h5py>=3.9.0" "pathos==0.3.0" "regions==0.7" \
                  "reproject==0.11.0" "sep==1.2.1" "tqdm==4.65.0" "matplotlib>=3.7"
      pip install -e .

Using Pipenv
------------

A ``Pipfile`` is included for users who prefer ``pipenv``:

.. code-block:: bash

   pip install pipenv
   pipenv install
   pipenv shell

Verifying the Installation
---------------------------

Start a Python session and check that the import succeeds:

.. code-block:: python

   >>> import farmer

You should see the startup banner followed by the prompt::

   ====================================================================
   T H E
    ________    _       _______     ____    ____  ________  _______
   ...
   You should start by running farmer.validate()!

If you see a ``RuntimeError`` mentioning ``config``, see :ref:`faq-config`.

Data Directory Setup
---------------------

The Farmer expects a specific directory layout for input and output files. The paths are configured in ``config/config.py``. Before running, create the required directories:

.. code-block:: bash

   mkdir -p data/external
   mkdir -p data/interim/bricks
   mkdir -p data/interim/psfmodels
   mkdir -p data/interim/logs
   mkdir -p data/output/figures
   mkdir -p data/output/catalogs
   mkdir -p data/output/ancillary

Then point ``PATH_DATA`` in ``config/config.py`` to your ``data/`` directory (absolute path recommended).

Building Documentation
-----------------------

If you want to build these docs locally:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme sphinx-copybutton
   cd docs
   make html

The output will be in ``docs/build/html/``.
