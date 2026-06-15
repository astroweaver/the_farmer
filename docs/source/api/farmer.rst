``farmer`` — Top-Level API
===========================

The ``farmer`` package exposes a concise set of functions that cover the full
photometry pipeline. Import the package to access them:

.. code-block:: python

   import farmer

On import, The Farmer locates ``config/config.py``, loads configuration, prints
the startup banner, and initialises the logging system. Call
:func:`farmer.validate` before any other function to confirm all configured
paths and PSF models are accessible.

.. automodule:: farmer
   :members:
   :special-members:
   :exclude-members: __builtins__, __doc__, __file__, __loader__, __name__,
                     __package__, __path__, __spec__, __version__
