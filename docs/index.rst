.. ynot documentation master file, created by
   sphinx-quickstart on Tue Oct 27 11:14:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the `ynot` documentation!
====================================

The `ynot` project aims to improve the astronomical echelle spectroscopy data reduction process through three key ideas:

1. Propagating the native pixel uncertainty, without interpolating noisy pixels
2. Employing an interpretable model in which all parameters are varied at the same time
3. Encouraging extensibility and modularity for a wide range of scientific applications

We invite you to engage with us at our `GitHub page
<http://www.github.com/gully/ynot>`_, with pull requests and discussions welcomed.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

    Installation <about/install>
    Application Programming Interface <api>

Currently this project requires NVIDIA GPUs.  To check if you have a GPU available:

.. code-block:: python

   import torch

   torch.cuda.is_available()



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
