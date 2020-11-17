.. _installation:

**********************************
Installing the development version
**********************************




.. note::

    Conda and pip are not available for this experimental research code.


Currently only the bleeding-edge developer version is available for beta testing.
We currently use a custom-patched version of
`ccdproc <https://ccdproc.readthedocs.io/en/latest/>`_ for reading in datasets.
You will need to install this patch-ed `ccdproc` before installing `ynot`:

Navigate the your source directory in the `ynot` repository::

    $ conda env create -f environment_torch1p6.yml
    $ conda activate environment_torch1p6
    $ git clone https://github.com/gully/ccdproc.git
    $ cd ccdproc
    $ git checkout unfixable_hdr_key_workaround
    $ python setup.py develop

Once that is complete, you can navigate to this project::

    $ cd your-ynot-path
    $ python setup.py develop


And voila!  It should work.  You can run the tests in `tests` to double-check
and benchmark GPU/CPU performance::

    $ py.test -vs



Requirements
============

The project may work with a variety of Python 3 minor versions, though none have been tested.  The project has been developed with:

- Python: 3.8
- PyTorch: 1.6 or later.
- ccdproc (custom-patched on Nov 17, 2020)
- CUDA 10.2
