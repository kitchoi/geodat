Download
============

You may download the source from its github repository:

.. code-block:: sh

   git clone https://github.com/kitchoi/geodat.git

You may also download the source directly as a `zip file <https://github.com/kitchoi/geodat/archive/master.zip>`_.


Installation
==================

If you do not have the Python package setuptools already installed, run this:

.. code-block:: sh

   wget "https://bootstrap.pypa.io/ez_setup.py" -O- | python

Then simply do

.. code-block:: sh

   python setup.py install

Python setuptools will attempt to install required packages.  But in case of missing system level libraries, error messages will be shown.  You may need to manually install some libraries on your machine.

Required system-level libraries for build
-------------------------------------------
     * liblapack-dev
     * libblas-dev
     * gfortran
     * libgeos-dev
     * libhdf5-serial-dev
     * libnetcdf-dev
     * python-sphere

Required Python packages (requirements.txt)
------------------------------------------------
     * matplotlib
     * numpy
     * six
     * scipy
     * python-dateutil
     * basemap
     * pyspharm
     * netCDF4

Language support
--------------------
     * Python 2.7

