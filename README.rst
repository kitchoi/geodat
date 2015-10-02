Geophysical Data Analysis Tool (GeoDAT)'s documentation
===================================================================

.. image:: https://travis-ci.org/kitchoi/geodat.svg?branch=master
    :target: https://travis-ci.org/kitchoi/geodat

.. image:: https://coveralls.io/repos/kitchoi/geodat/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/kitchoi/geodat?branch=master

GeoDAT is a python library that provides utilities for handling, analysing and visualising geophysical climate data.  Here are just a few of the functions of GeoDAT:

* Extract data from and write to NetCDF files
* Associate variables with dimensions
* Maintain the consistency between the variable data and the dimension axes
* Handle time and calendar
* Handle convergence of meridians on a latitude-longitude grid
* Frequently used functions such as time average, climatology etc.


Installation
=============

After downloading the source, simply do

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


Quick References
===================================================================

Extracting variable from a NetCDF file
---------------------------------------

You can use :py:func:`~geodat.nc.getvar` for extracting one variable and its associated dimensions from a NetCDF file.

.. autofunction:: geodat.nc.getvar

  >>> import geodat

  >>> # Extract sea surface temperature from a file sst.nc
  >>> # "sst" is the variable name
  >>> sst = geodat.nc.getvar("sst.nc", "sst") 
  >>> print sst
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 180, 360)>

Print basic info about a Variable
----------------------------------
  >>> sst.info()
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 180, 360)>
  Attributes:
      sst:_FillValue = 1e+20
      sst:standard_name = sst
      sst:description = HadISST 1.1 monthly average sea surface temperature Climatology
      sst:long_name = Monthly 1 degree resolution SST
      sst:units = degK
      sst:history = ; From file: sst.nc
  Dim: time = 15.5:349.7425 Unit: days since 0001-01-01 00:00:00
  Dim: lat = -89.5:89.5 Unit: degrees_north
  Dim: lon = 0.5:359.5 Unit: degrees_east


Computing time average
-----------------------
  >>> print sst
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 180, 360)>
  >>> sst_tave = sst.time_ave()
  >>> print sst_tave
  <geodat.nc.Variable sst(time,lat,lon), shape: (1, 180, 360)>
