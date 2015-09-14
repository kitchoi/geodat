.. Geophysical Data Analysis Tool documentation master file, created by
   sphinx-quickstart on Thu Sep 10 21:34:06 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Quick References
===================================================================

Extracting variable from a NetCDF file
--------------------

You can use :py:func:`~geodat.nc.getvar` for extracting one variable and its associated dimensions from a NetCDF file.

.. autofunction:: geodat.nc.getvar

  >>> import geodat

  >>> # Extract sea surface temperature from a file sst.nc
  >>> # "sst" is the variable name
  >>> sst = geodat.nc.getvar("sst.nc", "sst") 
  >>> print sst
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 180, 360)>

Print basic info about a Variable
--------------------
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
--------------------
  >>> print sst
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 180, 360)>
  >>> sst_tave = sst.time_ave()
  >>> print sst_tave
  <geodat.nc.Variable sst(time,lat,lon), shape: (1, 180, 360)>

