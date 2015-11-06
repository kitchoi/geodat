Tutorials
===========

.. contents::


Import the library
---------------------

  >>> import geodat


Extracting variable from a NetCDF file
-----------------------------------------

You can use :py:func:`~geodat.nc.getvar` for extracting one variable and its associated dimensions from a NetCDF file.

  >>> # Extract sea surface temperature from a file sst.nc
  >>> # "sst" is the variable name
  >>> sst = geodat.nc.getvar("sst.nc", "sst") 
  >>> print sst
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 180, 360)>


Getting a regional slide of your data
-----------------------------------------

Suppose you already have a Variable `sst`, you can get a regional slice of it by simply calling the variable as a function:

  >>> # This returns a new Variable
  >>> sst_regional = sst(lat=(-20., 20.), lon=(100., 220.))
  >>> sst_regional
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 40, 120)>
  >>> sst # unchanged
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 180, 360)>

The specified range is inclusive.  The example above will slice the region where 20S<=latitude<=20N and 100E<=longitude<=40W.

This is the same as calling :py:func:`~geodat.nc.Variable.getRegion`.

  >>> sst_regional = sst.getRegion(lat=(-20., 20.),
                                   lon=(100., 220.))

Other names referring to time, depth, latitude and longitude are also accepted:

==============    ==================================
Name              Other names (and their upper case)
==============    ==================================
Time              time, t
Longitude         longitude, lon, x
Latitude          latitude, lat, y
Depth             depth, dep, level, lev, z
==============    ==================================

You can also set the region when you load the data.

  >>> # Extract regional sea surface temperature from a file sst.nc
  >>> # Latitude: 20S-20N, Longitude: 100E-220E
  >>> sst = geodat.nc.getvar("sst.nc", "sst",
                             lat=(-20., 20.), lon=(100., 220.)) 
  >>> sst
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 40, 120)>

This is in fact equivalent to calling :py:func:`~geodat.nc.Variable.getRegion` after :py:func:`~geodat.nc.getvar` and assigning it to `sst`.



Print basic info about a Variable
------------------------------------
  >>> sst.info()
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 180, 360)>
  Attributes:
      sst:_FillValue = 1e+20
      sst:standard_name = sst
      sst:description = HadISST 1.1 monthly average SST Climatology
      sst:long_name = Monthly 1 degree resolution SST
      sst:units = degK
      sst:history = ; From file: sst.nc
  Dim: time = 15.5:349.7425 Unit: days since 0001-01-01 00:00:00
  Dim: lat = -89.5:89.5 Unit: degrees_north
  Dim: lon = 0.5:359.5 Unit: degrees_east


Computing time average
---------------------------
  >>> print sst
  <geodat.nc.Variable sst(time,lat,lon), shape: (120, 180, 360)>
  >>> sst_tave = sst.time_ave()
  >>> print sst_tave
  <geodat.nc.Variable sst(time,lat,lon), shape: (1, 180, 360)>

