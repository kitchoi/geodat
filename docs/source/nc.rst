.. Geophysical Data Analysis Tool documentation master file, created by
   sphinx-quickstart on Thu Sep 10 21:34:06 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**geodat.nc** - Variable and Dimension utilities
===================================================================

.. automodule:: geodat.nc

.. autoclass:: geodat.nc.Variable

.. autoclass:: geodat.nc.Dimension


See also

.. toctree::
   :maxdepth: 1
   
   nc_Variable
   nc_Dimension
   frequent_func_nc


Examples
----------

A physical variable of dimension (time,latitude,longitude) can be initialised::

  >>> ## Defining dimension sizes
  >>> ntime = 12; nlat=50; nlon=60

  >>> # Time, with units and calendar
  >>> time_dim = geodat.nc.Dimension(data=numpy.arange(ntime),
  >>>                                units="months since 0001-01-01",
  >>>                                dimname="time",
  >>>                                attributes={"calendar":"julian"})

  >>> # Latitudes, with units
  >>> lat_dim = geodat.nc.Dimension(data=numpy.linspace(-90.,90.,nlat),
  >>>                               units="degreeN", dimname="lat")

  >>> # Longitudes, with units
  >>> lon_dim = geodat.nc.Dimension(data=numpy.linspace(0.,360.,nlon),
  >>>                               units="degreeE", dimname="lon")

  >>> ## Defining the variable using the dimensions
  >>> var = geodat.nc.Variable(data=numpy.arange(float(ntime*nlat*nlon)).reshape(ntime,nlat,nlon),
  >>>                          dims=[time_dim,lat_dim,lon_dim],
  >>>                          varname="temp")

  >>> print var
  <geodat.nc.Variable temp(time,lat,lon), shape: (12, 50, 60)>

