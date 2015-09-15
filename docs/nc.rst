.. Geophysical Data Analysis Tool documentation master file, created by
   sphinx-quickstart on Thu Sep 10 21:34:06 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**geodat.nc** - Variable and Dimension utilities
===================================================================

.. automodule:: geodat.nc


**geodat.nc.Variable** - Variable container
----------

.. autoclass:: geodat.nc.Variable
   :members:

**geodat.nc.Dimension** - Dimension container
-----------------------------------------------

.. autoclass:: geodat.nc.Dimension
   :members:


Frequently used functions under **geodat.nc**
-----------------------------------------------

The following functions are shortcuts for applying frequently used functions from the other modules (such as :mod:`~geodat.monthly` and :mod:`~geodat.math`) on :mod:`~geodat.nc.Variable` instances.


Manipulation along time axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: geodat.nc
.. autosummary::
   :toctree: nc
   :template: autosummary/function.rst
   
   climatology
   anomaly
   running_climatology
   clim2long
   TimeSlices
   time_input_to_datetime
   time_array_to_dim
   create_monthly
   create_climatology_dimension
   create_monthly_dimension
   create_monthly_dimension2


Map regridding
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: nc
   :template: autosummary/function.rst

   pyferret_regrid
   regrid
   conform_regrid



Mathematical functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: nc
   :template: autosummary/function.rst
   
   nc_cal
   div
   gradient
   integrate


Statistical analysis and signal processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: nc
   :template: autosummary/function.rst

   ave
   wgt_ave
   wgt_sum
   gaus_filter
   spatial_corr
   regress


Indexing and Slicing
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: nc
   :template: autosummary/function.rst

   concatenate
   ensemble
   conform_region

File I/O
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: nc
   :template: autosummary/function.rst
   
   getvar
   savefile


Visualisation
^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: nc
   :template: autosummary/function.rst
   
   contour
   contourf
   quiver
   plot_vs_axis




Examples
----------

A physical variable of dimension (time,latitude,longitude) can be initialised::

  >>> ## Defining dimension sizes
  >>> ntime = 12; nlat=50; nlon=60

  >>> # Time, with units and calendar
  >>> time_dim = nc.Dimension(data=numpy.arange(ntime),
  >>>                         units="months since 0001-01-01",
  >>>                         dimname="time",
  >>>                         attributes={"calendar":"julian"})

  >>> # Latitudes, with units
  >>> lat_dim = nc.Dimension(data=numpy.linspace(-90.,90.,nlat),
  >>>                        units="degreeN",
  >>>                        dimname="lat")

  >>> # Longitudes, with units
  >>> lon_dim = nc.Dimension(data=numpy.linspace(0.,360.,nlon),
  >>>                        units="degreeE",
  >>>                        dimname="lon")

  >>> ## Defining the variable using the dimensions
  >>> var = nc.Variable(data=numpy.arange(float(ntime*nlat*nlon)).reshape(ntime,nlat,nlon),
  >>>                   dims=[time_dim,lat_dim,lon_dim],
  >>>                   varname="temp")

  >>> print var
  <nc.Variable temp(time,lat,lon), shape: (12, 50, 60)>

