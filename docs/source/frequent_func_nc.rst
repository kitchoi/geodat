Frequently used functions under **geodat.nc**
-----------------------------------------------

The following functions are shortcuts for applying frequently used functions from the other modules (such as :mod:`~geodat.monthly` and :mod:`~geodat.math`) on :mod:`~geodat.nc.Variable` instances.


Manipulation along time axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: geodat.nc
.. autosummary::
   :toctree: generated
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
   :toctree: generated
   :template: autosummary/function.rst

   pyferret_regrid
   regrid
   conform_regrid



Mathematical functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst
   
   nc_cal
   div
   gradient
   integrate


Statistical analysis and signal processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   wgt_ave
   wgt_sum
   gaus_filter
   spatial_corr
   regress


Indexing and Slicing
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   concatenate
   ensemble
   conform_region

File I/O
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst
   
   getvar
   savefile


Visualisation
^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst
   
   contour
   contourf
   quiver
   plot_vs_axis


Working with PyFerret
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   var2fer
   fer2var

