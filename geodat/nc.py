"""This module contains the main classes for handling geophysical (climate)
variables and dimensions.  It also reads and writes NetCDF files.

The :mod:`~geodat.nc.Dimension` and :mod:`~geodat.nc.Variable` classes in this
module act as containers of :py:mod:`~numpy` arrays which can be easily
accessed.
"""
import os
import sys
import copy
import warnings
import logging
from functools import wraps
import datetime

import numpy
import scipy.io.netcdf as netcdf
import pylab
from netCDF4 import num2date as _num2date
from netCDF4 import date2num as _date2num
from dateutil.relativedelta import relativedelta

import geodat.keepdims as keepdims
import geodat.arrays
import geodat.stat
import geodat.math
import geodat.monthly
import geodat.plot.mapplot


def getvar(filename, varname, *args, **kwargs):
    ''' Short hand for retrieving variable from a netcdf file

    Args:
        filename (str): Name of input file
        varname (str): Name of the variable

    Returns:
        geodat.nc.Variable

    Optional arguments and keyword arguments are parsed to geodat.nc.Variable

    Example:
    var = getvar("sst.nc","sst")

    '''
    return Variable(netcdf.netcdf_file(filename), varname, *args, **kwargs)


def dataset(filenames, result=None, *args, **kwargs):
    ''' Extract all variables in one file or more files

    Args:
        filenames (str or a list of str): Input files

    Returns:
        dict: str and geodat.nc.Variable pairs

    Optional arguments accepted by geodat.nc.Variable can be used here
    '''
    if result is None:
        result = {}
    if type(filenames) is not list:
        filenames = [filenames]
    for filename in filenames:
        file_handle = netcdf.netcdf_file(filename)
        for varname in file_handle.variables.keys():
            if varname in file_handle.dimensions:
                # Do not add dimensions to the dataset
                continue
            if result.has_key(varname):
                print varname+''' alread loaded.
                Overwrite[o] or rename[r] or skip?'''
                append_code = sys.stdin.readline()
                if append_code[0] == 'o':
                    result[varname] = Variable(file_handle, varname,
                                               *args, **kwargs)
                elif append_code[0] == 'r':
                    print "Enter new name:"
                    newname = sys.stdin.readline()[:-1]
                    result[newname] = Variable(file_handle, varname,
                                               *args, **kwargs)
                    result[newname].varname = newname
                else:
                    print " ".join("I am skipping the variable:",
                                   varname, "in", filename)
                    continue
            else:
                result[varname] = Variable(file_handle, varname,
                                           *args, **kwargs)
    return result


def _genereal_axis(axis):
    ''' Standardize keyword for axes time/lat/lon

        'T':  'time','t','TIME','T'
        'X':  'x','X','lon','LON','longitude','LONGITUDE'
        'Y':  'y','Y','lat','LAT','latitude','LATITUDE'
    Anything not recognized will be returned in upper case

    Args:
        axis (str)

    Returns:
        str

    Example: _genereal_axis('time') -> 'T'
    Example: _genereal_axis('dummy') -> 'DUMMY'
    '''
    invaxnames = {'tim':'T', 'lon':'X', 'lat':'Y',
                  'lev':'Z', 'dep':'Z'}
    if len(axis) > 1:
        return invaxnames[axis[:3].lower()]
    else:
        return axis.upper()


def _general_region(region):
    ''' Standardize keyword for regional slicing.

    Use by Variable.getRegion()
    'T':  'time','t','TIME','T'
    'X':  'x','X','lon','LON','longitude','LONGITUDE'
    'Y':  'y','Y','lat','LAT','latitude','LATITUDE'

    Args:
        region (dict)

    Returns:
        dict

    Example:
        _general_region({'TIME':(10.,1000.),'LAT':(-5.,5.)})
        --> {'T': (10.,1000.),'Y':(-5.,5.)}
    '''
    results = {}
    for iax, value in region.items():
        results[_genereal_axis(iax)] = value
    return results


def _assign_caxis_(dimunit):
    '''
    Assign cartesian_axis (T/Z/Y/X) to the axis with identifiable axis units.

    Axes without identifiable units will be set to None

    Args:
        unit (str)

    Returns:
        str
    '''
    assert type(dimunit) is str
    dimunit = dimunit.split()[0]
    conventions = {'T': ['second', 'seconds', 'sec', 'minute', 'minutes', 'min',
                         'hour', 'hours', 'h', 'day', 'days', 'd',
                         'month', 'months', 'mon', 'year', 'years', 'y'],
                   'Z': ['bar', 'millibar', 'decibar', 'atm', 'atmosphere',
                         'pascal', 'Pa', 'hpa',
                         'meters', 'meter', 'm', 'kilometer', 'km', 'density'],
                   'Y': ['degrees_north', 'degree_north', 'degree_n',
                         'degrees_n', 'degreen', 'degreesn'],
                   'X': ['degrees_east', 'degree_east', 'degree_e',
                         'degrees_e', 'degreee', 'degreese']}
    invaxunits = {unit.lower():ax
                  for ax, units in conventions.items()
                  for unit in units}
    return invaxunits.get(dimunit.lower(), None)


class Dimension(object):
    """
    A container for handling physical dimensions such as time, latitude.
    It can be indexed/sliced the same way as indexing a numpy array

    Attributes:
        data (numpy 1-d array): Array for the physical axis
        dimname (str): Name of the dimension, e.g. "time"
        units (str): Unit of the dimension, e.g. "days since 1990-01-01"
        attributes (dict): Attributes for the dimension

    Arguments:
        data (numpy 1-d array): Array for the physical axis
        dimname (str): Name of the dimension, e.g. "time"
        units (str): Unit of the dimension, e.g. "days since 1990-01-01"
        attributes (dict): Attributes for the dimension
        parent (Dimension): from which dimname,units,attributes are copied. If
          parent is provided and dimname/units is specified, the latter is used.
          If parent is provided and attributes is specified, the attributes
          from the parent is copied and then updated by the attributes argument.
    """
    def __init__(self, data, dimname=None, units=None,
                 attributes=None, parent=None):
        self.data = data
        self.units = units
        self.attributes = {}
        self.dimname = 'UNNAMED_DIM'
        if parent is not None:
            self.units = parent.units
            self.attributes.update(parent.attributes)
            self.dimname = parent.dimname
        if units is not None:
            self.units = units
        if attributes is not None:
            self.attributes.update(attributes)
        if dimname is not None:
            self.dimname = dimname
        if isinstance(data, netcdf.netcdf_variable):
            self.data = data.data
            self.units = getattr(data, 'units', None)
            self.attributes.update(data.__dict__['_attributes'])
        if type(data) is int:
            self.data = numpy.array(data)
        if attributes is not None:
            self.attributes.update(attributes)
        self.attributes.update(dict(units=str(self.units)))


    def __getitem__(self, sliceobj):
        return Dimension(data=self.data[sliceobj],
                         dimname=self.dimname,
                         units=self.units,
                         attributes=self.attributes)


    def info(self, detailed=False):
        """ Print brief info about the dimension

        if detailed is True, attributes and length of axis are also printed
        """
        info_str = 'Dim: '+ self.dimname
        if numpy.isscalar(self.data):
            info_str += ' = '+str(self.data)
        else:
            info_str += ' = '+ str(self.data[0]) + ':' + str(self.data[-1])
        info_str += ' Unit: ' + str(self.units)
        print info_str
        if detailed:
            print 'Length= ' + str(len(self.data))
            print 'Attributes:'
            for attname, val in self.attributes.items():
                print '   ' + self.dimname + ':' + attname + ' = ' + str(val)


    def getCAxis(self):
        """ Get cartesian axis (T/Z/Y/X) for a dimension instance

        if the dimension has a cartesian_axis attribute, the value of
        the attribute is returned.  Otherwise, the unit is used as a clue

        Example:
            dim.setattr("cartesian_axis","X")
            dim.getCAxis() --> "X"

        Example:
            dim.units = "months"
            dim.getCAxis() --> "T"

        Example:
            dim.units = "degreeN"
            dim.getCAxis() --> "Y"

        """
        atts = self.attributes
        cax = atts.get('axis', atts.get('cartesian_axis', None))
        if cax is None:
            if self.units is not None:
                cax = _assign_caxis_(self.units)
        return cax


    def setattr(self, att, value):
        ''' Java style setter for attributes
        '''
        self.attributes[att] = value


    def getattr(self, att, default=None):
        ''' Java style getter for attributes
        '''
        return self.attributes.get(att, default)


    def is_monotonic(self):
        ''' Return True if the axis is monotonic, False otherwise
        '''
        strict_monotonic_func = lambda data: (numpy.diff(data) > 0.).all() or \
                                (numpy.diff(data) < 0.).all()
        strict_monotonic = strict_monotonic_func(self.data)
        if not strict_monotonic:
            # Make sure it is not because of periodic boundary condition
            # of longitude
            if self.getCAxis() == 'X':
                x_diff = numpy.diff(self.data)
                if sum(x_diff > 0.) > sum(x_diff < 0.):
                    # More often increasing
                    # Roll backward
                    return strict_monotonic_func(
                        numpy.roll(self.data, (numpy.argmin(x_diff)+1)*-1))
                else:
                    # More often decreasing
                    # Roll forward
                    return strict_monotonic_func(
                        numpy.roll(self.data, numpy.argmin(x_diff)+1))
        return strict_monotonic


    def is_climo(self):
        ''' Return True if the axis is a climatological time axis '''
        if self.getCAxis() != 'T':
            return False
        months = self.time2array()[:, 1]
        is_climo = len(self.data) == 12 and \
                   months.std() > 3. and \
                   months.std() < 4.  # depends on the calendar,
        # sometimes there are duplicated months due to leap years
        return is_climo


    def time2array(self):
        ''' Given a dimension object, if it is a time axis,
        return ndarray of size (N,6) where N is the number of
        time point, and the six indices represent:
        YEAR,MONTH,DAY,HOUR,MINUTE,SECOND
        '''
        if self.units.split()[0] != 'months' and \
           self.units.split()[0] != 'month':
            alltimes = _num2date(self.data, self.units,
                                 self.attributes.get(
                                     'calendar', 'standard').lower())
            try:
                _ = iter(alltimes)
            except TypeError:
                alltimes = [alltimes, ]
            time_list = numpy.array([[t.year,
                                      t.month,
                                      t.day,
                                      t.hour,
                                      t.minute,
                                      t.second] for t in alltimes])
            return time_list
        else:
            # netcdftime does not handle month as the unit
            if 'since' not in self.units:
                raise Exception("the dimension, assumed to be a time"+\
                                " axis, should have a unit such as "+\
                                "\"days since 01-JAN-2000\"")
            if not self.is_monotonic():
                raise Exception("The axis is not monotonic!")
            if (numpy.diff(self.data) < 0.).all():
                raise Exception('''Time going backward...really?
                Haven't thought of how to take care of this''')

            t0 = self.time0()
            unit = self.units.split()[0]
            # this is redundant computation as the if-clause made sure the
            # unit is "month"
            units = unit if unit.endswith("s") else unit+"s"
            alltimes = [t0 + relativedelta(**{units:t}) for t in self.data]
            time_list = [[t.year,
                          t.month,
                          t.day,
                          t.hour,
                          t.minute,
                          t.second] for t in alltimes]
            return numpy.array(time_list)

    def time0(self):
        ''' Return a datetime.datetime object referring to the t0 of a time axis
        '''
        if self.getCAxis() != 'T':
            raise Exception("This axis is not a time axis")
        startpoint = self.units.split(' since ')[-1]
        # choose the appropriate time format since some data
        # does not specify the hour/minute/seconds
        if len(startpoint.split()) > 1:
            if len(startpoint.split()[-1].split(':')) == 3:
                date_format = '%Y-%m-%d %H:%M:%S'
            else:
                startpoint = startpoint.split()[0]
                date_format = '%Y-%m-%d'
        else:
            date_format = '%Y-%m-%d'
        return datetime.datetime.strptime(startpoint, date_format)


    def getDate(self, toggle=None, no_continuous_duplicate_month=False):
        ''' Return the time axis date in an array format of
        "Year,Month,Day,Hour,Minute,Second"
        Toggle one in Y/m/d/H/M/S to select a particular time format
        e.g. getDate('m') will return an array of the month of the time axis
        '''
        if self.getCAxis() != 'T':
            raise Exception("Dimension.getDate: not a time axis")

        if no_continuous_duplicate_month:
            if toggle != 'm':
                logging.warning('''no_continuous_duplicate_month only
                applies when toggle=m''')

        if toggle is not None:
            if len(toggle) != 1:
                raise Exception('''Toggle one axis only,
                choose one in Y/m/d/H/M/S, case matters''')

        timeformat = 'YmdHMS'
        timearray = self.time2array()

        if toggle:
            result = timearray[:, timeformat.index(toggle)]
            if toggle == 'm':
                months_diff = numpy.diff(result)
                if any(months_diff == 0):
                    if no_continuous_duplicate_month:
                        # if there are duplicated months,
                        # correct it using the days
                        for itime in range(len(timearray)):
                            day = timearray[itime, 2]
                            if day > 25:
                                logging.warning(
                                    'Month:{} Day:{} is refered as month {}'.\
                                    format(result[itime], day,
                                           numpy.mod(result[itime], 12)+1))
                                result[itime] = numpy.mod(result[itime], 12)+1
                            if day < 5:
                                logging.warning(
                                    'Month:{} Day:{} is refered as month {}'.\
                                    format(result[itime], day,
                                           numpy.mod(result[itime]-2, 12)+1))
                                result[itime] = numpy.mod(result[itime]-2, 12)+1
                    else:
                        logging.warning(
                            '''There are continuous duplicated months.
                            But not correcting for them.''')
        else:
            result = timearray

        return result


class Variable(object):
    """
    A container for handling physical variable together with its dimensions
    so that while the variable is manipulated (e.g. averaged along one axis),
    the information of the dimensions change accordingly.

    It can be indexed/sliced the same way as indexing a numpy array

    Attributes:
        data (numpy.ndarray or numpy.ma.core.MaskedArray): Data array of the
          variable
        varname (str): Name of the variable
        dims (list of Dimension instances): Dimensions of the variable
          consistent with the shape of the data array
        units (str): Unit of the variable
        attributes (dict): Attributes of the variable

    Arguments:
        reader  (netcdf.netcdf_file, optional): if given, the variable is read
          from the NetCDF file
        varname (str) : variable name
        data    (numpy.ndarray or numpy.ma.core.MaskedArray)
        dims    (a list of Dimension) : dimensions
        attributes (dict): attributes of the variables
        history (str): to be stored/appended to attributes['history']
        parent (Variable): from which varname, dims and attributes are copied;
          Copied `varname` and `dims` can be overwritten by assigning values in
          the arguments.  If `attributes` is copied from `parent`, the
          dictionary assigned to the argument `attributes` is used to update the
          copied `attributes`.  `parent` is left unchanged.
        lat (tuple): length=2, specify a meridional domain, e.g. lat=(-5,5)
        lon (tuple): length=2, specify a zonal domain, e.g. lon=(-170,-120.),
               modulo=360. is forcefully applied
        time (tuple): extract a temporal domain
        ensureMasked (bool): whether the array is masked using _FillValue
               upon initialization. default: False

    Examples:
        >>> var = Variable(netcdf.netcdf_file,"temperature")

        >>> var = Variable(netcdf.netcdf_file,"temperature",
                           lat=(-5.,5.),lon=(-170.,-120))

        >>> # Copy varname, dims, attributes from var
        >>> # If the dimension shape does not match data shape, raise an Error
        >>> var2 = Variable(data=numpy.array([1,2,3,4]),parent=var)

        >>> var = Variable(data=numpy.array([1,2,3,4]),
                           dims=[Dimension(data=numpy.array([0.,1.,2.,3.]),)],
                           varname='name')

    """

    def __init__(self, reader=None, varname=None, data=None, dims=None,
                 attributes=None, history=None, parent=None,
                 ensureMasked=False, **kwargs):
        # Initialize the most basic properties.
        # Anything else goes to the attribute dictionary
        self.data = data
        self.dims = dims   # a list of Dimension instances
        self.varname = varname
        self.attributes = {}
        self._ncfile = None
        if reader is not None and type(reader) is netcdf.netcdf_file:
            assert varname is not None
            try:
                varobj = reader.variables[varname]
            except KeyError:
                print 'Unknown variable name. Available variables: '+\
                    ','.join(reader.variables.keys())
                return None
            self.data = getattr(varobj, "data", None)
            self.dims = [Dimension(reader.variables[dim], dim)
                         if reader.variables.has_key(dim)
                         else Dimension(reader.dimensions[dim], dim)
                         for dim in varobj.dimensions]
            self.attributes.update(varobj.__dict__['_attributes'])
            self.addHistory('From file: '+reader.fp.name)
            self._ncfile = reader
        elif parent is not None:
            # Copy instance
            try:
                self.__copy_from_parent__(parent)
            except AttributeError:
                raise AttributeError("Type of parent is :"+str(type(parent))+\
                                     ". Expect geodat.nc.Variable")
        else:
            # no recognizable reader or parent variable is given;
            #data, varname should not be None
            if data is None:
                raise Exception('data should not be None')
            if varname is None:
                raise Exception('varname should not be None')

        # If parent is given, these will overwrite the properties
        # copied from parent
        # If parent is not given, the following initializes the instances
        if data is not None:
            self.data = data
        if dims is not None:
            self.dims = dims
        if varname is not None:
            self.varname = varname
        if attributes is not None:
            self.attributes.update(attributes)
        if self.dims is None:
            self.dims = [None,]*self.data.ndim
        if history is not None:
            self.addHistory(history)

        self.setRegion(**kwargs)
        self.masked = False

        # This is the one that takes the time while initializing variables
        if ensureMasked:
            self.ensureMasked()

        # Check to make sure the variable data shape matches the dimensions'
        self.check_shape_matches_dims()


    def check_shape_matches_dims(self):
        ''' Check if the shape of the data matches the dimensions

        Raise ValueError if the dimensions do not match
        '''
        var_data_shape = self.data.shape
        dim_shape = tuple([dim.data.size for dim in self.dims])
        if var_data_shape != dim_shape:
            raise ValueError("Dimension mismatch.\n"+
                             "data shape:{}, dimensions shape:{}".format(
                                 var_data_shape,dim_shape))


    def addHistory(self, string):
        """ Append to the history attribute in the variable.
        If history doesn't exist, create one
        """
        history = self.attributes.get('history', '')
        newhistory = history + '; '+ string
        self.setattr('history', newhistory)

    def __repr__(self):
        result = "<{}.{} ".format(__name__, type(self).__name__) + \
            self.varname +\
            '(' + ",".join(self.getDimnames()) + '), shape: ' +\
            str(self.data.shape) + '>'
        return result

    def info(self, detailed=False):
        """ Print brief info about the variable
        """
        # varname, dim, shape
        print self.__repr__()

        # Attributes:
        print "Attributes:"
        for attname, val in self.attributes.items():
            print "    " + self.varname + ":" +  attname + " = " + str(val)

        # Dimension info:
        for dim in self.dims:
            dim.info(detailed=detailed)

    def getCAxes(self):
        """ get the cartesian axes for all the dimensions.
        Return a list of cartesian axes.
        if it is undefined, replace with dummy: A,B,C,...(excludes: T/Z/X/Y)
        """
        dummies = list('ABCDEFGHIJKLMNOPQRSUVW')
        caxes = []
        for dim in self.dims:
            # Using try-catch is clearly not ideal
            # Previously the try block was an if-statement that
            # getCAxis is called only if dim is an instance of
            # geodat.nc.Dimension. However when the module is reload,
            # objects created before reloading is no longer an
            # instance of the reloaded module
            try:
                cax = dim.getCAxis()
            except AttributeError:
                cax = None
            if cax is None:
                cax = dummies.pop(0)
            caxes.append(cax)
        return caxes

    def getDimnames(self):
        """Return a list of dimension names
        """
        return [dim.dimname for dim in self.dims]

    def setattr(self, att, value):
        '''Set the value of an attribute of the variable

        Java style setter
        '''
        self.attributes[att] = value

    def getattr(self, att, default=None):
        ''' Return the value of an attribute of the variable

        Java style getter
        '''
        return self.attributes.get(att, default)

    def getAxes(self):
        ''' Return the dimensions of the variable as a list of numpy arrays
        In the order of dimension
        '''
        axes = []
        for idim in range(len(self.dims)):
            dim = self.dims[idim]
            if dim.data is None:
                dim.data = numpy.arange(1, self.data.shape[idim]+1)
            axis = dim.data
            axes.append(axis)
        return axes

    def getAxis(self, axis):
        ''' Return a numpy array of an axis of a variable
        Input:
        axis - if it is an integer, return the array for that dimension
               if it is a string, match it with the CAxes in the variable
                  and then return the array for the dimension
        '''
        if type(axis) is int:
            return self.dims[axis].data
        if type(axis) is str:
            caxes = self.getCAxes()
            axis = _genereal_axis(axis)
            if axis not in caxes:
                raise KeyError(self.varname+" has no "+axis+" axis")
            else:
                return self.dims[caxes.index(axis)].data
        else:
            raise ValueError("axis has to be either an integer or a string")

    def getDomain(self, axis=None):
        ''' Return the domain of the variable

        If the axis is a longitude axis, make all negative degree positive
        '''
        if axis is None:
            axis = self.getCAxes()
        domain = {}
        for ax_name in axis:
            coor = self.getAxis(ax_name)
            if ax_name == 'X':
                coor = coor.copy()
                coor[coor < 0.] += 360.
            domain[_genereal_axis(ax_name)] = (min(coor), max(coor))

        return domain

    def __copy_from_parent__(self, parent):
        """ Copy the dimensions, attributes and varname
        from a parent variable
        Use copy.copy instead of deepcopy
        """
        self.dims = copy.copy(parent.dims)
        self.attributes = copy.copy(parent.attributes)
        self.varname = copy.copy(parent.varname)

    def __broadcast_dim__(self, other, result):
        ''' Return a list of dimensions suitable for operations (__add__...)
        between self and other

        Arg:
            other (geodat.nc.Variable or numpy.ndarray attribute)
            result (numpy.ndarray) : the result of an operation e.g. __add__

        Returns:
            a list of geodat.nc.Dimension

        Example:
        varA <geodat.nc.Variable with shape (12,10)>
        varB <geodat.nc.Variable with shape (1,10)>
        varC = varA + varB
        varC <geodat.nc.Variable with shape (12,10)> inherits dimensions
        from varA

        But varC = varB + varA
        would require inheriting the first dimension of varA and the second
        dimension of varB

        '''
        dims = []
        for idim, (size1, size2) in enumerate(zip(self.data.shape,
                                                  result.shape)):
            if size1 == 1 and size2 > 1:
                dims.append(other.dims[idim])
            else:
                dims.append(self.dims[idim])
        return dims

    def __sub__(self, other):
        var1 = __getdata__(self)
        var2 = __getdata__(other)
        history = ""
        name1 = getattr(self, 'varname', str(self))
        name2 = getattr(other, 'varname', str(other))
        data = var1 - var2
        history = name1 + '-' + name2
        return Variable(data=data, dims=self.__broadcast_dim__(other, data),
                        parent=self, history=history)

    def __add__(self, other):
        var1 = __getdata__(self)
        var2 = __getdata__(other)
        history = ""
        name1 = getattr(self, 'varname', str(self))
        name2 = getattr(other, 'varname', str(other))
        data = var1 + var2
        history = name1 + '+' + name2
        return Variable(data=data, dims=self.__broadcast_dim__(other, data),
                        parent=self, history=history)

    def __div__(self, other):
        var1 = __getdata__(self)
        var2 = __getdata__(other)
        history = ""
        name1 = getattr(self, 'varname', str(self))
        name2 = getattr(other, 'varname', str(other))
        data = var1 / var2
        history = name1 + '/' + name2
        return Variable(data=data, dims=self.__broadcast_dim__(other, data),
                        parent=self, history=history)

    def __mul__(self, other):
        var1 = __getdata__(self)
        var2 = __getdata__(other)
        history = ""
        name1 = getattr(self, 'varname', str(self))
        name2 = getattr(other, 'varname', str(other))
        data = var1 * var2
        history = name1 + '*' + name2
        return Variable(data=data,
                        dims=self.__broadcast_dim__(other, data),
                        parent=self, history=history)


    def __getitem__(self, sliceobj):
        a = Variable(data=self.data, varname=self.varname, parent=self)
        sliceobj = numpy.index_exp[sliceobj]
        a.__slicing__(sliceobj)
        a.addHistory('__getitem__['+str(sliceobj)+']')
        return a


    def __setitem__(self, sliceobj, val):
        if type(sliceobj) is dict:
            sliceobj = self.getSlice(**sliceobj)
        self.data[sliceobj] = val

    def getRegion(self, **kwargs):
        ''' Return a new Variable object within the region specified.
        '''
        a = Variable(data=self.data, parent=self)
        a.setRegion(**kwargs)
        return a

    def getSlice(self, **kwargs):
        ''' Return a tuple of slice object corresponding a region specified.
        Example: variable.getSlice(lat=(-30.,30.))
        '''
        region = _general_region(kwargs)
        if len(region) > 0:
            return self.__createSlice__(region)
        else:
            return None

    def setRegion(self, **kwargs):
        ''' Change the region of interest for the variable
        This function slices the data.
        '''
        region = _general_region(kwargs)
        if len(region) > 0:
            self.__slicing__(self.__createSlice__(region))
            self.addHistory('setRegion('+str(region)+')')
        return self

    def setRegion_value(self, value, **kwargs): # Maybe unused
        ''' Set values for a particular region
        Example: variable.setRegion_value(0.,lat=(-90.,-30.))
        '''
        sl = self.getSlice(**kwargs)
        self[sl] = value
        return self

    def __createSlice__(self, region=None):
        ''' Generate a tuple of slice object for the given region
        specifications
        '''
        if region is None or len(region) == 0:
            return (slice(None),)*self.data.ndim

        sliceobj = ()
        # general selector method that returns a slice object
        selector = geodat.arrays.getSlice
        axes = self.getAxes()
        cartesian_axes = self.getCAxes()
        assert len(axes) == len(cartesian_axes) == self.data.ndim
        for axis in cartesian_axes:
            iax = cartesian_axes.index(axis)
            bounds = region.get(axis, None)
            if bounds is None:
                sliceobj += (slice(None),)
            elif isinstance(bounds, slice):
                sliceobj += (bounds,)
            elif isinstance(bounds, numpy.ndarray):
                sliceobj += (bounds,)
            else:
                if not isinstance(bounds, tuple):
                    bounds = (bounds,)
                if len(bounds) == 1:
                    bounds = (bounds[0], bounds[0])
                if axis == 'X':
                    modulo = 360.
                else:
                    modulo = self.dims[iax].attributes.get('modulo', None)
                sliceobj += (selector(axes[cartesian_axes.index(axis)],
                                      bounds[0], bounds[1], modulo=modulo),)
        return sliceobj


    def __slicing__(self, sliceobj):
        ''' Perform the slicing operation on both the data and axes
        '''
        self.data = self.data[sliceobj]
        sliceobj = list(sliceobj)
        newaxis_list = []
        for iax, sl in enumerate(sliceobj):
            if isinstance(sl, int):
                newaxis_list.append(numpy.newaxis)
            else:
                newaxis_list.append(slice(None))
            if not numpy.isscalar(self.dims[iax].data):
                self.dims[iax] = self.dims[iax][sl]
            # Make sure the dimension data is a numpy array
            if numpy.isscalar(self.dims[iax].data):
                self.dims[iax].data = numpy.array(
                    [self.dims[iax].data,],
                    dtype=self.dims[iax].data.dtype)
        if newaxis_list:
            self.data = self.data[newaxis_list]


    def getLatitude(self):
        ''' Return a numpy array that contains the latitude axis
        '''
        return self.getAxis('Y')

    def getLongitude(self):
        ''' Return a numpy array that contains the longitude axis
        '''
        return self.getAxis('X')

    def getTime(self):
        ''' Return a numpy array that contains the time axis
        '''
        return self.getAxis('T')

    def apply_mask(self, mask):
        ''' mask the variable's last axes with a mask
        This function changes the variable
        '''
        return apply_mask(self, mask)

    def climatology(self, *args, **kwargs):
        ''' Compute the climatology
        '''
        return climatology(self, *args, **kwargs)

    def zonal_ave(self):
        ''' Compute the zonal average
        Same as wgt_ave('X')
        '''
        return wgt_ave(self, 'X')

    def time_ave(self):
        ''' Compute the time average
        Same as wgt_ave('T')
        '''
        return wgt_ave(self, 'T')

    def lat_ave(self):
        ''' Compute meridional average
        Same as wgt_ave('Y')
        '''
        return wgt_ave(self, 'Y')

    def area_ave(self):
        ''' Compute area average
        Same as wgt_ave('XY')
        '''
        return wgt_ave(self, 'XY')

    def wgt_ave(self, axis=None):
        ''' Compute averge on one or more axes
        Input:
        axis   - either integer or a string (T/X/Y/Z/...)
        See getCAxes
        '''
        return wgt_ave(self, axis)

    def getMissingValue(self):
        ''' Return "missing_value" if defined in the attributes
        Otherwise "_FillValue" will be used as missing value
        If both are undefined, the numpy default for the variable
        data type is returned
        '''
        FillValue = self.getattr('_FillValue', None)
        missing_value = self.getattr('missing_value', None)
        default = numpy.ma.default_fill_value(self.data)
        return missing_value or FillValue or default

    def ensureMasked(self):
        ''' If the data in the variable is not a masked array and
        missing_value is present
        Read the data and convert the numpy ndarray into masked array,
        note that this will be slow if the data is large. But this will
        only be done once.

        Returns: None
        '''
        if self.masked:
            return None
        missing_value = self.getMissingValue()

        if isinstance(self.data, numpy.ma.core.MaskedArray):
            self.data = numpy.ma.array(self.data.filled(missing_value))
        self.data = numpy.ma.masked_values(self.data, missing_value)
        if not isinstance(self.data, numpy.ma.core.MaskedArray):
            raise AssertionError("Missing value: {}".format(missing_value))

        self.data.set_fill_value(missing_value)
        self.setattr('_FillValue', missing_value)
        self.masked = True
        return None


    def runave(self, N, axis=0, step=None):
        '''Running mean along an axis.
        N specifies the size of the window (assumed to be odd)
        '''
        self.ensureMasked()
        cartesian_axes = self.getCAxes()
        history = 'runave('+str(N)+','+str(axis)+',step='+str(step)+')'
        if type(axis) is str:
            axis = axis.upper()
            axis = cartesian_axes.index(axis)
            if self.dims[axis].is_monotonic():
                N = N/numpy.abs(numpy.diff(self.getAxes()[axis]).mean())
            else:
                logging.warning('''{var}'s {dim} is not monotonic.
                N is treated as integer'''.format(var=self.varname,
                                                  dim=self.dims[axis].dimname))
                if not isinstance(N, int):
                    raise Exception('''N is treated as step.
                    It has to be an integer.''')
        if N % 2 != 1:
            N = N + 1
        return Variable(data=geodat.stat.runave(self.data, N, axis, step),
                        parent=self, history=history)

    def squeeze(self):
        ''' Remove singlet dimensions
        '''
        var = Variable(data=self.data, parent=self)
        shape = self.data.shape

        var.dims = [var.dims[idim]
                    for idim in range(var.data.ndim)
                    if shape[idim] > 1]
        var.data = var.data.squeeze()
        assert var.data.ndim == len(var.dims)
        var.addHistory('squeeze()')
        return var


    def getDate(self, toggle=None, no_continuous_duplicate_month=False):
        ''' Return the time axis date in an array format as
        Year,Month,Day,Hour,Minute,Second
        Toggle one in Y/m/d/H/M/S to select a particular time format
        e.g. getDate('m') will return an array of the month of the time axis
        '''
        if 'T' not in self.getCAxes():
            raise Exception("There is no recognized time axis in Variable:"+\
                            self.varname)
        return self.dims[self.getCAxes().index('T')].getDate(
            toggle, no_continuous_duplicate_month)


def __getdata__(other):
    '''
    If the input is a geodat.nc.Variable, run the ensureMasked function
    and return the `data attribute
    Otherise, return the input

    Used by __sub__,__add__,...
    '''
    if isinstance(other, geodat.nc.Variable):
        other.ensureMasked()
        return other.data
    else:
        return other


def apply_mask(var, mask):
    ''' Mask the last axes of the data with a mask array
    example: apply_mask(v,land_mask>0.)
    The data of var is copied to a new variable that is being returned
    '''
    newvar = geodat.nc.Variable(data=var.data.copy(), parent=var)
    newvar.ensureMasked()
    newvar.data[..., mask] = numpy.ma.masked
    return newvar


def nc_cal(f):
    ''' A decorator that returns a variable object
    Accept only function that operates on a numpy array
    '''
    @wraps(f)
    def g(var, *args, **kwargs):
        history = "".join([f.__name__, args.__str__(), kwargs.__str__()])
        var.ensureMasked()
        return Variable(data=f(var.data, *args, **kwargs), parent=var,
                        history=history)
    return g


def ave(var, axis=None):
    '''Backup.  To be trashed
    '''
    var.ensureMasked()
    data = var.data
    cartesian_axes = var.getCAxes()
    dimnames = var.getDimnames()
    if axis is None:
        axis = range(len(dimnames))
    # If the input axis is a single integer, convert it into a list
    if type(axis) is int:
        axis = [axis]

    history = 'wgt_ave(axis='+','.join([str(ax) for ax in axis])+')'
    if type(axis) is str:
        axis = axis.upper()
        axis = [cartesian_axes.index(ax) for ax in axis]

    weights = numpy.ones(data.shape, dtype=data.dtype)
    has_Y = 'Y' in [cartesian_axes[iax] for iax in axis]
    if has_Y:
        sliceobj = [numpy.newaxis if cax != 'Y' else slice(None)
                    for cax in cartesian_axes]
        weights = geodat.stat.lat_weights(var.getLatitude())[sliceobj]

    data = data*weights
    data = keepdims.mean(data, axis=axis)/keepdims.mean(weights, axis=axis)
    dims = [Dimension(numpy.array([1,], dtype='i4'),
                      var.dims[iax].dimname, units=var.dims[iax].units)
            if iax in axis else var.dims[iax]
            for iax in range(var.data.ndim)]
    return Variable(data=data.astype(var.data.dtype),
                    dims=dims, parent=var, history=history)


def wgt_ave(var, axis=None):
    '''A more general routine for averaging

    The method first reads the axes (x/y/z/t) needed for averaging,
    finds the indices corresponding these axes, then uses the
    geodat.stat.wgt_ave to sort the axis and do the weighted average

    if the axis is a "Y" axis, weights are computed using the latitude
    axis in the variable.

    if no axis is given, all axes will be averaged over.

    Arg:
        var (Variable)

    Optional args:
        axis (str or a list of str or int) - axis to be averaged over
        weights (scalar or a numpy array)

        if axis is a string, e.g. "xy", the input argument weights will
        be overwritten

        E.g.
            (1) wgt_ave(Variable,'xy') will do the area average
    '''
    var.ensureMasked()
    data = var.data
    cartesian_axes = var.getCAxes()
    if axis is None:
        axis = range(len(cartesian_axes))
    # If the input axis is a single integer, convert it into a list
    if type(axis) is int:
        axis = [axis]

    history = 'wgt_ave(axis='+','.join([str(ax) for ax in axis])+')'
    if type(axis) is str:
        axis = axis.upper()
        axis = [cartesian_axes.index(ax) for ax in axis]

    # apply varied lat_weights only if 'XY' are included
    caxes = [cartesian_axes[ax] for ax in axis]
    has_XY = 'X' in caxes and 'Y' in caxes
    if has_XY:
        sliceobj = [numpy.newaxis if cax != 'Y' else slice(None)
                    for cax in cartesian_axes]
        lat_weights = geodat.stat.lat_weights(var.getLatitude())[sliceobj]
    else:
        lat_weights = 1.

    for iax in axis:
        if cartesian_axes[iax] in 'XYZ':
            assert var.dims[iax].is_monotonic()

    weights = reduce(lambda x, y: x[..., numpy.newaxis]*y,
                     [numpy.gradient(var.dims[iax].data)
                      if iax in axis and cartesian_axes[iax] in 'XYZ'
                      else numpy.ones_like(var.dims[iax].data)
                      for iax in range(var.data.ndim)])
    weights *= lat_weights
    weights = numpy.ma.masked_where(data.mask, weights)
    data = keepdims.mean(data*weights, axis=axis)/\
           keepdims.mean(weights, axis=axis)
    dims = [Dimension(numpy.array([1,], dtype='i4'),
                      var.dims[iax].dimname,
                      units=var.dims[iax].units)
            if iax in axis else var.dims[iax]
            for iax in range(var.data.ndim)]
    return Variable(data=data.astype(var.data.dtype),
                    dims=dims, parent=var, history=history)


def wgt_sum(var, axis=None):
    '''A more general routine for sum
    The method first reads the axes (x/y/z/t) needed,
    finds the indices corresponding these axes,
    if the axis is a "Y" axis, weights are computed using the latitude
    axis in the variable.
    if no axis is given, all axes will be summed over.

    Args:
        var (geodat.nc.Variable)
        axis (str/int/list of int, optional): along which the array is summed

    Examples:
        >>> # Area sum
        >>> wgt_sum(var,'xy')

        >>> # Sum along the first axis
        >>> wgt_sum(var,0)
    '''
    var.ensureMasked()
    data = var.data
    cartesian_axes = var.getCAxes()
    dimnames = var.getDimnames()
    if axis is None:
        axis = range(len(dimnames))
    # If the input axis is a single integer, convert it into a list
    if type(axis) is int:
        axis = [axis]

    history = 'wgt_sum(axis='+','.join([str(ax) for ax in axis])+')'
    if type(axis) is str:
        axis = axis.upper()
        axis = [cartesian_axes.index(ax) for ax in axis]

    weights = numpy.ones(data.shape, dtype=data.dtype)
    has_Y = 'Y' in [cartesian_axes[iax] for iax in axis]
    if has_Y:
        sliceobj = [numpy.newaxis if cax != 'Y' else slice(None)
                    for cax in cartesian_axes]
        weights = geodat.stat.lat_weights(var.getLatitude())[sliceobj]

    data = data*weights
    data = keepdims.sum(data, axis=axis)
    dims = [Dimension(numpy.array([1,], dtype='i4'),
                      var.dims[iax].dimname, units=var.dims[iax].units)
            if iax in axis else var.dims[iax]
            for iax in range(var.data.ndim)]
    return Variable(data=data.astype(var.data.dtype),
                    dims=dims, parent=var, history=history)


def time_input_to_datetime(time, calendar, units):
    ''' Return a datetime.datetime object given time as string
        Example: time_input_to_datetime("1999-01-01 00:00:00",
                 "julian","days since 0001-01-01")
    '''
    if isinstance(time, datetime.datetime):
        return time
    elif isinstance(time, str):
        try:
            return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return datetime.datetime.strptime(time, '%Y-%m-%d')
    else:
        return _num2date(time, units=units, calendar=calendar)


def time_array_to_dim(time_array, calendar, units, **kwargs):
    ''' Return a geodat.nc.Dimension object given a time array
    time_array = [ [ year, month, day, hour, minute, second ],...]
    calendar   = string ("standard","julian",...)
    units      = string (e.g. "days since 0001-01-01")
    '''
    times = numpy.array([_date2num(
        time_input_to_datetime(
            "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}".format(*time),
            calendar=calendar, units=units),
        calendar=calendar, units=units)
                         for time in time_array])
    return geodat.nc.Dimension(data=times, units=units,
                               attributes={'calendar':calendar}, **kwargs)


def create_monthly(calendar, units, time0, time_end=None):
    ''' Return a generator that return a scalar time value with
    the specified calendar and unit.

    time0 is the starting time.
    if time_end is not specified, the generator will not stop iterating.
    unit should take the form UNIT since DATETIME
    example: days since 0001-01-01 00:00:00

    Work in progress.  Need to be rewritten using relativedelta
    '''

    time0 = time_input_to_datetime(time0, calendar=calendar, units=units)
    if time_end is not None:
        time_end = time_input_to_datetime(time_end,
                                          calendar=calendar, units=units)

    calendar = calendar.lower()
    def days_to_next_month(time):
        '''Hard coded number of days between calendar months
        TODO: Should use relativedelta'''
        days = [29.5, 29.5, 30.5, 30.5,
                30.5, 30.5, 31.0, 30.5,
                30.5, 30.5, 30.5, 31.]
        isleap = lambda year: (year % 4) == 0
        if isleap(time.year) and (time.month == 1 or time.month == 2) and \
           calendar != 'noleap':
            return days[time.month-1]+0.5
        else:
            return days[time.month-1]

    def continue_iter(current_time, time_end):
        ''' Determine if the current time has passed the specified time_end'''
        if time_end is None:
            return True
        else:
            return current_time < _date2num(time_end, units=units,
                                            calendar=calendar)

    current_time = _date2num(time0, units=units, calendar=calendar)
    while continue_iter(current_time, time_end):
        yield current_time
        current_time += days_to_next_month(_num2date(current_time,
                                                     units=units,
                                                     calendar=calendar))


def create_climatology_dimension(calendar, units, time0=None, **dim_args):
    ''' Create a monthly dimension for climatology time axis

    Args:
        calendar (str) : e.g. "julian"
        units (str): e.g. "days since 0001-01-01 00:00:00"
        time0 (str): default "0001-01-16 00:00:00", the first value on the time
           axis

    Returns:
        geodat.nc.Dimension

    Optional keyword arguments are passed to geodat.nc.Dimension
    '''
    if time0 is None:
        time0 = '0001-01-16 00:00:00'
    time_generator = create_monthly(calendar, units, time0)
    times = [time_generator.next() for i in range(12)]
    return Dimension(data=numpy.array(times),
                     dimname='time',
                     units=units,
                     attributes={'modulo':""}, **dim_args)


def create_monthly_dimension(calendar, units, time0, time_end, **dim_args):
    time_generator = create_monthly(calendar, units, time0, time_end)
    times = [t for t in time_generator]
    return Dimension(data=numpy.array(times), units=units, **dim_args)


def create_monthly_dimension2(ref_dim=None):
    if ref_dim is None:
        time0 = datetime.datetime(1, 1, 1, 0, 0)
        units = 'days since 0001-1-1 0'
        calendar = 'standard'
        attributes = {'modulo':" "}
    else:
        time0 = ref_dim.time0()
        units = ref_dim.units
        calendar = ref_dim.attributes.\
                   get('calendar', 'standard').lower()
        attributes = ref_dim.attributes.copy()
        attributes['modulo'] = " "
    newaxis = [_date2num(time0+datetime.timedelta(days=int(day)),
                         units=units,
                         calendar=calendar)
               for day in numpy.linspace(15, 365-15, 12)]

    return Dimension(data=numpy.array(newaxis),
                     dimname='time',
                     units=units,
                     attributes=attributes)


def climatology(var, appendname=False,
                no_continuous_duplicate_month=True,
                *args, **kwargs):
    var.ensureMasked()
    data = var.data
    assert 'T' in var.getCAxes()
    months = var.getDate('m', no_continuous_duplicate_month)
    axis = var.getCAxes().index('T')

    clim_data = geodat.monthly.climatology(data=data, months=months, axis=axis,
                                           *args, **kwargs)
    history = 'climatology'
    long_name = var.getattr('long_name', '')
    if appendname:
        long_name += " climatology"
    dims = [dim for dim in var.dims]
    # units is forced to be "days since 0001-01-01 00:00:00" instead of
    # inheriting the var's time unit
    dims[axis] = create_climatology_dimension(
        calendar=var.dims[axis].getattr('calendar', 'standard').lower(),
        units='days since 0001-01-01 00:00:00',
        parent=var.dims[axis])
    return Variable(data=clim_data, dims=dims, parent=var,
                    history=history,
                    attributes=dict(long_name=long_name))


def anomaly(var, appendname=False, clim=None,
            no_continuous_duplicate_month=True):
    var.ensureMasked()
    data = var.data
    assert 'T' in var.getCAxes()
    months = var.getDate('m', no_continuous_duplicate_month)
    axis = var.getCAxes().index('T')
    if clim is None:
        anom_data = geodat.monthly.anomaly(data=data,
                                           months=months, axis=axis)[0]
    else:
        anom_data = geodat.monthly.anomaly(data=data, months=months,
                                           axis=axis, clim=clim.data)[0]
    history = 'anomaly'
    long_name = var.getattr('long_name', '')
    if appendname:
        long_name += " anomaly"
    dims = [dim for dim in var.dims]
    return Variable(data=anom_data, dims=dims,
                    parent=var, history=history,
                    attributes=dict(long_name=long_name))


def running_climatology(var, appendname, runave_window, step, need_anom=True):
    ''' Calculate the running climatology, with anomaly

    Args:
      var (geodat.nc.Variable)
      appendname (bool): whether to append "_c" to the varname of the output
      runave_window (int): size of the running average window
      step (int): step for slicing the array
      need_anom (bool): whether anomaly is returned

    Returns:
      climatology (geodat.nc.Variable), anomaly (None or geodat.nc.Variable if
      need_anom is True)

    Example:
      If the time axis is monthly, compute a running climatology
      with a 30-year window, with appended name and anomaly returned, like this::

      >>> running_climatology(var,True,30,12,True)
    '''
    climo = var.runave(runave_window, var.getCAxes().index('T'), step)
    climo.addHistory("Moving climatology with window:{}".format(runave_window))
    if appendname:
        climo.varname += '_c'
    if need_anom:
        anom = var - climo
        anom.addHistory("Anomaly on a moving climatology")
        if appendname:
            anom.varname += '_a'
    else:
        anom = None
    return climo, anom


def clim2long(clim, target):
    # Copy the target time dimension
    time_dim = target.dims[target.getCAxes().index("T")]
    time_idim = clim.getCAxes().index("T")
    new_dim = [time_dim
               if idim == time_idim
               else dim
               for idim, dim in enumerate(clim.dims)]
    return geodat.nc.Variable(data=geodat.monthly.clim2long(
        clim.data, 0, target.getDate("m", True)),
                              dims=new_dim,
                              attributes=clim.attributes,
                              history="geodat.nc.clim2long({},{})".\
                              format(clim.varname, target.varname),
                              varname=clim.varname)


def concatenate(variables, axis=0):
    ''' Concatenate a list of variables similar to numpy.concatenate

    Take care of numpy masked array and concatenate dimensions as well

    Args:
        variables (list of Variable)
        axis (int): along which the variables are concatenated

    Returns:
        geodat.nc.Variable
    '''
    for var in variables:
        var.ensureMasked()
    data = numpy.ma.concatenate([var.data for var in variables], axis=axis)

    # Concatenate dimensions
    dim_data = numpy.concatenate([var.dims[axis].data
                                  for var in variables])
    dims = [dim for dim in variables[0].dims]
    dims[axis] = Dimension(dim_data, parent=dims[axis])

    return Variable(data=data, dims=dims, varname=variables[0].varname,
                    parent=variables[0],
                    history=variables[0].getattr('history'))


def ensemble(variables, new_axis=None, new_axis_unit=None, **kwargs):
    ''' Given a list of variables, perform numpy.concatenate()
    New axis is added as the left most axis

    Optional arguments:
        new_axis (numpy array) : for the new axis
        new_axis_unit (str): defines the unit of the new axis

    Other keyword arguments are parsed to geodat.nc.Variable
    '''
    for d in variables:
        d.ensureMasked()
    ensdata = numpy.ma.concatenate([d.data[numpy.newaxis, ...]
                                    for d in variables], axis=0)
    if new_axis is None:
        new_axis = numpy.arange(1, len(variables)+1)
    dims = [Dimension(new_axis, dimname='ensemble', units=new_axis_unit),] + \
           variables[0].dims
    return Variable(data=ensdata, parent=variables[0], dims=dims, **kwargs)


def div(u, v, varname='div', long_name='divergence', **kwargs):
    ''' Compute wind divergence by central difference

    Args:
        u (geodat.nc.Variable) - zonal wind
        v (geodat.nc.Variable) - meridional wind

    Returns:
        geodat.nc.Variable
    '''
    lon = numpy.radians(u.getLongitude())
    lat = numpy.radians(u.getLatitude())
    xaxis = u.getCAxes().index('X')
    yaxis = u.getCAxes().index('Y')
    assert xaxis == v.getCAxes().index('X')
    assert yaxis == v.getCAxes().index('Y')
    # dx,dy
    R = 6371000.
    dx = numpy.cos(lat) * numpy.diff(lon)[0] * R  # a function of latitude
    dx_slice = (numpy.newaxis,)*yaxis + (slice(None),) \
                + (numpy.newaxis,)*(u.data.ndim-yaxis-1)
    dx = dx[dx_slice]
    dy = numpy.diff(lat)[0] * R

    return Variable(data=geodat.math.div(u.data, v.data, dx, dy, xaxis, yaxis),
                    varname=varname, parent=u, history='divergence',
                    attributes=dict(long_name=long_name), **kwargs)


def gradient(var, axis, mask_boundary=True, **kwargs):
    ''' Compute the gradient of a variable taking into account the convergence
    of meridians

    Args:
        var (geodat.nc.Variable)
        axis (str or int)  - the axis along which the gradient is computed
        mask_boundary (bool, default=True) - whether boundary values are masked

    Additional keyword arguments are parsed to geodat.nc.Variable

    Returns:
        geodat.nc.Variable
    '''
    if type(axis) is str:
        axis = var.getCAxes().index(axis.upper())
    R = 6371000.
    if var.getCAxes()[axis] == 'X' and 'Y' in var.getCAxes():
        yaxis = var.getCAxes().index('Y')
        lon = numpy.radians(var.getLongitude())
        lat = numpy.radians(var.getLatitude())
        lat_slice = (numpy.newaxis,)*yaxis + (slice(None),) \
            +  (numpy.newaxis,)*(var.data.ndim-yaxis-1)
        lon_slice = (numpy.newaxis,)*axis + (slice(None),) \
            + (numpy.newaxis,)*(var.data.ndim-axis-1)
        # a function of latitude
        dx = numpy.cos(lat)[lat_slice] * numpy.gradient(lon)[lon_slice] * R
    else:
        if var.getCAxes()[axis] == 'X' or var.getCAxes()[axis] == 'Y':
            dx = numpy.radians(numpy.gradient(var.getAxes()[axis])) * R
        else:
            dx = numpy.gradient(var.getAxes()[axis])

    return Variable(data=geodat.math.gradient(var.data, dx, axis,
                                              mask_boundary=mask_boundary),
                    parent=var,
                    history='gradient: '+var.getCAxes()[axis], **kwargs)


def integrate(var, axis, varname='int', versatile=False):
    ''' Integrate variable along one or more axes
    var - geodat.nc.Variable
    axis - a list of integer that select the dimension to be integrated along
    '''
    var.ensureMasked()
    if type(axis) is str:
        axis = axis.upper()
        axis = [var.getCAxes().index(ax) for ax in axis]

    if type(axis) is not list:
        axis = [axis]

    # Compute integration
    re_data = geodat.math.integrate(data=var.data, axes=var.getAxes(), iax=axis)

    # History attribute
    history = 'Integrated along axis:'+ \
              ''.join([var.getCAxes()[iax] for iax in axis])

    # It may take some time to compute integration, notify the user
    if versatile:
        print 'Integrating along axis...\b'

    # This long name is probably not needed
    long_name = var.attributes.get('long_name', '') + \
                                ' integrated on ' + \
                                ''.join([var.getCAxes()[iax] for iax in axis])

    result = Variable(data=re_data,
                      varname=varname, parent=var, history=history,
                      attributes=dict(long_name=long_name))

    # Reduce dimension to the mean of the domain
    for ax in axis:
        result.dims[ax].data = numpy.array([var.dims[ax].data.mean()],
                                           dtype=var.dims[ax].data.dtype)

    if versatile:
        print 'Done.'

    return result


def conform_region(*args):
    ''' Return a dictionary with the common lat-lon region
    Input:
    args:  a list (length > 1) of dictionary or geodat.nc.Variable
    the dictionary resembles the input for geodat.nc.Variable.getRegion()
    Return:
    a dictionary {'X': (min_lon,max_lon), 'Y': (min_lat,max_lat)}
    '''
    if len(args) == 1:
        raise Exception("Expect more than one domain in conform_region")

    args = list(args)
    for i in range(len(args)):
        # For backward compatibility, get the domains for Variable inputs
        try:
            argdomain = args[i].getDomain()
        except AttributeError:
            argdomain = args[i]
        # Generalise the form of the dictionary
        args[i] = _general_region(argdomain)

    minlon = max([domain.get('X', (numpy.inf*-1, numpy.inf))[0]
                  for domain in args])
    maxlon = min([domain.get('X', (numpy.inf*-1, numpy.inf))[1]
                  for domain in args])
    minlat = max([domain.get('Y', (numpy.inf*-1, numpy.inf))[0]
                  for domain in args])
    maxlat = min([domain.get('Y', (numpy.inf*-1, numpy.inf))[1]
                  for domain in args])
    return dict(lat=(minlat, maxlat), lon=(minlon, maxlon))


def conform_regrid(*args, **kwargs):
    ''' Given a list of variable
    Conform and regrid to match the region and grid
    Return a list of variables
    Unnamed optional argument go to conform_region
    Named optional arguments:
    ref  - specify a reference variable to regrid to

    The rest go to geodat.nc.pyferret_regrid
    '''
    # Conform the region first
    region = conform_region(*args)
    varstoregrid = [var.getRegion(**region) for var in args]
    axes = 'X' if all(['X' in var.getCAxes() for var in varstoregrid]) else ''
    axes += 'Y' if all(['Y' in var.getCAxes() for var in varstoregrid]) else ''
    if 'ref' in kwargs:
        ref = kwargs.pop('ref').getRegion(**region)
        regridded = [pyferret_regrid(var, ref)
                     for var in varstoregrid]
    else:
        # Reference is not given
        # The variable with the finest grid would be the reference
        # area = cos(theta) dtheta dphi
        def minarea(var, axes):
            mindelta = lambda v: numpy.abs(numpy.gradient(v)).min()
            if axes == 'XY':
                phi = numpy.radians(var.getLongitude())
                theta = numpy.radians(var.getLatitude())
                area = numpy.cos(theta)[numpy.newaxis, :]*\
                    numpy.gradient(phi)[:, numpy.newaxis]*\
                    numpy.gradient(theta)[numpy.newaxis, :]
                return numpy.abs(area).min()
            elif axes == 'X':
                return mindelta(var.getLongitude())
            elif axes == 'Y':
                return mindelta(var.getLatitude())
        ires = numpy.array([minarea(var, axes) for var in args]).argmin()
        ref = varstoregrid[ires]
        regridded = [pyferret_regrid(varstoregrid[i], ref, axis=axes, **kwargs)
                     if i != ires
                     else ref
                     for i in range(len(varstoregrid))]
    return regridded


def pyferret_regrid(var, ref_var=None, axis='XY', nlon=None, nlat=None,
                    transform="@lin", *args, **kwargs):
    ''' Use pyferret to perform regridding.

    Args:
        var (geodat.nc.Variable): input data
        ref_var (geodat.nc.Variable): provide the target grid
        axis (str): which axis needs regridding
        nlon (int): if ref_var is not provided, a cartesian latitude-longitude global grid is created as the target grid. nlon is the number of longitudes
        nlat (int): number of latitude, used with nlon and when ref_axis is None
        transform (str): Mode of regridding.  "@lin" means linear interpolation
                 "@ave" means preserving area mean.  See `Ferret doc`_
        verbose (bool)

    Either ref_var or (nlon and nlat) has to be specified

    Returns:
        geodat.nc.Variable

    .. _Ferret doc: http://ferret.pmel.noaa.gov/Ferret/documentation/users-guide
    '''
    import geodat.pyferret_func as _pyferret_func
    import sphere_grid.grid_func
    if ref_var is None:
        # If ref_var is not given, use nlon and nlat instead
        if nlon is None or nlat is None:
            raise Exception('''reference variable is not given.
            nlon and nlat need to be specified''')
        if ''.join(sorted(axis.upper())) != 'XY':
            raise Exception('''ref_var not given and therefore assumed
            regridding in the XY direction.
            The axis/axes you chose:'''+str(axis))
        # Create latitude and longitude using the sphere_grid and spharm modules
        lon, lat = sphere_grid.grid_func.grid_degree(NY=nlat, NX=nlon)
        lon = Dimension(data=lon, units="degrees_E", dimname="lon")
        lat = Dimension(data=lat, units="degrees_N", dimname="lat")
        # Create new dimensions
        dims = []
        for idim in range(len(var.dims)):
            if var.getCAxes()[idim] == 'X':
                dims.insert(idim, lon)
            elif var.getCAxes()[idim] == 'Y':
                dims.insert(idim, lat)
            else:
                dims.insert(idim, var.dims[idim])
        data_shape = [dim.data.shape[0] for dim in dims]
        ref_var = Variable(data=numpy.ones(data_shape, dtype=var.data.dtype),
                           dims=dims, parent=var)

    if axis == 'XY' and transform.lower() != '@ave':
        warnings.warn('''Regridding onto XY grid and
        transformation: {} is used.'''.format(transform))

    # Only a slice of ref_var is needed
    # No need to copy the entire variable
    # (reduce chance of running out of memory)
    ref_var_slice = tuple([slice(0, 1) if cax not in axis.upper()
                           else slice(None)
                           for cax in ref_var.getCAxes()])
    return _pyferret_func.regrid(var, ref_var[ref_var_slice].squeeze(), axis,
                                 transform=transform, *args, **kwargs)


def regrid(var, nlon, nlat, verbose=False):
    ''' Use spherical harmonic for regridding
    May produce riggles.

    Take an instance of geodat.nc.Variable,
    Deduce the lat-lon grid on a complete sphere,
    Return a regridded data on a spherical grid (nlat,nlon)

    Return: a geodat.nc.Variable instance

    TODO: grid.regrid now only handle 2D or 3D data,
    extend the function to handle rank-3+ data
    by flattening the extra dimension into one dimension
    '''
    import sphere_grid.grid_func
    ilat = var.getCAxes().index('Y')
    ilon = var.getCAxes().index('X')
    if var.data.ndim == 3:
        if verbose:
            print "Perform regridding on 3-D data."
        otherdim = [i for i in range(var.data.ndim)
                    if i != ilat and i != ilon][0]
        # new axis order:
        newaxorder = [ilat, ilon, otherdim]
        # transformed data
        trans_data = numpy.transpose(var.data, newaxorder)
        result = sphere_grid.grid_func.regrid(var.getLongitude(),
                                              var.getLatitude(),
                                              trans_data, nlon, nlat)
        # transform back
        newaxorder = sorted(range(var.data.ndim), key=lambda x: newaxorder[x])
        regridded = numpy.transpose(result, newaxorder)
    elif var.data.ndim > 3:
        raise Exception('Right now the regrid function only take 2D or 3D data')
    else:
        regridded = sphere_grid.grid_func.regrid(var.getLongitude(),
                                                 var.getLatitude(),
                                                 var.data, nlon, nlat)

    newlon, newlat = sphere_grid.grid_func.grid_degree(nlat, nlon)
    lon_d = Dimension(data=newlon, units=var.dims[ilon].units, dimname='lon')
    lat_d = Dimension(data=newlat, units=var.dims[ilat].units, dimname='lat')
    dims = []
    for i in range(var.data.ndim):
        if i == ilat:
            dims.append(lat_d)
        elif i == ilon:
            dims.append(lon_d)
        else:
            dims.append(var.dims[i])
    return Variable(data=regridded, dims=dims, parent=var, history='Regridded')


def gaus_filter(var, gausize):
    ''' Filter a variable spatially (i.e. X-Y)
    using a gaussian filter of size gausize

    Args:
    var (geodat.nc.Variable)
    gausize (int) - the size of the window for gaussian filtering

    Returns:
    geodat.nc.Variable
    '''
    from scipy.ndimage.filters import gaussian_filter
    if var.data.mask.any():
        warnings.warn('''There are masked values.
        They are assigned zero before filtering''')
        # Preserve the mask
        mask = var.data.mask.copy()
        var[var.data.mask] = 0.
        var.data.mask = False
        var.data.mask = mask

    newvar = Variable(data=gaussian_filter(var.data, gausize),
                      parent=var,
                      history="Gaussian filter size:"+str(gausize))
    return newvar



def savefile(filename, listofvar, overwrite=False,
             recordax=-1, appendall=False):
    '''
    filename         - a string that specifies the filename,
                       if it is not suffixed with .nc, .nc will be added
    list of variable - a list of geodat.nc.Variable objects, can be a single
                       geodat.nc.Variable
    overwrite        - a boolean.  Overwrite existing file if True.
                       default=False
    recordax         - an integer. Specify the axis that will be the
                       record axis, default = -1 (no record axis)
    appendall        - a boolean. Append to existing file if True.
                       default = False

    This function, however different from other functions in the module,
    uses NetCDF4 Dataset to write data
    '''
    import inspect
    from netCDF4 import Dataset

    # Handle endian
    endian_code = {'>':'big', '<':'little'}

    # if the file is not suffixed by .nc, add it
    if filename[-3:] != '.nc':
        filename += '.nc'

    # check if the file exists.
    # If it does, warn the user if the overwrite flag is not specified
    # savedfile = None

    if os.path.exists(filename):
        if overwrite and appendall:
            raise Exception('appendall and overwrite can\'t be both True.')

        if not overwrite and not appendall:
            print filename+" exists.  Overwrite or Append? [a/o]"
            yn = sys.stdin.readline()
            if yn[0].lower() == 'o':
                overwrite = True
            elif yn[0].lower() == 'a':
                appendall = True
            else:
                raise Exception('''File exists.  Action must be either
                to append or to overwrite''')

    if not os.path.exists(filename) or overwrite:
        # Create temporary file
        ncfile = Dataset(filename+'.tmp.nc', 'w', format='NETCDF3_CLASSIC')
    else:
        # Append existing file
        assert appendall and os.path.exists(filename)
        ncfile = Dataset(filename, 'a', format='NETCDF3_CLASSIC')

    # Add history to the file
    ncfile.history = 'Created from script: '+ inspect.stack()[1][1]

    # if listofvar is a single object, convert it into a list
    if type(listofvar) is not list:
        listofvar = [listofvar]

    for var in listofvar:
        varname = var.varname if var.varname is not None else 'var'
        if var.dims is None and var.data.ndim > 0:
            raise Exception("There is/are missing dimension(s) for "+\
                            var.varname)
        if var.dims is not None:
            # Save dimension arrays
            dimnames = var.getDimnames()
            axes = var.getAxes()
            for idim in range(len(var.dims)):
                if idim == recordax:
                    dimsize = None
                else:
                    #dimsize = var.dims[idim].data.size
                    dimsize = var.data.shape[idim]

                # check if the dimension has already been saved
                isnewdim = dimnames[idim] not in ncfile.dimensions
                if not isnewdim:  # the dimension name exists already
                    olddim = ncfile.variables[dimnames[idim]]
                    # check if the dimensions are in fact the same one,
                    # if not, create new dimension and rename it
                    if not numpy.allclose(axes[idim], olddim[:]) and \
                       idim != recordax:
                        isnewdim = True

                # create new dimension
                if isnewdim:
                    # Rename the dimension if it is unique but has name
                    # collision within the file
                    dimsuffix = ''
                    newDcount = 0
                    while dimnames[idim]+dimsuffix in ncfile.dimensions:
                        newDcount += 1
                        dimsuffix = str(newDcount)

                    dimnames[idim] += dimsuffix
                    ncfile.createDimension(dimnames[idim], dimsize)
                    ## saveddims.append(dimnames[idim])
                    endian = endian_code.get(axes[idim].dtype.byteorder,
                                             'native')
                    if not numpy.isscalar(axes[idim]):
                        dimvar = ncfile.createVariable(
                            dimnames[idim], numpy.dtype(axes[idim].dtype.name),
                            (dimnames[idim],), endian=endian)
                        dimvar[:] = axes[idim]
                        dimvar.setncatts(var.dims[idim].attributes)


        varappend = False or (varname in ncfile.variables and appendall)
        if varname in ncfile.variables and not varappend:
            # Check again
            print "Variable "+varname+" exists.  Append variable?  \
                [a - append]"
            append_code = sys.stdin.readline()
            if append_code[0] == 'a':
                varappend = True
            else:
                # Likely an unintended collision of variable names
                # Change it!
                print "Rename variable "+varname+" as :"
                varname = sys.stdin.readline()[:-1]
                var.varname = varname

        if not varappend:
            endian = endian_code.get(var.data.dtype.byteorder, 'native')
            _ = ncfile.createVariable(varname,
                                      numpy.dtype(var.data.dtype.name),
                                      dimnames, endian=endian,
                                      fill_value=var.getMissingValue())
        var.ensureMasked()
        data2save = var.data
        # print varname
        if varappend:
            if float(var.getMissingValue()) != \
               float(ncfile.variables[varname].getncattr('_FillValue')):
                print '''Warning: Existing var missing value: {}
                Appending var's missing value: {}'''.format(
                    var.getMissingValue(),
                    ncfile.variables[varname].getncattr('_FillValue'))
            oldvar = ncfile.variables[varname]
            olddim = ncfile.variables[var.getDimnames()[recordax]]
            oldnrec = ncfile.variables[varname].shape[recordax]
            newnrec = var.data.shape[recordax]
            s_lice = (slice(None),)*recordax + \
                     (slice(oldnrec, newnrec+oldnrec),)
            print "Appending variable: "+varname
            oldvar[s_lice] = data2save
            print "Appending dimensions: "+ var.getDimnames()[recordax]
            olddim[oldnrec:newnrec+oldnrec] = axes[recordax]
        else:
            if data2save.ndim == 0:
                ncfile.variables[varname].assignValue(data2save)
            else:
                slice_obj = (slice(None),)*data2save.ndim
                ncfile.variables[varname][slice_obj] = data2save

        # Update attributes
        atts = {att:val for att, val in var.attributes.items()
                if att != '_FillValue'}
        ncfile.variables[varname].setncatts(atts)

    ncfile.close()
    if overwrite or os.path.exists(filename) is False:
        os.rename(filename+'.tmp.nc', filename)
        print "Saved to file: "+filename
    elif appendall:
        print "Appended to file: "+filename
    else:
        print "Temporary file created: "+filename+".tmp.nc"


def TimeSlices(var, lower, upper, toggle, no_continuous_duplicate_month=False):
    """ Return a time segment of the variable according to the lower (inclusive)
    and upper limits (inclusive)

    Args:
      var (geodat.nc.Variable)
      lower (numeric): lower time limit
      upper (numeric): upper time limit
      toggle (str): Y/m/d/H/M/S to select a particular time format
      no_continuous_duplicate_month (bool): default False; make sure the
        difference between calendar months in the time axis is always larger
        than or equal to 1; only suitable for dealing with monthly data.

    Returns:
      geodat.nc.Variable

    Examples:
        >>> # time segments in Nov, Dec, Jan and Feb
        >>> TimeSlices(var,11.,2.,"m")

        >>> # time segments from year 1990 to 2000 (inclusive)
        >>> TimeSlices(var,1990,2000,"Y")

        >>> '''Say 01-01-0001 and 31-01-0001 are two adjacent time
        >>> steps as far as monthly data is concerned, the second
        >>> time step 31-01-0001 should be considered as the
        >>> beginning of February and not as January.  So we
        >>> want no_continuous_duplicate_month=True '''
        >>> TimeSlices(var,1,2,"m",True)
    """
    time = var.getDate(toggle, no_continuous_duplicate_month)
    taxis = var.getCAxes().index('T')
    if upper < lower:
        slices = (slice(None),)*taxis + \
            (numpy.logical_or(time >= lower, time <= upper),) +\
            (slice(None),)*(var.data.ndim-taxis-1)
    else:
        slices = (slice(None),)*taxis + \
            (numpy.logical_and(time >= lower, time <= upper),) +\
            (slice(None),)*(var.data.ndim-taxis-1)
    return var[slices]


def plot_vs_axis(var, axis, *args, **kwargs):
    axis = axis.upper()
    line = pylab.plot(var.getAxis(axis), var.data, *args, **kwargs)
    # Use date for the time axis
    if axis == 'T':
        times = var.getAxis(axis)
        iticks = range(0, len(times), len(times)/10)
        xticks = [times[i] for i in iticks]
        dates = ["{}-{}-{}".format(*var.getDate()[i]) for i in iticks]
        pylab.gca().set_xticks(xticks)
        pylab.gca().set_xticklabels(dates, rotation=20)
    return line


def UseMapplot(f_map, f_pylab):
    """ A decorator for using mapplot functions on an geodat.nc.Variable object
    f_map is a functino for making lat-lon plot
    f_pylab is the pylab genuine function
    """
    def plot_func(variable, *args, **kwargs):
        # args needed for quiver
        args = list(args)
        for i in range(len(args)):
            try:
                args[i] = args[i].data.squeeze()
            except AttributeError:
                pass
        var_squeeze = variable.squeeze()
        caxes = var_squeeze.getCAxes()
        data = var_squeeze.data
        if kwargs.has_key("basemap_kwargs"):
            basemap_kwargs = kwargs.pop("basemap_kwargs")
        else:
            basemap_kwargs = None
        if len(caxes) != 2:
            raise Exception('UseMapplot is supposed to be used on 2D data')
        if 'X' in caxes and 'Y' in caxes:
            # Lat-Lon plot
            lons = variable.getLongitude()
            lats = variable.getLatitude()
            m, cs = f_map(lons, lats, data, basemap_kwargs, *args, **kwargs)
            return m, cs
        elif caxes[-1] == 'Z':
            # Z axis is prefered as the vertical axis
            # and the data needs to be transposed
            data = data.T
            for i in range(len(args)):
                try:
                    args[i] = args[i].T
                except AttributeError:
                    pass
        y, x = var_squeeze.getAxes()
        return f_pylab(x, y, data, *args, **kwargs)
    return plot_func

contour = UseMapplot(geodat.plot.mapplot.contour, pylab.contour)
contourf = UseMapplot(geodat.plot.mapplot.contourf, pylab.contourf)
quiver = UseMapplot(geodat.plot.mapplot.quiver, pylab.quiver)
pcolor = UseMapplot(geodat.plot.mapplot.pcolor, pylab.pcolor)


def spatial_corr(var1, var2):
    return numpy.ma.corrcoef(var1.data.ravel(), var2.data.ravel())[0, 1]


def regress(var1, var2):
    return geodat.nc.Variable(data=geodat.signal.regress(var1.data,
                                                         var2.data)[0],
                              dims=var1.dims[1:],
                              varname="{}_{}".format(var1.varname,
                                                     var2.varname),
                              history="{} regress to {}".format(var1.varname,
                                                                var2.varname))
