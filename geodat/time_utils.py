import collections
import logging

import numpy

import datetime
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#-----------------------------------------------------------------
# If netCDF4 is not installed, some functions are not available
# When these functions are called, an Exception will be raised
#-----------------------------------------------------------------
try:
    import netCDF4 as _netCDF4
    _NETCDF4_IMPORTED = True
except ImportError:
    _NETCDF4_IMPORTED = False


def _throw_error(error):
    def new_func(*args,**kwargs):
        raise error
    return new_func

if _NETCDF4_IMPORTED:
    _num2date = _netCDF4.num2date
    _date2num = _netCDF4.date2num
    _netCDF4_Dataset = _netCDF4.Dataset
    _netCDF4_datetime = _netCDF4.netcdftime.datetime
else:
    logger.warning("Failed to import netCDF4 package. "+\
                   "Some functions may not work")
    _NETCDF4_IMPORT_ERROR = ImportError("The netCDF4 package is "+\
                                        "required but fail to import. "+\
                            "See https://pypi.python.org/pypi/netCDF4/0.8.2")
    _num2date = _date2num = _netCDF4_Dataset = \
                _throw_error(_NETCDF4_IMPORT_ERROR)



def extract_t0_from_unit(unit):
    """ Return a datetime.datetime object referring to the t0 of the time axis
    given a unit of the format "unit since YYYY-mm-dd..."

    Args:
       unit (str)

    Returns: datetime.datetime
    """
    startpoint = unit.split(' since ')[-1]
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


def num2date(time, unit, calendar):
    """ If unit does not start with "month", this function behaves like
    netCDF4.num2date and ensures that an iterable is returned.  If the unit
    starts with "month", a list of datetime is returned where the datetime
    instances are constructed using t0 and relativedelta(months)

    Args:
       time (numpy 1d array)
       unit (str): format "unit since YYYY-mm-dd..."
       calendar (str)

    Returns: an iterable of datetime.datetime instances
    """
    if not unit.startswith("month") and _NETCDF4_IMPORTED:
        alltimes = _num2date(time, unit, calendar.lower())
        if not isinstance(alltimes, collections.Iterable):
            alltimes = [alltimes, ]
    else:
        warning_msg = "Using datetime and relativedelta instead.  Special "+\
                      "calendars are not accounted for"

        if unit.startswith('month'):
            logger.warning("netCDF4.netcdftime cannot handle month units. "+\
                           warning_msg)

        if not _NETCDF4_IMPORTED:
            logger.warning("netCDF4 cannot be imported. "+warning_msg)

        # netcdftime does not handle month as the unit
        # or netCDF4 cannot be imported
        if 'since' not in unit:
            raise Exception("the dimension, assumed to be a time"+\
                            " axis, should have a unit such as "+\
                            "\"days since 01-JAN-2000\"")
        dtime = numpy.diff(time)

        if  (dtime > 0.).any() and (dtime < 0.).any():
            raise ValueError("The axis is not monotonic!")

        if (dtime < 0.).all():
            raise ValueError("Time going backward...really?")

        t0 = extract_t0_from_unit(unit) # is a datetime.datetime object
        # at this point we knew the unit is month
        alltimes = [t0 + relativedelta(months=t)
                    for t in time]
    return alltimes


def roll(time, unit, calendar, days=0, microseconds=0, seconds=0,
         **relativedelta_kwargs):
    """ Roll time axis by timedelta

    Args:
       time (numpy 1d array)
       unit (str)
       calendar (str)
       days (int)
       microseconds (int)
       seconds (int)

    Optional keyword arguments, if set, would be parsed to relativedelta together
    with `days`, `microseconds` and `seconds`

    Returns: numpy 1d array
    """
    time0 = extract_t0_from_unit(unit)
    if relativedelta_kwargs is not None:
        # use relativedelta instead
        timedelta = relativedelta(days=days, microseconds=microseconds,
                                  seconds=seconds, **relativedelta_kwargs)
    else:
        timedelta = datetime.timedelta(days=days, microseconds=microseconds,
                                       seconds=seconds)

    factor = 1
    try:
        new_time0 = time0 + timedelta
    except (OverflowError, ValueError):
        new_time0 = time0 - timedelta
        factor = -1

    delta_num = _date2num(new_time0, unit, calendar)*factor
    return time + delta_num
