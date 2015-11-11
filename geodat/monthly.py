import datetime
import logging

import numpy
import matplotlib.dates

from . import keepdims
from . import time_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_sliceobj(data, months, month, axis):
    ''' Get the slice that would extract the data in a particular month

    Input:
    data   - numpy.array
    months - numpy.array for the months on the time axis
    month  - float, the desire month
    axis   - integer, the dimension where the time axis is
    '''
    return (slice(None),)*axis + (months == month,) + \
        (slice(None),)*(data.ndim-axis-1)


def monthly(data, months, axis, func):
    """Apply func to each month.
    If months is given, it is used directly as the time axis.
    Otherwise the months are computed using matplotlib.dates assuming the
    units of the time axis is in DAYS.

    data    - numpy array
    months  - 1-d numpy array of length = length of time axis
    axis    - specify the location of the time axis
    func    - function applied to each month
              the function is given the argument "axis"
              if func is a list, each element is applied to a different month
              func[0] is the function applied to January,
              func[1] is the function applied to February...
    """
    if isinstance(data, numpy.ma.core.MaskedArray):
        npmod = numpy.ma
    else:
        npmod = numpy
    result = []
    if not isinstance(func, list):
        func = [func,]*12
    assert len(func) == 12
    for mon in range(1, 13):
        sliceobj = get_sliceobj(data, months, mon, axis)
        result.append(func[mon-1](data[sliceobj], axis=axis))
    if data.ndim > 1:
        return npmod.concatenate(result, axis=axis).astype(data.dtype)
    else:
        return npmod.array(result, dtype=data.dtype)


def climatology(data, months, axis=0):
    """Compute climatology
    If months is given, it is used directly as the time axis.
    Otherwise the months are computed using matplotlib.dates assuming the
    units of the time axis is in DAYS.

    data    - numpy array
    months  - 1-d numpy array of length = length of time axis
    axis    - specify the location of the time axis, default = 0
    """
    return monthly(data, months, axis, keepdims.mean)


def anomaly(data, months, axis=0, clim=None):
    """ Use climatology function to compute the climatology, and then compute
    the anomaly
    Anomalies and the climatology are returned.  If clim is given, the input
    climatology is used.

    See climatology()
    """
    if clim is None: clim = climatology(data, months=months, axis=axis)
    clim_long = clim2long(clim, axis, months)
    return data-clim_long, clim


def clim2long(clim, axis, months):
    ''' Extend the climatology to match the targeted time axis
    Input:
    climatology : numpy.ndarray
    axis : integer of where the time axis is
    months : integer of months
    '''
    months = months - 1
    assert clim.shape[axis] == 12
    sliceobj = (slice(None),)*axis + (months,)
    return clim[sliceobj]


def is_monthly(time, unit, calendar):
    """ Return a boolean whether the time axis is a monthly axis
    Criterion: as long as all the time steps are larger than 0.5 month and less
               than 1.5 months

    Args:
        time (numpy 1d array)
        unit (str): e.g. "days since 0001-01-01"
        calendar (str): e.g. "julian"

    Returns: bool
    """
    t0 = time_utils.extract_t0_from_unit(unit)

    def round_off_month(month, day):
        """ Round off to the closest month """
        # 15 days to 1 month, 1 month 14 days to 1 month, etc.
        return month + numpy.clip(numpy.ceil(day/15), 0, 1)

    dmonths = (round_off_month(d.month-t0.month, d.day-t0.day)
               for d in time_utils.num2date(numpy.diff(time), unit, calendar))

    return all(( int(dmonth) == 1 for dmonth in dmonths ))


def filter_monthly(time, unit, calendar):
    ''' Given a monthly time axis, check if there are adjacent calendar months
    that are identical because the days are close to the ends of calendar months.
    If so, correct for the duplicate by making reasonable guess about how The
    monthly data is distributed.

    For example, 31-01-0001 and 01-03-0001 would be interpreted as being in JAN
    and FEB.

    Args:
        time (numpy 1d array)
        unit (str): e.g. "days since 0001-01-01"
        calendar (str): e.g. "julian"

    Returns:
        months (numpy 1d array, dtype=numpy.int)
    '''
    if not is_monthly(time, unit, calendar):
        raise ValueError("Input time axis is not a monthly axis")

    alltimes = time_utils.num2date(time, unit, calendar)
    all_months = numpy.array([ t.month for t in alltimes ])

    # Difference between months
    month_diff = numpy.diff(all_months)

    # Filtering is not required
    if ((month_diff == 1) | (month_diff == -11)).all():
        return all_months

    """
    If most of the time stamps are close to the end of a calendar
    month, the problem can be fixed by rolling the time axis
    backward or forward by half a month
    """
    all_days = numpy.array([t.day for t in alltimes])
    if not (sum(numpy.logical_or(
            all_days > 25, all_days < 5)) > len(all_days)/2):
        raise RuntimeError("There are duplicated months and "+\
                           "no_continuous_duplicate_month is True. "+\
                           "But the calendar days provide no hint "+\
                           "for correction.")

    move_forward = sum(all_days > 25) < sum(all_days < 5)
    if isinstance(alltimes[0], datetime.date):
        # timedelta can be added directly
        if move_forward:
            alltimes = numpy.array(alltimes) + datetime.timedelta(days=15)
        else:
            alltimes = numpy.array(alltimes) - datetime.timedelta(days=15)
    else:
        new_times = time_utils.roll(time, unit, calendar,
                                    days=15*int(move_forward))

        alltimes = time_utils.num2date(new_times, unit, calendar)

    all_months = numpy.array([t.month for t in alltimes])

    if (numpy.diff(all_months) != 0).all():
        logger.info("Months are computed by shifting the time "+\
                    "axis {}".format("forward" if move_forward
                                     else "backward"))
        return all_months
    else:
        raise RuntimeError("Failed to correct for duplicated months by "+\
                           "shifting the time axis. "+\
                           "You may consider regridding.")
