import numpy
import matplotlib.dates

from . import keepdims

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
