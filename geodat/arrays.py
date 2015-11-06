import numpy
from scipy.interpolate import interp1d

from .parallelprocessing import run_in_parallel, extract_output

def getSlice(axis, lower, upper, modulo=None):
    '''Given a numpy array, return a slice that extracts the entries
    where lower<=array<=upper.

    The function also takes an optional argument "modulo"
    '''
    if modulo is not None:
        axis, lower, upper = (numpy.mod(a, modulo)
                              for a in [axis, lower, upper])

    if (axis > upper).all() or (axis < lower).all():
        raise ValueError("Not found for range: ({}, {}). modulo:{}".format(
            lower, upper, modulo))

    if numpy.isclose(lower, upper) and ~(axis == upper).any():
        raise ValueError("Not found for range: ({}, {}). modulo:{}".format(
            lower, upper, modulo))
    
    gt_eq_lower = axis >= lower
    lt_eq_upper = axis <= upper

    # Ordinary logical_and
    matched = gt_eq_lower & lt_eq_upper
    if matched.all():
        return slice(None)

    # Perhaps the axis is wrapped around
    if not matched.any():
        matched = gt_eq_lower ^ lt_eq_upper
    
    if not matched.any():
        raise ValueError("Not found for range: ({}, {}). modulo:{}".format(
            lower, upper, modulo))

    # Find the beginning and ending of the chunk that matches the range
    indices = numpy.where(numpy.diff(~matched))[0]
    if len(indices) == 1:
        i1, i2 = indices+1, len(axis)-1
    elif len(indices) == 2:
        i1, i2 = indices+1
    elif len(indices) > 2:
        raise RuntimeError("Too many chunks along the axis that fall within "+\
                           "the given range")
    else:
        raise NotImplementedError

    # Determine if the chunks are at the ends of the axis
    if all(matched[:i1]) and all(matched[i2:]):
        return numpy.array(range(i2, len(axis))+range(0, i1+1))
    else:
        return slice(i1, i2, None)


def find_1d(arr, criterion, nresult=1, result_func=lambda i, x: (i, x),
            fillvalue=None):
    '''
    arr (numpy.array)
    criterion (function that returns boolean), e.g. lambda v: v > 2
    nresult (str) - number of results (1 means return the first result)
    result_func (function that accepts (int,value))- by default returns the
                                                     index i and the value x
                                                     i.e. lambda i,x: (i,x)
    '''
    if arr.ndim != 1:
        raise Exception("Expect the array to be 1-D")
    if not isinstance(nresult, int):
        raise TypeError("nresult must be an integer")
    if nresult <= 0:
        raise ValueError("nresult must be > 0")
    result = []
    for i, x in enumerate(arr):
        if criterion(x):
            result.append(result_func(i, x))
            if len(result) == nresult:
                break
    if len(result) != nresult:
        result = tuple(result + [fillvalue,]*(nresult-result))
    if len(result) == 1:
        return result[0]
    else:
        return result


def apply_along_axis(func, axis, arr, chunk_size=10000, do_parallel=True,
                     *args, **kwargs):
    ''' This seems to be about 3 times faster than numpy.apply_along_axis '''
    if axis != -1 and axis != arr.ndim-1:
        # roll arr to make the required axis comes last
        # this should just be a view instead of creating a new array
        new_arr = numpy.rollaxis(arr, axis, arr.ndim)
    else:
        new_arr = arr
    rolled_shape = new_arr.shape
    if new_arr.ndim > 2:
        # reshape
        new_arr = new_arr.reshape(-1, new_arr.shape[-1])
    if new_arr.shape[0] > chunk_size and do_parallel:
        # Call apply_along_axis multiple time using multiprocessing
        parallel_fun = run_in_parallel(apply_along_axis)
        for arr_slice in getSlice_chunk(new_arr, istep=chunk_size):
            ps, queue_output = parallel_fun(func, 1, new_arr[arr_slice],
                                            *args, do_parallel=False, **kwargs)
        result = numpy.vstack(extract_output(ps, queue_output))
        queue_output.close()
    else:
        # Get the first result to find how long the reduced axis should be
        first_result = numpy.array(func(new_arr[0, :].squeeze(),
                                        *args, **kwargs))
        reduced_axis_len = first_result.size
        # this is a new array
        result = numpy.zeros((new_arr.shape[0], reduced_axis_len),
                             dtype=first_result.dtype)
        result[0, :] = first_result
        for iother in xrange(1, result.shape[0]):
            result[iother, :] = numpy.array(func(new_arr[iother, :].squeeze(),
                                                 *args, **kwargs))
    # reshape result back
    result = result.reshape(rolled_shape[:-1]+(-1,))
    # roll the result back
    return numpy.rollaxis(result, -1, axis)


def find_value(arr, axis, func, nresult=1):
    ''' Find the first nresult values that fulfils func '''
    return apply_along_axis(find_1d, axis, arr, criterion=func,
                            nresult=nresult,
                            result_func=lambda i, x: x,
                            fillvalue=numpy.nan)


def find_ind2(arr, axis, func, nresult=1):
    ''' Find the index(indices) of the first nresult value(s)
    that fulfils func '''
    return apply_along_axis(find_1d, axis, arr, criterion=func, nresult=nresult,
                            result_func=lambda i, x: i,
                            fillvalue=-9999)

def find_ind(arr, axis, func):
    ''' Find the first index where func(arr) is true along axis
    func has to return a numpy boolean array '''
    return numpy.argmax(func(arr), axis=axis)


def find_loc(arr, axis, value, x, kind='linear', bounds_error=False, **kwargs):
    ''' Find the x along an axis
    where arr == value using scipy.interpolation.interp1d
    '''
    # I have the old version of scipy so I need to sort arr_1d
    def new_func_1d(arr_1d, x=x):
        # Force monotonic increasing
        arr_ind = numpy.argsort(arr_1d)
        arr_1d = arr_1d[arr_ind]
        x = x[arr_ind]
        return interp1d(arr_1d, x,
                        kind=kind,
                        bounds_error=bounds_error)(value)
    return apply_along_axis(new_func_1d, axis, arr, **kwargs)


def find_loc2(arr, axis, value, x):
    ''' Find the x along an axis
    where arr == value using linear interpolation
    arr is assumed to be monotonic along the selected axis
    '''
    if len(x) != arr.shape[axis]:
        raise Exception('''Length of x ({}) should be the same as the selected
        dimension ({})'''.format(len(x), arr.shape[axis]))

    if numpy.ma.isMaskedArray(arr):
        npmod = numpy.ma
    else:
        npmod = numpy

    # force-cast arr into floating point values
    if not issubclass(arr.dtype.type, numpy.inexact):
        arr = arr.astype(numpy.float_)

    # Find the first zero crossing
    ix_inc = npmod.argmax(arr >= value, axis=axis)
    ix_dec = npmod.argmax(arr <= value, axis=axis)

    def get_y(*args, **kwargs):
        ''' to be used by numpy.fromfunction '''
        sl = list(args)
        sl.insert(axis, kwargs['ix'])
        return arr[tuple(sl)]

    y_inc_m1 = npmod.fromfunction(get_y, ix_inc.shape, dtype=int,
                                  ix=numpy.where(ix_inc > 0, ix_inc-1, 0))
    y_dec_m1 = npmod.fromfunction(get_y, ix_dec.shape, dtype=int,
                                  ix=numpy.where(ix_dec > 0, ix_dec-1, 0))
    ix_hi = npmod.where(((ix_inc <= ix_dec) & ((ix_inc != 0) | \
                                               (y_inc_m1.mask
                                                if numpy.ma.is_masked(y_inc_m1)
                                                else True))) |\
                         ((ix_inc >= ix_dec) & ((ix_dec == 0) | \
                                                (y_dec_m1.mask
                                                 if numpy.ma.is_masked(y_dec_m1)
                                                 else True))),
                        ix_inc, ix_dec)
    ix_lo = npmod.where(ix_hi > 0, ix_hi-1, 0)
    x_lo, x_hi = x[ix_lo], x[ix_hi]

    # y_lo and y_hi would have shape of (m,l) as well
    y_lo = npmod.fromfunction(get_y, ix_lo.shape, dtype=int, ix=ix_lo)
    y_hi = npmod.fromfunction(get_y, ix_hi.shape, dtype=int, ix=ix_hi)
    dx = x_hi - x_lo
    dy = y_hi - y_lo
    # Interpolate when y_lo != value
    # Out of bound value will be assigned nan
    result = npmod.where(y_lo == value, x_lo, (value-y_lo)*dx/dy + x_lo)
    return result



def find_loc2p1(arr, axis, value, x, masked_value=None):
    ''' Find the x along an axis
    where arr == value using linear interpolation
    arr is assumed to be monotonic along the selected axis
    '''
    if len(x) != arr.shape[axis]:
        raise Exception('''Length of x ({}) should be the same as the selected
        dimension ({})'''.format(len(x), arr.shape[axis]))
    def get_y(*args, **kwargs):
        sl = list(args)
        sl.insert(axis, kwargs['ix'])
        return arr[tuple(sl)]

    if numpy.ma.isMaskedArray(arr):
        npmod = numpy.ma
    else:
        npmod = numpy

    # View arr as floating point values
    if not issubclass(arr.dtype.type, numpy.inexact):
        arr = arr.view(numpy.float_)

    # Slicing
    sl_p1, sl_m1 = zip(*[(slice(None), slice(None)) if iax != axis
                         else (slice(1, None), slice(0, -1))
                         for iax in xrange(arr.ndim)])

    # Changing sign
    arr_m_sign = numpy.sign(arr - value)
    sign_changed = arr_m_sign[sl_p1]*arr_m_sign[sl_m1] <= 0

    # Make sure masked values are neglected
    # By providing a masked value, the input can be a numpy.ndarray
    # instead of a numpy.ma.ndarray
    # This may be faster than masking the array beforehand
    if masked_value is not None:
        sign_changed = sign_changed & \
                       (arr[sl_m1] != masked_value) & \
                       (arr[sl_p1] != masked_value)

    # If there is no zero crossing at all
    # Make ix_hi and ix_lo identical and dx/dy will be nan
    ix_lo = npmod.argmax(sign_changed, axis=axis)
    ix_hi = npmod.where(~npmod.any(sign_changed, axis=axis),
                        ix_lo, ix_lo+1)
    x_lo, x_hi = x[ix_lo], x[ix_hi]

    # y_lo and y_hi would have shape of (m,l) as well
    y_lo = npmod.fromfunction(get_y, ix_lo.shape, dtype=int, ix=ix_lo)
    y_hi = npmod.fromfunction(get_y, ix_hi.shape, dtype=int, ix=ix_hi)
    dx = x_hi - x_lo
    dy = y_hi - y_lo

    # Interpolate when y_lo != value
    # Out of bound value will be assigned nan
    result = npmod.where(y_lo == value, x_lo, (value-y_lo)*dx/dy+x_lo)
    return result


def find_loc3(arr, axis, value, x, kind='linear', bounds_error=False, **kwargs):
    ''' Find the x along an axis
    where arr == value using scipy.interpolation.interp1d
    '''
    # I have the old version of scipy so I need to sort arr_1d
    def new_func_1d(arr_1d, x=x):
        # Force monotonic increasing
        arr_ind = numpy.argsort(arr_1d)
        arr_1d = arr_1d[arr_ind]
        x = x[arr_ind]
        return numpy.array([interp1d(arr_1d, x,
                                     kind=kind,
                                     bounds_error=bounds_error,
                                     **kwargs)(value),])
    return numpy.apply_along_axis(new_func_1d, axis, arr)


def find_max(arr, axis):
    '''Just to make sure it replicates the builtin numpy function'''
    return apply_along_axis(numpy.max, axis, arr)


def getSlice_chunk(arr, niter=10, istep=None, idim_iter=0):
    ''' A generator that run the func iteratively in chunk
    Iterate through the *idim_iter*-th dimension along arr
    arr   = numpy.array
    niter = number of iteration (or minus 1)
    iarg_iter = the index of which element in args is to be iterated over
    istep would over-run niter
    For example,
    x = numpy.arange(34).reshape(17,2)
    gen = getSlice_chunk(x,niter=3,idim_iter=0)
    gen.next() ---> slice(0,5)
    gen.next() ---> slice(5,10)
    gen.next() ---> slice(10,15)
    gen.next() ---> slice(15,17)
    '''
    assert isinstance(niter, int)
    assert isinstance(idim_iter, int)
    icurrent = 0
    length = arr.shape[idim_iter]
    if istep is None:
        istep = length/niter
    while icurrent < length-1:
        sliceobj = (slice(None),)*idim_iter +\
                   (slice(icurrent, min(icurrent+istep, length)),)
        yield sliceobj
        icurrent += istep



def is_strict_monotonic_func(axis):
    ''' Check whether axis is a strictly monotonic function

    Args:
        axis (1d numpy.ndarray)

    Returns:
        bool
    '''
    return (numpy.diff(axis) > 0.).all() or (numpy.diff(axis) < 0.).all()


def fix_longitude(axis, modulo=360.):
    ''' Add modulo (default 360.) *in-place* to the points
    beyond which discontinuity occurs

    Arguments:
    axis -- numpy 1d array
    '''
    if axis.ndim != 1:
        raise TypeError("Input should be an 1-d array")

    if is_strict_monotonic_func(axis):
        return axis
    else:
        i_jump = numpy.argmin(numpy.diff(axis))+1
        axis[i_jump:] += modulo
        return fix_longitude(axis)
