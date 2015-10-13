import numpy
import scipy.interpolate
from scipy.ndimage import filters
import scipy.stats.mstats
import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def lat_weights(lats):
    """ Returns cos(lats)

    Required as a weighting for calculating areas on
    a regular cartesian lat-lon grid in which meridians
    converge.

    Arguments:
       lats (numpy ndarray): in degree

    Returns:
       numpy ndarray
    """
    wgts = numpy.cos(numpy.radians(lats))
    return wgts


def runave(data, N, axis=0, step=None):
    """ Compute a smooth box-car running average and masked
    edge values using scipy.ndimage.filters.convolve1d

    Args:
       data (numpy array)
       N (int) : width of the box car
       axis (int): axis along which the average is computed
                (See scipy.ndimage.filters.convolve1d)
       step (optional, int): skips between the box car samples

    Returns:
       numpy masked array (same shape as data)
    """
    if isinstance(data, numpy.ma.core.MaskedArray):
        data = data.filled(numpy.nan)
    if step is None:
        weights = numpy.ones((N,))
    else:
        if type(step) is not int:
            raise Exception("step should be an integer")
        weights = numpy.array(([1.]+[0.]*(step-1))*(N-1)+[1.])
    weights /= weights.sum()
    return numpy.ma.masked_invalid(filters.convolve1d(
        data, weights, mode='constant', cval=numpy.nan, axis=axis))


def skewness(data):
    """ Essentially the same as scipy.stats.skew or
        scipy.stats.mstats.skew depending on the input

    Input:
        data (numpy array or masked array)
    Returns:
        numpy array or masked array
    """
    if isinstance(data,numpy.ma.core.MaskedArray):
        npmod = numpy.ma
    else:
        npmod = numpy
    return npmod.power(data-npmod.mean(data), 3).mean()/\
        npmod.power(npmod.var(data), 3./2.)


def resample_xy(x, y, xnew, nx, ny):
    """ Given paired values of (x,y), randomly sample a set of
    values for ynew given an array xnew such that the joint distribution
    of (xnew, ynew) resembles that of (x,y)

    Arguments:
        x (numpy 1d array)
        y (numpy 1d array): shape should match x
        xnew (numpy 1d array)
        nx (int) : the number of bins applied to x
        ny (int) : the number of bins applied to y

    Returns:
        ynew (numpy 1d array): length = len(xnew)
    """
    if y.shape != x.shape:
        raise ValueError("shape of y should match the shape of x")

    xedges = numpy.linspace(x.min(), x.max(), nx+1)
    xnew_ibin = numpy.digitize(xnew, xedges)
    y_cdf = {}
    y_mids = {}
    ynew = numpy.empty_like(xnew, dtype=y.dtype)
    for ix, ibin in enumerate(xnew_ibin):
        # If the xnew is out of bound, return numpy.nan for that entry
        if xnew[ix] >= xedges[-1] or xnew[ix] < xedges[0]:
            logger.warn("Out of bound for x={:.2e}. "+\
                        "Return numpy.nan.".format(float(xnew[ix])))
            ynew[ix] = numpy.nan
            continue

        # Don't compute the cdf twice for the same bin
        # Compute it once and save it
        if ibin not in y_cdf:
            ys = y[(x > xedges[ibin-1]) & (x <= xedges[ibin])]
            y_cdf[ibin], y_mids[ibin] = cdf(ys,bins=ny)

        # Keep drawing a new y until the drawn value is valid
        while True:
            rand_num = numpy.random.uniform()
            try:
                ynew[ix] = scipy.interpolate.interp1d(
                    y_cdf[ibin], y_mids[ibin])(rand_num)
                break
            except ValueError:
                pass
    return ynew


def cdf(data, **kwargs):
    """ Compute a histogram and the corresponding cumulative
    frequency polygon.

    Args:
        data (numpy 1d array)
        option (str): currently accepts "hist" only

    Keyword arguments currently are passed to numpy.histogram

    Returns:
        bins, cdf : mid values of bins and the cumulative frequencies
    """
    # Use histogram for cdf
    h, x = numpy.histogram(data,**kwargs)
    x_mid = 0.5*(x[1:]+x[:-1])
    h_cum = numpy.cumsum(h)
    return h_cum/float(h_cum[-1]), x_mid
