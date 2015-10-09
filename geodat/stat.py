import numpy
import scipy.interpolate
from scipy.ndimage import filters
import scipy.stats.mstats
import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def lat_weights(lats):
    wgts = numpy.cos(numpy.radians(lats))
    return wgts

def runave(data,N,axis=0,step=None):
    if isinstance(data, numpy.ma.core.MaskedArray):
        data = data.filled(numpy.nan)
    if step is None:
        weights = numpy.ones((N,))
    else:
        if type(step) is not int:
            raise Exception("step should be an integer")
        weights = numpy.array(([1.,]+[0.,]*(step-1))*(N-1)+[1.,])
    weights /= weights.sum()
    return numpy.ma.masked_invalid(filters.convolve1d(data,weights,mode='constant',cval=numpy.nan,axis=axis))


def skewness(data):
    if isinstance(data,numpy.ma.core.MaskedArray):
        npmod = numpy.ma
    else:
        npmod = numpy
    return npmod.power(data - npmod.mean(data),3).mean()/npmod.power(npmod.var(data),3./2.)


def resample_xy(x,y,xnew,nx,ny):
    """ Given a pair of (x,y), resample the y based on the 
    distribution of x and xnew
    
    Keyword Arguments:
    x       -- numpy 1d array
    y       -- numpy 1d array (shape should match x)
    xnew    -- numpy 1d array (any length)
    nx      -- integer
        the number of bins applied to x
    ny      -- integer
        the number of bins applied to y
    
    Return:
    ynew -- numpy 1d array of length = len(xnew)
    
    """
    xedges = numpy.linspace(x.min(),x.max(),nx+1)
    xnew_ibin = numpy.digitize(xnew,xedges)
    y_cdf = {}
    y_mids = {}
    ynew = numpy.empty_like(xnew,dtype=y.dtype)
    for ix,ibin in enumerate(xnew_ibin):
        # If the xnew is out of bound, return numpy.nan for that entry
        if xnew[ix] >= xedges[-1] and xnew < xedges[0]:
            logger.warn("Out of bound for x={:.2e}. Return numpy.nan.".format(float(xnew[ix])))
            ynew[ix] = numpy.nan
            continue
        
        # Don't compute the cdf twice for the same bin
        # Compute it once and save it
        if ibin not in y_cdf:
            ys = y[(x>xedges[ibin-1]) & (x<=xedges[ibin])]
            y_cdf[ibin],y_mids[ibin] = cdf(ys,bins=ny)
        
        while True:
            rand_num = numpy.random.uniform()
            try:
                ynew[ix] = scipy.interpolate.interp1d(y_cdf[ibin],y_mids[ibin])(rand_num)
                break
            except ValueError:
                pass
    return ynew


def cdf(data,option="hist",**kwargs):
    '''
    data   -- numpy 1d array
    option -- "hist" or "empirical"
    '''
    if option.lower() == "hist":
        # Use histogram for cdf
        h,x = numpy.histogram(data,**kwargs)
        x_mid = 0.5*(x[1:]+x[:-1])
        h_cum = numpy.cumsum(h)
        return h_cum/float(h_cum[-1]), x_mid
    elif option.lower() == "empirical":
        raise TypeError("Not yet implemented, sorry!")
    else:
        raise ValueError("option has to be either 'hist' or 'empirical'")
