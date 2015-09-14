import numpy
import geodat.keepdims as keepdims
import scipy.stats

def regress(y, x, axis=0, reverse=False, return_corr=False):
    """regress(y,x,axis=0)
       where x is 1D of shape (M,), y is (...,M,...)
             axis specifies where M is in y

       Return: slope, intercept, p, corr
       reverse - if False (default) then y = slope*x + intercept
                 if True            then x = slope*y + intercept

    """
    if x.squeeze().ndim != 1:
        raise Exception("x is supposed to be 1D")

    x = x.squeeze()
    if y.shape[axis] != len(x):
        raise Exception("The {}-th axis of y(={}) should match the length"+\
                        "of x(={})".format(axis, y.shape[axis], len(x)))
    xtoy = (numpy.newaxis,)*axis + (slice(None),) + \
           (numpy.newaxis,)*(y.ndim-axis-1)

    if reverse:
        x_dev = x[xtoy] - x.mean()
        y_dev = y-keepdims.mean(y, axis)
        slope = keepdims.sum(x_dev*y_dev, axis)/\
            keepdims.sum(numpy.power(y-y.mean(), 2), axis)
        intercept = (keepdims.mean(x[xtoy], axis)-\
                     slope*keepdims.mean(y, axis)).squeeze()
        slope = slope.squeeze()
        intercept = intercept.squeeze()
    else:
        # original interpretation
        x_dev = x[xtoy]-x.mean()
        y_dev = y-keepdims.mean(y, axis)
        slope = keepdims.sum((x_dev*y_dev), axis)/\
                keepdims.sum(numpy.power(x-x.mean(), 2))
        intercept = (keepdims.mean(y, axis)-\
                     slope*keepdims.mean(x[xtoy], axis)).squeeze()
        slope = slope.squeeze()
        intercept = intercept.squeeze()

    def compute_sqrt_x2(x, axis=None):
        return numpy.ma.sqrt(keepdims.sum(numpy.power(x-x.mean(), 2), axis))


    def compute_nd(y, axis=None):
        if axis is None:
            nd = len(y)
        else:
            nd = y.shape[axis]
        nd *= numpy.ones([n for idim, n in enumerate(y.shape)
                          if idim != axis])
        if isinstance(y, numpy.ma.core.MaskedArray):
            if y.mask.any():
                nd = (~y.mask).sum(axis=axis)
        return nd

    if not return_corr:
        return slope, intercept
    else:
        sqrt_y2 = compute_sqrt_x2(y, axis=axis)
        sqrt_x2 = compute_sqrt_x2(x)
        if reverse:
            corr = (slope*sqrt_y2/sqrt_x2).squeeze()
        else:
            corr = (slope*sqrt_x2/sqrt_y2).squeeze()

        # calculate degree of freedom
        if isinstance(x, numpy.ma.core.MaskedArray):
            if x.mask.any():
                nd = compute_nd(x[xtoy]*y, axis=axis)
            else:
                nd = compute_nd(y, axis=axis)
        else:
            nd = compute_nd(y, axis=axis)
        ts = numpy.abs(corr)/numpy.ma.sqrt((1-corr*corr)/(nd-2))
        p = 1 - scipy.stats.t.sf(ts, nd)
        return slope, intercept, p, corr


def regress_xND(y, x, axis_x=0):
    if y.ndim != 1:
        raise Exception("y is expected to be 1D")
    if x.shape[axis_x] != len(y):
        raise Exception("Axis {}(={}) of x should match the length of "+\
                        "y (={})".format(axis_x, x.shape[axis_x], len(y)))
    slope = (x-keepdims.mean(x, axis=axis_x))*(y-y.mean())
    return slope


def ma_polyfit_fix(x, y, *args, **kwargs):
    ''' Temporary work around for numpy.ma.polyfit'''
    all_mask = y.mask.sum(axis=0) > 0
    new_y = y.copy()
    new_y.mask = False
    result = numpy.ma.array(numpy.ma.polyfit(x, new_y, *args, **kwargs))
    result[:, all_mask] = numpy.ma.masked
    return result


def detrend(y, x, axis=0):
    """detrend(y,x=None,axis=0)
    Make use of regress(y,x,axis)
    Return ydtd = y - slope*x[xtoy] - intercept
    """
    slope, intercept = regress(y, x, axis)
    xtoy = (numpy.newaxis,)*axis + (slice(None),) + \
           (numpy.newaxis,)*(y.ndim-axis-1)
    ydtd = y - slope*x[xtoy] - intercept
    return ydtd

def princomp(data, numpc=0, var_dim=1, normalise=True):
    ''' Compute the principal component for data
    Input:
    numpc   - number of PC to be extracted (default - 0 = ALL PC)
    var_dim - the dimension corresponding to the variable (default 1 - col)
    normalise - whether the vectors are normalised (default - True)
    Return:
    evecs (eigenvectors), evals (eigenvalues), score (projection)
    '''
    # Center the data
    m = keepdims.mean(data, axis=var_dim)
    data -= m
    # Covariance matrix
    C = numpy.cov(data.T)

    # Compute eigenvalues and eigenvectors
    evals, evecs = numpy.linalg.eig(C)

    # Sort them in descending order
    indices = numpy.argsort(evals)[::-1]
    evecs = evecs[:, indices]
    evals = evals[indices]

    if numpc > 0:
        if numpc > evecs.shape[1]:
            import warnings
            warnings.warn("The number of PC wanted exceeds the number of "+\
                          "observations. All PCs are returned.")
            numpc = evecs.shape[1]
        evecs = evecs[:, :numpc]

    # Normalise the PCs
    if normalise:
        for i in range(evecs.shape[1]):
            evecs[:, i] /= numpy.linalg.norm(evecs[:, i]) * numpy.sqrt(evals[i])

    score = numpy.dot(evecs.T, data) # projection of the data in the new space
    return evecs, evals, score
