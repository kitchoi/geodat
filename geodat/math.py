import numpy
from . import keepdims

def integrate(data, axes=None, iax=None, **kwargs):
    ''' Integrate data along a selected set of axes
    Dimension is maintained using keepdims.sum
    data - numpy.ndarray
    axes - a list of axes
    iax - a list of integers that select which axes are integrated along
    '''
    if axes is None:
        axes = [numpy.arange(idim) for idim in data.shape]
    if iax is None:
        iax = range(data.ndim)
    if type(iax) is not list:
        iax = [iax]
    inc = numpy.ones(data.shape, dtype=data.dtype)
    for ax in iax:
        ax_data = numpy.array(axes[ax], dtype=data.dtype)
        dx = numpy.gradient(ax_data)[(numpy.newaxis,)*ax + \
                                     (slice(None),) + \
                                     (numpy.newaxis,)*(data.ndim-ax-1)]
        inc *= numpy.abs(dx)

    # Perform integration
    return keepdims.sum(data*inc, axis=iax, **kwargs)


def gradient(f, dx=1., axis=0, mask_boundary=False):
    '''Compute central difference for a pariticular axis
    Input:
    f - numpy ndarray
    dx - spacing
    '''
    if f.shape[axis] <= 1:
        raise ValueError("Length of axis {} must be >1".format(axis))
    result = numpy.ma.zeros(f.shape, dtype=f.dtype)
    axis = axis % f.ndim
    sl_right = (slice(None),)*axis \
                + (slice(2, None),) + (slice(None),)*(f.ndim-axis-1)
    sl_left = (slice(None),)*axis \
              + (slice(0, -2),) + (slice(None),)*(f.ndim-axis-1)
    sl_center = (slice(None),)*axis \
                + (slice(1, -1),) + (slice(None),)*(f.ndim-axis-1)

    # Make sure all dimension of dx have len>1
    dx = dx.squeeze()

    # Broadcasting dx
    sl_dx_center = (numpy.newaxis,)*axis + (slice(1, -1),) \
                   + (numpy.newaxis,)*(f.ndim-axis-1)
    if numpy.isscalar(dx):
        result[sl_center] = (f[sl_right] - f[sl_left])/2./dx
    elif result.ndim == dx.ndim:
        result[sl_center] = (f[sl_right] - f[sl_left])/2./dx[sl_center]
    else:
        result[sl_center] = (f[sl_right] - f[sl_left])/2./dx[sl_dx_center]

    # Boundary values
    b1 = (slice(None),)*axis \
         + (slice(0, 1),) + (slice(None),)*(f.ndim-axis-1)
    b2 = (slice(None),)*axis \
         + (slice(-1, None),) + (slice(None),)*(f.ndim-axis-1)
    if mask_boundary:
        result[b1] = numpy.ma.masked
        result[b2] = numpy.ma.masked
    else:
        b1_p1 = (slice(None),)*axis \
                + (slice(1, 2),) + (slice(None),)*(f.ndim-axis-1)
        b2_m1 = (slice(None),)*axis \
                + (slice(-2, -1),) + (slice(None),)*(f.ndim-axis-1)
        sl_dx_0 = (numpy.newaxis,)*axis + (slice(0, 1),) \
                  + (numpy.newaxis,)*(f.ndim-axis-1)
        sl_dx_end = (numpy.newaxis,)*axis + (slice(-1, None),) \
                    + (numpy.newaxis,)*(f.ndim-axis-1)
        if numpy.isscalar(dx):
            result[b1] = (f[b1_p1] - f[b1])/dx
            result[b2] = (f[b2] - f[b2_m1])/dx
        elif result.ndim == dx.ndim:
            result[b1] = (f[b1_p1] - f[b1])/dx[b1]
            result[b2] = (f[b2] - f[b2_m1])/dx[b2]
        else:
            result[b1] = (f[b1_p1] - f[b1])/dx[sl_dx_0]
            result[b2] = (f[b2] - f[b2_m1])/dx[sl_dx_end]
    return result


def div(ux, uy, dx, dy, xaxis=-1, yaxis=-2):
    ''' Compute the divergence of a vector field ux,uy
    dx, dy are either 1-d array or a scalar
    xaxis - integer indicating the location of x axis (default = -1)
    yaxis - integer indicating the location of y axis (default = -2)
    '''
    result = gradient(ux, dx, axis=xaxis, mask_boundary=True) + \
             gradient(uy, dy, axis=yaxis, mask_boundary=True)
    return result


def _div(ux, uy, dx, dy):
    ''' Backup - Compute the divergence of a vector field ux,uy
    dx, dy are either 1-d array or a scalar
    xaxis - integer indicating the location of x axis (default = -1)
    yaxis - integer indicating the location of y axis (default = -2)
    '''
    ny, nx = ux.shape[-2:]
    if numpy.isscalar(dx) or dx.ndim <= 1:
        dx = numpy.resize(dx, (nx, ny)).T
    if numpy.isscalar(dy) or dy.ndim <= 1:
        dy = numpy.resize(dy, (ny, nx))
    result = numpy.ma.zeros(ux.shape)
    result[..., 1:-1, 1:-1] = \
        (ux[..., 1:-1, 2:]-ux[..., 1:-1, 0:-2])/2./dx[1:-1, 1:-1] \
        + (uy[..., 2:, 1:-1]-uy[..., 0:-2, 1:-1])/2./dy[1:-1, 1:-1]
    result[..., 0, :] = numpy.ma.masked
    result[..., -1, :] = numpy.ma.masked
    result[..., :, 0] = numpy.ma.masked
    result[..., :, -1] = numpy.ma.masked
    return result
