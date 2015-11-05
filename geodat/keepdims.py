'''This module is applied to numpy arithmetic functions and it has three main
purposes:

The first two purposes are supported in some numpy routines (but not their
masked array equivalence) since ver 1.7+:
1) deliver the same effect of setting keepdims=True
2) accept a tuple of int as axis

The third feature is useful for climate science:
3) propagate masked values instead of ignoring them

`keepdims` in this module is a decorator that can be applied to any numpy
routines such as numpy.mean, numpy.argmax, ... that accept axis as an argument.
Their numpy.ma counterparts would be called if the input array is a numpy masked
array.
'''

import warnings

import numpy

def keepdims(f):

    def new_f(arr,axis,*args,**kwargs):
        if isinstance(arr,numpy.ma.core.MaskedArray):
            npmod = numpy.ma
        else:
            npmod = numpy
        if hasattr(npmod,f.__name__):
            f_ma = getattr(npmod,f.__name__)
        else:
            f_ma = f
        result = f_ma(arr, axis, *args, **kwargs)
        return result


    def new_f_axis(arr, axis, *args,**kwargs):
        # Only one axis is requested
        result = new_f(arr, axis, *args,**kwargs)
        if isinstance(arr, numpy.ma.core.MaskedArray):
            npmod = numpy.ma
        else:
            npmod = numpy

        if arr.ndim > 0 and arr.ndim > result.ndim:
            return npmod.expand_dims(result, axis)

        return result

    def new_f_axes(arr, axis, *args, **kwargs):
        # dimensions of array
        ndims = arr.ndim

        # Rewrite negative indices to positive ones, and sort them
        axis = sorted({ax % ndims for ax in axis})
        nravel = len(axis)

        # new axis order
        newaxorder = [ i for i in range(axis[-1]) if i not in axis ]
        numaxfront = len(newaxorder)
        newaxorder += axis + [ i for i in range(numaxfront+nravel,ndims) ]

        # Do transpose
        arr = numpy.transpose(arr, newaxorder)

        # Reshape
        arr = arr.reshape(arr.shape[:numaxfront] + (-1,)
                            + arr.shape[numaxfront+nravel:])

        # apply the function
        result = new_f(arr, numaxfront, *args, **kwargs)

        if numpy.isscalar(result):
            if isinstance(arr,numpy.ma.core.MaskedArray):
                npmod = numpy.ma
            else:
                npmod = numpy
            return npmod.array(result).reshape([1]*ndims)
        else:# insert the axes back
            return result[[numpy.newaxis if i in axis else slice(None)
                           for i in range(ndims) ]]

    def new_f_dispatch(arr, axis=None, *args,**kwargs):
        """ Behave like the new version numpy `keepdims`=True

        Args:
           arr (numpy array)
           axis (int or a list of int): default all axes
        Other arguments are parsed to the numpy function

        Keyword arguments:
           progMask (bool): if a masked array's mask is to be propagated

        Other keyword arguments are passed to the numpy function
        """
        # Is the mask array (if there is one) propagated along the axis?
        progMask = kwargs.pop("progMask", False)

        if axis is None:
            if arr.ndim == 0:
                axis = 0
            else:
                axis=range(0, arr.ndim)

        if isinstance(axis, int):
            func = new_f_axis
        else:
            func = new_f_axes
        result = func(arr, axis, *args, **kwargs)

        # Propagate mask if kwargs['progMask'] is True
        if progMask and isinstance(arr, numpy.ma.core.MaskedArray):
            try:
                result.mask = keepdims(numpy.max)(arr.mask, axis)
            except ZeroDivisionError:
                warnings.warn('ZeroDivisionError while propagaing mask.')
            except ValueError:
                warnings.warn('ValueError while propaging mask.')

        # Preserve fill value
        if isinstance(arr,numpy.ma.core.MaskedArray):
            result.set_fill_value(arr.fill_value)

        return result
    return new_f_dispatch


mean = keepdims(numpy.mean)
average = keepdims(numpy.average)
min = keepdims(numpy.min)
max = keepdims(numpy.max)
std = keepdims(numpy.std)
argmax = keepdims(numpy.argmax)
argmin = keepdims(numpy.argmin)
trapz = keepdims(numpy.trapz)
sum = keepdims(numpy.sum)
