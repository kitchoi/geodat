from functools import wraps
import numpy
import pyferret
import geodat.nc as _NC


def Num2Fer(data, coords, dimunits,
            varname="UNKNOWN", data_units=None, missing_value=None,
            cartesian_axes=None, dimnames=None):
    ''' Create a dictionary that resemble the Ferret
    data variable structure to be passed to pyferret.putdata
    Necessary Input:
    data           - numpy.ndarray
    coords         - a list of numpy.ndarray
    dimunits       - a list of strings for the dimension units
                     (e.g. ['months','degrees_N','degrees_E'])

    Optional input:
    varname        - string
    data_units     - string
    missing_value  - numeric
    cartesian_axes - a list of characters that specifies
                     the cartesian axes (e.g. ['T','Y','X'])
                     If this is not specified, guesses will be made
                     using the dimension units (say unit month will
                     be interpreted for a [T]IME axis.
                     Specifying cartesian_axes overwirtes the
                     guesses.
    dimnames       - a list of strings for the dimension names
                     (e.g. ['time','lat','lon'])

    Length of cartesian_axes, dimnames, dimunits and coords need
    to agree with the number of dimensions of data

    Return:
    a dictionary
    '''
    if len(dimunits) != data.ndim:
        raise Exception("Number of dimunits does not match data.ndim")
    if len(coords) != data.ndim:
        raise Exception("Number of coords does not match data.ndim")
    fer_var = {}
    # Define the variable
    fer_var['data'] = data.copy()
    # Variable name
    fer_var['name'] = varname
    # Dataset
    fer_var['dset'] = None
    # Title = variable name
    fer_var['title'] = fer_var['name']
    # Set missing value
    if missing_value is not None:
        fer_var['missing_value'] = missing_value
    # Set data unit
    if data_units is not None:
        fer_var['data_unit'] = data_units
    # Determine the axis type
    cax2ax_type = {'X': pyferret.AXISTYPE_LONGITUDE,
                   'Y': pyferret.AXISTYPE_LATITUDE,
                   'Z': pyferret.AXISTYPE_LEVEL,
                   'T': pyferret.AXISTYPE_CUSTOM}
    # Make guessses for the axis type
    if cartesian_axes is None:
        cartesian_axes = [_NC._assign_caxis_(dimunit)
                          for dimunit in dimunits]

    if len(cartesian_axes) != data.ndim:
        raise Exception("Number of cartesian_axes/dimunits does"+\
                        " not match data.ndim")

    # Convert it to PyFerret convention
    fer_var['axis_types'] = [cax2ax_type[cax]
                             if cax in cax2ax_type.keys()
                             else pyferret.AXISTYPE_NORMAL
                             for cax in cartesian_axes]

    if dimnames is not None:
        if len(dimnames) != data.ndim:
            raise Exception("Number of dimnames does not match data.ndim")
        fer_var['axis_names'] = dimnames

    fer_var['axis_units'] = dimunits
    fer_var['axis_coords'] = coords
    # This will be used as the second argument to pyferret.putdata
    axis_pos_dict = {'X': pyferret.X_AXIS,
                     'Y': pyferret.Y_AXIS,
                     'Z': pyferret.Z_AXIS,
                     'T': pyferret.T_AXIS}
    # Force axis position
    fer_var['axis_pos'] = [axis_pos_dict[cax]
                           if cax in axis_pos_dict.keys()
                           else cartesian_axes.index(cax)
                           for cax in cartesian_axes]
    return fer_var


def Var2Fer(var, name=None):
    ''' Given a geodat.nc.Variable, return a dictionary
    that resemble the Ferret data variable structure
    to be passed to pyferret.putdata
    '''
    if name is None:
        varname = var.varname
    else:
        varname = name

    return Num2Fer(data=var.data, missing_value=var.getMissingValue(),
                   coords=var.getAxes(),
                   dimunits=[dim.units for dim in var.dims],
                   varname=varname, data_units=var.getattr('units', ''),
                   cartesian_axes=var.getCAxes(), dimnames=var.getDimnames())

def Fer2Var(var):
    ''' Convert the dictionary returned by pyferret.getdata into a
    geodat.nc.Variable

    Args:
    var  - a dictionary returned by pyferret.getdata

    Returns:
    geodat.nc.Variable
    '''
    result = Fer2Num(var)
    dims = [_NC.Dimension(data=result['coords'][i],
                          units=result['dimunits'][i],
                          dimname=result['dimnames'][i])
            for i in range(len(result['coords']))]
    newvar = _NC.Variable(data=result['data'], dims=dims,
                          varname=result['varname'],
                          history='From Ferret')
    return newvar


def Fer2Num(var):
    ''' Filter the dictionary returned by pyferret.getdata
    PyFerret usually returns data with extra singlet dimension
    Need to filter those
    Input:
    var       - a dictionary returned by pyferret.getdata

    Returns:
    A dictionary with the following items
    data      - a numpy ndarray
    varname   - the name of the variable
    coords    - a list of numpy ndarrays for the dimensions
    dimunits  - a list of strings, the units for the dimensions
    dimnames  - a list of strings, the names for the dimensions
    '''
    results = {}
    results['coords'] = [ax for ax in var['axis_coords']
                         if ax is not None]
    if var['axis_names'] is not None:
        results['dimnames'] = [var['axis_names'][i]
                               for i in range(len(var['axis_names']))
                               if var['axis_coords'][i] is not None]

    # If the axis_type is TIME, the axis_unit is the calendar type which
    # is not considered yet
    if pyferret.AXISTYPE_TIME in var['axis_types']:
        raise Exception("Immature function: axis_type from Ferret is TIME,"+\
                        "not CUSTOM; a situation not taken into yet.")

    results['dimunits'] = [var['axis_units'][i]
                           for i in range(len(var['axis_units']))
                           if var['axis_coords'][i] is not None]

    sliceobj = [0 if ax is None else slice(None)
                for ax in var['axis_coords']]
    results['data'] = var['data'][sliceobj]
    results['varname'] = var['title']
    return results


def send_var_run_get_output(input_vars, command, output_name,
                            verbose=False, output_caxes=None):
    pyferret.start(quiet=True, journal=verbose,
                   verify=False, server=True)

    # Convert from geodat.nc.Variable to dictionary to be
    # fed to Ferret
    input_fer_var = {varname: Var2Fer(var, name=varname)
                     for varname, var in input_vars.items()}

    # Put data
    for varname, fer_var in input_fer_var.items():
        pyferret.putdata(fer_var, axis_pos=fer_var['axis_pos'])
        if verbose:
            print "Put variable:"+varname
            pyferret.run('show grid '+fer_var['name'])

    # Run command
    pyferret.run(command)
    if verbose:
        print "Run command:"+command

    # Get results
    result_fer = pyferret.getdata(output_name)
    if verbose:
        print "Get {output_name} from FERRET".format(output_name=output_name)

    # Convert from ferret data structure to geodat.nc.Variable
    result = Fer2Var(result_fer)

    # Preserve dimension order (Ferret reverts the order)
    if output_caxes is not None:
        assert isinstance(output_caxes, list)
        neworder = [result.getCAxes().index(cax)
                    for cax in output_caxes]
        new_dims = [result.dims[result.getCAxes().index(cax)]
                    for cax in output_caxes]
        result.data = result.data.transpose(neworder)
        result.dims = new_dims

    result.addHistory(command)
    status = pyferret.stop()
    if verbose:
        if status:
            print "PyFerret stopped."
        else:
            print "PyFerret failed to stop."
    return result



def run_worker(f):
    ''' A workaround for clearing memory used by PyFerret
    '''
    import multiprocessing
    @wraps(f)
    def run_func(*args, **kwargs):
        P = multiprocessing.Pool(1)
        result = P.apply(f, args, kwargs)
        P.close()
        P.terminate()
        P.join()
        return result
    return run_func


def regrid_once(var, ref_var, axis,
                verbose=False, prerun=None, transform='@lin'):
    ''' Now only deal with regridding without the time axis
    Input:
    var (geodat.nc.Variable) : whose data will be regridded onto the grid
                             given by ref_var
    ref_var (geodat.nc.Variable): supplies the grid for regridding
    axis (str) :      - the axis for regridding, e.g. "X","Y","XY"
    verbose (bool) : whether to print progress (default: False)
    prerun (a list of strings) : commands to be run at the start (default: None)
    transform (str): "@lin" (DEFAULT: Linear interpolation) or
                     "@ave" (Conserve area average),...see Ferret doc

    Returns
    geodat.nc.Variable
    '''
    pyferret.start(quiet=True, journal=verbose,
                   verify=False, server=True)
    # commands to run before regridding
    if prerun is not None:
        if type(prerun) is str:
            pyferret.run(prerun)
        elif type(prerun) is list:
            for s in prerun:
                if type(s) is str:
                    pyferret.run(prerun)
                else:
                    raise Exception("prerun has to be either a string or "+\
                                    "a list of string")
        else:
            raise Exception("prerun has to be either a string or a list of"+\
                            "string")

    # Make sure axis is a string denoting X or Y axis
    # In case it is a integer, read the cartesian axis for that axis
    if type(axis) is not str:
        if numpy.iterable(axis) == 0:
            axis = var.getCAxes()[axis]
        else:
            axis = ''.join([var.getCAxes()[ax] for ax in axis])

    axis = axis.upper()
    # Construct the dictionary read by pyferret.putdata
    source_fer = Var2Fer(var, name='source')
    dest_fer = Var2Fer(ref_var, name='dest')
    if verbose:
        print source_fer
        print dest_fer
    pyferret.putdata(source_fer, axis_pos=source_fer['axis_pos'])
    if verbose:
        print "Put source variable"
        pyferret.run('show grid '+var.varname)
    pyferret.putdata(dest_fer, axis_pos=dest_fer['axis_pos'])
    if verbose:
        print "Put destination variable"
        pyferret.run('show grid '+ref_var.varname)

    pyfer_command = 'let result = source[g'+axis.lower()+'=dest'+transform+']'
    pyferret.run(pyfer_command)
    if verbose:
        print "Regridded in FERRET"
        pyferret.run('show grid result')

    # Get results
    result_ref = pyferret.getdata('result')
    if verbose: print "Get data from FERRET"
    # Convert from ferret data structure to geodat.nc.Variable
    tmp_result = Fer2Var(result_ref)
    tmp_result.varname = var.varname
    # Preserve dimension order (Ferret reverts the order)
    neworder = [tmp_result.getCAxes().index(cax)
                for cax in var.getCAxes()]
    dims = [tmp_result.dims[tmp_result.getCAxes().index(cax)]
            if cax in axis
            else var.dims[iax]
            for iax, cax in enumerate(var.getCAxes())]
    # Create the geodat.nc.Variable to be returned
    result = _NC.Variable(
        data=tmp_result.data.transpose(neworder).astype(var.data.dtype),
        dims=dims, parent=var,
        history='Regridding using '+axis+' of '+ref_var.varname)

    status = pyferret.stop()
    if verbose:
        if status:
            print "PyFerret stopped."
        else:
            print "PyFerret failed to stop."
    return result


def regrid_once_primitive(var, ref_var, axis,
                          verbose=False, prerun=None, transform='@ave'):
    ''' A generic function that regrids a variable without the dependence of
    geodat.nc.Variable

    Input:
    var (dict) : arguments for Num2Fer
                 Required keys: data,coords,dimunits
    ref_var (dict)  :  arguments for Num2Fer.
                       This supplies the grid for regridding
                       Required keys: coords,dimunits
    axis (str) : the axis for regridding (2D only), e.g. 'X'/'Y'/'XY'/"YX"
    verbose (bool) : whether to print progress (default: False)
    prerun (a list of str) : commands to be run at the start (default: None)
    transform (str): "@ave" (Conserve area average),
                     "@lin" (Linear interpolation),...see Ferret doc
    Return:
    a dictionary
    '''
    pyferret.start(quiet=True, journal=verbose,
                   verify=False, server=True)
    # commands to run before regridding
    if prerun is not None:
        if type(prerun) is str:
            pyferret.run(prerun)
        elif type(prerun) is list:
            for s in prerun:
                if type(s) is str:
                    pyferret.run(prerun)
                else:
                    raise Exception("prerun has to be either a string or "+\
                                    "a list of string")
        else:
            raise Exception("prerun has to be either a string or a list of "+\
                            "string")

    axis = axis.upper()
    # Make sure axis is a string denoting X or Y axis
    if axis not in ['X', 'Y', 'XY', 'YX']:
        raise Exception("Currently axis can only be X/Y/XY")

    # Construct the source data read by pyferret.putdata
    source_fer = Num2Fer(varname="source", **var)

    # Fill in unnecessary input for Ferret
    if not ref_var.has_key('data'):
        ref_var['data'] = numpy.zeros((1,)*len(ref_var['coords']))

    # Construct the destination data read by pyferret.putdata
    dest_fer = Num2Fer(varname="dest", **ref_var)

    if verbose:
        print source_fer
        print dest_fer
    pyferret.putdata(source_fer, axis_pos=source_fer['axis_pos'])
    if verbose:
        print "Put source variable"
        pyferret.run('show grid source')
    pyferret.putdata(dest_fer, axis_pos=dest_fer['axis_pos'])
    if verbose:
        print "Put destination variable"
        pyferret.run('show grid dest')

    pyfer_command = 'let result = source[g'+axis.lower()+'=dest'+transform+']'
    pyferret.run(pyfer_command)
    if verbose:
        print "Regridded in FERRET"
        pyferret.run('show grid result')

    # Get results
    result_ref = pyferret.getdata('result')
    if verbose: print "Get data from FERRET"
    # Convert from ferret data structure to geodat.nc.Variable
    tmp_result = Fer2Num(result_ref)
    if var.has_key('varname'):
        tmp_result['varname'] = var['varname']
    tmp_caxes = [_NC._assign_caxis_(dimunit)
                 for dimunit in tmp_result['dimunits']]
    var_caxes = [_NC._assign_caxis_(dimunit)
                 for dimunit in var['dimunits']]
    # Preserve dimension order (Ferret reverts the order)
    neworder = [tmp_caxes.index(cax)
                for cax in var_caxes]
    # Change the dimension order of the result to match with the input
    tmp_result['coords'] = [tmp_result['coords'][iax] for iax in neworder]
    tmp_result['dimunits'] = [tmp_result['dimunits'][iax] for iax in neworder]
    if tmp_result.has_key('dimnames'):
        tmp_result['dimnames'] = [tmp_result['dimnames'][iax]
                                  for iax in neworder]
    tmp_result['data'] = tmp_result['data'].transpose(neworder).astype(
        var['data'].dtype)
    # Return the input var with the data and dimensions replaced by
    # the regridded ones
    var.update(tmp_result)
    result = var
    status = pyferret.stop()
    if verbose:
        if status:
            print "PyFerret stopped."
        else:
            print "PyFerret failed to stop."
    return result


regrid = run_worker(regrid_once)
regrid_primitive = run_worker(regrid_once_primitive)

if __name__ == '__main__':
    import scipy.io.netcdf as netcdf
    ncfile_low = netcdf.netcdf_file("land_mask_lowres.nc")
    newvar = dict(data=ncfile_low.variables['land_mask'].data,
                  coords=[ncfile_low.variables[dim].data
                          for dim in ncfile_low.variables['land_mask'].\
                          dimensions],
                  dimunits=[ncfile_low.variables[dim].units
                            for dim in ncfile_low.variables['land_mask'].\
                            dimensions])
    ncfile_high = netcdf.netcdf_file("land_mask_highres.nc")
    var_high = dict(data=ncfile_high.variables['land_mask'].data,
                    coords=[ncfile_high.variables[dim].data
                            for dim in ncfile_high.variables['land_mask'].\
                            dimensions],
                    dimunits=[ncfile_high.variables[dim].units
                              for dim in ncfile_high.variables['land_mask'].\
                              dimensions])
    regridded = regrid_primitive(var_high, newvar, 'XY')
