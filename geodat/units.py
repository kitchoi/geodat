def assign_caxis(dimunit):
    '''
    Assign cartesian_axis (T/Z/Y/X) to the axis with identifiable axis units.

    Axes without identifiable units will be set to None

    Args:
        unit (str)

    Returns:
        str
    '''
    assert type(dimunit) is str
    dimunit = dimunit.split()[0]
    conventions = {'T': ['second', 'seconds', 'sec', 'minute', 'minutes', 'min',
                         'hour', 'hours', 'h', 'day', 'days', 'd',
                         'month', 'months', 'mon', 'year', 'years', 'y'],
                   'Z': ['bar', 'millibar', 'decibar', 'atm', 'atmosphere',
                         'pascal', 'Pa', 'hpa',
                         'meters', 'meter', 'm', 'kilometer', 'km', 'density'],
                   'Y': ['degrees_north', 'degree_north', 'degree_n',
                         'degrees_n', 'degreen', 'degreesn'],
                   'X': ['degrees_east', 'degree_east', 'degree_e',
                         'degrees_e', 'degreee', 'degreese']}
    invaxunits = {unit.lower():ax
                  for ax, units in conventions.items()
                  for unit in units}
    return invaxunits.get(dimunit.lower(), None)
