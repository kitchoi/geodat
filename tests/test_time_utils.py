import unittest

import numpy
import datetime

import geodat.time_utils as time_utils

from . import misc

#-----------------------------------------------------------------
# If netCDF4 is not installed, some functions are not available
# When these functions are called, an Exception will be raised
#-----------------------------------------------------------------
try:
    import netCDF4 as _netCDF4
    _NETCDF4_IMPORTED = True
except ImportError:
    _NETCDF4_IMPORTED = False


def _throw_error(error):
    def new_func(*args,**kwargs):
        raise error
    return new_func

if _NETCDF4_IMPORTED:
    _num2date = _netCDF4.num2date
    _date2num = _netCDF4.date2num
    _netCDF4_Dataset = _netCDF4.Dataset
    _netCDF4_datetime = _netCDF4.netcdftime.datetime
else:
    logger.warning("Failed to import netCDF4 package. "+\
                   "Some functions may not work")
    _NETCDF4_IMPORT_ERROR = ImportError("The netCDF4 package is "+\
                                        "required but fail to import. "+\
                            "See https://pypi.python.org/pypi/netCDF4/0.8.2")
    _num2date = _date2num = _netCDF4_Dataset = \
                _throw_error(_NETCDF4_IMPORT_ERROR)


class Time_utils_TestCase(unittest.TestCase):
    def test_extract_time0(self):
        """ Test if time0 can be read from the unit """
        expected = datetime.datetime(1,1,1)
        actual = time_utils.extract_t0_from_unit("days since 0001-01-01")
        self.assertEqual(actual, expected)
        actual = time_utils.extract_t0_from_unit(
            "days since 0001-01-01 00:00:00")
        self.assertEqual(actual, expected)

    @misc.expect_import_error_unless_module_exists("netCDF4")
    def test_num2date(self):
        """ Test if num2date handles both monthly and non-monthly axis """

        time = numpy.linspace(15., 350., 12)
        # If unit is not month, behaves like netCDF4.num2date
        expected = _num2date(time, "days since 0001-01-01", "julian")
        actual = time_utils.num2date(time, "days since 0001-01-01", "julian")
        self.assertTupleEqual(tuple(actual), tuple(expected))

        # If unit is month, add the months up using relativedelta
        time = numpy.arange(12)
        expected = numpy.array([datetime.datetime(year=1, month=month, day=1)
                                for month in range(1, 13)])
        actual = time_utils.num2date(numpy.arange(12), "months since 0001-01-01",
                                     "standard")
        self.assertTupleEqual(tuple(actual), tuple(expected))
