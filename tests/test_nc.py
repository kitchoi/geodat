import os
import importlib

import unittest
import numpy
import functools
import itertools
import urllib

import geodat.nc


def var_dummy(ntime,nlat,nlon):
    month_day = 365.25/12.
    time_dim = geodat.nc.Dimension(data=numpy.arange(394.,ntime*month_day+394.,
                                                     step=month_day,
                                                     dtype=numpy.float),
                                   units="days since 0001-01-01",
                                   dimname="time",
                                   attributes={"calendar":"julian"})
    lon_dim = geodat.nc.Dimension(data=numpy.linspace(0.,360.,nlon,
                                                      endpoint=False),
                                  units="degreeE",
                                  dimname="lon")
    lat_dim = geodat.nc.Dimension(data=numpy.linspace(-90.,90.,nlat),
                                  units="degreeN",
                                  dimname="lat")

    return geodat.nc.Variable(data=numpy.arange(float(ntime*nlat*nlon)).\
                              reshape(ntime,nlat,nlon),
                              dims=[time_dim,lat_dim,lon_dim],
                              varname="temp")



def is_module_exist(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def expectImportErrorUnlessModuleExists(module_name):
    ''' Runtime dependencies will lead to ImportError
    The library is supposed to check that and notify the
    user that a dependency is required for a function
    '''
    def new_test_func(func):
        def new_func(testcase, *args, **kwargs):
            try:
                func(testcase, *args, **kwargs)
            except ImportError as ierr:
                print module_name+" cannot be imported and an ImportError "+\
                    "is expected to be raised by the code.  The module name "+\
                    "should be included in the error message"
                testcase.assertRaisesRegexp(ierr, module_name)
        return new_func

    try:
        importlib.import_module(module_name)
    except ImportError:
        return new_test_func
    return lambda func: func


class NCVariableTestCase(unittest.TestCase):
    def setUp(self):
        " Create a geodat.nc.Variable for testing "
        self.ntime = 24
        self.nlat = 50
        self.nlon = 60
        self.var = var_dummy(self.ntime, self.nlat, self.nlon)

    def test_getCAxes(self):
        " Test if the cartesian axes can be identified using the units "
        self.assertEqual(self.var.getCAxes(), ["T", "Y", "X"])

    def test_getAxis(self):
        " Test if the axis/axes can be retrieved properly "
        self.assertTrue(numpy.allclose(self.var.getAxis("Y"),
                                       numpy.linspace(-90., 90., self.nlat)))
        self.assertTrue(numpy.allclose(self.var.getAxis("X"),
                                       numpy.linspace(0., 360., self.nlon,
                                                      endpoint=False)))
        self.assertTrue(numpy.allclose(
            self.var.getAxis("Y"), self.var.getLatitude()))
        self.assertTrue(numpy.allclose(
            self.var.getAxis("X"), self.var.getLongitude()))
        self.assertEqual(len(self.var.getAxes()), 3)

    def test_sliceVar(self):
        " geodat.nc.Variable can be sliced like a numpy array object "
        self.assertEqual(self.var[:2, :3, :4].data.shape, (2, 3, 4))

        sliced_var = self.var[2, :4, 4]

        # Make sure the number of dimensions and their sizes matches
        self.assertTrue(sliced_var.is_shape_matches_dims())

        # Singlet dimension should be maintained as well
        self.assertTrue(numpy.allclose(
            sliced_var.data,
            self.var.data[2, :4, 4][numpy.newaxis, ..., numpy.newaxis]))

        # Make sure the dimension values are as expected
        self.assertEqual(sliced_var.getTime(),
                         self.var.getTime()[2])
        self.assertEqual(sliced_var.getLongitude(),
                         self.var.getLongitude()[4])
        self.assertTrue(numpy.allclose(sliced_var.getLatitude(),
                                       self.var.getLatitude()[:4]))

    def test_getSlice(self):
        self.assertEqual(self.var.getSlice(latitude=(-45.,45.),
                                           longitude=(100.,200.)),
                         (slice(None, None, None),
                          slice(13, 37, None),
                          slice(17, 34, None)))

    def test_getRegion(self):
        self.assertEqual(self.var.getRegion(latitude=(-45., 45.),
                                            longitude=(100., 200.)).data.shape,
                         (24, 24, 17))

    def test_getRegion_no_side_effect(self):
        newvar = self.var.getRegion(latitude=(-45.,45.),
                                    longitude=(100.,200.))
        self.assertNotEqual(newvar.data.shape, self.var.data.shape)

    @expectImportErrorUnlessModuleExists("netCDF4")
    def test_getDate_month(self):
        months_iter = itertools.cycle(range(1,13))
        months = numpy.array([ months_iter.next()
                               for _ in range(self.var.data.shape[0])])
        self.assertTrue((self.var.getDate("m",True) == months).\
                        all())

    def test_getDate_monthly_unit(self):
        time_dim = geodat.nc.Dimension(
            data=numpy.arange(0, 24),
            units="month since 0001-01-01",
            dimname="time",
            attributes={"calendar":"standard"})
        month_iter = itertools.cycle(range(1,13))
        dates = numpy.array([[imon/12+1, month_iter.next()]
                             for imon in range(time_dim.data.size)])
        self.assertTrue(numpy.allclose(time_dim.getDate("Ym"), dates))

    def test_wgtave(self):
        self.assertAlmostEqual(float(self.var.wgt_ave().data),
                               35999.5,1)

    def test_timeave(self):
        self.assertTrue(numpy.allclose(
            self.var[..., :3, :4].time_ave().data.squeeze(),
            self.var.data[..., :3, :4].mean(axis=0)))

    def test_squeeze(self):
        self.assertEqual(self.var[0, :3, :4].squeeze().data.shape, (3, 4))

    def test_info(self):
        try:
            self.var.info();
            self.var.info(True);
        except Exception:
            self.assertTrue(False, "Failed to print info")
        self.assertTrue(True)

    def test_add_scalar(self):
        self.assertTrue(numpy.allclose((self.var + 2).data,
                                       self.var.data + 2))

    def test_radd_scalar(self):
        self.assertTrue(numpy.allclose((2 + self.var).data,
                                       self.var.data + 2))

    def test_sub_scalar(self):
        self.assertTrue(numpy.allclose((self.var - 2).data,
                                       self.var.data - 2))
    def test_rsub_scalar(self):
        self.assertTrue(numpy.allclose((2 - self.var).data,
                                       2 - self.var.data))

    def test_mul_scalar(self):
        self.assertTrue(numpy.allclose((self.var * 2).data,
                                       self.var.data * 2))

    def test_rmul_scalar(self):
        self.assertTrue(numpy.allclose((2 * self.var).data,
                                       self.var.data * 2))

    def test_div_scalar(self):
        self.assertTrue(numpy.allclose((self.var / 2).data,
                                       self.var.data / 2))
    def test_rdiv_scalar(self):
        self.assertTrue(numpy.allclose((2 / self.var).data,
                                       2 / self.var.data))

    def test_add_broadcast(self):
        self.assertTrue(numpy.allclose((self.var.time_ave() + self.var).data,
                                       self.var.data.mean(axis=0) + \
                                       self.var.data))

    @expectImportErrorUnlessModuleExists("netCDF4")
    def test_timeslices(self):
        nNDJF = sum(numpy.logical_or(self.var.getDate("m",True) >= 11,
                                     self.var.getDate("m",True) <= 2))
        self.assertEqual(geodat.nc.TimeSlices(self.var[:, :2, :3],
                                              11., 2., "m", True).data.shape[0],
                         nNDJF)

    @expectImportErrorUnlessModuleExists("netCDF4")
    def test_climatology(self):
        clim = geodat.nc.climatology(self.var)
        self.assertTrue(clim.dims[clim.getCAxes().index("T")].is_climo())

    @expectImportErrorUnlessModuleExists("netCDF4")
    def test_anomaly(self):
        clim = geodat.nc.climatology(self.var)
        anom = geodat.nc.anomaly(self.var, clim=clim)
        self.assertTrue(numpy.allclose(
            (geodat.nc.clim2long(clim,anom) + anom).data,
            self.var.data))

    def test_concatenate(self):
        newvar = geodat.nc.concatenate([self.var[4], self.var[5]])
        self.assertTrue(numpy.allclose(newvar.getTime(),
                                       self.var.getTime()[4:6]))

    def test_conform_region(self):
        regional = self.var.getRegion(lat=(-40.,40.),lon=(100.,200.))
        conform_region = geodat.nc.conform_region(self.var, regional)
        regional_domain = regional.getDomain("XY")
        self.assertTrue(conform_region["lat"] == regional_domain["Y"] and \
                        conform_region["lon"] == regional_domain["X"])

    @expectImportErrorUnlessModuleExists("pyferret")
    def test_conform_regrid(self):
        regional = self.var.getRegion(lat=(-40.,40.),lon=(100.,200.))
        conformed, regional = geodat.nc.conform_regrid(self.var, regional)
        self.assertTrue(numpy.allclose(conformed.getLatitude(),
                                       regional.getLatitude()) and \
                        numpy.allclose(conformed.getLongitude(),
                                       regioinal.getLongitude()))

    def test_is_monotonic(self):
        newvar = self.var.getRegion(lon=(100.,260.))
        self.assertTrue(newvar.dims[newvar.getCAxes().index("X")].is_monotonic())

    @expectImportErrorUnlessModuleExists("netCDF4")
    def test_savefile(self):
        geodat.nc.savefile("test_nc_file.nc",self.var)
        tmp = geodat.nc.getvar("test_nc_file.nc","temp")
        self.assertTrue(numpy.allclose(self.var.data,tmp.data))
        if os.path.exists("test_nc_file.nc"):
            os.remove("test_nc_file.nc")

    @expectImportErrorUnlessModuleExists("spharm")
    def test_regrid_spharm(self):
        regridded = geodat.nc.regrid(self.var,nlat=100,nlon=120)
        self.assertAlmostEqual(float(regridded.wgt_ave().data),
                               17999.5,1)

    @expectImportErrorUnlessModuleExists("pyferret")
    def test_regrid_pyferret(self):
        regridded = geodat.nc.pyferret_regrid(self.var[...,::2,::2], self.var)
        self.assertAlmostEqual(float(regridded.wgt_ave().data),
                               17999.0,1)

if __name__== "__main__":
    unittest.main(verbosity=2)
