from __future__ import print_function

import os
import importlib
import unittest
import functools
import itertools

import numpy
import urllib

import geodat.nc

from . import misc

def var_dummy(ntime, nlat, nlon):
    ''' Create a geodat.nc.Variable instance of shape (ntime, nlat, nlon)
    with times, latitudes and longitudes already initialised to cover the globe

    Time is a monthly axis with units "days since 0001-01-01", julian calendar

    Returns:
       geodat.nc.Variable
    '''
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
                              reshape(ntime, nlat, nlon),
                              dims=[time_dim, lat_dim, lon_dim],
                              varname="temp")


class DummyVariable(object):
    ''' Empty object (not inheriting from Variable) that has `varname`, `dims`,
    `attributes` as its attributes

    Use for testing the parent argument in initialising geodat.nc.Variable
    '''
    def __init__(self, varname=None, dims=None, attributes=None):
        if varname is not None:
            self.varname = varname
        if dims is not None:
            self.dims = dims
        if attributes is not None:
            self.attributes = attributes


class NCVariableTestCase(unittest.TestCase):
    ''' Test case for the geodat.nc.Variable'''
    TMP_FILE_NAME = "test_nc_tmp.nc"

    def setUp(self):
        ''' Create a geodat.nc.Variable for testing '''
        self.ntime = 24
        self.nlat = 50
        self.nlon = 60
        self.var = var_dummy(self.ntime, self.nlat, self.nlon)
        self.var_mean = self.var.data.mean()

    def tearDown(self):
        # Delete temporary file if it exists
        if os.path.exists(self.TMP_FILE_NAME):
            os.remove(self.TMP_FILE_NAME)

    def test_var_setup(self):
        ''' Test the other ways of creating a Variable '''
        # Inherit from a parent with new data
        newvar = geodat.nc.Variable(data=numpy.ones_like(self.var.data),
                                    parent=self.var)
        self.assertEqual(newvar.varname, self.var.varname)
        self.assertListEqual(newvar.dims, self.var.dims)
        self.assertDictEqual(newvar.attributes, self.var.attributes)

        # Inherit from a parent but update attributes
        newvar = geodat.nc.Variable(data=numpy.ones_like(self.var.data),
                                    parent=self.var,
                                    attributes=dict(is_new="yes"))
        self.assertEqual(newvar.varname, self.var.varname)
        self.assertListEqual(newvar.dims, self.var.dims)
        newvar.attributes.pop("is_new");
        self.assertDictEqual(newvar.attributes, self.var.attributes)

        # Inherit from a parent with new data, should complain about wrong shape
        with self.assertRaisesRegexp(ValueError, "Dimension mismatch"):
            newvar = geodat.nc.Variable(data=numpy.array(1.), parent=self.var)

        # dimension is not provided
        with self.assertRaises(AttributeError):
            newvar = geodat.nc.Variable(data=numpy.array([1,1,1]),
                                        varname="tmp")

        # parent can be anything with the right attributes
        newparent = DummyVariable(varname="tmp", dims=self.var.dims,
                          attributes=dict(newparent="yes"))
        newvar = geodat.nc.Variable(data=numpy.ones_like(self.var.data),
                                    parent=newparent)
        self.assertDictEqual(newvar.attributes, newparent.attributes)


        def complain_about_parent(parent, err):
            with self.assertRaises(err):
                newvar = geodat.nc.Variable(data=numpy.empty(10),
                                            parent=parent)

        # if parent does not have any of the (varname/dims/attributes)
        # complain!
        complain_about_parent(DummyVariable(varname="tmp", dims=self.var.dims),
                              AttributeError)
        complain_about_parent(DummyVariable(varname="tmp",
                                            attributes=self.var.attributes),
                              AttributeError)
        complain_about_parent(DummyVariable(dims=self.var.dims,
                                            attributes=self.var.attributes),
                              AttributeError)

        # if the attributes are of the wrong type, complain too
        complain_about_parent(DummyVariable(varname=1, dims=self.var.dims,
                                            attributes=self.var.attributes),
                              TypeError)
        complain_about_parent(DummyVariable(varname="tmp", dims=self.var.dims[0],
                                            attributes=self.var.attributes),
                              TypeError)
        complain_about_parent(DummyVariable(varname="tmp", dims=[1,2,3],
                                            attributes=self.var.attributes),
                              TypeError)
        complain_about_parent(DummyVariable(varname="tmp", dims=self.var.dims,
                                            attributes="bad_attribute"),
                              TypeError)


    def test_getCAxes(self):
        ''' Test if the cartesian axes can be identified using the units '''
        self.assertEqual(self.var.getCAxes(), ["T", "Y", "X"])


    def test_getAxis(self):
        " Test if the axis/axes can be retrieved properly "

        # Using property
        self.assertTrue(numpy.allclose(self.var.time,
                                       numpy.arange(394.,
                                                    self.ntime*365.25/12.+394.,
                                                    step=365.25/12.,
                                                    dtype=numpy.float)))
        self.assertTrue(numpy.allclose(self.var.lat,
                                       numpy.linspace(-90., 90., self.nlat)))
        self.assertTrue(numpy.allclose(self.var.lon,
                                       numpy.linspace(0., 360., self.nlon,
                                                      endpoint=False)))

        # Using getAxis
        self.assertTrue(numpy.allclose(self.var.getAxis("T"),
                                       numpy.arange(394.,
                                                    self.ntime*365.25/12.+394.,
                                                    step=365.25/12.,
                                                    dtype=numpy.float)))
        self.assertTrue(numpy.allclose(self.var.getAxis("Y"),
                                       numpy.linspace(-90., 90., self.nlat)))
        self.assertTrue(numpy.allclose(self.var.getAxis("X"),
                                       numpy.linspace(0., 360., self.nlon,
                                                      endpoint=False)))

        self.assertEqual(len(self.var.getAxes()), 3)


    def test_sliceVar(self):
        '''Test if geodat.nc.Variable can be sliced like a numpy array object'''
        self.assertEqual(self.var[:2, :3, :4].data.shape, (2, 3, 4))

        sliced_var = self.var[...,2]

        # Make sure the number of dimensions and their sizes matches
        self.assertTrue(sliced_var.is_shape_matches_dims())

        # Singlet dimension should be maintained as well
        self.assertTrue(numpy.allclose(
            sliced_var.data,
            self.var.data[...,2][..., numpy.newaxis]))

        sliced_var = self.var[...,2,:4]
        # Make sure the dimension values are as expected
        self.assertTrue(numpy.allclose(sliced_var.getTime(),
                                        self.var.getTime()))
        self.assertEqual(sliced_var.getLatitude(),
                         self.var.getLatitude()[2])
        self.assertTrue(numpy.allclose(sliced_var.getLongitude(),
                                       self.var.getLongitude()[:4]))


    def test_getSlice(self):
        '''Test if slice objects can be created properly using lat-lon ranges'''
        self.assertEqual(self.var.getSlice(latitude=(-45., 45.),
                                           longitude=(100., 200.)),
                         (slice(None, None, None),
                          slice(13, 37, None),
                          slice(17, 34, None)))

    def test_getRegion(self):
        '''Test if the sliced Variable has the right shape'''
        # Simpliest case
        self.assertTupleEqual(self.var.getRegion(lon=(100., 200.)).data.shape,
                              (24, 50, 17))

        # Negative longitude is taken care of
        self.assertTupleEqual(self.var.getRegion(lon=(100., -160.)).data.shape,
                              (24, 50, 17))

        # Cross the meridian
        self.assertTupleEqual(self.var.getRegion(lon=(-10., 10.)).data.shape,
                              (24, 50, 3))

        # One-sided
        self.assertTupleEqual(self.var(lat=(-45., 90.)).data.shape,
                              (24, 36, 60))

        # All within range
        self.assertTupleEqual(self.var(lat=(-90., 90.)).data.shape,
                              (24, 50, 60))

        # Out of range
        with self.assertRaisesRegexp(ValueError, "Not found"):
            self.var.getRegion(lat=(-100., -95.))

        with self.assertRaisesRegexp(ValueError, "Not found"):
            self.var.getRegion(lon=-10.)

        with self.assertRaisesRegexp(ValueError, "Not found"):
            self.var.getRegion(lat=(70., 71.))

        with self.assertRaisesRegexp(ValueError, "Not found"):
            self.var.getRegion(lat=(100., -100.))

        # longitude is rearranged
        self.var.dims[self.var.getIAxis("X")].data = numpy.hstack(
            (self.var.lon[self.nlon/2:], self.var.lon[:self.nlon/2]))

        # Cross the meridian
        self.assertTupleEqual(self.var.getRegion(lon=(-10., 10.)).data.shape,
                              (24, 50, 3))

        # Negative longitude is taken care of
        self.assertTupleEqual(self.var.getRegion(lon=(100., -160.)).data.shape,
                              (24, 50, 17))
        # Out of range
        with self.assertRaisesRegexp(ValueError, "Not found"):
            self.var.getRegion(lat=(-100., -95.))

        with self.assertRaisesRegexp(ValueError, "Not found"):
            self.var.getRegion(lon=-10.)

        with self.assertRaisesRegexp(ValueError, "Not found"):
            self.var.getRegion(lat=(70., 71.))

        with self.assertRaisesRegexp(ValueError, "Not found"):
            self.var.getRegion(lat=(100., -100.))

        # Go around the latitude circle twice, won't know what to do
        var_double_lon = geodat.nc.concatenate((self.var, self.var),
                                               self.var.getIAxis("X"))
        with self.assertRaisesRegexp(RuntimeError, "Too many chunks"):
            var_double_lon.getRegion(lon=(100., 120.))


    def test_getRegion_no_side_effect(self):
        '''getRegion should not have side effect'''
        newvar = self.var.getRegion(latitude=(-45.,45.),
                                    longitude=(100.,200.))
        self.assertNotEqual(newvar.data.shape, self.var.data.shape)


    def test_getDate(self):
        '''Test if getDate functions properly using standard calendar'''
        time_dim1 = geodat.nc.Dimension(
            data=numpy.arange(0, 24),
            units="month since 0001-01-01",
            dimname="time",
            attributes={"calendar":"standard"})

        month_iter = itertools.cycle(range(1,13))
        dates = numpy.array([[imon/12+1, month_iter.next()]
                             for imon in range(time_dim1.data.size)])

        # Should work on different format of monthly unit
        self.assertTrue(numpy.allclose(time_dim1.getDate("Ym"), dates))

        # Throw an error when the dimension is not a time axis
        with self.assertRaisesRegexp(RuntimeError, "not a time axis"):
            self.var.dims[1].getDate()

        # Throw an error when toggle is not one of Y/m/d/H/M/S
        with self.assertRaisesRegexp(ValueError, "toggle has to be one of"):
            time_dim1.getDate("E")

        # Throw an error when toggle is not iterable
        with self.assertRaisesRegexp(TypeError, "toggle has to be iterable"):
            time_dim1.getDate(0)

        # Throw an error if the unit format is wrong
        time_dim1.units = "month"
        with self.assertRaisesRegexp(Exception, "should have a unit such as"):
            time_dim1.getDate()


    def test_wgtave(self):
        self.assertAlmostEqual(float(self.var.wgt_ave().data), self.var_mean, 1)


    def test_timeave(self):
        self.assertTrue(numpy.allclose(
            self.var[..., :3, :4].time_ave().data.squeeze(),
            self.var.data[..., :3, :4].mean(axis=0)))


    def test_squeeze(self):
        self.assertEqual(self.var[0, :3, :4].squeeze().data.shape, (3, 4))


    def test_info(self):
        with open(os.devnull, "w") as stdnull:
            self.var.info(file_out=stdnull);
            self.var.info(True, file_out=stdnull);
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
        self.assertTrue(numpy.allclose((2. / (self.var + 1.)).data,
                                       2. / (self.var.data + 1.)))


    def test_add_broadcast(self):
        self.assertTrue(numpy.allclose((self.var.time_ave() + self.var).data,
                                       self.var.data.mean(axis=0) + \
                                       self.var.data))


    @misc.expect_import_error_unless_module_exists("netCDF4")
    def test_timeslices(self):
        nNDJF = sum(numpy.logical_or(self.var.dims[0].getMonthly() >= 11,
                                     self.var.dims[0].getMonthly() <= 2))
        self.assertEqual(geodat.nc.TimeSlices(self.var[:, :2, :3],
                                              11., 2., "m", True).data.shape[0],
                         nNDJF)


    @misc.expect_import_error_unless_module_exists("netCDF4")
    def test_climatology(self):
        clim = geodat.nc.climatology(self.var)
        self.assertTrue(clim.dims[clim.getCAxes().index("T")].is_climo())


    @misc.expect_import_error_unless_module_exists("netCDF4")
    def test_anomaly(self):
        clim = geodat.nc.climatology(self.var)
        anom = geodat.nc.anomaly(self.var, clim=clim)
        self.assertTrue(numpy.allclose(
            (geodat.nc.clim2long(clim,anom) + anom).data,
            self.var.data))


    def test_concatenate(self):
        newvar = geodat.nc.concatenate([self.var[4], self.var[5:8]])
        self.assertTrue(numpy.allclose(newvar.getTime(),
                                       self.var.getTime()[4:8]))


    def test_conform_region(self):
        regional = self.var.getRegion(lat=(-40.,40.),lon=(100.,200.))
        conform_region = geodat.nc.conform_region(self.var, regional)
        regional_domain = regional.getDomain("XY")
        self.assertTupleEqual(conform_region["lat"], regional_domain["Y"])
        self.assertTupleEqual(conform_region["lon"], regional_domain["X"])


    @misc.expect_import_error_unless_module_exists("pyferret")
    def test_conform_regrid(self):
        regional = self.var.getRegion(lat=(-40.,40.), lon=(100.,200.))
        conformed, regional = geodat.nc.conform_regrid(self.var, regional)
        self.assertTrue(numpy.allclose(conformed.getLatitude(),
                                       regional.getLatitude()) and \
                        numpy.allclose(conformed.getLongitude(),
                                       regional.getLongitude()))


    def test_is_monotonic(self):
        newvar = self.var.getRegion(lon=(100.,260.))
        self.assertTrue(newvar.dims[newvar.getCAxes().index("X")].is_monotonic())
        # Special case when longitude can be negative passed the date line
        lon_data = newvar.getDim("X").data
        lon_data[lon_data>180.]-=360.
        self.assertTrue(newvar.getDim("X").is_monotonic())


    @misc.expect_import_error_unless_module_exists("netCDF4")
    def test_savefile(self):
        geodat.nc.savefile(self.TMP_FILE_NAME, self.var)
        geodat.nc.savefile(self.TMP_FILE_NAME, self.var, overwrite=True)
        geodat.nc.savefile(self.TMP_FILE_NAME, self.var[0:2], recordax=0,
                           overwrite=True)
        geodat.nc.savefile(self.TMP_FILE_NAME, self.var[2:], recordax=0,
                           appendall=True)
        tmp_var = geodat.nc.getvar(self.TMP_FILE_NAME, self.var.varname)

        # Verify
        self.assertTrue(tmp_var.data.shape, self.var.data.shape)
        self.assertTrue(numpy.allclose(tmp_var.data, self.var.data))

        geodat.nc.savefile(self.TMP_FILE_NAME, self.var, overwrite=True)
        newvar = self.var[10:]
        newvar.varname="temp2"
        geodat.nc.savefile(self.TMP_FILE_NAME, newvar, appendall=True)
        tmp_var = geodat.nc.getvar(self.TMP_FILE_NAME, self.var.varname)
        tmp_var2 = geodat.nc.getvar(self.TMP_FILE_NAME, "temp2")

        # Verify
        self.assertTrue(tmp_var.data.shape, self.var.data.shape)
        self.assertTrue(tmp_var.varname, "temp")
        self.assertTrue(tmp_var.getDimnames()[0], "time")

        self.assertTrue(tmp_var2.data.shape, self.var.data[10:].shape)
        self.assertTrue(tmp_var2.varname, "temp2")
        self.assertTrue(tmp_var2.getDimnames()[0], "time1")

        # Clean up
        if os.path.exists(self.TMP_FILE_NAME):
            os.remove(self.TMP_FILE_NAME)

    @misc.expect_import_error_unless_module_exists("spharm")
    def test_regrid_spharm(self):
        regridded = geodat.nc.regrid(self.var,nlat=100,nlon=120)
        self.assertAlmostEqual(float(regridded.wgt_ave().data),
                               float(self.var.wgt_ave().data), 1)


    @misc.expect_import_error_unless_module_exists("pyferret")
    def test_regrid_pyferret(self):
        regridded = geodat.nc.pyferret_regrid(self.var, nlat=100, nlon=120)
        actual = float(regridded.wgt_ave().data)
        expected = float(self.var.wgt_ave().data)
        self.assertAlmostEqual(actual, expected, 1)

        # Regrid back
        regridded = geodat.nc.pyferret_regrid(regridded, self.var)
        self.assertAlmostEqual(actual, expected, 1)


    def test_wgt_sum(self):
        self.assertTrue(numpy.allclose(geodat.nc.wgt_sum(self.var).data,
                                       numpy.array([[[1.62e+09]]]), rtol=0.01))
        self.assertTrue(numpy.allclose(
            geodat.nc.wgt_sum(self.var, axis=[1,2]).data[0],
            numpy.array([[2805596.]]), rtol=0.01))

    def test_runave(self):
        expected = numpy.ma.array([numpy.ma.masked, numpy.ma.masked,
                                   7530.0, 10530.0, 13530.0, 16530.0,
                                   19530.0, 22530.0, 25530.0, 28530.0,
                                   31530.0, 34530.0, 37530.0, 40530.0,
                                   43530.0, 46530.0, 49530.0, 52530.0,
                                   55530.0, 58530.0, 61530.0, 64530.0,
                                   numpy.ma.masked, numpy.ma.masked])
        actual = self.var.runave(5, 0).data[:, 25, 30]
        self.assertTrue(numpy.ma.allclose(expected, actual))

        # Same results using absolute axis spacing
        actual = self.var.runave(150., "T").data[:, 25, 30]
        self.assertTrue(numpy.ma.allclose(expected, actual))

        expected = numpy.ma.array([numpy.ma.masked, numpy.ma.masked,
                                   numpy.ma.masked, numpy.ma.masked,
                                   13530.0, 16530.0, 19530.0, 22530.0,
                                   25530.0, 28530.0, 31530.0, 34530.0,
                                   37530.0, 40530.0, 43530.0, 46530.0,
                                   49530.0, 52530.0, 55530.0, 58530.0,
                                   numpy.ma.masked, numpy.ma.masked,
                                   numpy.ma.masked, numpy.ma.masked])
        actual = self.var.runave(150., "T", step=2).data[:, 25, 30]
        self.assertTrue(numpy.ma.allclose(expected, actual))

        with self.assertRaisesRegexp(Exception, "step should be an integer"):
            actual = self.var.runave(150., "T", step=2.).data[:, 25, 30]


    def test_ensemble(self):
        """ Test the generation of ensemble axis"""
        actual = geodat.nc.ensemble([self.var[0].squeeze(),
                                     self.var[1].squeeze()])
        self.assertEqual(actual.data.shape[0], 2)
        self.assertTrue(numpy.allclose(actual.dims[0].data, numpy.arange(1,3)))


    def test_div(self):
        """ Test computing divergence """
        actual = geodat.nc.div(self.var[0, :5, :4].squeeze(),
                               self.var[1, :5, :4].squeeze())
        expected = numpy.ma.empty((5,4), dtype=numpy.float)
        expected[1:-1, 1:-1] = numpy.ma.array(
            [[0.000170287289868, 0.000170287289868],
             [0.000158612374383, 0.000158612374383],
             [0.00015473153082, 0.00015473153082]])

        expected[0, :] = expected[-1, :] = expected[:, 0] = expected[:, -1] = \
                        numpy.ma.masked
        self.assertTrue(numpy.ma.allclose(actual.data, expected, rtol=1e-5))


class NCFileIOTestCase(unittest.TestCase):
    TEST_DATA_NAME = "tests/data/test_nc_data.nc"
    TEST_DATA_SOURCE = "http://kychoi.org/geodat/test_data/sst_parts.nc"

    @property
    def does_test_data_exist(self):
        """Download data if test data does not exist

        Args:
           filename (str): path to the test data

        Returns:
           boolean: True if the data exists or is successfully downloaded
        """
        if not os.path.exists(self.TEST_DATA_NAME):
            print("Test data not found.  Downloading...", end="")
            # Create the test data directory if it does not exist
            if not os.path.exists(os.path.dirname(self.TEST_DATA_NAME)):
                os.makedirs(os.path.dirname(self.TEST_DATA_NAME))

            # Download the data
            try:
                _, response = urllib.urlretrieve(self.TEST_DATA_SOURCE,
                                                 self.TEST_DATA_NAME)
            except IOError:
                print("Failed. IOError during urllib")
                return False

            if "netcdf" not in response.gettype():
                # Failed to download
                if os.path.exists(self.TEST_DATA_NAME):
                    os.remove(self.TEST_DATA_NAME)
                print("Failed")
                return False

            print("OK")  # Completed download
        return True


    def test_openfile(self):
        ''' Test opening file and extracting variables from netCDF files'''

        if not self.does_test_data_exist:
            self.skipTest("Test data does not exist.  Failed to download.")

        # Check if a single variable can be extracted
        var = geodat.nc.getvar(self.TEST_DATA_NAME, "sstMAM")
        self.assertTupleEqual(var.data.shape, (3, 180, 360))

        # Check if all of the variables can be extracted
        dataset = geodat.nc.dataset(self.TEST_DATA_NAME)
        expected = sorted(["sstSON", "sstMAM", "sstDJF", "sstJJA"])
        self.assertListEqual(expected, sorted(dataset.keys()))

        # Test overwriting loaded value
        dataset = geodat.nc.dataset([self.TEST_DATA_NAME,
                                     self.TEST_DATA_NAME], "o")
        self.assertListEqual(expected, sorted(dataset.keys()))

        # Test skipping loaded value
        dataset = geodat.nc.dataset([self.TEST_DATA_NAME,
                                     self.TEST_DATA_NAME], "s")
        self.assertListEqual(expected, sorted(dataset.keys()))

        with self.assertRaisesRegexp(ValueError, "Invalid choice"):
            geodat.nc.dataset([self.TEST_DATA_NAME,
                               self.TEST_DATA_NAME], "v")


if __name__== "__main__":
    unittest.main(verbosity=2)
