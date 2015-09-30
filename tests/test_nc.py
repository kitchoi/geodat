import os

import unittest
import numpy
import functools
import itertools

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


def skipUnlessNetCDF4Exists():
    try:
        import netCDF4
    except ImportError:
        return unittest.skip("Cannot load netCDF4")
    return lambda func: func


def skipUnlessSpharmExists():
    try:
        import spharm
    except ImportError:
        return unittest.skip("Cannot load spharm")
    return lambda func: func


def skipUnlessPyFerretExists():
    try:
        import pyferret
    except ImportError:
        return unittest.skip("Cannot load pyferret")
    return lambda func: func


class NCVariableTestCase(unittest.TestCase):
    def setUp(self):
        self.var = var_dummy(24,50,60)
        self.__name__ = "temp"

    def test_getvar_data(self):
        self.assertEqual(self.var.data.shape,(24,50,60))
    
    def test_getCAxes(self):
        self.assertEqual(self.var.getCAxes(),["T","Y","X"])
    
    def test_sliceVar(self):
        self.assertEqual(self.var[:2,:3,:4].data.shape,(2,3,4))

    def test_getDate_month(self):
        months_iter = itertools.cycle(range(1,13))
        months = numpy.array([ months_iter.next() 
                               for _ in range(self.var.data.shape[0])])
        self.assertTrue((self.var.getDate("m",True) == months).\
                        all())

    def test_wgtave(self):
        self.assertAlmostEqual(float(self.var.wgt_ave().data),
                               35999.5,1)
    
    def test_timeave(self):
        self.assertEqual(self.var[...,:3,:4].time_ave().data.shape,(1,3,4))

    def test_add_scalar(self):
        self.assertTrue(numpy.allclose((self.var + 2).data,
                                       self.var.data + 2))

    def test_sub_scalar(self):
        self.assertTrue(numpy.allclose((self.var - 2).data,
                                       self.var.data - 2))

    def test_mul_scalar(self):
        self.assertTrue(numpy.allclose((self.var * 2).data,
                                       self.var.data * 2))

    def test_div_scalar(self):
        self.assertTrue(numpy.allclose((self.var / 2).data,
                                       self.var.data / 2))

    def test_add_broadcast(self):
        self.assertTrue(numpy.allclose((self.var.time_ave() + self.var).data,
                                       self.var.data.mean(axis=0) + \
                                       self.var.data))

    @skipUnlessNetCDF4Exists()
    def test_savefile(self):
        geodat.nc.savefile("test_nc_file.nc",self.var)
        tmp = geodat.nc.getvar("test_nc_file.nc","temp")
        self.assertTrue(numpy.allclose(self.var.data,tmp.data))
        if os.path.exists("test_nc_file.nc"):
            os.remove("test_nc_file.nc")

    @skipUnlessSpharmExists()
    def test_regrid_spharm(self):
        regridded = geodat.nc.regrid(self.var,nlat=100,nlon=120)
        self.assertAlmostEqual(float(regridded.wgt_ave().data),
                               17999.5,1)
    
    @skipUnlessPyFerretExists()
    def test_regrid_pyferret(self):
        regridded = geodat.nc.pyferret_regrid(self.var[...,::2,::2], self.var)
        self.assertAlmostEqual(float(regridded.wgt_ave().data),
                               17999.0,1)


if __name__== "__main__":
    unittest.main(verbosity=2)
