import unittest

import numpy
import scipy.stats as scipy_stats

import geodat.stat as geodat_stat

class StatTestCase(unittest.TestCase):
    def test_skewness(self):
        '''Test if stat.skewness gives the same results as scipy.stats.skew'''
        x = numpy.arange(20.)
        self.assertTrue(numpy.allclose(scipy_stats.skew(x),
                                       geodat_stat.skewness(x)))
        x = numpy.ma.arange(20.)
        x[3] = numpy.ma.masked
        self.assertTrue(numpy.ma.allclose(scipy_stats.mstats.skew(x),
                                          geodat_stat.skewness(x)))

    def test_runave(self):
        x = numpy.arange(20.)
        answer = numpy.ma.arange(20.)
        answer[:2] = answer[-2:] = numpy.ma.masked
        self.assertTrue(numpy.ma.allclose(answer, geodat_stat.runave(x, 5),
                                          rtol=0.1))
        answer[:4] = answer[-4:] = numpy.ma.masked
        self.assertTrue(numpy.ma.allclose(
            answer, geodat_stat.runave(x, 5, step=2), rtol=0.1))


    def test_lat_weights(self):
        x = numpy.arange(70.,80.)
        answer = [ 0.34202, 0.3255, 0.30901,
                   0.29237, 0.2756, 0.25881,
                   0.24192, 0.2249, 0.20791,
                   0.19080 ]
        self.assertTrue(numpy.allclose(answer, geodat_stat.lat_weights(x),
                                       rtol=0.1))


if __name__== "__main__":
    unittest.main(verbosity=2)
