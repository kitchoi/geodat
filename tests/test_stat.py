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

    def test_resample_xy(self):
        '''Test if resample_xy returns an array that has the shape requested.'''
        # Shape is what is tested so far
        # To test for the resemblance of the joint pdf requires larger sample
        # and therefore would slow down tests
        x = numpy.random.random(50)
        y = numpy.random.random(40)
        xnew = numpy.random.random(20)

        with self.assertRaisesRegexp(ValueError, "shape of y should match"):
            ynew = geodat_stat.resample_xy(x, y, xnew, 10, 10)

        y = numpy.random.random(50)
        ynew = geodat_stat.resample_xy(x, y, xnew, 10, 10)
        self.assertTupleEqual(ynew.shape, xnew.shape)


    def test_cdf(self):
        """ Test if the cumulative frequency function works"""
        x = numpy.arange(30)
        a, b = geodat_stat.cdf(x)
        a_ans = numpy.array([ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                              0.7, 0.8, 0.9, 1.0 ])
        b_ans = numpy.array([ 1.45, 4.35, 7.25, 10.15, 13.05,
                              15.95, 18.85, 21.75, 24.65, 27.55 ])
        self.assertTrue(numpy.allclose(a, a_ans, rtol=0.01))
        self.assertTrue(numpy.allclose(b, b_ans, rtol=0.01))


if __name__== "__main__":
    unittest.main(verbosity=2)
