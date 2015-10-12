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
        answer = x.copy()
        answer[:2] = answer[-2:] = numpy.ma.masked
        self.assertTrue(numpy.allclose(answer, geodat_stat.runave(x, 5),
                                       rtol=0.1))
        answer[:4] = answer[-4:] = numpy.ma.masked
        self.assertTrue(numpy.allclose(answer, geodat_stat.runave(x, 5, step=2),
                                       rtol=0.1))


    def test_lat_weights(self):
        x = numpy.arange(70.,80.)
        answer = [ 0.342020143326, 0.325568154457, 0.309016994375,
                   0.292371704723, 0.275637355817, 0.258819045103,
                   0.2419218956, 0.224951054344, 0.207911690818,
                   0.190808995377 ]
        self.assertTrue(numpy.allclose(answer, geodat_stat.lat_weights(x),
                                       rtol=0.1))


if __name__== "__main__":
    unittest.main(verbosity=2)
