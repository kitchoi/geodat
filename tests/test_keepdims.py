from __future__ import print_function

import unittest

import numpy

import geodat.keepdims as keepdims


class KeepdimsTestCase(unittest.TestCase):
    ''' Test case for geodat.keepdims '''
    def setUp(self):
        self.ndarray = numpy.arange(60).reshape(3, 4, 5)
        self.ma_ndarray = numpy.ma.arange(60).reshape(3, 4, 5)
        self.ma_ndarray[1, 1, :] = numpy.ma.masked

    def test_mean(self):
        self.assertAlmostEqual(float(keepdims.mean(self.ndarray)), 29.5)
        self.assertAlmostEqual(float(keepdims.mean(self.ma_ndarray)), 29.727272,
                               places=5)
        self.assertTupleEqual(keepdims.mean(self.ndarray, axis=0).shape,
                              (1, 4, 5))
        self.assertTupleEqual(keepdims.mean(self.ma_ndarray, axis=0).shape,
                              (1, 4, 5))
        self.assertTupleEqual(keepdims.mean(self.ndarray, axis=[0, 1]).shape,
                              (1, 1, 5))
        self.assertTupleEqual(keepdims.mean(self.ma_ndarray, axis=[0, 1]).shape,
                              (1, 1, 5))
        self.assertTupleEqual(keepdims.mean(self.ndarray, axis=[0, 1, 2]).shape,
                              (1, 1, 1))
        self.assertTupleEqual(keepdims.mean(self.ma_ndarray,
                                            axis=[0, 1, 2]).shape,
                              (1, 1, 1))
