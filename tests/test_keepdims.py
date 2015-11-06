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
        self.assertAlmostEqual(keepdims.mean(numpy.array(1.)), 1.)

    def test_arbitrary(self):
        def new_func(arr, axis):
            return arr.max(axis=axis)

        # keepdims can be applied to any function that accepts a numpy array
        # and an input `axis`
        func_keepdims = keepdims.keepdims(new_func)
        self.assertTrue(numpy.allclose(func_keepdims(self.ma_ndarray, axis=0),
                                       keepdims.max(self.ma_ndarray, axis=0)))

    def test_progMask(self):
        ''' Test if progMask works as expected '''
        self.assertTrue(keepdims.mean(self.ma_ndarray, axis=[0, 1],
                                      progMask=True).mask.all())
        self.assertTrue(keepdims.mean(self.ma_ndarray, axis=2,
                                      progMask=True)[1, 1].squeeze() \
                        is numpy.ma.masked)
        masked = keepdims.mean(self.ma_ndarray, axis=1, progMask=True)
        self.assertTrue(masked[1].mask.all())
        self.assertFalse(masked[0].mask.any())
        self.assertFalse(masked[2].mask.any())

