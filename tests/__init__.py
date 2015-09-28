import os.path
import unittest


def get_tests():
    return full_suite()


def full_suite():
    from .test_nc import NCVariableTestCase

    ncsuite = unittest.TestLoader().loadTestsFromTestCase(NCVariableTestCase)

    return unittest.TestSuite([ncsuite])
