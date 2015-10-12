import os.path
import unittest


def get_tests():
    return full_suite()


def full_suite():
    from .test_nc import NCVariableTestCase
    from .test_stat import StatTestCase
    all_suites = []
    all_suites.append(unittest.TestLoader().loadTestsFromTestCase(
        NCVariableTestCase))
    all_suites.append(unittest.TestLoader().loadTestsFromTestCase(
        StatTestCase))
    
    return unittest.TestSuite(all_suites)
