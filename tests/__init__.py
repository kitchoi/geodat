import os.path
import unittest


def full_suite():
    import test_nc
    import test_stat
    import test_keepdims

    return unittest.TestSuite([unittest.TestLoader().loadTestsFromModule(module)
                               for module in [test_nc, test_stat,
                                              test_keepdims]])
