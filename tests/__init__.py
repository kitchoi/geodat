import os.path
import unittest

import test_nc
import test_stat
import test_keepdims
import test_time_utils
import test_monthly

def full_suite():
    return unittest.TestSuite([unittest.TestLoader().loadTestsFromModule(module)
                               for module in [test_nc, test_stat,
                                              test_keepdims, test_time_utils,
                                              test_monthly]])
