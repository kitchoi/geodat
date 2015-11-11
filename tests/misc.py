import unittest
import importlib

def expect_import_error_unless_module_exists(module_name):
    ''' Runtime dependencies will lead to ImportError
    The library is supposed to check that and notify the
    user that a dependency is required for a function

    Args:
       module_name (str): Name of the required module

    Returns:
       decorator that accepts a single callable argument
    '''
    def new_test_func(test_func):
        def new_func(testcase, *args, **kwargs):
            with testcase.assertRaisesRegexp(ImportError, module_name):
                test_func(testcase, *args, **kwargs)
            testcase.skipTest(module_name+" cannot be imported. "+\
                              "ImportError raised")
        return new_func

    try:
        importlib.import_module(module_name)
    except ImportError:
        return new_test_func
    return lambda test_func: test_func

