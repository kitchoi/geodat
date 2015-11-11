import unittest
import numpy
import itertools

import geodat.monthly

class Monthly_TestCase(unittest.TestCase):
    def test_filter_monthly(self):
        # This time axis, each time stamp is close to the beginning of each Month
        # Dates: JAN 1, JAN 31, MAR 1, MAR 31...
        # Correct by shifting the time axis forward
        time = numpy.array([ 0., 29.5, 59., 89.5, 120., 150.5, 181. ,  212. ,
                             242.5, 273., 303.5, 334., 365., 394.5, 424., 454.5,
                             485., 515.5, 546., 577., 607.5, 638., 668.5, 699. ])
        unit = "days since 0001-01-01"
        calendar = "standard"
        month_iter = itertools.cycle(range(1, 13))
        expected = tuple((month_iter.next()
                          for imon in range(len(time))))
        actual = geodat.monthly.filter_monthly(time, unit, calendar)
        self.assertTupleEqual(tuple(actual), expected)

        # This time axis is close to the end of each month
        # Dates: JAN 31, MAR 2, APR 1, MAY 1,...
        
