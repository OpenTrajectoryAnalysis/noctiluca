import unittest

from test_core import *
from test_analysis import *
from test_io import *
from test_util import *
from test_plot import *

if __name__ == '__main__':
    unittest.main(module=__file__.split('/')[-1][:-3])
