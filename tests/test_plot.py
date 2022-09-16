import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass

import numpy as np
np.seterr(all='raise') # pay attention to details

import unittest
from unittest.mock import patch

from context import noctiluca as nl

"""
exec "norm jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kcc]kV?__all__j>>
"""
__all__ = [
    'TestPlotting',
]

class myTestCase(unittest.TestCase):
    def assert_array_equal(self, array1, array2):
        try:
            np.testing.assert_array_equal(array1, array2)
            res = True
        except AssertionError as err: # pragma: no cover
            res = False
            print(err)
        self.assertTrue(res)

class TestPlotting(myTestCase):
    def test_traj_vs_time(self):
        for N in range(1, 4):
            for d in range(1, 4):
                traj = nl.Trajectory(np.ones((N, 10, d)))
                out = nl.plot.vstime(traj)
                self.assertEqual(len(out), traj.N*traj.d)

                if N == 1:
                    self.assertEqual(out[0].get_label(), "d=0")
                else:
                    self.assertEqual(out[0].get_label(), "N=0, d=0")

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
