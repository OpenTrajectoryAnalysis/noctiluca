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

from multiprocessing import Pool

"""
exec "norm jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kcc]kV?__all__j>>
"""
__all__ = [
    'TestP2',
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

class TestP2(myTestCase):
    def setUp(self):
        self.traj = nl.Trajectory([1, 2, 3, 4, np.nan, 6])
        raw_data = np.cumsum(np.random.normal(size=(10, 10)), axis=0)
        self.ds = nl.TaggedSet([nl.Trajectory(raw_data[n]) for n in range(len(raw_data))], hasTags=False)

    def test_MSDtraj(self):
        nl.analysis.MSD(self.traj)
        msd = self.traj.meta['MSD']['data']
        self.assert_array_equal(msd, np.array([0, 1, 4, 9, 16, 25]))
        self.assert_array_equal(self.traj.meta['MSD']['N'], np.array([5, 3, 3, 2, 1, 1]))

        del self.traj.meta['MSD']
        msd = nl.analysis.MSD(self.traj, TA=False, recalculate=True)
        self.assert_array_equal(msd, np.array([0, 1, 4, 9, np.nan, 25]))
        self.assert_array_equal(self.traj.meta['MSD']['N'], np.array([1, 1, 1, 1, 0, 1]))

        traj3d = nl.Trajectory(np.arange(30).reshape(-1, 3))
        nl.analysis.MSD(traj3d)
        msd = traj3d.meta['MSD']['data']
        self.assert_array_equal(msd, 27*np.arange(len(traj3d))**2)

    def test_MSDdataset(self):
        msd, N = nl.analysis.MSD(self.ds, giveN=True)
        self.assertEqual(len(msd), 10)
        self.assert_array_equal(N, len(self.ds)*np.linspace(10, 1, 10))

        msd, var = nl.analysis.MSD(self.ds, givevar=True)
        self.assert_array_equal(np.isnan(var), 10*[False])

        self.assert_array_equal(msd, nl.analysis.MSD(self.ds))

    def test_covariances_and_correlations(self):
        acov = nl.analysis.ACov(self.traj)
        self.assert_array_equal(acov, np.array([13.2, 20/3, 35/3, 11, 12, 6]))
        self.assert_array_equal(self.traj.meta['ACov']['data'], np.array([13.2, 20/3, 35/3, 11, 12, 6]))

        acorr = nl.analysis.ACorr(self.traj)
        self.assert_array_equal(acorr, np.array([13.2, 20/3, 35/3, 11, 12, 6])/13.2)
        self.assert_array_equal(self.traj.meta['ACorr']['data'], np.array([13.2, 20/3, 35/3, 11, 12, 6])/13.2)

        vacov = nl.analysis.VACov(self.traj)
        self.assert_array_equal(vacov, np.array([1, 1, 1, np.nan, np.nan]))
        self.assert_array_equal(self.traj.meta['VACov']['data'], np.array([1, 1, 1, np.nan, np.nan]))

        vacorr = nl.analysis.VACorr(self.traj)
        self.assert_array_equal(vacov, np.array([1, 1, 1, np.nan, np.nan]))
        self.assert_array_equal(self.traj.meta['VACorr']['data'], np.array([1, 1, 1, np.nan, np.nan]))

    def test_new_p2(self):
        def AD(xm, xn):
            return np.sum(np.abs(xm-xn), axis=-1)

        MAD = nl.analysis.p2.P2traj(self.traj, function=AD, writeto=None)['data']
        with self.assertRaises(KeyError):
            _ = self.traj.meta['P2']
        nl.analysis.p2.P2(self.traj, function=AD)

        self.assert_array_equal(MAD, np.array([0, 1, 2, 3, 4, 5]))
        self.assert_array_equal(MAD, self.traj.meta['P2']['data'])

        _ = nl.analysis.p2.P2dataset(self.ds, function=AD)

    def test_parallelization(self):
        with nl.Parallelize(n=1):
            msd = nl.analysis.MSD(self.ds, chunksize=1)

        for traj in self.ds:
            self.assertIn('MSD', traj.meta)
            self.assertIn('data', traj.meta['MSD'])
            self.assertIn('N', traj.meta['MSD'])

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
