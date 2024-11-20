import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
np.seterr(all='raise') # pay attention to details
import pandas as pd

import unittest
from unittest.mock import patch

from context import noctiluca as nl

"""
  exec "norm jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kcc]kV?__all__j>>
"""
__all__ = [
    'TestClean',
    'TestUtil',
    'TestMCMC',
    'TestMCMCMCMCRun',
    'TestUserinput',
    'TestSyncAsync',
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

class TestClean(myTestCase):
    def setUp(self):
        # Split with threshold = 2 => trajectories of lengths [2, 3, 3]
        self.traj = nl.Trajectory([1., 2, 4.1, 4.5, 3, -0.5, -1, -0.7])

        # Split with threshold = 3 => 3 trajectories with lengths [4, 5, 6]
        self.ds = nl.TaggedSet()
        self.ds.add(nl.Trajectory([0, 0.75, 0.5, 0.3, 5.4, 5.5, 5.3, -2.0, 5.4]))
        self.ds.add(nl.Trajectory([1.2, 1.4, np.nan, np.nan, 10.0, 10.2]))

    def test_split_trajectory(self):
        split_trajs = nl.util.clean.split_trajectory_at_big_steps(self.traj, 2)
        self.assert_array_equal(np.sort([len(traj) for traj in split_trajs]), np.array([2, 3, 3]))

    def test_split_dataset(self):
        split_ds = nl.util.clean.split_dataset_at_big_steps(self.ds, 3)
        self.assert_array_equal(np.sort([len(traj) for traj in split_ds]), np.array([4, 5, 6]))

class TestUtil(myTestCase):
    def test_distribute_noiselevel(self):
        self.assert_array_equal(nl.util.distribute_noiselevel(14, [5, 10, 15]),
                                [1, 2, 3])

    def test_log_derivative(self):
        x = [1, 2, 5, np.nan, 30]
        y = np.array(x)**1.438
        xnew, dy = nl.util.log_derivative(y, x, resampling_density=1.5)
        self.assertTrue(np.all(np.abs(dy - 1.438) < 1e-7))

        xnew, dy = nl.util.log_derivative(y)

class TestMCMC(myTestCase):
    myprint = print

    @patch('builtins.print')
    def test_mcmc(self, mock_print):
        # Simply use the example from the docs
        class normMCMC(nl.util.mcmc.Sampler):
            callback_tracker = 0

            def __init__(self, stepsize=0.1):
                self.stepsize = stepsize

            def propose_update(self, current_value):
                proposed_value = current_value + np.random.normal(scale=self.stepsize)
                logp_forward = -0.5*(proposed_value - current_value)**2/self.stepsize**2 - 0.5*np.log(2*np.pi*self.stepsize**2)
                logp_backward = -0.5*(current_value - proposed_value)**2/self.stepsize**2 - 0.5*np.log(2*np.pi*self.stepsize**2)
                return proposed_value, logp_forward, logp_backward
                
            def logL(self, value):
                return -0.5*value**2 - 0.5*np.log(2*np.pi)
            
            def callback_logging(self, current_value, best_value):
                self.callback_tracker += np.abs(current_value) + np.abs(best_value)
                print("test")

            def callback_stopping(self, myrun):
                return len(myrun.logLs) > 7
        
        mc = normMCMC()
        mc.configure(iterations=10,
                     burn_in=5,
                     log_every=2,
                     show_progress=False,)
        myrun = mc.run(1)

        self.assertEqual(mock_print.call_count, 10)
        self.assertEqual(len(myrun.logLs), 10)
        self.assertEqual(len(myrun.samples), 5)
        self.assertGreater(mc.callback_tracker, 0)

        # Stopping
        mc.config['check_stopping_every'] = 2
        myrun = mc.run(1)

        self.assertEqual(len(myrun.logLs), 9)
        self.assertEqual(len(myrun.samples), 4)

        mc.config['check_stopping_every'] = -1

        # infinite likelihoods
        mc.logL = lambda value: -np.inf
        myrun = mc.run(1)
        self.assertEqual(np.sum(np.diff(myrun.samples) != 0), 4)

        mc.logL = lambda value: np.nan
        with self.assertRaises(RuntimeError):
            myrun = mc.run(1)

        mc.logL = lambda value: np.inf
        with self.assertRaises(RuntimeError):
            myrun = mc.run(1)

class TestMCMCMCMCRun(myTestCase):
    def setUp(self):
        sample0 = [0]
        sample1 = [1]
        sample2 = [2]

        self.samples = 5*[sample0] + 2*[sample1] + 3*[sample2]
        self.logLs = np.array(10*[-10] + 5*[-5] + 2*[-2] + 3*[-3])
        self.myrun = nl.util.mcmc.MCMCRun(self.logLs, self.samples)

    def test_init_noparams(self):
        myrun = nl.util.mcmc.MCMCRun()
        self.assertEqual(len(myrun.samples), 0)
        myrun.samples.append(1)
        myrun = nl.util.mcmc.MCMCRun()
        self.assertEqual(len(myrun.samples), 0)

    def test_logLs_trunc(self):
        self.assert_array_equal(self.myrun.logLs_trunc(), self.logLs[10:20])

    def test_best_sample_logL(self):
        best, bestL = self.myrun.best_sample_logL()
        self.assertEqual(bestL, -2)
        self.assertListEqual(best, [1])

    def test_acceptance_rate(self):
        acc = 2/9
        self.assertEqual(self.myrun.acceptance_rate('sample_equality'), acc)
        self.assertEqual(self.myrun.acceptance_rate('sample_identity'), acc)
        self.assertEqual(self.myrun.acceptance_rate('likelihood_equality'), acc)

    def test_evaluate(self):
        tmp = self.myrun.evaluate(lambda sample : 2*sample)
        self.assertIs(tmp[0], tmp[1])
        self.assertTupleEqual(np.array(tmp).shape, (10, 2))

class TestUserinput(myTestCase):
    def test_make_Trajectory(self):
        make_Trajectory = nl.util.userinput.make_Trajectory

        dat = nl.Trajectory([1, 2, 3])
        self.assertIs(make_Trajectory(dat), dat)

        dat = pd.DataFrame(data=[1, 2, 3], columns=['x'])
        traj = make_Trajectory(dat)
        self.assertEqual(traj.T, 3)
        with self.assertRaises(ValueError):
            make_Trajectory(pd.DataFrame(data=[1, 2, 3], columns=['y']))
        with self.assertRaises(ValueError):
            make_Trajectory(pd.DataFrame(data=[1, 2, 3], columns=['x'], index=['a', 4, 5]))

        dat = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=['x', 'y', 'w'])
        traj = make_Trajectory(dat)
        self.assertEqual(traj.d, 2)
        traj = make_Trajectory(dat, pos_columns=['y', 'w'])
        self.assertEqual(traj.d, 2)
        self.assertEqual(traj[0][1], 3)
        traj = make_Trajectory(dat, pos_columns=[['y'], ['x']])
        self.assertEqual(traj.N, 2)

        dat = pd.Series(data=[1, 2, 3], index=[4, 5, 7])
        traj = make_Trajectory(dat)
        self.assertTrue(np.isnan(traj[2][0]))
        with self.assertRaises(ValueError):
            make_Trajectory(pd.Series(data=[1, 2, 3], index=[1, 1.5, 2]))

    def test_make_TaggedSet(self):
        make_TaggedSet = nl.util.userinput.make_TaggedSet

        dat = nl.TaggedSet([1, 2, 3], hasTags=False)
        self.assertIs(make_TaggedSet(dat), dat)

        dat = pd.DataFrame(data=[[1, 2, 3, 1, 1],
                                 [4, 5, 6, 1, 2],
                                 [7, 8, 9, 1, 4],
                                 [9, 8, 7, 2, 5],
                                 [6, 5, 4, 2, 6],
                                ],
                           columns=['x', 'y', 'z', 'particle', 'frame'],
                          )
        ds = make_TaggedSet(dat)
        self.assertEqual(len(ds), 2)
        self.assertEqual(len(ds[0]), 4)

        dat = ([i] for i in range(10))
        ds = make_TaggedSet(dat)
        self.assertEqual(len(ds), 10)

import threading
class TestSyncAsync(myTestCase):
    @nl.util.sync_async()
    async def coro(self):
        return 'success'

    @nl.util.sync_async('kw_async')
    async def coro2(self):
        return 'success'

    def test_sync(self):
        self.assertEqual(self.coro(), 'success')

    def test_async(self):
        async def main():
            self.assertEqual(await self.coro(run_async=True), 'success')
        asyncio.run(main())

        async def main():
            self.assertEqual(await self.coro2(kw_async=True), 'success')
        asyncio.run(main())

    def test_sync_in_async_context(self):
        async def main():
            self.assertEqual(self.coro(), 'success')
        asyncio.run(main())

    def test_async_different_thread(self):
        async def main():
            def running_in_thread():
                self.assertEqual(self.coro(), 'success')

            with ThreadPoolExecutor() as executor:
                futu = executor.submit(running_in_thread)
                self.assertIsNone(futu.result())

        asyncio.run(main())


if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
