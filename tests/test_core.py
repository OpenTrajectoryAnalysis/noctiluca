import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError: # pragma: no cover
    pass

import numpy as np
np.seterr(all='raise') # pay attention to details

import unittest
from unittest.mock import patch

from context import noctiluca
Trajectory = noctiluca.Trajectory
TaggedSet = noctiluca.TaggedSet
parallel = noctiluca.parallel
del noctiluca

"""
exec "norm jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kcc]kV?__all__j>>
"""
__all__ = [
    'Test0Trajectory',
    'Test1Trajectory',
    'Test0TaggedSet',
    'Test1TaggedSet',
    'TestParallel',
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

class Test0Trajectory(myTestCase):
    def test_init(self):
        _ = Trajectory()

        traj = Trajectory(np.zeros((2, 5, 2)), t=[1, 2, 4, 5, 7])
        self.assertEqual(traj.T, 7)
        self.assertTrue(np.all(np.isnan(traj[[2, 5]])))

        traj = Trajectory(0)
        self.assertTupleEqual(traj.data.shape, (1, 1, 1))
        traj = Trajectory([1, 2, 3])
        self.assertTupleEqual(traj.data.shape, (1, 3, 1))
        traj = Trajectory([[1, 2, 3], [4, 5, 6]])
        self.assertTupleEqual(traj.data.shape, (1, 2, 3))
        traj = Trajectory([[[1, 2, 3]], [[1, 2, 3]]])
        self.assertTupleEqual(traj.data.shape, (2, 1, 3))

        with self.assertRaises(ValueError):
            traj = Trajectory(np.zeros((1, 1, 1, 1)))

        with self.assertRaises(ValueError):
            traj = Trajectory([3], t=[[1, 2]])
        with self.assertRaises(ValueError):
            traj = Trajectory([3], t=[1, 2])

        traj = Trajectory(np.zeros((2, 3, 1)), localization_error=849, parity='even', moo=5)
        self.assertEqual(traj.localization_error, 849)
        self.assertEqual(traj.parity, 'even')
        self.assertEqual(traj.meta['moo'], 5)

class Test1Trajectory(myTestCase):
    def setUp(self):
        self.T = 10
        self.Ns = [1, 1, 1, 2, 2, 2]
        self.ds = [1, 2, 3, 1, 2, 3]
        self.trajs = [Trajectory(np.zeros((N, self.T, d)), localization_error=np.ones((d,)), parity='even')
                      for N, d in zip(self.Ns, self.ds)]
        self.trajs[5].localization_error = np.ones((2, 3)) # To check that shape

    def test_count_valid_frames(self):
        traj = Trajectory(np.zeros((2, 5, 2)), t=[1, 2, 4, 5, 7])
        self.assertEqual(traj.count_valid_frames(), 5)

        traj.data[0, 2, :] = 0
        self.assertEqual(traj.count_valid_frames(), 5)

        traj.data[1, 2, :] = 0
        self.assertEqual(traj.count_valid_frames(), 6)

    def test_interface(self):
        for traj, N, d in zip(self.trajs, self.Ns, self.ds):
            # Basic
            self.assertEqual(len(traj), self.T)
            self.assertEqual(traj.N, N)
            self.assertEqual(traj.T, self.T)
            self.assertEqual(traj.T, traj.F) # no missing frames
            self.assertEqual(traj.d, d)

            if N == 1:
                self.assertTupleEqual(traj[3:5].shape, (2, d))
                self.assert_array_equal(traj[2], np.zeros((d,)))
            else:
                self.assertTupleEqual(traj[3:5].shape, (N, 2, d))
                self.assert_array_equal(traj[2], np.zeros((N, d)))

            # Modifiers
            _ = traj.rescale(2)

            traj.meta['MSD'] = {'data' : np.array([5])}
            times2 = traj.rescale(2, keepmeta=['MSD'])
            self.assertIn('MSD', times2.meta)
            self.assert_array_equal(traj.data*2, times2.data)
            plus1 = traj.offset(1, keepmeta=['MSD'])
            self.assert_array_equal(traj.data+1, plus1.data)
            plus1 = traj.offset([1])
            self.assert_array_equal(traj.data+1, plus1.data)
            plus1 = traj.offset([[1]])
            self.assert_array_equal(traj.data+1, plus1.data)

            if N == 2:
                rel = traj.relative(keepmeta=['MSD'])
                self.assertTupleEqual(rel.data.shape, (1, self.T, d))
                self.assertIn('MSD', rel.meta)
                mag = rel.abs(keepmeta=['MSD'])
                self.assertTupleEqual(mag.data.shape, (1, self.T, 1))
                dif = rel.diff(dt=2, keepmeta=['MSD'])
                self.assertTupleEqual(dif.data.shape, (1, self.T-2, d))
                dim = traj.dims([0], keepmeta=['MSD'])
                self.assertTupleEqual(dim.data.shape, (2, self.T, 1))
            elif N == 1:
                mag = traj.abs()
                self.assertNotIn('MSD', mag.meta)
                self.assertTupleEqual(mag.data.shape, (1, self.T, 1))
                with self.assertRaises(ValueError):
                    rel = traj.relative()
                dif = traj.diff(dt=3)
                self.assertTupleEqual(dif.data.shape, (1, self.T-3, d))
                if d >= 2:
                    dim = traj.dims([0, 1])
                    self.assertTupleEqual(dim.data.shape, (1, self.T, 2))

    def test_relative_Nloci(self):
        traj = Trajectory(np.arange(30).reshape((3, 5, 2)))
        dif_sequential = traj.relative()
        self.assert_array_equal(dif_sequential.data, traj.data[1:] - traj.data[:-1])
        dif_ref1 = traj.relative(ref=1)
        self.assert_array_equal(dif_ref1.data, traj.data[[0, 2]] - traj.data[[1]])

        with self.assertRaises(ValueError):
            _ = traj.relative(ref=5)

class Test0TaggedSet(unittest.TestCase):
    def test_init(self):
        ls = TaggedSet()

        ls = TaggedSet(zip([1, 2, 3], [["a", "b"], "a", ["b", "c"]]))
        self.assertListEqual(ls._data, [1, 2, 3])
        self.assertListEqual(ls._tags, [{"a", "b"}, {"a"}, {"b", "c"}])

        ls = TaggedSet([1, 2, 3], hasTags=False)
        
    def test_add(self):
        ls = TaggedSet()
        ls.add(1)
        ls.add(2, 'a')
        ls.add(3, {'b', 'c'})
        
class Test1TaggedSet(unittest.TestCase):
    def setUp(self):
        self.ls = TaggedSet(zip([1, 2, 3], [["a", "b"], "a", ["b", "c"]]))

    def test_len(self):
        self.assertEqual(len(self.ls), 3)
        
    def test_iteration(self):
        for ind, val in enumerate(self.ls):
            self.assertEqual(val, self.ls._data[ind])

        for ind, (val, tags) in enumerate(self.ls(giveTags=True)):
            self.assertEqual(val, self.ls._data[ind])
            self.assertSetEqual(tags, self.ls._tags[ind])

        all_values = {1, 2, 3}
        for val in self.ls(randomize=True):
            all_values.remove(val)
        self.assertSetEqual(all_values, set())

    def test_elementaccess(self):
        self.assertEqual(self.ls[1], 2)

    def test_mergein(self):
        newls = TaggedSet(zip([4, 5, 6], [["d"], [], "e"]))
        self.ls.mergein(newls, additionalTags='new')
        self.assertListEqual(self.ls._data, [1, 2, 3, 4, 5, 6])
        self.assertSetEqual(self.ls.tagset(), {'a', 'b', 'c', 'd', 'e', 'new'})

        self.ls |= newls
        self.assertEqual(len(self.ls), 9)
        
        self.ls.makeSelection(tags='new')
        self.assertSetEqual(set(self.ls), {4, 5, 6})

    def test_makeTagsSet(self):
        self.assertSetEqual(TaggedSet.makeTagsSet("foo"), {"foo"})
        self.assertSetEqual(TaggedSet.makeTagsSet(["foo"]), {"foo"})
        with self.assertRaises(ValueError):
            TaggedSet.makeTagsSet(1)

    def test_selection(self):
        for _ in range(5):
            self.ls.makeSelection(nrand=1, random_seed=6542)
            self.assertEqual(len(self.ls), 1)
            self.ls.makeSelection(1)
            self.assertEqual(len(self.ls), 1)
        for _ in range(5):
            self.ls.makeSelection(prand=0.5)
            self.assertEqual(len(self.ls), 1)
            self.ls.makeSelection(0.5)
            self.assertEqual(len(self.ls), 1)

        self.ls.makeSelection(tags="a")
        self.assertSetEqual({*self.ls}, {1, 2})
        self.ls.makeSelection("a")
        self.assertSetEqual({*self.ls}, {1, 2})

        self.ls.makeSelection(tags=["a", "b"], logic=all)
        self.assertSetEqual({*self.ls}, {1})
        self.ls.makeSelection(["a", "b"], logic=all)
        self.assertSetEqual({*self.ls}, {1})

        self.ls.makeSelection(tags=["a", "b"], logic=any)
        self.assertSetEqual({*self.ls}, {1, 2, 3})
        self.ls.makeSelection(["a", "b"], logic=any)
        self.assertSetEqual({*self.ls}, {1, 2, 3})

        def sel(datum, tags):
            return datum >= 2
        self.ls.makeSelection(selector=sel)
        self.assertSetEqual({*self.ls}, {2, 3})
        self.ls.makeSelection(sel)
        self.assertSetEqual({*self.ls}, {2, 3})

        self.ls.refineSelection(tags='c')
        self.assertSetEqual({*self.ls}, {3})
        self.ls.refineSelection('c')
        self.assertSetEqual({*self.ls}, {3})

        with self.assertRaises(ValueError):
            self.ls.makeSelection(np.array([1])[0]) # doesn't work, bc numpy uses np.int instead of int

        sel = self.ls.saveSelection()
        self.assertListEqual(sel, [False, False, True])

        self.ls.makeSelection()
        self.assertEqual(len(self.ls), 3)
        self.ls.restoreSelection(sel)
        self.assertEqual(len(self.ls), 1)
        self.assertSetEqual({*self.ls}, {3})

        copied = self.ls.copySelection()
        copied.makeSelection()
        self.assertEqual(len(copied), 1)

    def test_deleteSelection(self):
        self.ls.makeSelection(tags='a')
        self.ls.deleteSelection()
        self.assertEqual(len(self.ls), 1)
        self.assertEqual(self.ls[0], 3)

    def test_addTags(self):
        self.ls.addTags('moo')
        self.assertSetEqual(self.ls.tagset(), {'a', 'b', 'c', 'moo'})

    def test_tagset(self):
        self.assertSetEqual(self.ls.tagset(), {'a', 'b', 'c'})

    def test_apply(self):
        # need a TaggedSet of mutable objects to check that they remain unaltered
        mutls = TaggedSet(zip(['hello', 'World', '!'], [["a", "b"], "a", ["b", "c"]]))
        mutls.makeSelection(tags="a")
        newls = mutls.apply(lambda word : word+'_moo')

        self.assertListEqual(mutls._data, ['hello', 'World', '!'])
        self.assertListEqual(newls._data, ['hello_moo', 'World_moo'])

        # check inplace modification
        def fun(i):
            return i+1
        self.ls.apply(fun, inplace=True)
        self.assertListEqual(self.ls._data, [2, 3, 4])

    def test_map_unique(self):
        def funTrue(x):
            return True
        def fun2(x):
            return 2*x

        self.assertTrue(self.ls.map_unique(funTrue))
        with self.assertRaises(RuntimeError):
            self.ls.map_unique(fun2)
        self.assertIsNone(TaggedSet().map_unique(funTrue))

def parfun(args=None, err=False): # needs to be pickleable, which TestCase is not
    if err:
        raise RuntimeError
    return 'success'

class TestParallel(myTestCase):
    def test_vanilla(self):
        # These tests are a bit useless...
        with parallel.Parallelize(n=1):
            ls = list(parallel._map(len, [[1], [1, 2]]))
            self.assertListEqual(ls, [1, 2])

    def test_parfun(self):
        with parallel.Parallelize(n=1):
            res = list(parallel._map(parfun, np.arange(2)))

        self.assertListEqual(res, ['success', 'success'])

    def test_chunksize(self):
        with parallel.Parallelize(n=1):
            res = list(parallel._map(parfun, np.arange(2), chunksize=0))
            self.assertListEqual(res, ['success', 'success'])

            res = list(parallel._map(parfun, np.arange(2), chunksize=-1))
            self.assertListEqual(res, ['success', 'success'])

    def test_dummy_executor(self):
        executor = parallel.DummyExecutor()
        self.assertEqual(executor.submit(parfun).result(), 'success')
        with self.assertRaises(RuntimeError):
            res = executor.submit(parfun, err=True).result()

    def test_executor(self):
        def to_run():
            return parallel._executor.submit(parfun).result()

        self.assertEqual(to_run(), 'success')
        with parallel.Parallelize(n=1):
            self.assertEqual(to_run(), 'success')

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
