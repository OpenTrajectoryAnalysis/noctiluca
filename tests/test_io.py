import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass
from pathlib import Path

import numpy as np
np.seterr(all='raise') # pay attention to details

import unittest
from unittest.mock import patch

from context import noctiluca as nl
import h5py

"""
exec "norm jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kcc]kV?__all__j>>
"""
__all__ = [
    'TestLoad',
    'TestWrite',
    'TestHDF5',
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

class TestLoad(myTestCase):
    def test_evalSPT(self):
        testdata="""
1.0	2.3	1	5	
1.5	2.1	2	5	
2.1	1.5	3	5	
1.9	1.7	5	5	
0.5	9.3	-5	10	
0.4	8.5	-4	10	
1.2	9.1	-6	10	
"""[1:-1] # Cut off '\n' at beginning and end

        path = Path('./testdata')
        path.mkdir(exist_ok=True)
        filename = str(path / "evalSPT.csv")
        with open(filename, mode='w') as f:
            f.write(testdata)

        ds = nl.io.load.evalSPT(filename, tags={'test'})

        ds.makeSelection(selector = lambda traj, _ : len(traj) <= 3)
        self.assert_array_equal(ds[0][:], np.array([[1.2, 9.1], [0.5, 9.3], [0.4, 8.5]]))

        ds.makeSelection(selector = lambda traj, _ : len(traj) > 3)
        self.assert_array_equal(ds[0][:], np.array([[1.0, 2.3], [1.5, 2.1], [2.1, 1.5], [np.nan, np.nan], [1.9, 1.7]]))

    def test_csv(self):
        testdata="""
line,id,noise,x,y,frame,x2,y2,meta_mean,meta_unique,meta_nanmean
1,32,10,0.6,0.8,10,0.5,1.0,1,5,1
2,32,1.5,0.7,0.5,9,0.7,0.8,2,5,nan
3,20,5.3,1.2,1.3,11,-0.8,-1.2,3,6,5
"""[1:-1]

        path = Path('./testdata')
        path.mkdir(exist_ok=True)
        filename = str(path / "twoLocus.csv")
        with open(filename, mode='w') as f:
            f.write(testdata)

        # First two are just dummies
        with self.assertRaises(ValueError): # too many columns specified
            ds = nl.io.load.csv(
                    filename, [None, 'id', 'noise', 'x', 'y', 'z', 't', 'x2', 'y2', 'z2', 'meta_mean', 'meta_unique'],
                    meta_post={'meta_mean' : 'mean', 'meta_unique' : 'unique'}, delimiter=',', skip_header=1)

        with self.assertRaises(AssertionError): # y2 does not have a y to go with it
            ds = nl.io.load.csv(
                    filename, [None, 'id', 'noise', 'x', 't', 'x2', 'y2', 'meta_mean', 'meta_unique'],
                    meta_post={'meta_mean' : 'mean', 'meta_unique' : 'unique'}, delimiter=',', skip_header=1)

        ds = nl.io.load.csv(
                filename, [None, 'id', 'noise', 'x', 'y', 't', 'x2', 'y2', 'meta_mean', 'meta_unique', 'meta_nanmean'],
                meta_post={'meta_mean' : 'mean',
                           'meta_unique' : 'unique',
                           'meta_nanmean' : 'nanmean',
                           }, delimiter=',', skip_header=1)

        ds.makeSelection(selector = lambda traj, _ : len(traj) >= 2)
        self.assert_array_equal(ds[0][:], np.array([[[0.7, 0.5], [0.6, 0.8]], [[0.7, 0.8], [0.5, 1.0]]]))
        self.assert_array_equal(ds[0].meta['noise'], np.array([1.5, 10]))
        self.assertEqual(ds[0].meta['meta_mean'], 1.5)
        self.assertEqual(ds[0].meta['meta_unique'], 5)
        self.assertEqual(ds[0].meta['meta_nanmean'], 1)

        ds.makeSelection(selector = lambda traj, _ : len(traj) < 2)
        self.assert_array_equal(ds[0][:], np.array([[[1.2, 1.3]], [[-0.8, -1.2]]]))
        self.assert_array_equal(ds[0].meta['noise'], np.array([5.3]))
        self.assertEqual(ds[0].meta['meta_mean'], 3)
        self.assertEqual(ds[0].meta['meta_unique'], 6)
        self.assertEqual(ds[0].meta['meta_nanmean'], 5)

        with self.assertRaises(RuntimeError):
            ds = nl.io.load.csv(
                    filename, [None, 'id', 'noise', 'x', 'y', 't', 'x2', 'y2', 'meta_mean', 'meta_unique'],
                    meta_post={'meta_mean' : 'unique', 'meta_unique' : 'unique'}, delimiter=',', skip_header=1)

class TestWrite(myTestCase):
    def setUp(self):
        self.ds = nl.TaggedSet()
        self.ds.add(nl.Trajectory([0, 0.75, 0.5, 0.3, 5.4, 5.5, 5.3, -2.0, 5.4]))
        self.ds.add(nl.Trajectory([1.2, 1.4, np.nan, np.nan, 10.0, 10.2]))

    def test_csv(self):
        path = Path('./testdata')
        path.mkdir(exist_ok=True)
        filename = str(path / 'test_write.csv')
        nl.io.write.csv(self.ds, filename)

        with open(filename, 'r') as f:
            self.assertTrue(f.read() == 'id\tframe\tx\n0\t0\t0.0\n0\t1\t0.75\n0\t2\t0.5\n0\t3\t0.3\n0\t4\t5.4\n0\t5\t5.5\n0\t6\t5.3\n0\t7\t-2.0\n0\t8\t5.4\n1\t0\t1.2\n1\t1\t1.4\n1\t4\t10.0\n1\t5\t10.2\n')

        ds = nl.TaggedSet()
        ds.add(nl.Trajectory(np.arange(20).reshape((2, 5, 2))))
        nl.io.write.csv(ds, filename)

        with open(filename, 'r') as f:
            self.assertTrue(f.read() == 'id\tframe\tx\ty\tx2\ty2\n0\t0\t0\t1\t10\t11\n0\t1\t2\t3\t12\t13\n0\t2\t4\t5\t14\t15\n0\t3\t6\t7\t16\t17\n0\t4\t8\t9\t18\t19\n')

    def test_mat(self):
        path = Path('./testdata')
        path.mkdir(exist_ok=True)
        filename = str(path / 'test_write.mat')
        nl.io.write.mat(self.ds, filename)

class TestHDF5(myTestCase):
    def test_simple(self):
        data = {
            'array' : np.array([1, 2, 3]),
            'noarray' : np.array(3.5),
            'bool' : True,
            'int' : 5,
            'float' : 5.3,
            'complex' : 1+3j,
            'str' : "Hello World",
            'Trajectory' : nl.Trajectory([1, 2, 4, 5], meta_test='moo'),
            'TaggedSet' : nl.TaggedSet(),
            'empty_tuple' : tuple(),
            'None' : None,
        }
        data['TaggedSet'].add(5, ['moo', 'foo', 'bar'])
        data['TaggedSet'].add(8.7)
        data['TaggedSet'].add(nl.Trajectory([1.5, 3.8]), 'traj')

        path = Path('./testdata')
        path.mkdir(exist_ok=True)
        filename = str(path / 'test.hdf5')
        nl.io.write.hdf5(data, filename)

        # Immediately reload
        data_read = nl.io.load.hdf5(filename)
        self.assertEqual(data.keys(), data_read.keys())

        for key in data:
            if key == 'Trajectory':
                traj = data[key]
                traj_read = data_read[key]
                self.assert_array_equal(traj.data, traj_read.data)
                for meta_key in traj.meta:
                    self.assertEqual(traj.meta[meta_key], traj_read.meta[meta_key])
            elif key == 'TaggedSet':
                self.assertListEqual(data[key]._tags, data_read[key]._tags)
                self.assertListEqual(data[key]._selected, data_read[key]._selected)

                data[key].makeSelection(tags='moo')
                data_read[key].makeSelection(tags='moo')
                self.assertEqual(data[key][0], data_read[key][0])
                
                data[key].makeSelection(tags='traj')
                data_read[key].makeSelection(tags='traj')
                self.assert_array_equal(data[key][0].data, data_read[key][0].data)
            elif key == 'array':
                self.assert_array_equal(data[key], data_read[key])
            elif key == 'empty_tuple':
                self.assertEqual(type(data[key]), type(data_read[key]))
                self.assertEqual(len(data_read[key]), 0)
            elif key == 'None':
                self.assertIs(data_read[key], None)
            else:
                self.assertEqual(data[key], data_read[key])

        # Test partial writing
        nl.io.write.hdf5(None, filename, '/None_group/test/')

        # Test partial reading
        self.assertTrue(nl.io.load.hdf5(filename, '/{bool}'))
        self.assertTrue(nl.io.load.hdf5(filename, '/bool'))
        self.assertEqual(nl.io.load.hdf5(filename, '{float}'), data['float'])
        self.assertIsNone(nl.io.load.hdf5(filename, 'None'))
        self.assertEqual(nl.io.load.hdf5(filename, 'empty_tuple/{_HDF5_ORIG_TYPE_}'), 'tuple')

        # Silent overwrite
        nl.io.write.hdf5({}, filename, '/None_group')

    def test_errors(self):
        path = Path('./testdata')
        path.mkdir(exist_ok=True)
        filename = path / 'hdf5_dummy.hdf5'

        class Test:
            pass
        with self.assertRaises(RuntimeError):
            nl.io.write.hdf5(Test(), filename)

        nl.io.write.hdf5(5, filename, group='/test/test')
        res = nl.io.load.hdf5(filename)
        self.assertEqual(res['test']['test'], 5)

    def test_ls(self):
        path = Path('./testdata')
        path.mkdir(exist_ok=True)
        filename = str(path / 'test.hdf5') # this is bad style, relying on a file written in another test...
        ls = nl.io.hdf5.ls(filename)
        self.assertIn('TaggedSet', ls)
        self.assertIn('[array]', ls)
        self.assertIn('{bool = True}', ls)
        self.assertIn('{_HDF5_ORIG_TYPE_ = dict}', ls)

        ls = nl.io.hdf5.ls(filename, '/Trajectory')
        self.assertIn('[data]', ls)
        self.assertIn('meta', ls)

        ls = nl.io.hdf5.ls(filename, depth=2)
        self.assertIn('TaggedSet/[_selected]', ls)
        self.assertIn('Trajectory/meta', ls)
        self.assertIn('empty_tuple/{_HDF5_ORIG_TYPE_ = tuple}', ls)

        self.assertTrue(nl.io.hdf5.ls(filename, '{bool}'))
        self.assertTrue(nl.io.hdf5.ls(filename, '/{bool}'))
        self.assertEqual(nl.io.hdf5.ls(filename, 'empty_tuple/{_HDF5_ORIG_TYPE_}'), 'tuple')

        # Just for completeness
        self.assertTupleEqual(nl.io.hdf5.check_group_or_attr(None), ('/', None))

    def test_write_subTaggetSet(self):
        path = Path('./testdata')
        path.mkdir(exist_ok=True)
        filename = str(path / 'hdf5_dummy.hdf5')

        # Have to make sure that data._data is not converted to numpy array or stored as attributes
        # One way to ensure this is to use dicts, which will be stored as groups
        data = nl.TaggedSet((({'i':i}, 'small' if i < 10 else 'large') for i in range(20)))
        self.assertEqual(len(data), 20)

        nl.io.write.hdf5({}, filename) # empty out the file
        nl.io.write.hdf5(data, filename, 'data_full')
        data.makeSelection(tags='small')

        # A few failing attempts
        with self.assertRaises(ValueError):
            nl.io.write.hdf5_subTaggedSet(data, filename, 'data_small') # forgot refTaggedSet
        with self.assertRaises(ValueError):
            nl.io.write.hdf5_subTaggedSet(data, filename, group='/', refTaggedSet='data_full') # forgot name for new entry

        nl.io.write.hdf5_subTaggedSet(data, filename, 'data_small', refTaggedSet='data_full') # this is how it's done

        read = nl.io.load.hdf5(filename, 'data_small')
        read.makeSelection()
        self.assertEqual(len(read), 10)

        with h5py.File(filename, 'r') as f:
            for i in range(10):
                self.assertEqual(f[f'data_full/_data/{i}'], f[f'data_small/_data/{i}'])
                self.assertNotEqual(f[f'data_full/_tags/{i}'], f[f'data_small/_tags/{i}'])

        # Check that silent overwrite works
        nl.io.write.hdf5_subTaggedSet(data, filename, 'data_small', refTaggedSet='data_full')

        # Test the error cases
        data = nl.TaggedSet(((i, 'small' if i < 10 else 'large') for i in range(20)))
        nl.io.write.hdf5({}, filename) # empty out the file
        nl.io.write.hdf5(data, filename, 'data_full')
        data.makeSelection(tags='small')
        with self.assertRaises(ValueError):
            nl.io.write.hdf5_subTaggedSet(data, filename, 'data_small', refTaggedSet='data_full')

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
