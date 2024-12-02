"""
Parallelization for noctiluca functions
"""
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, Executor, Future
from functools import wraps

class DummyExecutor(Executor):
    # A dummy executor that just runs everything synchronously
    # adapted from https://stackoverflow.com/a/10436851/12975599
    def submit(self, fun, *args, **kwargs):
        futu = Future()
        try:
            result = fun(*args, **kwargs)
        except BaseException as e:
            futu.set_exception(e)
        else:
            futu.set_result(result)

        return futu

class Parallelize:
    """
    Context manager for parallelization

    We indicate in docstrings that certain functions are "parallel-aware", if
    they make use of this functionality

    Parameters
    ----------
    n : int, optional
        how many cores to use; defaults to maximum available
    chunksize : int, optional
        chunk size for parallel execution, when used. Note that the default of
        1 might make things very slow (as also noted in the `!multiprocessing`
        docs).

    Examples
    --------
    >>> import noctiluca as nl
    ...
    ... with nl.Parallelize():
    ...     # parallel-aware stuff goes here

    Notes
    -----
    It is generally recommended to avoid hyperthreading when parallelizing
    computations; this can be achieved by running
    >>> import os
    ... os.environ['OMP_NUM_THREADS'] = '1'

    at the very beginning of your code, before any other imports (technically:
    before `!multiprocessing` is imported, which e.g. this module does).
    """
    def __init__(self, n=None):
        self.n = n

    def __enter__(self):
        global _executor, _map
        _map = self.map
        _executor = ProcessPoolExecutor(max_workers=self.n)

    def __exit__(self, type, value, traceback):
        global _executor, _map
        _map = Parallelize.serial_map
        _executor.shutdown(wait=False)
        _executor = DummyExecutor()
        return False # raise anything that might have happened

    def map(self, func, todo, chunksize=None):
        if chunksize is None:
            chunksize = _chunksize

        if chunksize < 0:
            return map(func, todo)
        else:
            if chunksize == 0:
                chunksize = len(todo)

            return _executor.map(func, todo, chunksize=chunksize)

    @staticmethod
    def serial_map(*args, chunksize=None, **kwargs):
        return map(*args, **kwargs)

_chunksize = 1
_map = Parallelize.serial_map
_executor = DummyExecutor()

def chunky(kwarg_name='chunksize', default=1):
    """
    A decorator for functions using the Parallelize interface

    This decorator adds a kwarg 'chunksize' (or custom name) and sets
    `!parallel._chunksize` accordingly before running the function. This
    facilitates user control over parallelization level.

    Parameters
    ----------
    kwarg_name : str
        the name for the kwarg to add
    default : int
        the default chunksize value
    """
    def decorator(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            global _chunksize
            chunksize_save = _chunksize

            try:
                _chunksize = kwargs[kwarg_name]
            except KeyError:
                _chunksize = default
            else:
                del kwargs[kwarg_name]

            try:
                return fun(*args, **kwargs)
            finally:
                _chunksize = chunksize_save
        return wrapper
    return decorator
