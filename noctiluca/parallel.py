"""
Parallelization for noctiluca functions
"""
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, Executor, Future

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

_map = map
_umap = map
_executor = DummyExecutor()

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
    def __init__(self, n=None, chunksize=1):
        self.n = n
        self.chunksize = chunksize

    def __enter__(self):
        global _map, _umap, _executor
        _map = self.map
        _umap = self.umap
        _executor = ProcessPoolExecutor(max_workers=self.n)

    def __exit__(self, type, value, traceback):
        global _map, _umap, _executor
        _map = map
        _umap = map
        _executor.shutdown()
        _executor = DummyExecutor()
        return False # raise anything that might have happened

    def map(self, func, iterable):
        with Pool(self.n) as mypool:
            imap = mypool.imap(func, iterable, self.chunksize)
            for X in imap: yield X

    def umap(self, func, iterable):
        with Pool(self.n) as mypool:
            imap = mypool.imap_unordered(func, iterable, self.chunksize)
            for X in imap: yield X
