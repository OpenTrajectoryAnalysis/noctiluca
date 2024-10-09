"""
Parallelization for noctiluca functions
"""
from multiprocessing import Pool

_map = map
_umap = map

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
        global _map, _umap
        _map = self.map
        _umap = self.umap

    def __exit__(self, type, value, traceback):
        global _map, _umap
        _map = map
        _umap = map
        return False # raise anything that might have happened

    def map(self, func, iterable):
        with Pool(self.n) as mypool:
            imap = mypool.imap(func, iterable, self.chunksize)
            for X in imap: yield X

    def umap(self, func, iterable):
        with Pool(self.n) as mypool:
            imap = mypool.imap_unordered(func, iterable, self.chunksize)
            for X in imap: yield X
