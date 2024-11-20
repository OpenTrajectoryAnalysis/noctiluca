"""
Random assortment of useful auxiliary stuff
"""
import asyncio
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.interpolate

# This is imported in the util module as *, so don't clutter
"""
  exec "norm jjd}O" | let @a="\n'" | exec "g/^def /norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kcc]kV?__all__j>>
"""
__all__ = [
    'distribute_noiselevel',
    'log_derivative',
    'sync_async',
]

def distribute_noiselevel(noise2, pixelsize):
    """
    Get coordinate-wise localization error from scalar noise level

    Parameters
    ----------
    noise2 : float
        the total noise level. In terms of coordinate-wise errors this is
        ``noise2 = Δx^2 + Δy^2 + Δz^2``
    pixelsize : array-like
        linear extension of pixel/voxel in each direction

    Returns
    -------
    localization_error : (d,) np.array
        localization error distributed to the coordinate directions,
        proportionately to the pixel size, i.e. ``Δx``, ``Δy``, ``Δz`` such
        that ``noise2 = Δx^2 + Δy^2 + Δz^2``.
    """
    voxel = np.asarray(pixelsize)
    noise_in_px = np.sqrt(noise2/np.sum(voxel**2))
    return noise_in_px*voxel

def log_derivative(y, x=None, resampling_density=2):
    """
    Calculate loglog-derivative.

    We resample the given data to log-spaced x.

    Parameters
    ----------
    y : array-like
        the function values whose derivative we are interested in
    x : array-like, optional
        the independent variable for the data in y. Will default to
        ``np.arange(len(y))`` (and thus ignore the first data point).
    resampling_density : float, optional
        how tight to space the log-resampled points. A value of 1 corresponds
        to the spacing between the first two data points, higher values
        decrease spacing.

    Returns
    -------
    x : np.array
        the log-resampled abscissa
    dlog : np.array
        the calculated log-derivative
    """
    if x is None:
        x = np.arange(len(y))

    with np.errstate(divide='ignore'):
        x = np.log(x)
        y = np.log(y)
    ind_valid = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[ind_valid]
    y = y[ind_valid]

    dlogx = (x[1] - x[0])/resampling_density
    xnew = np.arange(np.min(x), np.max(x), dlogx)
    ynew = scipy.interpolate.interp1d(x, y)(xnew)

    return np.exp(xnew[:-1] + dlogx/2), np.diff(ynew)/dlogx

def sync_async(kw_async_on='run_async'):
    """
    Decorator(-factory) to make (async) coroutines execute synchronously

    We add a keyword argument that allows switching between synchronous and
    asynchronous execution. Synchronous is the default, such that the decorated
    function/coroutine integrates seamlessly with synchronous libraries

    Adapted from https://stackoverflow.com/a/78911765/12975599

    Parameters
    ----------
    kw_async_on : str, optional
        the name of the keyword argument to add for switching between
        sync/async

    Examples
    --------
    >>> @sync_async()                       # note "()"
    ... async def coro():
    ...     # [...] asynchronous code
    ...     return 'success'
    ...
    ... result = coro()                     # in synchronous context
    ... # OR
    ... result = await coro(run_async=True) # in asynchronous context
    """
    def decorator(coro):
        @wraps(coro)
        def wrapper(*args, **kwargs):
            try:
                async_on = kwargs[kw_async_on]
            except KeyError:
                async_on = False
            else:
                del kwargs[kw_async_on]

            if async_on:
                return coro(*args, **kwargs)
            else:
                # Check whether we can just asyncio.run, or whether we need to
                # push it to a separate thread to avoid interfering with a
                # running event loop
                try:
                    _ = asyncio.get_running_loop()
                except RuntimeError as err:
                    return asyncio.run(coro(*args, **kwargs))

                with ThreadPoolExecutor() as pool:
                    future = pool.submit(lambda : asyncio.run(coro(*args, **kwargs)))
                    return future.result()

        return wrapper
    return decorator
