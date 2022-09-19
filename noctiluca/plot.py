"""
Provide simple plotting capabilities
"""
import numpy as np
from matplotlib import pyplot as plt

def vstime(traj, ax=None, maskNaNs=True, label='', **kwargs):
    """
    Plot a given trajectory vs time

    Parameters
    ----------
    traj : Trajectory
        the trajectory to plot
    ax : matplotlib axes, optional
        the axes to plot into. Defaults to ``plt.gca()``.
    maskNaNs : bool, optional
        whether to plot through missing values. `matplotlib` by default leaves
        gaps when it encounters `nan` values; this can be either quite useful
        to stay aware of missing data, or equally suboptimal, since a single
        valid frame surronded by `nan`'s will be missing from the plot (with
        default settings). Since the latter is a bigger problem, by default we
        plot a continuous line and just ignore missing frames; set ``maskNaNs =
        False`` to plot with gaps.
    label : str or None, optional
        prefix for plot label. Curves will be labelled with ``label+[N=..]+
        d=..``. Set ``label=None`` to prevent labelling.
    kwargs : keyword arguments
        forwarded to ``ax.plot()``.
    
    Returns
    -------
    list of Line2D
        the output of ``ax.plot()``, ordered by ``(N, d)``. Use this to
        customize the plot.
    """
    if ax is None:
        ax = plt.gca()

    tplot = np.arange(traj.T)
    if maskNaNs:
        tplot = tplot[~np.any(np.isnan(traj.data), axis=(0, 2))]

    outs = []
    for N in range(traj.N):
        for d in range(traj.d):
            if label is not None:
                kwargs['label'] = label+(f"N={N}, " if traj.N > 1 else "")+f"d={d}"
            outs += ax.plot(tplot, traj.data[N, tplot, d], **kwargs)

    return outs

def spatial(traj, ax=None, dims=(0, 1), maskNaNs=True, label='', **kwargs):
    """
    Plot a trajectory in space

    Parameters
    ----------
    traj : Trajectory
        the trajectory to plot
    ax : matplotlib axes, optional
        the axes to plot into. Defaults to ``plt.gca()``.
    dims : 2-tuple of int, optional
        which dimensions to plot
    maskNaNs : bool, optional
        whether to plot through missing values. `matplotlib` by default leaves
        gaps when it encounters `nan` values; this can be either quite useful
        to stay aware of missing data, or equally suboptimal, since a single
        valid frame surronded by `nan`'s will be missing from the plot (with
        default settings). Since the latter is a bigger problem, by default we
        plot a continuous line and just ignore missing frames; set ``maskNaNs =
        False`` to plot with gaps.
    label : str or None, optional
        prefix for plot label. Curves will be labelled with ``label+[N=..]``.
        Set ``label=None`` to prevent labelling.
    kwargs : keyword arguments
        forwarded to ``ax.plot()``.
    
    Returns
    -------
    list of Line2D
        the output of ``ax.plot()``, ordered by ``(N, d)``. Use this to
        customize the plot.
    """
    if ax is None:
        ax = plt.gca()

    if not len(dims) == 2:
        raise ValueError(f"Need exactly two dimensions to plot spatial, got {dims}")
    elif np.max(dims) >= traj.d:
        raise ValueError(f"Cannot plot dimensions {dims} for trajectory with d = {traj.d}")

    tplot = np.arange(traj.T)
    if maskNaNs:
        tplot = tplot[~np.any(np.isnan(traj.data), axis=(0, 2))]

    outs = []
    for N in range(traj.N):
        if label is not None:
            kwargs['label'] = label+f"N={N}"
        outs += ax.plot(traj.data[N, tplot, dims[0]],
                        traj.data[N, tplot, dims[1]],
                        **kwargs)

    return outs
