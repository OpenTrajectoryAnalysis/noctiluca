"""
Provide simple plotting capabilities
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

from . import analysis

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

    Notes
    -----
    If `!ax` is specified, the trajectory plot is simply added to the existing
    axes; if it is unspecified, we plot to ``plt.gca()`` and adjust some optics
    (e.g. constrain xticklabels to be integers)
    """
    if ax is None:
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

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

    Notes
    -----
    If `!ax` is specified, the trajectory plot is simply added to the existing
    axes; if it is unspecified, we plot to ``plt.gca()`` and adjust some optics
    (e.g. constrain xticklabels to be integers)
    """
    if ax is None:
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

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

def msd_overview(dataset, ax=None, dt=1., **kwargs):
    """
    Plot individual and ensemble MSDs of the given dataset

    All keyword arguments are forwarded to ``plt.plot()`` for plotting of the
    individual trajectory MSDs

    Parameters
    ----------
    dataset : `TaggedSet` of `Trajectory`
        the dataset to use
    ax : matplotlib axes, optional
        default is to plot to ``plt.gca()``
    dt : float, optional
        the time step between two frames of the trajectory; this will simply
        rescale the horizontal axis of the plot.

    Returns
    -------
    list of Line2D
        all the lines added to the axes. The last entry in the list is the plot
        of the ensemble mean
    """
    if ax is None:
        ax = plt.gca()

    try:
        ensemble_label = kwargs['label']
        del kwargs['label']
    except KeyError:
        ensemble_label = 'ensemble mean'

    # individual trajectories
    lines = []
    for traj in dataset:
        msd = analysis.MSD(traj)
        tmsd = dt*np.arange(len(msd))
        lines += ax.plot(tmsd[1:], msd[1:], **kwargs)

    # ensemble mean
    msd = analysis.MSD(dataset)
    tmsd = dt*np.arange(len(msd))
    lines += ax.plot(tmsd[1:], msd[1:],
                     color='k',
                     linewidth=2,
                     label=ensemble_label,
                     )

    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    return lines
