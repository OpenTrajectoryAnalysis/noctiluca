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
