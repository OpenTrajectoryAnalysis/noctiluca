"""
Convert user data to noctiluca objects (Trajectory, TaggedSet)

This module provides convenience functions to convert from various formats in
which users might provide their data to the objects that noctiluca and
libraries building on it use. These functions are designed to be used under the
hood in downstream libraries, such that beginning users do not have to go the
extra step of creating Trajectory/TaggedSet objects from their data. In the
long run it is of course recommended to get familiar with these objects, as
they provide some useful additional functionality.

For trajectories, the following inputs are accepted:

+ ``noctiluca.Trajectory``, which is returned unchanged
+ ``pandas.DataFrame``; columns should be named ``'x', 'y', 'z'`` to indicate
  dimension, possibly with a suffix identifying different loci. So for example

  - ``('x', 'y', 'x2', 'y2')`` would indicate a trajectory with two loci in two
    spatial dimensions
  - equivalent column specifications to the above include ``('x_left',
    'x_right', 'y_left', 'y_right')``, ``(x0, y0, x1, y1)``, ``('x_red',
    'y_red', 'x_green', y_green')``
  - the precise regular expression to identify columns containing positional
    data is ``'(x|y|z)(\d*|_.*)'``.
  - for each locus all dimensions should exist, i.e. ``('x0', 'y0', 'x0')``
    would be invalid
  - dimensions should be labelled starting with ``'x'``, i.e. a dataframe
    containing just a ``'y'`` column would be invalid

  Beyond the positional data, the dataframe may contain a column ``'frame'``
  indicating the frame number that each detection belongs to. This column will
  be coerced to integer.
+ numpy array-like; anything that casts well to a numpy array is accepted. This
  case is simply forwarded to the ``noctiluca.Trajectory`` constructor, which
  accepts array-like structures. Possible shapes for array-like inputs:

  - ``(N, T, d)`` for a trajectory with ``N`` loci in ``d`` spatial dimensions
    over ``T`` time steps
  - ``(T, d)`` for a single-locus trajectory
  - ``(T,)`` for a trajectory with a single locus in a single spatial
    dimension.

+ ``pandas.Series``, which is cast to a one dimensional numpy array. In doing
  so, we use the index of the series as frame number, i.e. ``pd.Series(data=[1,
  2, 3], index=[5, 7, 8])`` would be read as a trajectory with 4 frames, where
  the second one is missing.

For datasets (i.e. a collection of trajectories) we accept

+ ``noctiluca.TaggedSet``, which is returned unchanged
+ any iterable of structures that `make_Trajectory` can deal with. Specifically
  this includes a list of numpy arrays (which then can be of different length /
  dimension) or a big numpy array, where the different trajectories are indexed
  along the first dimension.
+ ``pandas.DataFrame`` with column names conforming to the patterns outlined
  above for `make_Trajectory`, plus an additional column ``'particle'``
  indicating which detections belong to which trajectory. The name
  ``'particle'`` for this column was chosen for compatibility with `!trackpy`.
  Note that contrary to `make_Trajectory`, the specification of the ``'frame'``
  column is now mandatory.

More control over the input process (e.g. the assumed column names for pandas
dataframes) can be attained by using the functions in this module explicitly,
in which case additional keyword arguments can be specified. For yet finer
control we recommend directly using the ``noctiluca.Trajectory`` and
``noctiluca.TaggedSet`` classes.
"""
import re

import numpy as np
import pandas as pd

from ..trajectory import Trajectory
from ..taggedset import TaggedSet

def make_Trajectory(inpt, **kwargs):
    """
    Create a ``noctiluca.Trajectory`` from user specified data

    Parameters
    ----------
    inpt : <various>
        see `module docstring <noctiluca.util.userinput>` for description.

    Keyword Arguments
    -----------------
    t_column : str
        name of the column containing frame numbers when giving a
        ``pandas.DataFrame`` as `!inpt`
    pos_columns : (list of) list of str
        names of the columns containing the positional data when giving a
        ``pandas.DataFrame`` as `!inpt`. This should be either a list of
        identifiers for the different dimensions (e.g. ``['x', 'y']``) or a
        list of such lists, if the trajectory has multiple loci (e.g. ``[['x1',
        'y1'], ['x2', 'y2']]``).

    Returns
    -------
    noctiluca.Trajectory
    """
    # Preproc
    if isinstance(inpt, pd.Series):
        inpt = inpt.dropna()
        ind = inpt.index.to_numpy()
        if not np.issubdtype(ind.dtype, np.integer):
            raise ValueError("Could not interpret index of pandas.Series object as integer frame numbers")

        ind -= np.min(ind)
        arr = np.empty(np.max(ind)+1, dtype=float)
        arr[:] = np.nan
        arr[ind] = inpt
        inpt = arr

    # Main processing logic
    if isinstance(inpt, Trajectory):
        return inpt
    elif isinstance(inpt, pd.DataFrame):
        # Find the right entries in the dataframe
        if 't_column' not in kwargs: # str or None, then assume contiguous data
            kwargs['t_column'] = 'frame'
        if 'pos_columns' not in kwargs: # (list of) list of str
            pcols = [col for col in inpt.columns if re.fullmatch(r'(x|y|z)(\d*|_.*)', col)]
            d = max([ord(col[0]) - ord('x') for col in pcols]) + 1
            locus_ids = np.unique([col[1:] for col in pcols])
            N = len(locus_ids)

            pcols_expect = [[chr(ord('x')+dim)+locus_id for dim in range(d)]
                            for locus_id in locus_ids]
            if any(col not in pcols for dcols in pcols_expect for col in dcols) or len(pcols) > N*d:
                raise ValueError(f"Detected positional columns {pcols} are incompatible with expected/inferred names {pcols_expect}")

            kwargs['pos_columns'] = pcols_expect

        # Prepare for assembly
        try:
            frame = inpt[kwargs['t_column']].to_numpy().astype(int)
        except KeyError:
            frame = inpt.index.to_numpy()
            if not np.issubdtype(frame.dtype, np.integer):
                raise ValueError("Could not interpret index of pandas.DataFrame object as integer frame numbers")
        frame = frame - np.min(frame)
        T = np.max(frame) + 1

        pcol = np.array(kwargs['pos_columns'])
        if len(pcol.shape) == 1:
            pcol = np.expand_dims(pcol, 0)
        N, d = pcol.shape

        # Assemble
        data = np.empty((N, T, d), dtype=float)
        data[:] = np.nan
        for n in range(N):
            for dim in range(d):
                data[n, frame, dim] = inpt[pcol[n, dim]]

        return Trajectory(data)
    else: # assume input is castable to numpy array and just expose Trajectory() constructor
        return Trajectory(inpt, **kwargs)
        
def make_TaggedSet(inpt, **kwargs):
    """
    Convert a collection of trajectories to ``noctiluca.TaggedSet``

    Parameters
    ----------
    inpt : <various>
        see `module docstring <noctiluca.util.userinput>` for description.

    Keyword Arguments
    -----------------
    id_column : str
        the name of the column containing trajectory identifiers (when
        specifying a ``pandas.DataFrame``). The data in this column does not
        have to be numerical. Defaults to ``'particle'``.
    **kwargs : <other keyword arguments>
        forwarded to `make_Trajectory`. Note ``'t_column'`` and
        ``'pos_columns'``.
    """
    if isinstance(inpt, TaggedSet):
        return inpt
    elif isinstance(inpt, pd.DataFrame):
        if 'id_column' not in kwargs:
            kwargs['id_column'] = 'particle' # from trackpy
        if 't_column' not in kwargs:
            kwargs['t_column'] = 'frame'

        ids = inpt[kwargs['id_column']].unique()
        out = TaggedSet()
        for id in ids:
            ind = inpt[kwargs['id_column']] == id
            out.add(make_Trajectory(inpt.loc[ind], **kwargs))

        return out
    else: # assume iterable of something that make_Trajectory() can deal with
        return TaggedSet((make_Trajectory(x, **kwargs) for x in inpt), hasTags=False)
