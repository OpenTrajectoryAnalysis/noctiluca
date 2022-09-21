from copy import deepcopy
import random

from . import parallel

class TaggedSet():
    """
    A set with tags for each object.

    This class can be used as an iterator in constructs such as ``for datum in
    myset: ...``. The subset to be iterated over can be adjusted using
    `makeSelection`. If you also need the tags for each datum, use
    ``myset(giveTags=True)``.

    For processing data (in the sense of "applying a function to all of them")
    it is often useful to simply use the built-in `!map`. If the function in
    question has to overwrite the data, you can use `apply`. Note that
    in-place modification can be achieved with `!map()`.

    Parameters
    ----------
    iterable : iterable of data or (datum, tags) pairs, optional
        most useful things to use here: an actual list, or a generator.
        Remember to set `!hasTags` accordingly.
    hasTags : bool, optional
        whether the iterable in the first argument gives data only or (datum,
        tags) pairs.

    Notes
    -----
    For a TaggedSet ``myset``, the following operations are defined:

    ``len(myset)``
        return number of data in current selection
    ``myset(giveTags)``
        return a generator yielding single data or (datum, tags) pairs,
        depending on the value of `!giveTags` (False by default).
    ``iter(myset)``
        shortcut for ``myset(giveTags=False)``. This enables the construction
        ``for datum in myset: ...``
    ``myset[i]``
        element access within the current selection. You can use indices from
        ``0`` to ``len(myset)-1``.
    ``myset |= otherset``
        add the data in `!otherset` into `!myset`. This is a shortcut for
        ``myset.mergein(otherset)``.
    """

    ### Utility ###

    @staticmethod
    def makeTagsSet(tags):
        """
        Input reformatting, mostly internal use.

        Make sure that the frequently used `!tags` argument is a set of strings.

        Parameters
        ----------
        tags : str, list of str, set of str, or None

        Returns
        -------
        set of str
        """
        if tags is None:
            return set()
        elif isinstance(tags, str):
            tags = {tags}
        elif isinstance(tags, list):
            tags = set(tags)
        elif not isinstance(tags, set):
            raise ValueError("Did not understand type of 'tags' ({})".format(str(type(tags))))

        return tags

    ### Setting up ###

    def __init__(self, iterable=[], hasTags=True):
        self._data = []
        self._tags = []
        self._selected = []

        if hasTags:
            for datum, tags in iterable:
                self.add(datum, tags)
        else:
            for datum in iterable:
                self.add(datum)

    def add(self, datum, tags=None):
        """
        Append the given datum with the given tags.

        Parameters
        ----------
        datum : object
            the datum to append
        tags : str, list of str, or set of str, optional
            the tags to attach to this new datum

        Notes
        -----
        The newly added datum will be part of the current selection.
        """
        tags = self.makeTagsSet(tags)

        self._data.append(datum)
        self._tags.append(tags)
        self._selected.append(True)

    ### Basic usage ###

    def __iter__(self):
        return self()

    def __call__(self, giveTags=False, randomize=False):
        """
        Iterate through the current selection of the set.

        Parameters
        ----------
        giveTags : bool, optional
            whether to yield only data or (datum, tags) pairs.
        randomize : bool, optional
            set to ``True`` to randomize the order in which data are yielded.

        Yields
        ------
        datum
            the data in the current selection
        set of str, optional
            the tags associated with the datum

        See also
        --------
        makeSelection
        """
        indices = range(len(self._data))
        if randomize:
            indices = random.sample(indices, k=len(indices))

        for i in indices:
            if self._selected[i]:
                if giveTags:
                    yield (self._data[i], self._tags[i])
                else:
                    yield self._data[i]

    def __len__(self):
        return sum(self._selected)

    def __getitem__(self, ind):
        for i, datum in enumerate(self):
            if i >= ind:
                return datum

    ### Selection system ###

    def makeSelection(self, arg=None, **kwargs):
        """
        Mark a subset of the current set as 'current selection'. For most
        purposes, the set will behave as if it contained only these data.

        There are multiple ways to select data. Which one is used depends on
        the kwargs given. With increasing preference, these methods are

        - select randomly: use the kwargs `!nrand` or `!prand`
        - select by tag: use the kwargs `!tags` and `!logic`
        - select with a user-specified function: use kwarg `!selector`

        Call this without arguments to reset the selection to the whole
        dataset.

        Parameters
        ----------
        arg : <various>
            catch-all argument that dispatches to any of the below by type
            (e.g. ``str`` is interpreted as single tag, ``float`` as `!prand`,
            any iterable as `!tags`, etc.)
        nrand : int
            number of trajectories to select at random
        prand : float, in [0, 1]
            fraction of trajectories to select at random
        random_seed : int, str, or bytes, optional
            seed for random selection. Note that this can be a string.
        tags : str, list of str, or set of str
            the tags to select. How these go together will be determined by
            `!logic`.
        logic : callable with signature bool = logic(<list of bool>)
            the logic for handling multiple tags. Set this to (the built-in)
            `!all` to select the data being tagged with all the given tags, or
            to `!any` (the default) to select the data having any of the given
            tags.
        selector : callable with signature bool = selector(datum, tags)
            should expect the datum and a set of tags as input and return True
            if the datum is to be selected, False otherwise.

        Other Parameters
        ----------------
        refining : bool, optional
            set to True to apply the current selection scheme only to those
            data that are already part of the selection. This can be used to
            successively refine a selection by using different selection
            methods (e.g. first by tag, then by some other criterion).

        See also
        --------
        refineSelection, saveSelection, restoreSelection
        """
        # dispatch arg if specified
        if arg is not None:
            if type(arg) == int:
                kwargs['nrand'] = arg
            elif type(arg) == float:
                kwargs['prand'] = arg
            elif type(arg) == str:
                kwargs['tags'] = arg
            elif callable(arg):
                kwargs['selector'] = arg
            elif hasattr(arg, '__iter__'):
                kwargs['tags'] = arg
            else:
                raise ValueError(f"Did not understand argument of type {type(arg)}")

            return self.makeSelection(**kwargs)

        assert len(self._data) == len(self._tags) == len(self._selected)
        if not 'refining' in kwargs.keys():
            kwargs['refining'] = False

        # Define selector function according to given arguments
        if 'selector' in kwargs.keys():
            selector = kwargs['selector']
        elif 'tags' in kwargs.keys():
            kwargs['tags'] = TaggedSet.makeTagsSet(kwargs['tags'])
            if not 'logic' in kwargs.keys():
                kwargs['logic'] = any
            def selector(datum, tags):
                return kwargs['logic']([tag in tags for tag in kwargs['tags']])
        elif 'nrand' in kwargs.keys() or \
             'prand' in kwargs.keys():
            curlen = sum(self._selected) if kwargs['refining'] else len(self._data)
            try:
                nrand = kwargs['nrand']
            except KeyError:
                nrand = int(kwargs['prand']*curlen)
            if 'random_seed' in kwargs.keys():
                random.seed(kwargs['random_seed'])
            toselect = random.sample(range(curlen), nrand)
            def selector(datum, tags):
                # Note: have to hack a bit to get a static variable
                try:
                    return selector.cnt in toselect
                except AttributeError:
                    selector.cnt = 0
                    return selector.cnt in toselect
                finally:
                    selector.cnt += 1
        else:
            def selector(datum, tags):
                return True

        for i, (datum, tags, selected) in enumerate(zip(self._data, self._tags, self._selected)):
            if selected or not kwargs['refining']:
                self._selected[i] = bool(selector(datum, tags))

    def refineSelection(self, *args, **kwargs):
        """
        A wrapper for ``makeSelection(..., refining=True)``.

        See also
        --------
        makeSelection
        """
        kwargs['refining'] = True
        self.makeSelection(*args, **kwargs)

    def saveSelection(self):
        """
        Return a copy of the current selection for reference.

        This is useful when planning to undo subsequent selections without
        having to redo the whole selection from scratch.

        Returns
        -------
        list of bool
            the current selection.

        See also
        --------
        restoreSelection, makeSelection
        """
        return deepcopy(self._selected)

    def restoreSelection(self, selection):
        """
        Restore a previously saved selection.

        Parameters
        ----------
        selection : list of bool
            the selection that was saved using `saveSelection`. Will be copied.

        See also
        --------
        saveSelection, makeSelection
        """
        self._selected = deepcopy(selection)

    def copySelection(self):
        """
        Generate a new (copied) set from the current selection.

        Returns
        -------
        The new set

        See also
        --------
        makeSelection
        """
        def gen():
            for traj, tags in self(giveTags=True):
                yield(deepcopy(traj), deepcopy(tags))

        return TaggedSet(gen())

    def deleteSelection(self):
        """
        Remove the selected entries from the set

        See also
        --------
        makeSelection, copySelection

        Notes
        -----
        Since everything in the current selection is deleted, the selection
        will be reset.
        """
        self._data = [datum for datum, selected in zip(self._data, self._selected) if not selected]
        self._tags = [tagset for tagset, selected in zip(self._tags, self._selected) if not selected]
        self._selected = len(self._data)*[True]

    ### Managing multiple TaggedSets ###

    def mergein(self, other, additionalTags=None):
        """
        Add the contents of the TaggedSet `!other` to the caller.

        Parameters
        ----------
        other : TaggedSet
            the set whose data to add
        additionalTags : str, list of str, or set of str, optional
            additional tag(s) to add to all of the new data.

        Notes
        -----
        This can also be invoked as ``self |= other``. In that case no
        additional tags can be added.
        """
        if not issubclass(type(other), TaggedSet): # pragma: no cover
            raise TypeError("Can only merge a TaggedSet")

        additionalTags = TaggedSet.makeTagsSet(additionalTags)
        newTags = [tags | additionalTags for tags in other._tags]

        self._data += other._data
        self._tags += newTags
        self._selected += other._selected

    def __ior__(self, other):
        self.mergein(other)
        return self

    ### Global tag management ###

    def addTags(self, tags):
        """
        Add new tag(s) to all data in the current selection

        Parameters
        ----------
        tags : str, list of str, or set of str
            the tag(s) to add
        """
        tags = TaggedSet.makeTagsSet(tags)

        for _, curtags in self(giveTags=True):
            curtags |= tags

    def tagset(self):
        """
        Return the set of all tags in the current selection.

        Returns
        -------
        set of str
            set of all tags in the current selection
        """
        tagset = set()
        for _, tags in self(giveTags=True):
            tagset |= tags

        return tagset


    ### Processing data in the set ###

    @staticmethod
    def _apply_single(args):
        i, datum, tags, fun = args
        return i, fun(datum), tags

    def apply(self, fun, inplace=False):
        """
        Apply a given function to all data.

        Parameters
        ----------
        fun: callable
            should have signature ``datum = fun(datum)``. Will be mapped across
            the current selection.
        inplace : bool, optional
            set to ``True`` to run in place, i.e. overwrite the original
            entries. Otherwise, a new ``TaggedSet`` with the processed data
            will be returned

        Returns
        -------
        TaggedSet or None

        See also
        --------
        builtins.map

        Notes
        -----
        This function is applicable when modifying the actual data in the
        ``TaggedSet``; for simply applying a function to all data in the set,
        use python's built-in ``map()``. The benefit of ``apply()`` over a
        construction like ``TaggedSet(map(fun, data))`` is just that it keeps
        the tags in order.

        This function is parallel-aware (ordered).
        """
        all_iter = enumerate(zip(self._data, self._selected, self._tags))
        todo = [(i, datum, tags, fun) for i, (datum, selected, tags) in all_iter if selected]
        imap = parallel._map(TaggedSet._apply_single, todo)

        if inplace:
            for i, datum, _ in imap:
                self._data[i] = datum

            return None
        else:
            out = TaggedSet()
            for _, datum, tags in imap:
                out.add(datum, tags)

            return out

    def map_unique(self, fun):
        """
        Apply a function to all data and check that the result is unique.

        Parameters
        ----------
        fun : callable with signature ``ret = fun(datum)``
            the function to apply. The return value can be any object for which
            ``ret1 == ret2`` is defined.

        Returns
        -------
        ret : object
            if `!fun` returns the same value on all data, that value.

        Raises
        ------
        RuntimeError
            if the return value of `!fun` is not the same on all data.

        See also
        --------
        builtins.map

        Notes
        -----
        Returns ``None`` if ``len(self) == 0`` (i.e. the caller / its current
        selection does not contain any data)
        """
        it = map(fun, self)
        try:
            first = next(it)
        except StopIteration:
            return None
        if all(first == rest for rest in it):
            return first
        else:
            raise RuntimeError("TaggedSet.map_unique() called on data that do not give uniform values")
