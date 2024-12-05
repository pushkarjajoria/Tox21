from itertools import count, product
import numpy as np


class SparseBoolArray:

    def __init__(self, shape, indices=None, broadcast=()):
        self.shape = tuple(shape)
        self.broadcast = tuple(np.unique(broadcast))
        assert len(self.broadcast) == len(broadcast), f"Duplicate dimensions in broadcast: {broadcast}"
        self.regular = self._reduce_it(np.arange(len(self.shape)), self.broadcast)
        if indices is None:
            self.indices = np.zeros((0, len(self.regular)), dtype=np.int64)
        else:
            self.indices = np.array(indices, dtype=np.int64)
        assert self.indices.shape[1] == len(self.regular), \
            f"Index array has wrong shape {self.indices.shape} for array of shape {self.shape} " \
            f"with broadcasting dimensions {self.broadcast}: " \
            f"second dimension should be of size {len(self.regular)} but is {self.indices.shape[1]}"

    @staticmethod
    def from_numpy(ndarray):
        indices = np.nonzero(ndarray)
        indices = np.array(indices).T
        return SparseBoolArray(ndarray.shape, indices)

    def expand(self, dims, sizes=None):
        if sizes is None:
            sizes = (1,) * len(dims)
        broadcast = np.zeros(len(self.shape), dtype=bool)
        if self.broadcast:
            broadcast[(np.array(self.broadcast),)] = True
        broadcast = np.array(self._expand_it(broadcast, dims, fill_value=True))
        broadcast = np.nonzero(broadcast)[0]
        shape = self._expand_it(self.shape, dims, fill_value=sizes, iter_fill=True)
        return SparseBoolArray(shape=shape, indices=self.indices.copy(), broadcast=broadcast)

    @classmethod
    def _expand_it(cls, it, dims, fill_value=None, iter_fill=False):
        t = []
        it = iter(it)
        if iter_fill:
            fill_value = iter(fill_value)
        for idx in count():
            if idx in dims:
                if iter_fill:
                    t.append(next(fill_value))
                else:
                    t.append(fill_value)
            else:
                try:
                    t.append(next(it))
                except StopIteration:
                    break
        return tuple(t)

    @classmethod
    def _reduce_it(cls, it, dims):
        return tuple(np.delete(it, dims))
        # return tuple(x for idx, x in enumerate(it) if idx not in dims)

    def _complement(self, it, size=None):
        if size is None:
            size = len(self.shape)
        return SparseBoolArray._reduce_it(np.arange(size), it)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, indices={self.indices}, broadcast={self.broadcast})"

    def _is_empty(self):
        return not bool(self.indices.shape[0])

    def copy(self):
        return SparseBoolArray(indices=self.indices.copy(), shape=self.shape, broadcast=self.broadcast)

    def to_dense(self, zeros=np.zeros):
        arr = zeros(self.shape, dtype=bool)
        if self.broadcast:
            indices = self._expand_it(self.indices.transpose(), self.broadcast, fill_value=slice(None))
        else:
            indices = tuple(self.indices.transpose())
        if not self._is_empty():
            arr[indices] = True
        return arr

    def fill_dims(self, dims, empty_ok=False):
        """
        Return a new array of same shape but with specified broadcast dimensions filled. The two arrays describe the
        same underlying values, but filled dimensions list them explicitly.
        :param dims: tuple of ints specifying dimensions to fill (must be broadcast dimensions)
        :param empty_ok: whether dims may be an empty tuple, in which case the original array is returned NOT a copy
        :return: filled array
        """
        if not isinstance(dims, tuple):
            raise TypeError(f"'dims' must be a tuple of integers but is {dims} of type {type(dims)}")
        if len(dims) == 0:
            if empty_ok:
                return self
            else:
                raise ValueError("'dims' is empty; use empty_ok=True to return oritinal array for that case")
        if not set(dims).issubset(set(self.broadcast)):
            raise ValueError(f"Some of the dimensions to fill {dims} are not broadcast dimensions {self.broadcast}")
        new_regular, new_broadcast, new_indices = self.regular, self.broadcast, self.indices
        for d in dims:
            new_regular, new_broadcast, new_indices = self._fill_dims_helper(d=d, s=self.shape[d],
                                                                             regular=new_regular,
                                                                             broadcast=new_broadcast,
                                                                             indices=new_indices)
        return SparseBoolArray(shape=self.shape, indices=new_indices, broadcast=new_broadcast)

    @classmethod
    def _fill_dims_helper(cls, d, s, regular, broadcast, indices):
        """
        fill a broadcasting dimension my repeating existing indices while adding new running index
        :param d: index of the filling dimension
        :param s: size of the filling dimension
        :param regular: indices of old regular dimensions
        :param broadcast: indices of old broadcasting dimensions
        :param indices: old indices
        :return: new_regular, new_broadcast, new_indices
        """
        # the indices of the new regular dimensions (with d added)
        new_regular = tuple(sorted(regular + (d,)))
        # the indices of the new broadcast dimensions (with d removed)
        new_broadcast = tuple(x for x in broadcast if x != d)
        # the index of the filling dimension in the index array (which only lists regular dimensions)
        i = np.argwhere(np.array(new_regular) == d)[0, 0]
        # construct new indices as 2D list of indices
        # - use broadcasting for assigning values
        # - later reshape to 1D list of indices
        new_indices = np.empty((indices.shape[0], s, len(new_regular)))
        # assign indices of old dimensions that come BEFORE filled dimension
        if i > 0:
            new_indices[:, :, :i] = indices[:, None, :i]
        # assign indices of old dimensions that come AFTER filled dimension
        new_indices[:, :, i + 1:] = indices[:, None, i:]
        # assign filling indices
        new_indices[:, :, i] = np.arange(s)[None, :]
        return new_regular, new_broadcast, new_indices.reshape(indices.shape[0] * s, len(new_regular))

    def any(self, dims=None):
        """
        ...resulting array may be uncoalesced
        :param dims:
        :return:
        """
        if dims is None:
            return bool(len(self.indices))
        elif dims == ():
            return self.copy()
        else:
            assert isinstance(dims, tuple), "dims must be a tuple"
            assert len(dims) == len(np.unique(dims)), f"dims has duplicate entries: {dims}"
            all_dims = np.arange(len(self.shape))
            remaining_dims = self._reduce_it(all_dims, dims)
            remaining_shape = tuple(np.array(self.shape)[np.array(remaining_dims)])
            remaining_regular_indices = self._reduce_it(
                self._expand_it(np.arange(len(self.regular)), self.broadcast),
                dims + self.broadcast)
            if self._is_empty():
                indices = None
            else:
                indices = self.indices.copy()[:, remaining_regular_indices]
            if self.broadcast:
                new_broadcast = np.zeros(len(self.shape), dtype=bool)
                new_broadcast[(np.array(self.broadcast),)] = True
                new_broadcast = np.delete(new_broadcast, dims)
                new_broadcast = np.argwhere(new_broadcast)
            else:
                new_broadcast = ()
            return SparseBoolArray(indices=indices, shape=remaining_shape, broadcast=new_broadcast)

    def coalesce(self):
        """Return a coalesced copy of the array (i.e. without duplicate indices)"""
        return SparseBoolArray(shape=self.shape, indices=np.unique(self.indices, axis=0), broadcast=self.broadcast)

    @classmethod
    def _intersect2D(cls, A, B, copy_A=True, copy_B=True):
        # https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
        # copy to have consistent memory layout
        if copy_A:
            A = A.copy()
        if copy_B:
            B = B.copy()
        nrows, ncols = A.shape
        dtype = dict(names=['f{}'.format(i) for i in range(ncols)],
                     formats=ncols * [A.dtype])
        C = np.intersect1d(A.view(dtype), B.view(dtype))
        return C.view(A.dtype).reshape(-1, ncols)

    def logical_and(self, other):
        assert isinstance(other, SparseBoolArray)
        assert self.shape == other.shape, f"Arrays have different shape: {self.shape} versus {other.shape}"
        shape = self.shape
        if self.broadcast or other.broadcast:
            # get dimensions that
            # - both arrays broadcast along
            # - none of both broadcasts along (both are regular)
            # - at least one broadcasts along (complement of both regular)
            both_broadcast = tuple(np.intersect1d(self.broadcast, other.broadcast))
            both_regular = tuple(np.intersect1d(self.regular, other.regular))
            some_broadcast = self._complement(both_regular)
            # indices of dimensions in intersection space
            reduced_dims_1 = SparseBoolArray._expand_it(np.arange(len(self.regular)), self.broadcast)
            reduced_dims_1 = SparseBoolArray._reduce_it(reduced_dims_1, some_broadcast)
            reduced_dims_1 = np.array(reduced_dims_1)
            reduced_dims_2 = SparseBoolArray._expand_it(np.arange(len(other.regular)), other.broadcast)
            reduced_dims_2 = SparseBoolArray._reduce_it(reduced_dims_2, some_broadcast)
            reduced_dims_2 = np.array(reduced_dims_2)
            # collect indices grouped by location in intersection space
            index_groups = {}
            if both_regular:
                for s_idx, indices, reduced in [(0, self.indices, self.indices[:, reduced_dims_1]),
                                                (1, other.indices, other.indices[:, reduced_dims_2])]:
                    for i, r in zip(indices, reduced):
                        r = tuple(r)
                        if r not in index_groups:
                            index_groups[r] = [[], []]
                        index_groups[r][s_idx].append(i)
            else:
                index_groups[None] = [self.indices, other.indices]
            # for a given group, all possible combinations are valid
            all_indices = []
            for indices_1, indices_2 in index_groups.values():
                if len(indices_1) and len(indices_2):
                    indices_1 = np.array(indices_1)
                    indices_2 = np.array(indices_2)
                    new_indices = np.full((indices_1.shape[0], indices_2.shape[0], len(shape)), fill_value=-1,
                                          dtype=np.int64)
                    new_indices[:, :, self.regular] = indices_1[:, None, :]
                    new_indices[:, :, other.regular] = indices_2[None, :, :]
                    new_indices = new_indices.reshape(-1, len(shape))
                    new_regular = SparseBoolArray._reduce_it(np.arange(len(shape)), both_broadcast)
                    new_indices = new_indices[:, new_regular]
                    all_indices.append(new_indices)
            if all_indices:
                indices = np.concatenate(all_indices, axis=0)
            else:
                indices = None
            return SparseBoolArray(indices=indices, shape=shape, broadcast=both_broadcast)
        else:
            return SparseBoolArray(indices=self._intersect2D(self.indices, other.indices), shape=shape)

    def logical_or(self, other):
        """
        ...resulting array may be uncoalesced
        :param self:
        :param other:
        :return:
        """
        assert isinstance(other, SparseBoolArray)
        assert self.shape == other.shape, f"Arrays have different shape: {self.shape} versus {other.shape}"
        shape = self.shape
        if self.broadcast or other.broadcast:
            # dimension along which both broadcast (result will broadcast along those too)
            both_broadcast = tuple(np.intersect1d(self.broadcast, other.broadcast))
            # dimensions along at least one in regular (result will be regular along those)
            some_regular = self._complement(both_broadcast)
            # fill broadcast dimensions that need to be regular in result
            s1_filled = self.fill_dims(tuple(np.intersect1d(self.broadcast, some_regular)), empty_ok=True)
            s2_filled = other.fill_dims(tuple(np.intersect1d(other.broadcast, some_regular)), empty_ok=True)
            return SparseBoolArray(shape=shape,
                                   indices=np.concatenate((s1_filled.indices, s2_filled.indices), axis=0),
                                   broadcast=both_broadcast)
        else:
            return SparseBoolArray(shape=shape, indices=np.concatenate((self.indices, other.indices), axis=0))


def logical_and(s1: SparseBoolArray, s2: SparseBoolArray):
    assert isinstance(s1, SparseBoolArray)
    return s1.logical_and(s2)


def logical_or(s1: SparseBoolArray, s2: SparseBoolArray):
    assert isinstance(s1, SparseBoolArray)
    return s1.logical_or(s2)
