import contextlib
from collections.abc import Iterable
from typing import Any
from typing import Callable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing_extensions import Self
from typing_extensions import TypedDict
from typing_extensions import override

try:
    from optree import tree_flatten as _tree_flatten
    from optree import tree_unflatten as _tree_unflatten
except:
    from mlbol.utils.tree._pytree import tree_flatten as _tree_flatten
    from mlbol.utils.tree._pytree import tree_unflatten as _tree_unflatten

__all__ = ["TreeLoader"]


def _get_length(obj: object) -> Optional[int]:
    """Try to get the length of an object, return ``None`` otherwise."""
    try:
        length = len(obj)
    except (TypeError, NotImplementedError):
        length = None
    return length


def _get_iterables_lengths(iterables: List[Iterable]) -> List[Union[int, float]]:
    return [
        (float("inf") if (length := _get_length(iterable)) is None else length)
        for iterable in iterables
    ]


_ITERATOR_RETURN = Tuple[Any, int, Union[int, str]]  # batch, batch_idx, dataloader_idx


class _TreeIterator(Iterator[_ITERATOR_RETURN]):
    def __init__(
        self,
        iterables: List[Iterable],
        limits: Optional[List[Union[int, float]]] = None,
    ) -> None:
        if limits is not None and len(limits) != len(iterables):
            raise ValueError(
                f"Mismatch in number of limits ({len(limits)}) and number of iterables ({len(iterables)})"
            )
        self.iterables = iterables
        self.iterators: List[Iterator] = []
        self._idx = 0  # what would be batch_idx
        self.limits = limits

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        raise NotImplementedError

    @override
    def __iter__(self) -> Self:
        self.iterators = [iter(iterable) for iterable in self.iterables]
        self._idx = 0
        return self

    def __len__(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        self.iterators = []
        self._idx = 0


class _MaxSizeCycle(_TreeIterator):
    def __init__(
        self,
        iterables: List[Iterable],
        limits: Optional[List[Union[int, float]]] = None,
    ) -> None:
        super().__init__(iterables, limits)
        self._consumed: List[bool] = []

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterators)
        out = [None] * n  # values per iterator
        for i in range(n):
            try:
                out[i] = next(self.iterators[i])
            except StopIteration:
                self._consumed[i] = True
                if all(self._consumed):
                    raise
                # reset the consumed dataloader
                self.iterators[i] = iter(self.iterables[i])
                out[i] = next(self.iterators[i])
        index = self._idx
        self._idx += 1
        return out, index, 0

    @override
    def __iter__(self) -> Self:
        super().__iter__()
        self._consumed = [False] * len(self.iterables)
        return self

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return max(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[return-value]
        return max(lengths)  # type: ignore[return-value]

    @override
    def reset(self) -> None:
        super().reset()
        self._consumed = []


class _MinSize(_TreeIterator):
    @override
    def __next__(self) -> _ITERATOR_RETURN:
        out = [next(it) for it in self.iterators]
        index = self._idx
        self._idx += 1
        return out, index, 0

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        return min(lengths + self.limits) if self.limits is not None else min(lengths)  # type: ignore[return-value]


class _Sequential(_TreeIterator):
    def __init__(
        self,
        iterables: List[Iterable],
        limits: Optional[List[Union[int, float]]] = None,
    ) -> None:
        super().__init__(iterables, limits)
        self._iterator_idx = 0  # what would be dataloader_idx

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterables)
        if n == 0 or self._iterator_idx >= n:
            raise StopIteration

        # if limits are set, go to the correct iterator
        if self.limits is not None:
            while self.limits[self._iterator_idx] <= self._idx:
                self._use_next_iterator()
                if self._iterator_idx >= n:
                    raise StopIteration

        try:
            out = next(self.iterators[0])
        except StopIteration:
            # try the next iterator
            self._use_next_iterator()
            return self.__next__()
        index = self._idx
        self._idx += 1
        return out, index, self._iterator_idx

    @override
    def __iter__(self) -> Self:
        self._iterator_idx = 0
        self._idx = 0
        self._load_current_iterator()
        return self

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return sum(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[misc]
        return sum(lengths)  # type: ignore[arg-type]

    @override
    def reset(self) -> None:
        super().reset()
        self._iterator_idx = 0

    def _load_current_iterator(self) -> None:
        # Load a single DataLoader, prevents multiple sets of workers from starting unnecessarily
        if self._iterator_idx < len(self.iterables):
            self.iterators = [iter(self.iterables[self._iterator_idx])]
        else:
            # No more iterables to step through, return an empty list
            self.iterators = []

    def _use_next_iterator(self) -> None:
        self._iterator_idx += 1
        self._idx = 0
        self._load_current_iterator()


class _MaxSize(_TreeIterator):
    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterators)
        out = [None] * n
        all_exhausted = True
        for i in range(n):
            with contextlib.suppress(StopIteration):
                out[i] = next(self.iterators[i])
                all_exhausted = False
        if all_exhausted:
            raise StopIteration
        index = self._idx
        self._idx += 1
        return out, index, 0

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return max(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[return-value]
        return max(lengths)  # type: ignore[return-value]


class _ZipMode(TypedDict):
    fn: Callable[[List[int]], int]
    iterator: Type[_TreeIterator]


_SUPPORTED_MODES = {
    "min_size": _ZipMode(fn=min, iterator=_MinSize),
    "max_size_cycle": _ZipMode(fn=max, iterator=_MaxSizeCycle),
    "max_size": _ZipMode(fn=max, iterator=_MaxSize),
    "sequential": _ZipMode(fn=sum, iterator=_Sequential),
}
_LITERAL_SUPPORTED_MODES = Literal[
    "min_size", "max_size_cycle", "max_size", "sequential"
]


class TreeLoader(Iterable):
    """Combines different iterables under specific sampling modes.

    Parameters
    ----------
    iterables : pytree
        The iterable or collection of iterables to sample from.
    mode : {'min_size', 'max_size_cycle', 'max_size', 'sequential'}, default is 'min_size'
        The mode to use. The following modes are supported:
        * ``min_size``: stops after the shortest iterable (the one with the lowest number of items) is done.
        * ``max_size_cycle``: stops after the longest iterable (the one with most items) is done, while cycling
          through the rest of the iterables.
        * ``max_size``: stops after the longest iterable (the one with most items) is done, while returning None
          for the exhausted iterables.
        * ``sequential``: completely consumes each iterable sequentially, and returns a triplet
          ``(data, idx, iterable_idx)``
    """

    def __init__(
        self, iterables: Any, mode: _LITERAL_SUPPORTED_MODES = "min_size"
    ) -> None:
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode {mode!r}, please select one of: {list(_SUPPORTED_MODES)}."
            )
        self._iterables = iterables
        self._flattened, self._spec = _tree_flatten(iterables)
        self._mode = mode
        self._iterator: Optional[_TreeIterator] = None
        self._limits: Optional[List[Union[int, float]]] = None

    @property
    def iterables(self) -> Any:
        """Return the original collection of iterables."""
        return self._iterables

    @property
    def flattened(self) -> List[Any]:
        """Return the flat list of iterables."""
        return self._flattened

    @flattened.setter
    def flattened(self, flattened: List[Any]) -> None:
        """Setter to conveniently update the list of iterables."""
        if len(flattened) != len(self._flattened):
            raise ValueError(
                f"Mismatch in flattened length ({len(flattened)}) and existing length ({len(self._flattened)})"
            )
        # update the iterable collection
        self._iterables = _tree_unflatten(treespec=self._spec, leaves=flattened)
        self._flattened = flattened

    @property
    def limits(self) -> Optional[List[Union[int, float]]]:
        """Optional limits per iterator."""
        return self._limits

    @limits.setter
    def limits(
        self, limits: Optional[Union[int, float, List[Union[int, float]]]]
    ) -> None:
        if isinstance(limits, (int, float)):
            limits = [limits] * len(self.flattened)
        elif isinstance(limits, list) and len(limits) != len(self.flattened):
            raise ValueError(
                f"Mismatch in number of limits ({len(limits)}) and number of iterables ({len(self.flattened)})"
            )
        self._limits = limits

    def __next__(self) -> _ITERATOR_RETURN:
        assert self._iterator is not None
        out = next(self._iterator)
        if isinstance(self._iterator, _Sequential):
            return out
        out, batch_idx, dataloader_idx = out
        return (
            _tree_unflatten(treespec=self._spec, leaves=out),
            batch_idx,
            dataloader_idx,
        )

    @override
    def __iter__(self) -> Self:
        cls = _SUPPORTED_MODES[self._mode]["iterator"]
        iterator = cls(self.flattened, self._limits)
        iter(iterator)
        self._iterator = iterator
        return self

    def __len__(self) -> int:
        """Compute the number of batches."""
        if self._iterator is None:
            raise RuntimeError("Please call `iter(combined_loader)` first.")
        return len(self._iterator)

    def reset(self) -> None:
        """Reset the state and shutdown any workers."""
        if self._iterator is not None:
            self._iterator.reset()
            self._iterator = None
