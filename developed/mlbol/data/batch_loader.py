import numpy as np
from typing import Any
from typing import Optional
from typing import Iterator
from typing import Iterable
from typing_extensions import override
from typing_extensions import Self
from mlbol.data.dataset import Dataset

__all__ = ["BatchLoader", "BatchIter", "SingleProcessIter"]

_ITERATOR_RETURN = Any


class BatchIter(Iterator[_ITERATOR_RETURN]):
    """Base class for batch iterators."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int],
        drop_last: bool,
        shuffle: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_idx = 0

    @override
    def __len__(self) -> int:
        num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        if self.drop_last and len(self.dataset) % self.batch_size != 0:
            num_batches -= 1
        return num_batches

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        """Generator function to yield batches of data."""
        raise NotImplementedError

    @override
    def __iter__(self) -> Self:
        """Returns an iterator over batches of data."""
        self.batch_idx = 0
        return self


class SingleProcessIter(BatchIter):
    """Single-process iterator for batches of data."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int],
        drop_last: bool,
        shuffle: bool,
    ) -> None:
        super().__init__(dataset, batch_size, drop_last, shuffle)
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        if self.batch_idx >= len(self):
            raise StopIteration
        start_idx = self.batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        self.batch_idx += 1
        return self.dataset[self.indices[start_idx:end_idx]]


class BatchLoader(Iterable):
    """BatchLoader class for loading batches of data from a dataset."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = None,
        drop_last: bool = False,
        shuffle: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = len(self.dataset) if batch_size is None else batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.iterator = SingleProcessIter(
            self.dataset, self.batch_size, self.drop_last, self.shuffle
        )

    @override
    def __len__(self) -> int:
        """Returns the number of batches that can be generated."""
        return len(self.iterator)

    @override
    def __iter__(self) -> Iterator[_ITERATOR_RETURN]:
        """Returns an iterator over batches of data."""
        return self.iterator.__iter__()
