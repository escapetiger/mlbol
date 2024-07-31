import math
import bisect
import warnings
import numpy as np
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import List
from typing import Any
from typing import Iterable
from typing import Dict
from typing import Optional
from typing import TypeVar
from itertools import accumulate
from mlbol.dtensor import Tensor
from mlbol.dtensor import size

_IndexLike = TypeVar("_IndexLike")

__all__ = [
    "Dataset",
    "ConcatDataset",
    "TensorDataset",
    "SequenceTensorDataset",
    "MappingTensorDataset",
    "Subset",
    "partition",
]


class Dataset:
    """An abstract class representing a dataset.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses should also overwrite :meth:`__len__`,
    which is expected to return the size of the dataset.
    """

    def __len__(self) -> int:
        """Return the size of the dataset."""
        raise NotImplementedError

    def __getitem__(self, index: _IndexLike) -> Any:
        """Fetch a data sample with a given index.

        Parameters
        ----------
        index : index_like
            Index of the data sample to retrieve.

        Returns
        -------
        Any
            The requested data sample.
        """
        raise NotImplementedError

    def __add__(self, other: "Dataset") -> "ConcatDataset":
        return ConcatDataset([self, other])


class ConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Parameters
    ----------
    datasets : Iterable[Dataset]
        List of datasets to be concatenated
    """

    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence: Iterable[Dataset]) -> List[int]:
        """
        Compute the cumulative sum of the lengths of the datasets.

        Parameters
        ----------
        sequence : Iterable[Dataset]
            List of datasets.

        Returns
        -------
        List[int]
            Cumulative sizes of the datasets.
        """
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        """
        Initialize ConcatDataset with a list of datasets.

        Parameters
        ----------
        datasets : Iterable[Dataset]
            List of datasets to be concatenated.
        """
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self) -> int:
        """
        Return the total length of the concatenated dataset.

        Returns
        -------
        int
            Total length of the concatenated dataset.
        """
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: _IndexLike) -> Any:
        """
        Retrieve a data sample from the concatenated dataset.

        Parameters
        ----------
        index : _IndexLike
            Index or slice to retrieve the data sample(s).

        Returns
        -------
        Any
            The requested data sample or samples.
        """
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if all(x is None for x in (start, stop, step)):
                # Return all elements
                return [self[i] for i in range(len(self))]
            indices = range(start, stop, step) if step else range(start, stop)
            return [self[i] for i in indices]

        if index < 0:
            if -index > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            index = len(self) + index
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class TensorDataset(Dataset):
    """Dataset wrapping a tensor.

    Each sample will be retrieved by indexing a tensor along the first dimension.

    Parameters
    ----------
    tensors : Tuple[Tensor]
        Tensors that have the same size of the first dimension.
    """

    size: int
    tensor: Tensor

    def __init__(
        self, tensor: Tensor, dims: Optional[Dict[str, tuple[int, ...]]] = None
    ) -> None:
        self.dims = dims
        self.size = size(tensor, 0)
        self.tensor = tensor

    def __getitem__(self, index: _IndexLike) -> Tuple[Tensor]:
        """Retrieve a sample from the dataset.

        Parameters
        ----------
        index : index_like
            Index of the sample to retrieve.

        Returns
        -------
        Tensor
            A slice of tensor along the first dimension.
        """
        return self.tensor[index]

    def __len__(self) -> int:
        """Return the size of the dataset.

        Returns
        -------
        int
            The size of the dataset.
        """
        return self.size


class SequenceTensorDataset(Dataset):
    """Dataset wrapping a sequence of tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Parameters
    ----------
    tensors : Tuple[Tensor]
        Tensors that have the same size of the first dimension.
    """

    size: int
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        self.size = size(tensors[0], 0)
        assert all(
            self.size == size(tensor, 0) for tensor in tensors
        ), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index: _IndexLike) -> Tuple[Tensor]:
        """Retrieve a sample from the dataset.

        Parameters
        ----------
        index : index_like
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[Tensor]
            A tuple containing the sample data.
        """
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        """Return the size of the dataset.

        Returns
        -------
        int
            The size of the dataset.
        """
        return self.size


class MappingTensorDataset(Dataset):
    """Dataset wrapping a mapping of str to tensor.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Parameters
    ----------
    tensors : Dict[str, Tensor]
        Tensors that have the same size of the first dimension.
    """

    size: int
    tensors: Dict[str, Tensor]

    def __init__(self, **tensors: Tensor) -> None:
        self.size = size(next(iter(tensors.values())), 0)
        assert all(
            self.size == size(tensor, 0) for tensor in tensors.values()
        ), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index: _IndexLike) -> Dict[str, Tensor]:
        """Retrieve a sample from the dataset.

        Parameters
        ----------
        index : index_like
            Index of the sample to retrieve.

        Returns
        -------
        Dict[str, Tensor]
            A tuple containing the sample data.
        """
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self) -> int:
        """Return the size of the dataset.

        Returns
        -------
        int
            The size of the dataset.
        """
        return self.size


class Subset(Dataset):
    """Subset of a dataset at specified indices.

    Parameters
    ----------
    dataset : Dataset
        The whole Dataset.
    indices : Sequence[int]
        Indices in the whole set selected for subset.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: _IndexLike) -> Any:
        """Fetch a data sample from the subset.

        Parameters
        ----------
        index : index_like
            Index of the data sample to retrieve.

        Returns
        -------
        Any
            The requested data sample.
        """
        if isinstance(index, list):
            return self.dataset[[self.indices[i] for i in index]]
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        """Return the size of the subset.

        Returns
        -------
        int
            The size of the subset.
        """
        return len(self.indices)


def partition(
    dataset: Dataset, lengths: Sequence[Union[int, float]], shuffle: bool = False
) -> List[Subset]:
    """Split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given, the lengths will be computed
    automatically as floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths until there are no remainders
    left.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    lengths : Sequence[Union[int, float]]
        Lengths of splits to produce. If float, interpreted as fraction of total size.
    shuffle : bool, optional
        Whether to shuffle the data before splitting, by default False.

    Returns
    -------
    List[Subset]
        A list of subsets.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = np.arange(sum(lengths))  # type: ignore[call-overload]
    if shuffle:
        np.random.shuffle(indices)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]
