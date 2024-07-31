from functools import wraps
from typing import Callable
from typing import ParamSpec
from typing import Concatenate
from typing import TypeVar
from mlbol.dtensor import as_tensor

__all__ = ["out_as_tensor"]

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


def out_as_tensor(
    func: Callable[Param, RetType]
) -> Callable[Concatenate[str, Param], RetType]:
    @wraps(func)
    def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
        original_out = func(*args, **kwargs)
        try:
            if isinstance(original_out, tuple):
                out = tuple(as_tensor(arr) for arr in original_out)
            else:
                out = as_tensor(original_out)
            return out
        except:
            return original_out

    return wrapper
