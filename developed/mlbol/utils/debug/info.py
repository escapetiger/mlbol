import logging
import inspect
import warnings
from functools import wraps
from typing import Any
from typing import Optional
from typing import Callable
from typing import ParamSpec
from typing import Concatenate
from typing import TypeVar

Param = ParamSpec("Param")
RetType = TypeVar("RetType")

__all__ = [
    "current_func_name",
    "get_num_args",
    "raise_not_implemented_error",
    "deprecated",
]


def current_func_name() -> str:
    """This is a prompt to get background function's name inside the function."""
    return inspect.currentframe().f_back.f_code.co_name


def get_num_args(func: Callable[Param, RetType]) -> int:
    """Get the number of arguments of a Python function.

    References:

    - https://stackoverflow.com/questions/847936/how-can-i-find-the-number-of-arguments-of-a-python-function
    """
    # If the function is a class method decorated with functools.wraps, then "self" will
    # be in parameters, as inspect.signature follows wrapper chains by default, see
    # https://stackoverflow.com/questions/308999/what-does-functools-wraps-do
    #
    # Example:
    #
    # import inspect
    # from functools import wraps
    #
    # def dummy(f):
    #     print(f)
    #     print(inspect.signature(f))
    #
    #     @wraps(f)
    #     def wrapper(*args, **kwargs):
    #         f(*args, **kwargs)
    #
    #     print(wrapper)
    #     print(inspect.signature(wrapper))
    #     return wrapper
    #
    # class A:
    #     @dummy  # See the difference by commenting out this line
    #     def f(self, x):
    #         pass
    #
    # print(A.f)
    # print(inspect.signature(A.f))
    #
    # a = A()
    # g = dummy(a.f)
    params = inspect.signature(func).parameters
    return len(params) - ("self" in params)


def raise_not_implemented_error(
    logger: logging.Logger, obj: Optional[Any] = None
) -> None:
    for_which = f" for {obj}" if obj is not None else ""
    logger.error(f"{inspect.stack()[1][3]} is undefined{for_which}.")
    raise NotImplementedError


def deprecated(
    func: Callable[Param, RetType]
) -> Callable[Concatenate[str, Param], RetType]:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @wraps(func)
    def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return wrapper
