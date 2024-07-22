import logging
from contextlib import contextmanager
from time import time
from functools import wraps
from typing import Callable
from typing import Generator
from typing import Any
from typing import Optional
from typing import ParamSpec
from typing import Concatenate
from typing import TypeVar

Param = ParamSpec("Param")
RetType = TypeVar("RetType")

__all__ = ["time_context", "time_func"]


@contextmanager
def time_context(context_name: Optional[str] = None) -> Generator[None, Any, None]:
    """This is a contextmanager to record the elapsed time of a code block."""
    t0 = time()
    try:
        yield
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while timing the code block {context_name}: {e}"
        )
    finally:
        t1 = time()
        prefix = f"{context_name} - " if context_name is not None else ""
        logging.info(f"{prefix}Elapsed time: {t1-t0:.2f} seconds")


def time_func(
    func: Callable[Param, RetType],
) -> Callable[Concatenate[str, Param], RetType]:
    """This is a decorator to record the elapsed time for a function."""

    @wraps(func)
    def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> RetType:
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        logging.info(f"Elapsed time for {func.__name__}: {t1-t0:.2f} seconds")
        return result

    return wrapper
