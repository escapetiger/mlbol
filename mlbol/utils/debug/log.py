import inspect
import pathlib
import logging


__all__ = ["log_info", "log_debug", "log_assert"]

log_path = pathlib.Path("tmp") / "logs"
log_path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, level: int) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.FileHandler(f"{log_path}/{name}.log", mode="w")
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return log


def log_info(name: str) -> logging.Logger:
    return get_logger(name, logging.INFO)


def log_debug(name: str) -> logging.Logger:
    return get_logger(name, logging.DEBUG)


def log_assert(
    bool_: bool,
    message: str = "",
    logger: logging.Logger = None,
    logger_name: str = "",
    verbose: bool = False,
):
    """Use this as a replacement for assert if you want the failing of the
    assert statement to be logged."""
    if logger is None:
        logger = logging.getLogger(logger_name)
    try:
        assert bool_, message
    except AssertionError:
        # construct an exception message from the code of the calling frame
        last_stackframe = inspect.stack()[-2]
        source_file, line_no, func = last_stackframe[1:4]
        source = (
            "Traceback (most recent call last):\n"
            + '  File "%s", line %s, in %s\n    ' % (source_file, line_no, func)
        )
        if verbose:
            # include more lines than that where the statement was made
            source_code = open(source_file).readlines()
            source += "".join(source_code[line_no - 3 : line_no + 1])
        else:
            source += last_stackframe[-2][0].strip()
        if logger.level == logging.INFO:
            logger.info("%s\n%s" % (message, source))
        elif logger.level == logging.DEBUG:
            logger.debug("%s\n%s" % (message, source))
        raise AssertionError("%s\n%s" % (message, source))


