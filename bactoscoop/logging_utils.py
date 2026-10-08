import logging
import os
import sys

from tqdm import tqdm


LOGGER_NAME = "logger"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def _resolve_log_level(level_name):
    return getattr(logging, str(level_name).upper(), logging.INFO)


def get_bactoscoop_logger():
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(_resolve_log_level(os.environ.get("BACTOSCOOP_LOG_LEVEL", "INFO")))
    return logger


def configure_bactoscoop_logging(level="INFO"):
    os.environ["BACTOSCOOP_LOG_LEVEL"] = str(level).upper()
    logger = get_bactoscoop_logger()
    logger.setLevel(_resolve_log_level(level))
    return logger


def tqdm_if_verbose(iterable, **kwargs):
    kwargs.setdefault(
        "disable",
        not get_bactoscoop_logger().isEnabledFor(logging.INFO),
    )
    return tqdm(iterable, **kwargs)
