import logging
import sys
from pathlib import Path
from typing import Optional

# Global logging level - can be controlled by config
_GLOBAL_LOG_LEVEL = logging.INFO


def configure_logging(
        verbose: bool = False,
        log_file: Optional[str] = None,
        log_level: Optional[int] = None
):
    global _GLOBAL_LOG_LEVEL

    # Determine log level
    if log_level is not None:
        _GLOBAL_LOG_LEVEL = log_level
    elif verbose:
        _GLOBAL_LOG_LEVEL = logging.DEBUG
    else:
        _GLOBAL_LOG_LEVEL = logging.INFO

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(_GLOBAL_LOG_LEVEL)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_GLOBAL_LOG_LEVEL)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        # Create logs directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(_GLOBAL_LOG_LEVEL)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(_GLOBAL_LOG_LEVEL)
    return logger


# Convenience function for simple logging setup
def setup_simple_logging(verbose: bool = False):
    """
    Simple logging setup for quick use.

    Args:
        verbose: If True, show DEBUG messages; if False, show INFO and above
    """
    configure_logging(verbose=verbose)