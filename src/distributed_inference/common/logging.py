"""Structured logging for the distributed inference system.

Uses Python's logging module with rich formatting for console output.
Each component gets a named logger with consistent formatting.
"""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


_initialized = False
_console = Console(stderr=True)


def setup_logging(
    level: str = "INFO",
    component: Optional[str] = None,
) -> None:
    """Initialize logging with rich console output.

    Should be called once at startup. Subsequent calls are no-ops.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
        component: Optional component name prefix for the root logger.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure rich handler for pretty console output
    handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    handler.setLevel(log_level)

    # Format: just the message (rich handles time/level)
    fmt = "%(message)s"
    if component:
        fmt = f"[bold cyan]\\[{component}][/] %(message)s"

    handler.setFormatter(logging.Formatter(fmt))

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a specific module/component.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
