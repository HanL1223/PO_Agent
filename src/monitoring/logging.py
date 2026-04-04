"""
Structured Logging for production
    1. JSON output: Machine-parseable logs for CloudWatch, Datadog, Splunk
    2. Context binding: Attach request_id, user_id, etc. to every log line
    3. Processors: Transform log entries through a pipeline (add timestamps,
       filter fields, format output)
    4. Development mode: Pretty-printed coloured output for local debugging

    Every API request gets a unique correlation_id (UUID). This ID is:
        - Attached to every log line during that request
        - Returned in the API response headers
        - Used to trace a single request across all log entries

USAGE:
    from src.monitoring.logging import get_logger

    logger = get_logger(__name__)
    logger.info("processing_request", user_request="Create auth ticket", space="CSCI")

    # Output (JSON mode):
    # {"event": "processing_request", "user_request": "Create auth ticket",
    #  "space": "CSCI", "timestamp": "2024-...", "level": "info", "logger": "src.agents"}

    # Output (dev mode):
    # 2024-... [info] processing_request  user_request=Create auth ticket  space=CSCI
"""


import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

from src.config import settings

#Context Variable
#ContextVars are thread-safe and async-safe storage.
# Each async task or thread gets its own copy of the variable.
# method to attach a correlation_id to an entire request lifecycle.

#Context variables
correlation_id_var : ContextVar[str] = ContextVar("correlation_id",default = "no-correlation_id")
request_space_var:ContextVar[str] = ContextVar("request_sapce",default="unknown")

def new_correlation_id() -> str:
    """generate and set a new correlation ID for current request context"""
    cid= str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid

def set_request_space(space: str)-> None:
    """Set the Jira space for the current request context"""
    request_space_var.set(space)

#Structlog processor
def add_correlation_id(
        logger: Any, method: str, event_dict: dict[str,Any]
) -> dict[str,Any]:
    """
    Add the current correlation ID to every log entry
    """
    event_dict["correlation_id"] = correlation_id_var.get()
    return event_dict


def add_request_space(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add the current Jira space to log entries."""
    space = request_space_var.get()
    if space != "unknown":
        event_dict["space"] = space
    return event_dict



#Logging configuration

def configure_logging(
        log_level:str = "INFO",
        json_output:bool = False
) -> None:
    """
    Configure structlog and stadnard logging

    """
    if json_output or settings.server.is_production:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    # Shared processors for both structlog and standard logging
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_correlation_id,
        add_request_space,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging to use structlog formatting
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Set root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Silence noisy third-party loggers
    for noisy_logger in ["httpx", "httpcore", "chromadb", "urllib3"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for a module.

    pass __name__ as the logger name. This
    automatically captures the module path (e.g., "src.agents.graph")
    which makes it easy to filter logs by component.

    Usage:
        logger = get_logger(__name__)
        logger.info("indexing_complete", chunks=150, duration_s=2.3)
    """
    return structlog.get_logger(name)