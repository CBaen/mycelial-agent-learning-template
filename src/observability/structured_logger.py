"""
Structured Logging for MAE

This module provides a comprehensive structured logging system with:
- JSON formatting for machine-readable logs
- Correlation IDs for request tracing
- Context enrichment with automatic field injection
- Log level configuration
- Integration with Python's logging module

Key features:
- Correlation ID propagation across async boundaries
- Context managers for automatic field enrichment
- Thread-safe context storage
- Configurable output formats (JSON, text)
- Integration with existing logging infrastructure
"""

import logging
import json
import time
import uuid
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    """Thread-local logging context."""
    correlation_id: Optional[str] = None
    agent_id: Optional[str] = None
    request_id: Optional[str] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        result = {}
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.request_id:
            result["request_id"] = self.request_id
        result.update(self.custom_fields)
        return result


class ContextStorage:
    """Thread-safe storage for logging context."""

    def __init__(self):
        self._local = threading.local()

    def get_context(self) -> LogContext:
        """Get current thread's logging context."""
        if not hasattr(self._local, 'context'):
            self._local.context = LogContext()
        return self._local.context

    def set_context(self, context: LogContext):
        """Set current thread's logging context."""
        self._local.context = context

    def clear_context(self):
        """Clear current thread's logging context."""
        self._local.context = LogContext()


@dataclass
class LoggerConfig:
    """Configuration for structured logger."""
    name: str = "mae"
    level: LogLevel = LogLevel.INFO
    enable_json: bool = True
    enable_correlation_ids: bool = True
    enable_timestamps: bool = True
    enable_context: bool = True
    additional_fields: Dict[str, Any] = field(default_factory=dict)
    output_file: Optional[str] = None


class StructuredLogger:
    """
    Structured logger with JSON formatting and correlation IDs.

    Provides comprehensive logging with:
    - JSON formatting for machine-readable logs
    - Automatic correlation ID generation and propagation
    - Context enrichment with custom fields
    - Thread-safe context storage
    - Integration with Python's logging module

    Example:
        >>> logger = StructuredLogger(name="mae.agents")
        >>> logger.info("Agent started", agent_id="agent_1", status="active")
        >>>
        >>> with logger.context(agent_id="agent_1"):
        ...     logger.info("Processing request")  # Includes agent_id
        >>>
        >>> with logger.correlation_id() as cid:
        ...     logger.info("Request started")
        ...     logger.info("Request completed")  # Same correlation_id
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[LoggerConfig] = None,
        context_storage: Optional[ContextStorage] = None
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name (defaults to "mae")
            config: LoggerConfig instance
            context_storage: ContextStorage instance (creates new if None)
        """
        self.config = config or LoggerConfig(name=name or "mae")
        self.context_storage = context_storage or ContextStorage()

        # Initialize Python logger
        self.logger = logging.getLogger(self.config.name)
        self.logger.setLevel(self._get_python_level(self.config.level))

        # Configure handler
        self._configure_handler()

    def _get_python_level(self, level: LogLevel) -> int:
        """Convert LogLevel to Python logging level."""
        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARN: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        return mapping[level]

    def _configure_handler(self):
        """Configure logging handler."""
        # Remove existing handlers
        self.logger.handlers.clear()

        # Create handler
        if self.config.output_file:
            handler = logging.FileHandler(self.config.output_file)
        else:
            handler = logging.StreamHandler()

        # Set formatter
        if self.config.enable_json:
            handler.setFormatter(JSONFormatter(self))
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))

        self.logger.addHandler(handler)

    def _build_log_entry(
        self,
        level: LogLevel,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Build structured log entry."""
        entry = {
            "message": message,
            "level": level.value
        }

        # Add timestamp
        if self.config.enable_timestamps:
            entry["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add context
        if self.config.enable_context:
            context = self.context_storage.get_context()
            entry.update(context.to_dict())

        # Add additional fields from config
        entry.update(self.config.additional_fields)

        # Add kwargs
        entry.update(kwargs)

        return entry

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        entry = self._build_log_entry(LogLevel.DEBUG, message, **kwargs)
        self.logger.debug(json.dumps(entry) if self.config.enable_json else message)

    def info(self, message: str, **kwargs):
        """Log info message."""
        entry = self._build_log_entry(LogLevel.INFO, message, **kwargs)
        self.logger.info(json.dumps(entry) if self.config.enable_json else message)

    def warn(self, message: str, **kwargs):
        """Log warning message."""
        entry = self._build_log_entry(LogLevel.WARN, message, **kwargs)
        self.logger.warning(json.dumps(entry) if self.config.enable_json else message)

    def error(self, message: str, **kwargs):
        """Log error message."""
        entry = self._build_log_entry(LogLevel.ERROR, message, **kwargs)
        self.logger.error(json.dumps(entry) if self.config.enable_json else message)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        entry = self._build_log_entry(LogLevel.CRITICAL, message, **kwargs)
        self.logger.critical(json.dumps(entry) if self.config.enable_json else message)

    @contextmanager
    def context(self, **fields):
        """
        Context manager for automatic field enrichment.

        Example:
            >>> with logger.context(agent_id="agent_1", request_id="req_123"):
            ...     logger.info("Processing")  # Includes agent_id and request_id
        """
        # Get current context
        current_context = self.context_storage.get_context()

        # Save original custom fields
        original_fields = current_context.custom_fields.copy()

        # Add new fields
        current_context.custom_fields.update(fields)

        try:
            yield
        finally:
            # Restore original fields
            current_context.custom_fields = original_fields

    @contextmanager
    def correlation_id(self, correlation_id: Optional[str] = None):
        """
        Context manager for correlation ID propagation.

        Args:
            correlation_id: Explicit correlation ID (generates new if None)

        Returns:
            str: The correlation ID being used

        Example:
            >>> with logger.correlation_id() as cid:
            ...     logger.info("Request started")
            ...     logger.info("Request completed")  # Same correlation_id
        """
        if not self.config.enable_correlation_ids:
            yield None
            return

        # Get current context
        current_context = self.context_storage.get_context()

        # Save original correlation ID
        original_cid = current_context.correlation_id

        # Set new correlation ID
        cid = correlation_id or str(uuid.uuid4())
        current_context.correlation_id = cid

        try:
            yield cid
        finally:
            # Restore original correlation ID
            current_context.correlation_id = original_cid

    @contextmanager
    def agent_context(self, agent_id: str):
        """
        Context manager for agent-scoped logging.

        Example:
            >>> with logger.agent_context("agent_1"):
            ...     logger.info("Agent processing")  # Includes agent_id="agent_1"
        """
        current_context = self.context_storage.get_context()
        original_agent_id = current_context.agent_id

        current_context.agent_id = agent_id

        try:
            yield
        finally:
            current_context.agent_id = original_agent_id

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID from context."""
        return self.context_storage.get_context().correlation_id

    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID in context."""
        self.context_storage.get_context().correlation_id = correlation_id

    def clear_context(self):
        """Clear all context fields."""
        self.context_storage.clear_context()

    def set_level(self, level: LogLevel):
        """Set logging level."""
        self.config.level = level
        self.logger.setLevel(self._get_python_level(level))


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logs."""

    def __init__(self, structured_logger: StructuredLogger):
        super().__init__()
        self.structured_logger = structured_logger

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Message is already JSON string from StructuredLogger
        return record.getMessage()


# Global logger instance
_default_logger: Optional[StructuredLogger] = None


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name (uses default if None)

    Returns:
        StructuredLogger instance
    """
    global _default_logger

    if name is None:
        if _default_logger is None:
            _default_logger = StructuredLogger()
        return _default_logger

    return StructuredLogger(name=name)


def configure_logging(config: LoggerConfig):
    """
    Configure global logging settings.

    Args:
        config: LoggerConfig instance
    """
    global _default_logger
    _default_logger = StructuredLogger(config=config)
