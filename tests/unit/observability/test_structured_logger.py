"""
Unit Tests for Structured Logging

Tests the StructuredLogger class for:
- JSON formatting
- Correlation ID generation and propagation
- Context enrichment
- Log levels
- Thread safety
- Integration with Python logging
"""

import pytest
import json
import logging
import threading

from src.observability.structured_logger import (
    StructuredLogger, LoggerConfig, LogLevel,
    LogContext, ContextStorage, get_logger, configure_logging
)


class TestLoggerInitialization:
    """Test StructuredLogger initialization."""

    def test_default_initialization(self):
        """Test logger initializes with defaults."""
        logger = StructuredLogger()

        assert logger.config.name == "mae"
        assert logger.config.level == LogLevel.INFO
        assert logger.config.enable_json is True
        assert logger.config.enable_correlation_ids is True

    def test_custom_initialization(self):
        """Test logger with custom configuration."""
        config = LoggerConfig(
            name="test.logger",
            level=LogLevel.DEBUG,
            enable_json=False,
            additional_fields={"environment": "test"}
        )

        logger = StructuredLogger(config=config)

        assert logger.config.name == "test.logger"
        assert logger.config.level == LogLevel.DEBUG
        assert logger.config.enable_json is False
        assert logger.config.additional_fields["environment"] == "test"

    def test_logger_with_name(self):
        """Test logger initialization with name parameter."""
        logger = StructuredLogger(name="custom.logger")
        assert logger.config.name == "custom.logger"


class TestLogContext:
    """Test LogContext functionality."""

    def test_context_to_dict(self):
        """Test context conversion to dictionary."""
        context = LogContext(
            correlation_id="cid_123",
            agent_id="agent_1",
            request_id="req_456",
            custom_fields={"key": "value"}
        )

        result = context.to_dict()

        assert result["correlation_id"] == "cid_123"
        assert result["agent_id"] == "agent_1"
        assert result["request_id"] == "req_456"
        assert result["key"] == "value"

    def test_context_to_dict_partial(self):
        """Test context with only some fields set."""
        context = LogContext(agent_id="agent_1")

        result = context.to_dict()

        assert "agent_id" in result
        assert "correlation_id" not in result
        assert "request_id" not in result


class TestContextStorage:
    """Test ContextStorage thread-local storage."""

    def test_get_context_creates_default(self):
        """Test getting context creates default if not exists."""
        storage = ContextStorage()
        context = storage.get_context()

        assert isinstance(context, LogContext)
        assert context.correlation_id is None

    def test_set_and_get_context(self):
        """Test setting and getting context."""
        storage = ContextStorage()
        context = LogContext(correlation_id="test_123")

        storage.set_context(context)
        retrieved = storage.get_context()

        assert retrieved.correlation_id == "test_123"

    def test_clear_context(self):
        """Test clearing context."""
        storage = ContextStorage()
        context = LogContext(correlation_id="test_123")
        storage.set_context(context)

        storage.clear_context()
        retrieved = storage.get_context()

        assert retrieved.correlation_id is None

    def test_thread_isolation(self):
        """Test context is isolated per thread."""
        storage = ContextStorage()
        results = {}

        def set_context(thread_id: str):
            context = LogContext(correlation_id=thread_id)
            storage.set_context(context)
            # Small delay to ensure threads overlap
            import time
            time.sleep(0.01)
            results[thread_id] = storage.get_context().correlation_id

        threads = [
            threading.Thread(target=set_context, args=(f"thread_{i}",))
            for i in range(5)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Each thread should have its own context
        for i in range(5):
            assert results[f"thread_{i}"] == f"thread_{i}"


class TestLogLevels:
    """Test log level functionality."""

    def test_debug_log(self, caplog):
        """Test debug logging."""
        config = LoggerConfig(level=LogLevel.DEBUG, enable_json=True)
        logger = StructuredLogger(config=config)

        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message", key="value")

        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)
        assert log_entry["message"] == "Debug message"
        assert log_entry["level"] == "DEBUG"
        assert log_entry["key"] == "value"

    def test_info_log(self, caplog):
        """Test info logging."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            logger.info("Info message", status="ok")

        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)
        assert log_entry["message"] == "Info message"
        assert log_entry["level"] == "INFO"

    def test_warn_log(self, caplog):
        """Test warning logging."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            logger.warn("Warning message", code="WARN_001")

        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)
        assert log_entry["message"] == "Warning message"
        assert log_entry["level"] == "WARN"

    def test_error_log(self, caplog):
        """Test error logging."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            logger.error("Error message", error_code="ERR_001")

        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)
        assert log_entry["message"] == "Error message"
        assert log_entry["level"] == "ERROR"

    def test_critical_log(self, caplog):
        """Test critical logging."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            logger.critical("Critical message", severity="high")

        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)
        assert log_entry["message"] == "Critical message"
        assert log_entry["level"] == "CRITICAL"

    def test_log_level_filtering(self, caplog):
        """Test that logs below level are filtered."""
        config = LoggerConfig(level=LogLevel.WARN)
        logger = StructuredLogger(config=config)

        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warn("Warning message")

        # Should only have warning
        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)
        assert log_entry["message"] == "Warning message"


class TestJSONFormatting:
    """Test JSON formatting functionality."""

    def test_json_output(self, caplog):
        """Test logs are formatted as JSON."""
        config = LoggerConfig(enable_json=True)
        logger = StructuredLogger(config=config)

        with caplog.at_level(logging.INFO):
            logger.info("Test message", field1="value1", field2=123)

        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)

        assert log_entry["message"] == "Test message"
        assert log_entry["level"] == "INFO"
        assert log_entry["field1"] == "value1"
        assert log_entry["field2"] == 123

    def test_json_with_timestamp(self, caplog):
        """Test JSON logs include timestamp."""
        config = LoggerConfig(enable_json=True, enable_timestamps=True)
        logger = StructuredLogger(config=config)

        with caplog.at_level(logging.INFO):
            logger.info("Test message")

        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)

        assert "timestamp" in log_entry
        assert log_entry["timestamp"].endswith("Z")  # ISO format UTC

    def test_json_disabled(self, caplog):
        """Test plain text output when JSON disabled."""
        config = LoggerConfig(enable_json=False)
        logger = StructuredLogger(config=config)

        with caplog.at_level(logging.INFO):
            logger.info("Plain text message")

        assert len(caplog.records) == 1
        message = caplog.records[0].message
        assert "Plain text message" in message
        # Should not be valid JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(message)


class TestCorrelationIDs:
    """Test correlation ID functionality."""

    def test_correlation_id_context(self, caplog):
        """Test correlation ID context manager."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            with logger.correlation_id() as cid:
                logger.info("Message 1")
                logger.info("Message 2")

        assert len(caplog.records) == 2

        log1 = json.loads(caplog.records[0].message)
        log2 = json.loads(caplog.records[1].message)

        assert log1["correlation_id"] == cid
        assert log2["correlation_id"] == cid
        assert log1["correlation_id"] == log2["correlation_id"]

    def test_explicit_correlation_id(self, caplog):
        """Test explicit correlation ID."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            with logger.correlation_id("custom_cid_123") as cid:
                logger.info("Test message")

        assert cid == "custom_cid_123"
        assert len(caplog.records) == 1

        log_entry = json.loads(caplog.records[0].message)
        assert log_entry["correlation_id"] == "custom_cid_123"

    def test_nested_correlation_ids(self, caplog):
        """Test nested correlation ID contexts."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            with logger.correlation_id("outer") as outer_cid:
                logger.info("Outer message")

                with logger.correlation_id("inner") as inner_cid:
                    logger.info("Inner message")

                logger.info("Outer again")

        assert len(caplog.records) == 3

        log1 = json.loads(caplog.records[0].message)
        log2 = json.loads(caplog.records[1].message)
        log3 = json.loads(caplog.records[2].message)

        assert log1["correlation_id"] == "outer"
        assert log2["correlation_id"] == "inner"
        assert log3["correlation_id"] == "outer"

    def test_get_correlation_id(self):
        """Test getting correlation ID."""
        logger = StructuredLogger()

        with logger.correlation_id("test_cid"):
            assert logger.get_correlation_id() == "test_cid"

    def test_set_correlation_id(self):
        """Test setting correlation ID."""
        logger = StructuredLogger()

        logger.set_correlation_id("manual_cid")
        assert logger.get_correlation_id() == "manual_cid"


class TestContextEnrichment:
    """Test context enrichment functionality."""

    def test_context_manager(self, caplog):
        """Test context manager adds fields."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            with logger.context(agent_id="agent_1", request_id="req_123"):
                logger.info("Test message")

        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)

        assert log_entry["agent_id"] == "agent_1"
        assert log_entry["request_id"] == "req_123"

    def test_nested_contexts(self, caplog):
        """Test nested context enrichment."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            with logger.context(field1="value1"):
                logger.info("Message 1")

                with logger.context(field2="value2"):
                    logger.info("Message 2")

                logger.info("Message 3")

        assert len(caplog.records) == 3

        log1 = json.loads(caplog.records[0].message)
        log2 = json.loads(caplog.records[1].message)
        log3 = json.loads(caplog.records[2].message)

        # First message has field1
        assert log1["field1"] == "value1"
        assert "field2" not in log1

        # Second message has both fields
        assert log2["field1"] == "value1"
        assert log2["field2"] == "value2"

        # Third message has only field1 again
        assert log3["field1"] == "value1"
        assert "field2" not in log3

    def test_agent_context(self, caplog):
        """Test agent context manager."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            with logger.agent_context("agent_1"):
                logger.info("Agent message")

        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)

        assert log_entry["agent_id"] == "agent_1"

    def test_context_cleanup(self, caplog):
        """Test context is cleaned up after exit."""
        logger = StructuredLogger()

        with caplog.at_level(logging.INFO):
            with logger.context(temp_field="temp_value"):
                logger.info("Inside context")

            logger.info("Outside context")

        assert len(caplog.records) == 2

        log1 = json.loads(caplog.records[0].message)
        log2 = json.loads(caplog.records[1].message)

        assert "temp_field" in log1
        assert "temp_field" not in log2


class TestAdditionalFields:
    """Test additional fields configuration."""

    def test_additional_fields(self, caplog):
        """Test additional fields are included in all logs."""
        config = LoggerConfig(
            additional_fields={
                "environment": "test",
                "service": "mae",
                "version": "1.0.0"
            }
        )
        logger = StructuredLogger(config=config)

        with caplog.at_level(logging.INFO):
            logger.info("Test message")

        assert len(caplog.records) == 1
        log_entry = json.loads(caplog.records[0].message)

        assert log_entry["environment"] == "test"
        assert log_entry["service"] == "mae"
        assert log_entry["version"] == "1.0.0"


class TestThreadSafety:
    """Test thread safety of structured logging."""

    def test_concurrent_logging(self, caplog):
        """Test concurrent logging from multiple threads."""
        logger = StructuredLogger()

        def log_messages(thread_id: int):
            with logger.agent_context(f"agent_{thread_id}"):
                for i in range(10):
                    logger.info(f"Message {i}", thread_id=thread_id)

        with caplog.at_level(logging.INFO):
            threads = [threading.Thread(target=log_messages, args=(i,)) for i in range(5)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        # Should have 50 log lines (5 threads * 10 messages)
        assert len(caplog.records) == 50

        # Parse and verify structure
        for record in caplog.records:
            log_entry = json.loads(record.message)
            assert "message" in log_entry
            assert "agent_id" in log_entry
            assert "thread_id" in log_entry


class TestGlobalLogger:
    """Test global logger functionality."""

    def test_get_logger_default(self):
        """Test getting default logger."""
        logger = get_logger()
        assert isinstance(logger, StructuredLogger)

    def test_get_logger_with_name(self):
        """Test getting named logger."""
        logger = get_logger("test.module")
        assert logger.config.name == "test.module"

    def test_configure_logging(self):
        """Test configuring global logging."""
        config = LoggerConfig(
            name="custom",
            level=LogLevel.DEBUG
        )

        configure_logging(config)
        logger = get_logger()

        assert logger.config.name == "custom"
        assert logger.config.level == LogLevel.DEBUG


class TestLoggerIntegration:
    """Integration tests for structured logger."""

    def test_complete_workflow(self, caplog):
        """Test complete logging workflow."""
        logger = StructuredLogger(name="test.integration")

        with caplog.at_level(logging.INFO):
            with logger.correlation_id() as cid:
                with logger.agent_context("agent_1"):
                    logger.info("Agent started")

                    with logger.context(task="learning"):
                        logger.info("Learning step", step=1, reward=10.5)
                        logger.info("Learning step", step=2, reward=15.0)

                    logger.info("Agent completed")

        assert len(caplog.records) == 4

        # Parse all logs
        logs = [json.loads(record.message) for record in caplog.records]

        # All should have same correlation ID
        assert all(log["correlation_id"] == cid for log in logs)

        # All should have agent_id
        assert all(log["agent_id"] == "agent_1" for log in logs)

        # Middle two should have task field
        assert "task" not in logs[0]
        assert logs[1]["task"] == "learning"
        assert logs[2]["task"] == "learning"
        assert "task" not in logs[3]
