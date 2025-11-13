"""
OpenTelemetry Tracing for MAE

This module provides distributed tracing capabilities using OpenTelemetry for:
- Span creation and management
- Context propagation across agent boundaries
- Trace instrumentation for multi-agent workflows
- Integration with Jaeger/Zipkin exporters

Key features:
- Automatic span creation with context managers
- Attribute and event recording
- Parent-child span relationships
- Low-overhead tracing (<5ms overhead target)
- Graceful degradation without OpenTelemetry
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Mock classes for graceful degradation
    class SpanKind:
        INTERNAL = "INTERNAL"
        SERVER = "SERVER"
        CLIENT = "CLIENT"
        PRODUCER = "PRODUCER"
        CONSUMER = "CONSUMER"

    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"
        UNSET = "UNSET"


logger = logging.getLogger(__name__)


class ExporterType(Enum):
    """Supported trace exporter types."""
    CONSOLE = "console"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"


@dataclass
class TracingConfig:
    """Configuration for distributed tracing."""
    enabled: bool = True
    service_name: str = "mae"
    exporter_type: ExporterType = ExporterType.CONSOLE
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    zipkin_endpoint: str = "http://localhost:9411/api/v2/spans"
    sample_rate: float = 1.0  # 0.0 to 1.0
    max_attributes: int = 128
    max_events: int = 128
    additional_attributes: Dict[str, Any] = field(default_factory=dict)


class TracingProvider:
    """
    OpenTelemetry tracing provider for MAE.

    Provides distributed tracing capabilities:
    - Span creation and management
    - Context propagation
    - Trace instrumentation
    - Multiple exporter support (Console, Jaeger, Zipkin)

    Example:
        >>> provider = TracingProvider(config=TracingConfig(
        ...     service_name="mae.agents",
        ...     exporter_type=ExporterType.JAEGER
        ... ))
        >>>
        >>> with provider.span("process_request") as span:
        ...     span.set_attribute("agent_id", "agent_1")
        ...     # Do work
        ...     span.add_event("work_completed")
    """

    def __init__(
        self,
        config: Optional[TracingConfig] = None,
        tracer_provider: Optional['TracerProvider'] = None
    ):
        """
        Initialize tracing provider.

        Args:
            config: TracingConfig instance
            tracer_provider: Custom TracerProvider (creates new if None)
        """
        self.config = config or TracingConfig()
        self.tracer_provider = tracer_provider
        self.tracer = None
        self._span_count = 0
        self._start_time = time.time()

        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available. Tracing will be mocked.")
            return

        if self.config.enabled:
            self._initialize_tracer()

    def _initialize_tracer(self):
        """Initialize OpenTelemetry tracer with configured exporter."""
        if not OPENTELEMETRY_AVAILABLE:
            return

        # Create resource with service name and attributes
        resource_attrs = {
            SERVICE_NAME: self.config.service_name,
            **self.config.additional_attributes
        }
        resource = Resource.create(resource_attrs)

        # Create tracer provider if not provided
        if self.tracer_provider is None:
            self.tracer_provider = TracerProvider(resource=resource)

            # Configure exporter
            exporter = self._create_exporter()
            if exporter:
                span_processor = BatchSpanProcessor(exporter)
                self.tracer_provider.add_span_processor(span_processor)

            # Set as global tracer provider
            trace.set_tracer_provider(self.tracer_provider)

        # Get tracer instance
        self.tracer = trace.get_tracer(
            instrumenting_module_name=__name__,
            instrumenting_library_version="1.0.0"
        )

        logger.info(f"Tracing initialized: {self.config.service_name} -> {self.config.exporter_type.value}")

    def _create_exporter(self) -> Optional['SpanExporter']:
        """Create span exporter based on configuration."""
        if not OPENTELEMETRY_AVAILABLE:
            return None

        try:
            if self.config.exporter_type == ExporterType.CONSOLE:
                return ConsoleSpanExporter()

            elif self.config.exporter_type == ExporterType.JAEGER:
                return JaegerExporter(
                    collector_endpoint=self.config.jaeger_endpoint
                )

            elif self.config.exporter_type == ExporterType.ZIPKIN:
                return ZipkinExporter(
                    endpoint=self.config.zipkin_endpoint
                )

            else:
                logger.error(f"Unknown exporter type: {self.config.exporter_type}")
                return ConsoleSpanExporter()

        except Exception as e:
            logger.error(f"Error creating exporter: {e}. Falling back to console.")
            return ConsoleSpanExporter()

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: SpanKind = SpanKind.INTERNAL
    ):
        """
        Context manager for creating spans.

        Args:
            name: Span name
            attributes: Optional attributes to add to span
            kind: Span kind (INTERNAL, SERVER, CLIENT, etc.)

        Yields:
            Span object

        Example:
            >>> with provider.span("agent_step", attributes={"agent_id": "agent_1"}):
            ...     # Do work
            ...     pass
        """
        if not self.config.enabled or not OPENTELEMETRY_AVAILABLE or self.tracer is None:
            # Create mock span with attributes
            mock_span = MockSpan(name)
            if attributes:
                mock_span.set_attributes(attributes)
            self._span_count += 1
            yield mock_span
            return

        span = self.tracer.start_span(name, kind=kind)

        # Add attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        self._span_count += 1

        try:
            # Make span active in context
            token = trace.set_span_in_context(span)
            yield span
        except Exception as e:
            # Record exception in span
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            span.end()

    @contextmanager
    def agent_span(self, agent_id: str, operation: str, **attributes):
        """
        Convenience method for agent-scoped spans.

        Args:
            agent_id: Agent identifier
            operation: Operation name
            **attributes: Additional attributes

        Example:
            >>> with provider.agent_span("agent_1", "learn"):
            ...     # Agent learning step
            ...     pass
        """
        span_name = f"agent.{operation}"
        span_attributes = {
            "agent.id": agent_id,
            "agent.operation": operation,
            **attributes
        }

        with self.span(span_name, attributes=span_attributes) as span:
            yield span

    @contextmanager
    def communication_span(
        self,
        source_agent: str,
        target_agent: str,
        message_type: str,
        **attributes
    ):
        """
        Convenience method for communication spans.

        Args:
            source_agent: Source agent ID
            target_agent: Target agent ID
            message_type: Type of message
            **attributes: Additional attributes

        Example:
            >>> with provider.communication_span("agent_1", "agent_2", "BROADCAST"):
            ...     # Send message
            ...     pass
        """
        span_name = f"communication.{message_type.lower()}"
        span_attributes = {
            "communication.source": source_agent,
            "communication.target": target_agent,
            "communication.type": message_type,
            **attributes
        }

        with self.span(span_name, attributes=span_attributes, kind=SpanKind.CLIENT) as span:
            yield span

    @contextmanager
    def memory_span(self, agent_id: str, operation: str, **attributes):
        """
        Convenience method for memory operation spans.

        Args:
            agent_id: Agent identifier
            operation: Memory operation (replay, consolidation, etc.)
            **attributes: Additional attributes

        Example:
            >>> with provider.memory_span("agent_1", "replay", batch_size=32):
            ...     # Memory replay
            ...     pass
        """
        span_name = f"memory.{operation}"
        span_attributes = {
            "memory.agent_id": agent_id,
            "memory.operation": operation,
            **attributes
        }

        with self.span(span_name, attributes=span_attributes) as span:
            yield span

    def get_current_span(self):
        """Get current active span."""
        if not OPENTELEMETRY_AVAILABLE or not self.config.enabled:
            return MockSpan("current")

        return trace.get_current_span()

    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID as hex string."""
        if not OPENTELEMETRY_AVAILABLE or not self.config.enabled:
            return None

        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, '032x')
        return None

    def get_span_id(self) -> Optional[str]:
        """Get current span ID as hex string."""
        if not OPENTELEMETRY_AVAILABLE or not self.config.enabled:
            return None

        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().span_id, '016x')
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        return {
            "enabled": self.config.enabled,
            "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
            "service_name": self.config.service_name,
            "exporter_type": self.config.exporter_type.value,
            "spans_created": self._span_count,
            "uptime_seconds": time.time() - self._start_time
        }

    def flush(self, timeout_millis: int = 30000):
        """
        Flush pending spans to exporter.

        Args:
            timeout_millis: Timeout in milliseconds
        """
        if not OPENTELEMETRY_AVAILABLE or not self.config.enabled:
            return

        if self.tracer_provider:
            self.tracer_provider.force_flush(timeout_millis)

    def shutdown(self, timeout_millis: int = 30000):
        """
        Shutdown tracing provider.

        Args:
            timeout_millis: Timeout in milliseconds
        """
        if not OPENTELEMETRY_AVAILABLE or not self.config.enabled:
            return

        if self.tracer_provider:
            self.tracer_provider.shutdown()
            logger.info("Tracing provider shutdown complete")


class MockSpan:
    """Mock span for when OpenTelemetry is not available."""

    def __init__(self, name: str):
        self.name = name
        self.attributes = {}
        self.events = []

    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]):
        """Set multiple span attributes."""
        self.attributes.update(attributes)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span."""
        self.events.append({"name": name, "attributes": attributes or {}})

    def set_status(self, status):
        """Set span status."""
        pass

    def record_exception(self, exception: Exception):
        """Record exception in span."""
        pass

    def end(self):
        """End span."""
        pass

    def get_span_context(self):
        """Get span context."""
        return MockSpanContext()


class MockSpanContext:
    """Mock span context."""

    @property
    def is_valid(self) -> bool:
        return False

    @property
    def trace_id(self) -> int:
        return 0

    @property
    def span_id(self) -> int:
        return 0


# Global tracing provider instance
_default_provider: Optional[TracingProvider] = None


def get_tracer(service_name: Optional[str] = None) -> TracingProvider:
    """
    Get or create a tracing provider.

    Args:
        service_name: Service name (uses default if None)

    Returns:
        TracingProvider instance
    """
    global _default_provider

    if service_name is None:
        if _default_provider is None:
            _default_provider = TracingProvider()
        return _default_provider

    return TracingProvider(config=TracingConfig(service_name=service_name))


def configure_tracing(config: TracingConfig):
    """
    Configure global tracing settings.

    Args:
        config: TracingConfig instance
    """
    global _default_provider
    _default_provider = TracingProvider(config=config)
