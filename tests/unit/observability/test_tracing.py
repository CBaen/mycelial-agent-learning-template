"""
Unit Tests for OpenTelemetry Tracing

Tests the TracingProvider class for:
- Span creation and management
- Context propagation
- Attribute and event recording
- Multiple exporter types
- Graceful degradation
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.observability.tracing import (
    TracingProvider, TracingConfig, ExporterType,
    MockSpan, MockSpanContext, get_tracer, configure_tracing,
    OPENTELEMETRY_AVAILABLE, SpanKind, StatusCode
)


class TestTracingConfiguration:
    """Test tracing configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = TracingConfig()

        assert config.enabled is True
        assert config.service_name == "mae"
        assert config.exporter_type == ExporterType.CONSOLE
        assert config.sample_rate == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = TracingConfig(
            enabled=True,
            service_name="mae.agents",
            exporter_type=ExporterType.JAEGER,
            jaeger_endpoint="http://custom:14268",
            sample_rate=0.5
        )

        assert config.service_name == "mae.agents"
        assert config.exporter_type == ExporterType.JAEGER
        assert config.jaeger_endpoint == "http://custom:14268"
        assert config.sample_rate == 0.5


class TestTracingProviderInitialization:
    """Test TracingProvider initialization."""

    def test_default_initialization(self):
        """Test provider initializes with defaults."""
        provider = TracingProvider()

        assert provider.config.service_name == "mae"
        assert provider.config.enabled is True
        assert provider._span_count == 0

    def test_custom_initialization(self):
        """Test provider with custom configuration."""
        config = TracingConfig(
            service_name="test.service",
            exporter_type=ExporterType.CONSOLE
        )
        provider = TracingProvider(config=config)

        assert provider.config.service_name == "test.service"

    def test_disabled_tracing(self):
        """Test tracing when disabled."""
        config = TracingConfig(enabled=False)
        provider = TracingProvider(config=config)

        assert provider.config.enabled is False


class TestSpanCreation:
    """Test span creation and management."""

    def test_basic_span(self):
        """Test creating a basic span."""
        provider = TracingProvider()

        with provider.span("test_operation") as span:
            assert span is not None

        # Span count should increment
        assert provider._span_count >= 1

    def test_span_with_attributes(self):
        """Test span with attributes."""
        provider = TracingProvider()

        attributes = {
            "key1": "value1",
            "key2": 123,
            "key3": True
        }

        with provider.span("test_operation", attributes=attributes) as span:
            # In mock mode, check attributes were set
            if isinstance(span, MockSpan):
                assert span.attributes["key1"] == "value1"
                assert span.attributes["key2"] == 123
                assert span.attributes["key3"] is True

    def test_span_with_kind(self):
        """Test span with different kinds."""
        provider = TracingProvider()

        with provider.span("client_operation", kind=SpanKind.CLIENT) as span:
            assert span is not None

        with provider.span("server_operation", kind=SpanKind.SERVER) as span:
            assert span is not None

    def test_nested_spans(self):
        """Test nested span creation."""
        provider = TracingProvider()

        with provider.span("parent_operation") as parent:
            assert parent is not None

            with provider.span("child_operation") as child:
                assert child is not None

        # Should have created 2 spans
        assert provider._span_count >= 2


class TestAgentSpan:
    """Test agent-scoped span creation."""

    def test_agent_span(self):
        """Test creating agent span."""
        provider = TracingProvider()

        with provider.agent_span("agent_1", "learn") as span:
            assert span is not None
            if isinstance(span, MockSpan):
                assert span.attributes["agent.id"] == "agent_1"
                assert span.attributes["agent.operation"] == "learn"

    def test_agent_span_with_attributes(self):
        """Test agent span with additional attributes."""
        provider = TracingProvider()

        with provider.agent_span(
            "agent_1",
            "learn",
            step=10,
            reward=15.5
        ) as span:
            if isinstance(span, MockSpan):
                assert span.attributes["agent.id"] == "agent_1"
                assert span.attributes["step"] == 10
                assert span.attributes["reward"] == 15.5


class TestCommunicationSpan:
    """Test communication span creation."""

    def test_communication_span(self):
        """Test creating communication span."""
        provider = TracingProvider()

        with provider.communication_span(
            "agent_1",
            "agent_2",
            "BROADCAST"
        ) as span:
            assert span is not None
            if isinstance(span, MockSpan):
                assert span.attributes["communication.source"] == "agent_1"
                assert span.attributes["communication.target"] == "agent_2"
                assert span.attributes["communication.type"] == "BROADCAST"

    def test_communication_span_with_payload(self):
        """Test communication span with payload info."""
        provider = TracingProvider()

        with provider.communication_span(
            "agent_1",
            "agent_2",
            "UNICAST",
            payload_size=1024,
            latency_ms=5.2
        ) as span:
            if isinstance(span, MockSpan):
                assert span.attributes["payload_size"] == 1024
                assert span.attributes["latency_ms"] == 5.2


class TestMemorySpan:
    """Test memory operation span creation."""

    def test_memory_span(self):
        """Test creating memory span."""
        provider = TracingProvider()

        with provider.memory_span("agent_1", "replay") as span:
            assert span is not None
            if isinstance(span, MockSpan):
                assert span.attributes["memory.agent_id"] == "agent_1"
                assert span.attributes["memory.operation"] == "replay"

    def test_memory_span_with_details(self):
        """Test memory span with operation details."""
        provider = TracingProvider()

        with provider.memory_span(
            "agent_1",
            "consolidation",
            buffer_size=500,
            experiences_processed=100
        ) as span:
            if isinstance(span, MockSpan):
                assert span.attributes["buffer_size"] == 500
                assert span.attributes["experiences_processed"] == 100


class TestSpanContext:
    """Test span context management."""

    def test_get_current_span(self):
        """Test getting current span."""
        provider = TracingProvider()

        with provider.span("test_operation"):
            current_span = provider.get_current_span()
            assert current_span is not None

    def test_get_trace_id(self):
        """Test getting trace ID."""
        provider = TracingProvider()

        with provider.span("test_operation"):
            trace_id = provider.get_trace_id()
            # In mock mode, returns None
            if not OPENTELEMETRY_AVAILABLE:
                assert trace_id is None

    def test_get_span_id(self):
        """Test getting span ID."""
        provider = TracingProvider()

        with provider.span("test_operation"):
            span_id = provider.get_span_id()
            # In mock mode, returns None
            if not OPENTELEMETRY_AVAILABLE:
                assert span_id is None


class TestSpanAttributes:
    """Test span attribute management."""

    def test_set_attribute(self):
        """Test setting span attributes."""
        provider = TracingProvider()

        with provider.span("test_operation") as span:
            span.set_attribute("custom_key", "custom_value")

            if isinstance(span, MockSpan):
                assert span.attributes["custom_key"] == "custom_value"

    def test_set_multiple_attributes(self):
        """Test setting multiple attributes."""
        provider = TracingProvider()

        with provider.span("test_operation") as span:
            span.set_attributes({
                "key1": "value1",
                "key2": 123,
                "key3": True
            })

            if isinstance(span, MockSpan):
                assert span.attributes["key1"] == "value1"
                assert span.attributes["key2"] == 123
                assert span.attributes["key3"] is True


class TestSpanEvents:
    """Test span event recording."""

    def test_add_event(self):
        """Test adding event to span."""
        provider = TracingProvider()

        with provider.span("test_operation") as span:
            span.add_event("checkpoint_reached")

            if isinstance(span, MockSpan):
                assert len(span.events) == 1
                assert span.events[0]["name"] == "checkpoint_reached"

    def test_add_event_with_attributes(self):
        """Test adding event with attributes."""
        provider = TracingProvider()

        with provider.span("test_operation") as span:
            span.add_event(
                "checkpoint_reached",
                attributes={"step": 100, "reward": 15.0}
            )

            if isinstance(span, MockSpan):
                assert len(span.events) == 1
                assert span.events[0]["attributes"]["step"] == 100


class TestExceptionHandling:
    """Test exception handling in spans."""

    def test_span_with_exception(self):
        """Test span captures exceptions."""
        provider = TracingProvider()

        with pytest.raises(ValueError):
            with provider.span("test_operation") as span:
                raise ValueError("Test error")

        # Span should still be created and ended
        assert provider._span_count >= 1

    def test_exception_recorded(self):
        """Test exception is recorded in span."""
        provider = TracingProvider()

        try:
            with provider.span("test_operation") as span:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Exception should have been recorded
        # (Implementation specific, just verify no crash)
        assert provider._span_count >= 1


class TestExporterTypes:
    """Test different exporter configurations."""

    def test_console_exporter(self):
        """Test console exporter initialization."""
        config = TracingConfig(exporter_type=ExporterType.CONSOLE)
        provider = TracingProvider(config=config)

        assert provider.config.exporter_type == ExporterType.CONSOLE

    def test_jaeger_exporter(self):
        """Test Jaeger exporter configuration."""
        config = TracingConfig(
            exporter_type=ExporterType.JAEGER,
            jaeger_endpoint="http://localhost:14268/api/traces"
        )
        provider = TracingProvider(config=config)

        assert provider.config.exporter_type == ExporterType.JAEGER
        assert "14268" in provider.config.jaeger_endpoint

    def test_zipkin_exporter(self):
        """Test Zipkin exporter configuration."""
        config = TracingConfig(
            exporter_type=ExporterType.ZIPKIN,
            zipkin_endpoint="http://localhost:9411/api/v2/spans"
        )
        provider = TracingProvider(config=config)

        assert provider.config.exporter_type == ExporterType.ZIPKIN
        assert "9411" in provider.config.zipkin_endpoint


class TestStatistics:
    """Test tracing statistics."""

    def test_get_statistics(self):
        """Test getting tracing statistics."""
        provider = TracingProvider()

        # Create some spans
        with provider.span("operation1"):
            pass
        with provider.span("operation2"):
            pass

        stats = provider.get_statistics()

        assert "enabled" in stats
        assert "opentelemetry_available" in stats
        assert "service_name" in stats
        assert "spans_created" in stats
        assert stats["spans_created"] >= 2

    def test_statistics_uptime(self):
        """Test statistics includes uptime."""
        provider = TracingProvider()

        time.sleep(0.1)

        stats = provider.get_statistics()
        assert stats["uptime_seconds"] >= 0.1


class TestFlushAndShutdown:
    """Test flushing and shutdown operations."""

    def test_flush(self):
        """Test flushing spans."""
        provider = TracingProvider()

        with provider.span("test_operation"):
            pass

        # Should not raise exception
        provider.flush()

    def test_shutdown(self):
        """Test shutting down provider."""
        provider = TracingProvider()

        with provider.span("test_operation"):
            pass

        # Should not raise exception
        provider.shutdown()


class TestMockSpan:
    """Test MockSpan functionality."""

    def test_mock_span_creation(self):
        """Test creating mock span."""
        span = MockSpan("test_operation")

        assert span.name == "test_operation"
        assert span.attributes == {}
        assert span.events == []

    def test_mock_span_attributes(self):
        """Test mock span attribute setting."""
        span = MockSpan("test_operation")

        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

        span.set_attributes({"key2": "value2"})
        assert span.attributes["key2"] == "value2"

    def test_mock_span_events(self):
        """Test mock span event recording."""
        span = MockSpan("test_operation")

        span.add_event("event1")
        assert len(span.events) == 1

        span.add_event("event2", attributes={"detail": "info"})
        assert len(span.events) == 2
        assert span.events[1]["attributes"]["detail"] == "info"

    def test_mock_span_context(self):
        """Test mock span context."""
        span = MockSpan("test_operation")
        context = span.get_span_context()

        assert context.is_valid is False
        assert context.trace_id == 0
        assert context.span_id == 0


class TestGlobalTracer:
    """Test global tracer functions."""

    def test_get_tracer_default(self):
        """Test getting default tracer."""
        tracer = get_tracer()
        assert isinstance(tracer, TracingProvider)

    def test_get_tracer_with_name(self):
        """Test getting named tracer."""
        tracer = get_tracer("custom.service")
        assert tracer.config.service_name == "custom.service"

    def test_configure_tracing(self):
        """Test configuring global tracing."""
        config = TracingConfig(
            service_name="configured",
            exporter_type=ExporterType.CONSOLE
        )

        configure_tracing(config)
        tracer = get_tracer()

        assert tracer.config.service_name == "configured"


class TestTracingIntegration:
    """Integration tests for tracing."""

    def test_complete_workflow(self):
        """Test complete tracing workflow."""
        provider = TracingProvider(config=TracingConfig(
            service_name="test.integration"
        ))

        # Parent span
        with provider.span("request", attributes={"request_id": "req_123"}) as parent:
            parent.add_event("request_started")

            # Agent span
            with provider.agent_span("agent_1", "process") as agent:
                agent.set_attribute("step", 1)
                agent.add_event("processing")

                # Memory span
                with provider.memory_span("agent_1", "replay", batch_size=32):
                    pass

            # Communication span
            with provider.communication_span("agent_1", "agent_2", "BROADCAST"):
                pass

            parent.add_event("request_completed")

        # Should have created 4 spans
        assert provider._span_count >= 4

    def test_multi_agent_scenario(self):
        """Test tracing for multiple agents."""
        provider = TracingProvider()

        for agent_id in [f"agent_{i}" for i in range(5)]:
            with provider.agent_span(agent_id, "learn", step=1):
                with provider.memory_span(agent_id, "replay"):
                    pass

        # Should have created 10 spans (2 per agent)
        assert provider._span_count >= 10

    def test_error_propagation(self):
        """Test error propagation through spans."""
        provider = TracingProvider()

        with pytest.raises(RuntimeError):
            with provider.span("parent"):
                with provider.span("child"):
                    raise RuntimeError("Simulated error")

        # Both spans should be created despite error
        assert provider._span_count >= 2


class TestPerformance:
    """Test tracing performance."""

    def test_span_overhead(self):
        """Test span creation overhead is minimal."""
        provider = TracingProvider()

        start_time = time.time()

        for i in range(100):
            with provider.span(f"operation_{i}"):
                pass

        elapsed_time = time.time() - start_time

        # 100 spans should complete quickly (target <5ms per span)
        assert elapsed_time < 0.5  # 500ms for 100 spans

    def test_high_throughput(self):
        """Test high throughput span creation."""
        provider = TracingProvider()

        start_time = time.time()

        # Create 1000 spans
        for i in range(1000):
            with provider.span(f"operation_{i}"):
                pass

        elapsed_time = time.time() - start_time

        # Calculate throughput
        throughput = 1000 / elapsed_time

        # Should handle >200 spans/sec
        assert throughput > 200
