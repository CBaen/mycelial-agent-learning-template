"""
Unit Tests for Prometheus Metrics Collection

Tests the MetricsCollector class for:
- Agent performance metrics
- System health metrics
- Communication metrics
- Memory metrics
- Metric export and statistics
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.observability.metrics import MetricsCollector, MetricsConfig, PROMETHEUS_AVAILABLE


class TestMetricsCollectorInitialization:
    """Test MetricsCollector initialization."""

    def test_default_initialization(self):
        """Test collector initializes with defaults."""
        collector = MetricsCollector()

        assert collector.namespace == "mae"
        assert collector.subsystem == "system"
        assert collector.config.enabled is True
        assert collector.metrics_count == 0
        assert collector.get_uptime() >= 0

    def test_custom_initialization(self):
        """Test collector with custom configuration."""
        config = MetricsConfig(
            enabled=True,
            collection_interval=2.0,
            enable_agent_metrics=True
        )

        collector = MetricsCollector(
            namespace="test",
            subsystem="agents",
            config=config
        )

        assert collector.namespace == "test"
        assert collector.subsystem == "agents"
        assert collector.config.collection_interval == 2.0

    def test_disabled_metrics(self):
        """Test collector with metrics disabled."""
        config = MetricsConfig(enabled=False)
        collector = MetricsCollector(config=config)

        # Recording should not increase count
        collector.record_agent_reward("agent_1", 10.0)
        assert collector.metrics_count == 0


class TestAgentMetrics:
    """Test agent performance metrics recording."""

    def test_record_learning_step(self):
        """Test recording learning steps."""
        collector = MetricsCollector()

        collector.record_learning_step("agent_1", 0.001)
        collector.record_learning_step("agent_1", 0.0005)

        assert collector.metrics_count >= 2

    def test_record_agent_reward(self):
        """Test recording agent rewards."""
        collector = MetricsCollector()

        collector.record_agent_reward("agent_1", 10.5)
        collector.record_agent_reward("agent_2", 5.0)
        collector.record_agent_reward("agent_1", 15.0)

        assert collector.metrics_count >= 3

    def test_record_convergence(self):
        """Test recording convergence scores."""
        collector = MetricsCollector()

        collector.record_convergence("agent_1", 0.85)
        assert collector.metrics_count >= 1

    def test_record_satisfaction(self):
        """Test recording satisfaction scores."""
        collector = MetricsCollector()

        collector.record_satisfaction("agent_1", 0.75)
        collector.record_satisfaction("agent_2", 0.90)

        assert collector.metrics_count >= 2

    def test_record_agent_xp(self):
        """Test recording agent XP (gamification)."""
        collector = MetricsCollector()

        collector.record_agent_xp("agent_1", 1000)
        collector.record_agent_xp("agent_1", 1500)

        assert collector.metrics_count >= 2

    def test_disabled_agent_metrics(self):
        """Test agent metrics when disabled."""
        config = MetricsConfig(enable_agent_metrics=False)
        collector = MetricsCollector(config=config)

        collector.record_agent_reward("agent_1", 10.0)
        assert collector.metrics_count == 0


class TestSystemMetrics:
    """Test system health metrics recording."""

    def test_record_system_metrics(self):
        """Test recording system metrics (CPU, memory)."""
        collector = MetricsCollector()

        collector.record_system_metrics()

        # Should record CPU and memory
        assert collector.metrics_count >= 2

    def test_record_request_latency(self):
        """Test recording request latency."""
        collector = MetricsCollector()

        collector.record_request_latency("get_policy", 0.025)
        collector.record_request_latency("update_agent", 0.050)

        assert collector.metrics_count >= 2

    def test_record_error(self):
        """Test recording errors."""
        collector = MetricsCollector()

        collector.record_error("ValueError")
        collector.record_error("TimeoutError")
        collector.record_error("ValueError")

        assert collector.metrics_count >= 3

    def test_disabled_system_metrics(self):
        """Test system metrics when disabled."""
        config = MetricsConfig(enable_system_metrics=False)
        collector = MetricsCollector(config=config)

        collector.record_system_metrics()
        collector.record_request_latency("test", 0.01)

        assert collector.metrics_count == 0


class TestCommunicationMetrics:
    """Test communication metrics recording."""

    def test_record_message_sent(self):
        """Test recording messages sent."""
        collector = MetricsCollector()

        collector.record_message_sent("agent_1", "BROADCAST")
        collector.record_message_sent("agent_2", "UNICAST")

        assert collector.metrics_count >= 2

    def test_record_message_received(self):
        """Test recording messages received."""
        collector = MetricsCollector()

        collector.record_message_received("agent_1", "BROADCAST")
        collector.record_message_received("agent_1", "UNICAST")

        assert collector.metrics_count >= 2

    def test_record_message_latency(self):
        """Test recording message latency."""
        collector = MetricsCollector()

        collector.record_message_latency("BROADCAST", 0.0005)
        collector.record_message_latency("UNICAST", 0.001)

        assert collector.metrics_count >= 2

    def test_record_gnn_routing_efficiency(self):
        """Test recording GNN routing efficiency."""
        collector = MetricsCollector()

        collector.record_gnn_routing_efficiency(0.85)
        collector.record_gnn_routing_efficiency(0.90)

        assert collector.metrics_count >= 2

    def test_disabled_communication_metrics(self):
        """Test communication metrics when disabled."""
        config = MetricsConfig(enable_communication_metrics=False)
        collector = MetricsCollector(config=config)

        collector.record_message_sent("agent_1", "BROADCAST")
        assert collector.metrics_count == 0


class TestMemoryMetrics:
    """Test memory subsystem metrics recording."""

    def test_record_episodic_buffer_state(self):
        """Test recording episodic buffer state."""
        collector = MetricsCollector()

        collector.record_episodic_buffer_state("agent_1", 500, 1000)
        collector.record_episodic_buffer_state("agent_2", 750, 1000)

        assert collector.metrics_count >= 4  # 2 metrics per call

    def test_record_replay(self):
        """Test recording memory replay."""
        collector = MetricsCollector()

        collector.record_replay("agent_1")
        collector.record_replay("agent_1")
        collector.record_replay("agent_2")

        assert collector.metrics_count >= 3

    def test_record_consolidation(self):
        """Test recording memory consolidation."""
        collector = MetricsCollector()

        collector.record_consolidation("agent_1")
        collector.record_consolidation("agent_2")

        assert collector.metrics_count >= 2

    def test_disabled_memory_metrics(self):
        """Test memory metrics when disabled."""
        config = MetricsConfig(enable_memory_metrics=False)
        collector = MetricsCollector(config=config)

        collector.record_replay("agent_1")
        assert collector.metrics_count == 0


class TestMetricsExport:
    """Test metrics export and statistics."""

    def test_export_metrics(self):
        """Test exporting metrics in Prometheus format."""
        collector = MetricsCollector()

        # Record some metrics
        collector.record_agent_reward("agent_1", 10.0)
        collector.record_system_metrics()

        # Export metrics
        metrics_data = collector.export_metrics()

        assert isinstance(metrics_data, bytes)
        if PROMETHEUS_AVAILABLE:
            assert len(metrics_data) > 0

    def test_get_metrics_count(self):
        """Test getting metrics count."""
        collector = MetricsCollector()

        initial_count = collector.get_metrics_count()
        collector.record_agent_reward("agent_1", 10.0)
        collector.record_learning_step("agent_1", 0.001)

        assert collector.get_metrics_count() > initial_count

    def test_get_uptime(self):
        """Test getting collector uptime."""
        collector = MetricsCollector()

        time.sleep(0.1)
        uptime = collector.get_uptime()

        assert uptime >= 0.1

    def test_get_statistics(self):
        """Test getting collector statistics."""
        collector = MetricsCollector()

        stats = collector.get_statistics()

        assert "enabled" in stats
        assert "prometheus_available" in stats
        assert "metrics_recorded" in stats
        assert "uptime_seconds" in stats
        assert stats["namespace"] == "mae"
        assert stats["subsystem"] == "system"

    def test_reset_metrics(self):
        """Test resetting metrics."""
        collector = MetricsCollector()

        # Record some metrics
        collector.record_agent_reward("agent_1", 10.0)
        assert collector.metrics_count > 0

        # Reset
        collector.reset()
        assert collector.metrics_count == 0


class TestMetricsThreadSafety:
    """Test thread safety of metrics collection."""

    def test_concurrent_recording(self):
        """Test concurrent metric recording."""
        import threading

        collector = MetricsCollector()

        def record_metrics():
            for i in range(100):
                collector.record_agent_reward(f"agent_{i % 10}", float(i))

        threads = [threading.Thread(target=record_metrics) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have recorded 1000 metrics (10 threads * 100 each)
        assert collector.metrics_count == 1000


class TestMetricsPerformance:
    """Test metrics collection performance."""

    def test_recording_overhead(self):
        """Test that metrics recording is fast (<1ms per metric)."""
        collector = MetricsCollector()

        start_time = time.time()

        for i in range(100):
            collector.record_agent_reward(f"agent_{i}", float(i))

        elapsed_time = time.time() - start_time

        # 100 metrics should complete in <100ms (1ms per metric target)
        assert elapsed_time < 0.1

    def test_high_throughput(self):
        """Test high throughput metrics recording (1000+ metrics/sec)."""
        collector = MetricsCollector()

        start_time = time.time()

        # Record 10,000 metrics
        for i in range(10000):
            collector.record_agent_reward(f"agent_{i % 100}", float(i))

        elapsed_time = time.time() - start_time

        # Calculate throughput
        throughput = 10000 / elapsed_time

        # Should exceed 1,000 metrics/sec
        assert throughput > 1000


class TestMetricsIntegration:
    """Integration tests for metrics collector."""

    def test_complete_workflow(self):
        """Test complete metrics collection workflow."""
        collector = MetricsCollector()

        # Agent metrics
        collector.record_learning_step("agent_1", 0.001)
        collector.record_agent_reward("agent_1", 10.5)
        collector.record_convergence("agent_1", 0.85)

        # System metrics
        collector.record_system_metrics()
        collector.record_request_latency("get_policy", 0.025)

        # Communication metrics
        collector.record_message_sent("agent_1", "BROADCAST")
        collector.record_message_latency("BROADCAST", 0.0005)

        # Memory metrics
        collector.record_episodic_buffer_state("agent_1", 500, 1000)
        collector.record_replay("agent_1")

        # Export metrics
        metrics_data = collector.export_metrics()

        assert collector.metrics_count >= 9
        assert isinstance(metrics_data, bytes)

    def test_multi_agent_scenario(self):
        """Test metrics collection for multiple agents."""
        collector = MetricsCollector()

        # Simulate 10 agents
        for agent_id in [f"agent_{i}" for i in range(10)]:
            collector.record_learning_step(agent_id, 0.001)
            collector.record_agent_reward(agent_id, 10.0)
            collector.record_episodic_buffer_state(agent_id, 500, 1000)

        # Should record 30 metrics (3 per agent)
        assert collector.metrics_count >= 30
