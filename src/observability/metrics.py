"""
Prometheus Metrics Collection for MAE

This module provides a comprehensive metrics collection system using
Prometheus client library for production observability.

Key metrics:
- Agent performance (learning rate, rewards, convergence)
- System health (CPU, memory, latency)
- Communication (message throughput, GNN routing efficiency)
- Memory (episodic buffer utilization, replay frequency)
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from threading import Lock

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Provide mock classes for testing without prometheus
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class CollectorRegistry:
        def __init__(self): pass

    def generate_latest(registry): return b""
    CONTENT_TYPE_LATEST = "text/plain"


logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    enabled: bool = True
    collection_interval: float = 1.0  # seconds
    enable_system_metrics: bool = True
    enable_agent_metrics: bool = True
    enable_communication_metrics: bool = True
    enable_memory_metrics: bool = True
    custom_labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Prometheus metrics collector for MAE.

    Provides comprehensive metrics collection across:
    - Agent performance (learning, rewards, convergence)
    - System health (CPU, memory, latency)
    - Communication (messages, routing, signals)
    - Memory (episodic buffer, replay)

    Example:
        >>> collector = MetricsCollector(namespace="mae", subsystem="agents")
        >>> collector.record_agent_reward("agent_1", 10.5)
        >>> collector.record_learning_step("agent_1", 0.001)
        >>> metrics_data = collector.export_metrics()
    """

    def __init__(
        self,
        namespace: str = "mae",
        subsystem: str = "system",
        config: Optional[MetricsConfig] = None,
        registry: Optional[CollectorRegistry] = None
    ):
        """
        Initialize metrics collector.

        Args:
            namespace: Prometheus namespace (e.g., "mae")
            subsystem: Prometheus subsystem (e.g., "agents", "system")
            config: MetricsConfig instance
            registry: Prometheus registry (creates new if None)
        """
        self.namespace = namespace
        self.subsystem = subsystem
        self.config = config or MetricsConfig()
        self.registry = registry or CollectorRegistry()
        self.lock = Lock()

        # Track metrics state
        self.metrics_count = 0
        self.start_time = time.time()

        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Metrics will be mocked.")

        # Initialize metric collectors
        self._init_agent_metrics()
        self._init_system_metrics()
        self._init_communication_metrics()
        self._init_memory_metrics()

        logger.info(f"MetricsCollector initialized: {namespace}.{subsystem}")

    def _init_agent_metrics(self):
        """Initialize agent performance metrics."""
        # Agent learning metrics
        self.learning_steps = Counter(
            f"{self.namespace}_agent_learning_steps_total",
            "Total learning steps per agent",
            ["agent_id"],
            registry=self.registry
        )

        self.rewards = Histogram(
            f"{self.namespace}_agent_reward",
            "Agent rewards distribution",
            ["agent_id"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            registry=self.registry
        )

        self.learning_rate = Gauge(
            f"{self.namespace}_agent_learning_rate",
            "Current learning rate",
            ["agent_id"],
            registry=self.registry
        )

        self.convergence_score = Gauge(
            f"{self.namespace}_agent_convergence_score",
            "Agent convergence score (0-1)",
            ["agent_id"],
            registry=self.registry
        )

        self.satisfaction_score = Gauge(
            f"{self.namespace}_agent_satisfaction",
            "Agent satisfaction score (Big Rock 4)",
            ["agent_id"],
            registry=self.registry
        )

        self.agent_xp = Gauge(
            f"{self.namespace}_agent_xp",
            "Agent experience points (gamification)",
            ["agent_id"],
            registry=self.registry
        )

    def _init_system_metrics(self):
        """Initialize system health metrics."""
        self.cpu_usage = Gauge(
            f"{self.namespace}_system_cpu_percent",
            "CPU usage percentage",
            registry=self.registry
        )

        self.memory_usage = Gauge(
            f"{self.namespace}_system_memory_bytes",
            "Memory usage in bytes",
            registry=self.registry
        )

        self.request_latency = Histogram(
            f"{self.namespace}_system_request_latency_seconds",
            "Request latency distribution",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )

        self.error_count = Counter(
            f"{self.namespace}_system_errors_total",
            "Total error count",
            ["error_type"],
            registry=self.registry
        )

    def _init_communication_metrics(self):
        """Initialize communication metrics."""
        self.messages_sent = Counter(
            f"{self.namespace}_communication_messages_sent_total",
            "Total messages sent",
            ["message_type", "agent_id"],
            registry=self.registry
        )

        self.messages_received = Counter(
            f"{self.namespace}_communication_messages_received_total",
            "Total messages received",
            ["message_type", "agent_id"],
            registry=self.registry
        )

        self.message_latency = Histogram(
            f"{self.namespace}_communication_message_latency_seconds",
            "Message delivery latency",
            ["message_type"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            registry=self.registry
        )

        self.gnn_routing_efficiency = Gauge(
            f"{self.namespace}_communication_gnn_routing_efficiency",
            "GNN routing efficiency (0-1)",
            registry=self.registry
        )

    def _init_memory_metrics(self):
        """Initialize memory subsystem metrics."""
        self.episodic_buffer_size = Gauge(
            f"{self.namespace}_memory_episodic_buffer_size",
            "Current episodic buffer size",
            ["agent_id"],
            registry=self.registry
        )

        self.episodic_buffer_capacity = Gauge(
            f"{self.namespace}_memory_episodic_buffer_capacity",
            "Episodic buffer capacity",
            ["agent_id"],
            registry=self.registry
        )

        self.replay_frequency = Counter(
            f"{self.namespace}_memory_replay_total",
            "Total replay operations",
            ["agent_id"],
            registry=self.registry
        )

        self.consolidation_count = Counter(
            f"{self.namespace}_memory_consolidation_total",
            "Total memory consolidation operations",
            ["agent_id"],
            registry=self.registry
        )

    # Agent Metrics Recording Methods

    def record_learning_step(self, agent_id: str, learning_rate: float):
        """Record a learning step for an agent."""
        if not self.config.enabled or not self.config.enable_agent_metrics:
            return

        with self.lock:
            self.learning_steps.labels(agent_id=agent_id).inc()
            self.learning_rate.labels(agent_id=agent_id).set(learning_rate)
            self.metrics_count += 1

    def record_agent_reward(self, agent_id: str, reward: float):
        """Record agent reward."""
        if not self.config.enabled or not self.config.enable_agent_metrics:
            return

        with self.lock:
            self.rewards.labels(agent_id=agent_id).observe(reward)
            self.metrics_count += 1

    def record_convergence(self, agent_id: str, score: float):
        """Record agent convergence score (0-1)."""
        if not self.config.enabled or not self.config.enable_agent_metrics:
            return

        with self.lock:
            self.convergence_score.labels(agent_id=agent_id).set(score)
            self.metrics_count += 1

    def record_satisfaction(self, agent_id: str, satisfaction: float):
        """Record agent satisfaction score."""
        if not self.config.enabled or not self.config.enable_agent_metrics:
            return

        with self.lock:
            self.satisfaction_score.labels(agent_id=agent_id).set(satisfaction)
            self.metrics_count += 1

    def record_agent_xp(self, agent_id: str, xp: int):
        """Record agent experience points (gamification)."""
        if not self.config.enabled or not self.config.enable_agent_metrics:
            return

        with self.lock:
            self.agent_xp.labels(agent_id=agent_id).set(xp)
            self.metrics_count += 1

    # System Metrics Recording Methods

    def record_system_metrics(self):
        """Record current system metrics (CPU, memory)."""
        if not self.config.enabled or not self.config.enable_system_metrics:
            return

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.Process().memory_info()

            with self.lock:
                self.cpu_usage.set(cpu_percent)
                self.memory_usage.set(memory_info.rss)
                self.metrics_count += 2
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def record_request_latency(self, operation: str, latency_seconds: float):
        """Record request latency."""
        if not self.config.enabled or not self.config.enable_system_metrics:
            return

        with self.lock:
            self.request_latency.labels(operation=operation).observe(latency_seconds)
            self.metrics_count += 1

    def record_error(self, error_type: str):
        """Record an error occurrence."""
        if not self.config.enabled:
            return

        with self.lock:
            self.error_count.labels(error_type=error_type).inc()
            self.metrics_count += 1

    # Communication Metrics Recording Methods

    def record_message_sent(self, agent_id: str, message_type: str):
        """Record message sent."""
        if not self.config.enabled or not self.config.enable_communication_metrics:
            return

        with self.lock:
            self.messages_sent.labels(message_type=message_type, agent_id=agent_id).inc()
            self.metrics_count += 1

    def record_message_received(self, agent_id: str, message_type: str):
        """Record message received."""
        if not self.config.enabled or not self.config.enable_communication_metrics:
            return

        with self.lock:
            self.messages_received.labels(message_type=message_type, agent_id=agent_id).inc()
            self.metrics_count += 1

    def record_message_latency(self, message_type: str, latency_seconds: float):
        """Record message delivery latency."""
        if not self.config.enabled or not self.config.enable_communication_metrics:
            return

        with self.lock:
            self.message_latency.labels(message_type=message_type).observe(latency_seconds)
            self.metrics_count += 1

    def record_gnn_routing_efficiency(self, efficiency: float):
        """Record GNN routing efficiency (0-1)."""
        if not self.config.enabled or not self.config.enable_communication_metrics:
            return

        with self.lock:
            self.gnn_routing_efficiency.set(efficiency)
            self.metrics_count += 1

    # Memory Metrics Recording Methods

    def record_episodic_buffer_state(self, agent_id: str, size: int, capacity: int):
        """Record episodic buffer state."""
        if not self.config.enabled or not self.config.enable_memory_metrics:
            return

        with self.lock:
            self.episodic_buffer_size.labels(agent_id=agent_id).set(size)
            self.episodic_buffer_capacity.labels(agent_id=agent_id).set(capacity)
            self.metrics_count += 2

    def record_replay(self, agent_id: str):
        """Record memory replay operation."""
        if not self.config.enabled or not self.config.enable_memory_metrics:
            return

        with self.lock:
            self.replay_frequency.labels(agent_id=agent_id).inc()
            self.metrics_count += 1

    def record_consolidation(self, agent_id: str):
        """Record memory consolidation operation."""
        if not self.config.enabled or not self.config.enable_memory_metrics:
            return

        with self.lock:
            self.consolidation_count.labels(agent_id=agent_id).inc()
            self.metrics_count += 1

    # Export and Utility Methods

    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.

        Returns:
            bytes: Prometheus-formatted metrics data
        """
        if not PROMETHEUS_AVAILABLE:
            return b"# Prometheus client not available\n"

        return generate_latest(self.registry)

    def get_metrics_count(self) -> int:
        """Get total number of metrics recorded."""
        return self.metrics_count

    def get_uptime(self) -> float:
        """Get collector uptime in seconds."""
        return time.time() - self.start_time

    def reset(self):
        """Reset all metrics (for testing)."""
        logger.warning("Resetting all metrics")
        self.__init__(
            namespace=self.namespace,
            subsystem=self.subsystem,
            config=self.config
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get metrics collector statistics."""
        return {
            "enabled": self.config.enabled,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "metrics_recorded": self.metrics_count,
            "uptime_seconds": self.get_uptime(),
            "namespace": self.namespace,
            "subsystem": self.subsystem
        }
