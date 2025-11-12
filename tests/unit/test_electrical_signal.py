"""
Unit tests for Electrical Signaling Layer (Big Rock 5)

Tests the ElectricalSignalBus, Signal, SignalPriority, and related functionality.

Author: MAE Development Team
Date: 2025-11-12
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.core.electrical_signal import (
    ElectricalSignalBus,
    Signal,
    SignalPriority,
    SignalMetrics,
    RateLimiter
)
from src.core.signal_types import SignalType, get_signal_info, validate_signal_payload


class TestSignal:
    """Test the Signal dataclass"""

    def test_signal_creation(self):
        """Test creating a signal"""
        signal = Signal(
            signal_type=SignalType.DANGER,
            source_agent_id="agent_1",
            timestamp=time.time(),
            payload={"risk_level": 0.9},
            priority=SignalPriority.CRITICAL
        )

        assert signal.signal_type == SignalType.DANGER
        assert signal.source_agent_id == "agent_1"
        assert signal.payload["risk_level"] == 0.9
        assert signal.priority == SignalPriority.CRITICAL
        assert signal.signal_id is not None

    def test_signal_expiry(self):
        """Test signal TTL expiration"""
        # Signal with 0.1 second TTL
        signal = Signal(
            signal_type=SignalType.HEARTBEAT,
            source_agent_id="agent_1",
            timestamp=time.time(),
            payload={},
            ttl=0.1
        )

        # Should not be expired immediately
        assert not signal.is_expired()

        # Wait for expiration
        time.sleep(0.15)
        assert signal.is_expired()

    def test_signal_no_expiry(self):
        """Test signal with infinite TTL"""
        signal = Signal(
            signal_type=SignalType.DANGER,
            source_agent_id="agent_1",
            timestamp=time.time() - 1000,  # Old signal
            payload={},
            ttl=0.0  # Infinite
        )

        assert not signal.is_expired()

    def test_signal_age(self):
        """Test signal age calculation"""
        signal = Signal(
            signal_type=SignalType.OPPORTUNITY,
            source_agent_id="agent_1",
            timestamp=time.time() - 0.5,  # 0.5 seconds ago
            payload={}
        )

        age = signal.age_ms()
        assert 450 < age < 550  # ~500ms with some tolerance


class TestRateLimiter:
    """Test the RateLimiter class"""

    def test_rate_limiter_allows_initial_requests(self):
        """Test that rate limiter allows initial requests"""
        limiter = RateLimiter(rate_per_second=10, burst_size=5)

        # Should allow first 5 requests
        for i in range(5):
            assert limiter.allow(f"agent_{i}")

    def test_rate_limiter_blocks_excess_burst(self):
        """Test that rate limiter blocks requests exceeding burst"""
        limiter = RateLimiter(rate_per_second=10, burst_size=5)

        agent_id = "agent_1"

        # Allow burst
        for i in range(5):
            assert limiter.allow(agent_id)

        # Exceed burst
        assert not limiter.allow(agent_id)

    def test_rate_limiter_refills_over_time(self):
        """Test that rate limiter refills tokens over time"""
        limiter = RateLimiter(rate_per_second=100, burst_size=10)

        agent_id = "agent_1"

        # Consume all tokens
        for i in range(10):
            assert limiter.allow(agent_id)

        # Should be blocked
        assert not limiter.allow(agent_id)

        # Wait for refill (0.1 second = 10 tokens at 100/sec)
        time.sleep(0.15)

        # Should allow again
        assert limiter.allow(agent_id)

    def test_rate_limiter_per_agent(self):
        """Test that rate limiting is per-agent"""
        limiter = RateLimiter(rate_per_second=10, burst_size=2)

        # Agent 1 uses burst
        assert limiter.allow("agent_1")
        assert limiter.allow("agent_1")
        assert not limiter.allow("agent_1")

        # Agent 2 should still have burst available
        assert limiter.allow("agent_2")
        assert limiter.allow("agent_2")


class TestSignalMetrics:
    """Test the SignalMetrics class"""

    def test_metrics_initialization(self):
        """Test metrics starts at zero"""
        metrics = SignalMetrics()

        assert metrics.total_signals_emitted == 0
        assert metrics.total_signals_delivered == 0
        assert metrics.total_signals_dropped == 0

    def test_metrics_record_emission(self):
        """Test recording signal emissions"""
        metrics = SignalMetrics()

        signal = Signal(
            signal_type=SignalType.DANGER,
            source_agent_id="agent_1",
            timestamp=time.time(),
            payload={},
            priority=SignalPriority.CRITICAL
        )

        metrics.record_emission(signal)

        assert metrics.total_signals_emitted == 1
        assert metrics.signals_by_type[SignalType.DANGER] == 1
        assert metrics.signals_by_priority[SignalPriority.CRITICAL] == 1

    def test_metrics_record_delivery(self):
        """Test recording signal deliveries"""
        metrics = SignalMetrics()

        signal = Signal(
            signal_type=SignalType.OPPORTUNITY,
            source_agent_id="agent_1",
            timestamp=time.time(),
            payload={}
        )

        metrics.record_delivery(signal, delivery_time_ms=0.5)

        assert metrics.total_signals_delivered == 1
        assert 0.5 in metrics.propagation_times_ms

    def test_metrics_get_stats(self):
        """Test getting metrics statistics"""
        metrics = SignalMetrics()

        signal = Signal(
            signal_type=SignalType.DANGER,
            source_agent_id="agent_1",
            timestamp=time.time(),
            payload={},
            priority=SignalPriority.HIGH
        )

        metrics.record_emission(signal)
        metrics.record_delivery(signal, 0.8)

        stats = metrics.get_stats()

        assert stats['total_emitted'] == 1
        assert stats['total_delivered'] == 1
        assert stats['avg_propagation_ms'] == 0.8
        assert stats['delivery_rate'] == 100.0


class TestElectricalSignalBus:
    """Test the ElectricalSignalBus class"""

    @pytest.fixture
    def signal_bus(self):
        """Create a signal bus for testing"""
        return ElectricalSignalBus(
            max_history=100,
            enable_async=False,  # Disable async for deterministic tests
            rate_limit_per_agent=1000,
            enable_monitoring=True
        )

    def test_bus_initialization(self, signal_bus):
        """Test signal bus initializes correctly"""
        assert signal_bus is not None
        assert signal_bus.enable_monitoring
        assert signal_bus.metrics is not None

    def test_emit_signal(self, signal_bus):
        """Test emitting a signal"""
        success = signal_bus.emit_signal(
            signal_type=SignalType.DANGER,
            source_agent_id="agent_1",
            payload={"risk_level": 0.9},
            priority=SignalPriority.CRITICAL
        )

        assert success
        assert signal_bus.metrics.total_signals_emitted == 1

    def test_subscribe_and_receive(self, signal_bus):
        """Test subscribing to and receiving signals"""
        received_signals = []

        def callback(signal: Signal):
            received_signals.append(signal)

        # Subscribe
        success = signal_bus.subscribe(
            signal_type=SignalType.DANGER,
            agent_id="agent_2",
            callback=callback,
            min_priority=SignalPriority.CRITICAL
        )

        assert success

        # Emit signal
        signal_bus.emit_signal(
            signal_type=SignalType.DANGER,
            source_agent_id="agent_1",
            payload={"risk_level": 0.9},
            priority=SignalPriority.CRITICAL
        )

        # Wait briefly for propagation (even in sync mode)
        time.sleep(0.01)

        # Should have received signal
        assert len(received_signals) == 1
        assert received_signals[0].signal_type == SignalType.DANGER
        assert received_signals[0].payload["risk_level"] == 0.9

    def test_no_self_signaling(self, signal_bus):
        """Test that agents don't receive their own signals"""
        received_signals = []

        def callback(signal: Signal):
            received_signals.append(signal)

        # Subscribe agent_1
        signal_bus.subscribe(
            signal_type=SignalType.OPPORTUNITY,
            agent_id="agent_1",
            callback=callback
        )

        # Agent_1 emits signal
        signal_bus.emit_signal(
            signal_type=SignalType.OPPORTUNITY,
            source_agent_id="agent_1",
            payload={"reward": 10}
        )

        time.sleep(0.01)

        # Should NOT receive own signal
        assert len(received_signals) == 0

    def test_priority_filtering(self, signal_bus):
        """Test that signals are filtered by priority"""
        received_signals = []

        def callback(signal: Signal):
            received_signals.append(signal)

        # Subscribe with HIGH min priority
        signal_bus.subscribe(
            signal_type=SignalType.STATUS_UPDATE,
            agent_id="agent_2",
            callback=callback,
            min_priority=SignalPriority.HIGH
        )

        # Emit LOW priority signal (should be filtered)
        signal_bus.emit_signal(
            signal_type=SignalType.STATUS_UPDATE,
            source_agent_id="agent_1",
            payload={"status": "ok"},
            priority=SignalPriority.LOW
        )

        time.sleep(0.01)
        assert len(received_signals) == 0

        # Emit HIGH priority signal (should pass)
        signal_bus.emit_signal(
            signal_type=SignalType.STATUS_UPDATE,
            source_agent_id="agent_1",
            payload={"status": "critical"},
            priority=SignalPriority.HIGH
        )

        time.sleep(0.01)
        assert len(received_signals) == 1

    def test_unsubscribe(self, signal_bus):
        """Test unsubscribing from signals"""
        received_signals = []

        def callback(signal: Signal):
            received_signals.append(signal)

        # Subscribe
        signal_bus.subscribe(
            signal_type=SignalType.HEARTBEAT,
            agent_id="agent_2",
            callback=callback
        )

        # Unsubscribe
        success = signal_bus.unsubscribe(SignalType.HEARTBEAT, "agent_2")
        assert success

        # Emit signal
        signal_bus.emit_signal(
            signal_type=SignalType.HEARTBEAT,
            source_agent_id="agent_1",
            payload={}
        )

        time.sleep(0.01)

        # Should not receive after unsubscribe
        assert len(received_signals) == 0

    def test_rate_limiting(self, signal_bus):
        """Test that rate limiting prevents excessive signals"""
        # Create bus with low rate limit and small burst
        from src.core.electrical_signal import RateLimiter

        bus = ElectricalSignalBus(
            rate_limit_per_agent=10,
            enable_async=False,
            enable_monitoring=True
        )

        # Replace rate limiter with one with smaller burst
        bus.rate_limiter = RateLimiter(rate_per_second=10, burst_size=3)

        # Emit 3 signals (should all succeed due to burst size)
        for i in range(3):
            success = bus.emit_signal(
                signal_type=SignalType.HEARTBEAT,
                source_agent_id="agent_1",
                payload={}
            )
            assert success

        # 4th signal should be rate limited
        success = bus.emit_signal(
            signal_type=SignalType.HEARTBEAT,
            source_agent_id="agent_1",
            payload={}
        )
        assert not success

        # Check metrics
        assert bus.metrics.rate_limit_violations >= 1

    def test_signal_history(self, signal_bus):
        """Test retrieving signal history"""
        # Emit several signals
        for i in range(5):
            signal_bus.emit_signal(
                signal_type=SignalType.STATUS_UPDATE,
                source_agent_id=f"agent_{i}",
                payload={"seq": i}
            )

        # Get history
        history = signal_bus.get_signal_history(limit=10)

        assert len(history) == 5
        # Should be in reverse order (most recent first)
        assert history[0].payload["seq"] == 4

    def test_signal_history_filtering(self, signal_bus):
        """Test filtering signal history by type"""
        # Emit different signal types
        signal_bus.emit_signal(
            signal_type=SignalType.DANGER,
            source_agent_id="agent_1",
            payload={}
        )
        signal_bus.emit_signal(
            signal_type=SignalType.OPPORTUNITY,
            source_agent_id="agent_1",
            payload={}
        )

        # Get only DANGER signals
        history = signal_bus.get_signal_history(signal_type=SignalType.DANGER)

        assert len(history) == 1
        assert history[0].signal_type == SignalType.DANGER

    def test_get_metrics(self, signal_bus):
        """Test getting bus metrics"""
        signal_bus.emit_signal(
            signal_type=SignalType.CONVERGENCE,
            source_agent_id="agent_1",
            payload={}
        )

        metrics = signal_bus.get_metrics()

        assert metrics is not None
        assert metrics['total_emitted'] == 1
        assert SignalType.CONVERGENCE in metrics['signals_by_type']

    def test_get_subscriber_count(self, signal_bus):
        """Test getting subscriber counts"""
        # Initially no subscribers
        assert signal_bus.get_subscriber_count() == 0

        # Add subscriber
        signal_bus.subscribe(
            SignalType.DANGER,
            "agent_1",
            lambda s: None
        )

        assert signal_bus.get_subscriber_count() == 1
        assert signal_bus.get_subscriber_count(SignalType.DANGER) == 1
        assert signal_bus.get_subscriber_count(SignalType.OPPORTUNITY) == 0

    def test_shutdown(self, signal_bus):
        """Test bus shutdown"""
        signal_bus.emit_signal(
            SignalType.HEARTBEAT,
            "agent_1",
            {}
        )

        signal_bus.shutdown()

        # History should be cleared
        history = signal_bus.get_signal_history()
        assert len(history) == 0


class TestSignalTypes:
    """Test signal type utilities"""

    def test_get_signal_info(self):
        """Test retrieving signal type info"""
        info = get_signal_info(SignalType.DANGER)

        assert info is not None
        assert info.name == "DANGER"
        assert info.category == "Critical Alert"
        assert info.default_priority == "CRITICAL"

    def test_validate_signal_payload(self):
        """Test payload validation"""
        # Valid payload
        valid, error = validate_signal_payload(
            SignalType.DANGER,
            {
                'risk_level': 0.9,
                'risk_type': 'policy_divergence',
                'description': 'High risk detected',
                'recommended_action': 'Isolate agent'
            }
        )

        assert valid
        assert error == ""

        # Invalid payload (missing keys)
        valid, error = validate_signal_payload(
            SignalType.DANGER,
            {'risk_level': 0.9}
        )

        assert not valid
        assert "Missing required keys" in error

    def test_unknown_signal_type(self):
        """Test handling unknown signal types"""
        info = get_signal_info("UNKNOWN_SIGNAL")
        assert info is None

        valid, error = validate_signal_payload("UNKNOWN_SIGNAL", {})
        assert not valid
        assert "Unknown signal type" in error


class TestSignalBusPerformance:
    """Test signal bus performance characteristics"""

    def test_signal_latency(self):
        """Test that signal propagation meets sub-millisecond target"""
        bus = ElectricalSignalBus(
            enable_async=False,
            enable_monitoring=True
        )

        received_times = []

        def callback(signal: Signal):
            received_times.append(time.time())

        bus.subscribe(SignalType.DANGER, "agent_2", callback)

        # Emit signal and record time
        emit_time = time.time()
        bus.emit_signal(
            SignalType.DANGER,
            "agent_1",
            {"risk_level": 0.9},
            priority=SignalPriority.CRITICAL
        )

        time.sleep(0.001)  # Wait 1ms

        # Check latency
        if received_times:
            latency_ms = (received_times[0] - emit_time) * 1000
            # In synchronous mode, should be < 1ms
            assert latency_ms < 1.0, f"Latency {latency_ms}ms exceeds 1ms target"

    def test_high_throughput(self):
        """Test handling high signal throughput"""
        bus = ElectricalSignalBus(
            enable_async=False,
            rate_limit_per_agent=10000,
            enable_monitoring=True
        )

        start_time = time.time()

        # Emit 1000 signals
        for i in range(1000):
            bus.emit_signal(
                SignalType.HEARTBEAT,
                f"agent_{i % 10}",
                {"seq": i}
            )

        elapsed = time.time() - start_time

        # Should handle 1000 signals quickly (< 1 second)
        assert elapsed < 1.0, f"Throughput test took {elapsed}s (target: <1s)"

        # Check metrics
        metrics = bus.get_metrics()
        assert metrics['total_emitted'] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
