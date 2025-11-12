"""
Electrical Signaling Layer for MAE v3.0 (Big Rock 5)

Inspired by mycelial action potentials, this module provides ultra-fast
sub-millisecond signaling for critical events in the agent network.

Performance Targets:
- Latency: 0.1-1ms (vs 10-50ms for Redis)
- Throughput: 100,000+ signals/sec
- Overhead: <1% CPU in steady state

Author: MAE Development Team
Date: 2025-11-12
"""

import time
import threading
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Callable, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SignalPriority(Enum):
    """Priority levels for electrical signals"""
    CRITICAL = 0  # Emergency signals (DANGER, SYSTEM_FAILURE)
    HIGH = 1      # Important signals (OPPORTUNITY, CONVERGENCE)
    NORMAL = 2    # Standard signals (RESOURCE_AVAILABLE)
    LOW = 3       # Background signals (HEARTBEAT, STATUS_UPDATE)


@dataclass
class Signal:
    """
    Represents an electrical signal propagating through the mycelial network.

    Attributes:
        signal_type: Type of signal (DANGER, OPPORTUNITY, etc.)
        source_agent_id: ID of agent that emitted the signal
        timestamp: Unix timestamp when signal was created
        payload: Dictionary containing signal-specific data
        priority: Signal priority level
        ttl: Time-to-live in seconds (0 = infinite)
        signal_id: Unique identifier for this signal
    """
    signal_type: str
    source_agent_id: str
    timestamp: float
    payload: Dict[str, Any]
    priority: SignalPriority = SignalPriority.NORMAL
    ttl: float = 0.0
    signal_id: str = field(default_factory=lambda: f"sig_{time.time_ns()}")

    def is_expired(self) -> bool:
        """Check if signal has exceeded its TTL"""
        if self.ttl <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def age_ms(self) -> float:
        """Get signal age in milliseconds"""
        return (time.time() - self.timestamp) * 1000


class SignalMetrics:
    """Tracks performance metrics for the electrical signaling system"""

    def __init__(self):
        self.total_signals_emitted: int = 0
        self.total_signals_delivered: int = 0
        self.total_signals_dropped: int = 0
        self.signals_by_type: Dict[str, int] = defaultdict(int)
        self.signals_by_priority: Dict[SignalPriority, int] = defaultdict(int)
        self.propagation_times_ms: List[float] = []
        self.rate_limit_violations: int = 0
        self.lock = threading.Lock()

    def record_emission(self, signal: Signal):
        """Record signal emission"""
        with self.lock:
            self.total_signals_emitted += 1
            self.signals_by_type[signal.signal_type] += 1
            self.signals_by_priority[signal.priority] += 1

    def record_delivery(self, signal: Signal, delivery_time_ms: float):
        """Record successful signal delivery"""
        with self.lock:
            self.total_signals_delivered += 1
            self.propagation_times_ms.append(delivery_time_ms)
            # Keep only last 1000 measurements
            if len(self.propagation_times_ms) > 1000:
                self.propagation_times_ms.pop(0)

    def record_drop(self):
        """Record dropped signal"""
        with self.lock:
            self.total_signals_dropped += 1

    def record_rate_limit_violation(self):
        """Record rate limit violation"""
        with self.lock:
            self.rate_limit_violations += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics statistics"""
        with self.lock:
            avg_propagation = sum(self.propagation_times_ms) / len(self.propagation_times_ms) if self.propagation_times_ms else 0
            p99_propagation = sorted(self.propagation_times_ms)[int(len(self.propagation_times_ms) * 0.99)] if self.propagation_times_ms else 0

            return {
                'total_emitted': self.total_signals_emitted,
                'total_delivered': self.total_signals_delivered,
                'total_dropped': self.total_signals_dropped,
                'rate_limit_violations': self.rate_limit_violations,
                'avg_propagation_ms': round(avg_propagation, 3),
                'p99_propagation_ms': round(p99_propagation, 3),
                'delivery_rate': round(self.total_signals_delivered / self.total_signals_emitted * 100, 2) if self.total_signals_emitted > 0 else 0,
                'signals_by_type': dict(self.signals_by_type),
                'signals_by_priority': {p.name: count for p, count in self.signals_by_priority.items()}
            }


class RateLimiter:
    """Token bucket rate limiter for per-agent signal emission"""

    def __init__(self, rate_per_second: int = 1000, burst_size: int = 100):
        """
        Initialize rate limiter.

        Args:
            rate_per_second: Maximum sustained rate (tokens per second)
            burst_size: Maximum burst capacity
        """
        self.rate = rate_per_second
        self.burst_size = burst_size
        self.agent_buckets: Dict[str, Tuple[float, float]] = {}  # agent_id -> (tokens, last_update)
        self.lock = threading.Lock()

    def allow(self, agent_id: str) -> bool:
        """
        Check if agent is allowed to emit a signal.

        Args:
            agent_id: ID of agent attempting to emit

        Returns:
            True if allowed, False if rate limited
        """
        with self.lock:
            now = time.time()

            if agent_id not in self.agent_buckets:
                # New agent, give them a full bucket
                self.agent_buckets[agent_id] = (self.burst_size - 1, now)
                return True

            tokens, last_update = self.agent_buckets[agent_id]

            # Refill tokens based on elapsed time
            elapsed = now - last_update
            tokens = min(self.burst_size, tokens + elapsed * self.rate)

            if tokens >= 1.0:
                # Allow signal and consume token
                self.agent_buckets[agent_id] = (tokens - 1, now)
                return True
            else:
                # Rate limited
                self.agent_buckets[agent_id] = (tokens, now)
                return False


class ElectricalSignalBus:
    """
    Ultra-fast electrical signaling bus for the mycelial network.

    Provides sub-millisecond in-memory signal propagation with:
    - Priority-based signal processing
    - Rate limiting per agent
    - Async callback execution
    - Thread-safe operations
    - Performance monitoring

    This is inspired by mycelial action potentials which propagate
    at high speed for critical coordination events.
    """

    def __init__(
        self,
        max_history: int = 1000,
        enable_async: bool = True,
        rate_limit_per_agent: int = 1000,
        enable_monitoring: bool = True,
        max_workers: int = 10
    ):
        """
        Initialize the electrical signal bus.

        Args:
            max_history: Maximum number of signals to keep in history
            enable_async: Enable async callback execution
            rate_limit_per_agent: Max signals per second per agent
            enable_monitoring: Enable performance monitoring
            max_workers: Max thread pool workers for async execution
        """
        # Subscriber management
        self.subscribers: Dict[str, List[Tuple[str, Callable, SignalPriority]]] = defaultdict(list)
        # signal_type -> [(agent_id, callback, min_priority), ...]

        # Signal history
        self.signal_history: deque = deque(maxlen=max_history)

        # Thread safety
        self.lock = threading.RLock()

        # Async execution
        self.enable_async = enable_async
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if enable_async else None

        # Rate limiting
        self.rate_limiter = RateLimiter(rate_per_second=rate_limit_per_agent)

        # Monitoring
        self.enable_monitoring = enable_monitoring
        self.metrics = SignalMetrics() if enable_monitoring else None

        logger.info(f"ElectricalSignalBus initialized: async={enable_async}, rate_limit={rate_limit_per_agent}/sec")

    def subscribe(
        self,
        signal_type: str,
        agent_id: str,
        callback: Callable[[Signal], None],
        min_priority: SignalPriority = SignalPriority.LOW
    ) -> bool:
        """
        Subscribe to a specific signal type.

        Args:
            signal_type: Type of signal to subscribe to
            agent_id: ID of subscribing agent
            callback: Function to call when signal received
            min_priority: Only receive signals at or above this priority

        Returns:
            True if subscription successful
        """
        with self.lock:
            # Check if already subscribed
            for existing_agent_id, _, _ in self.subscribers[signal_type]:
                if existing_agent_id == agent_id:
                    logger.warning(f"Agent {agent_id} already subscribed to {signal_type}")
                    return False

            self.subscribers[signal_type].append((agent_id, callback, min_priority))
            logger.debug(f"Agent {agent_id} subscribed to {signal_type} (min_priority={min_priority.name})")
            return True

    def unsubscribe(self, signal_type: str, agent_id: str) -> bool:
        """
        Unsubscribe from a signal type.

        Args:
            signal_type: Type of signal to unsubscribe from
            agent_id: ID of unsubscribing agent

        Returns:
            True if unsubscription successful
        """
        with self.lock:
            if signal_type not in self.subscribers:
                return False

            original_count = len(self.subscribers[signal_type])
            self.subscribers[signal_type] = [
                (aid, cb, pri) for aid, cb, pri in self.subscribers[signal_type]
                if aid != agent_id
            ]

            success = len(self.subscribers[signal_type]) < original_count
            if success:
                logger.debug(f"Agent {agent_id} unsubscribed from {signal_type}")
            return success

    def emit_signal(
        self,
        signal_type: str,
        source_agent_id: str,
        payload: Dict[str, Any],
        priority: SignalPriority = SignalPriority.NORMAL,
        ttl: float = 0.0
    ) -> bool:
        """
        Emit an electrical signal to all subscribers.

        This method has sub-millisecond latency and propagates
        the signal instantly to all subscribed agents.

        Args:
            signal_type: Type of signal to emit
            source_agent_id: ID of agent emitting signal
            payload: Signal data
            priority: Signal priority level
            ttl: Time-to-live in seconds (0 = infinite)

        Returns:
            True if signal was emitted, False if rate limited
        """
        # Rate limiting check
        if not self.rate_limiter.allow(source_agent_id):
            if self.enable_monitoring:
                self.metrics.record_rate_limit_violation()
            logger.warning(f"Rate limit exceeded for agent {source_agent_id}")
            return False

        # Create signal
        signal = Signal(
            signal_type=signal_type,
            source_agent_id=source_agent_id,
            timestamp=time.time(),
            payload=payload,
            priority=priority,
            ttl=ttl
        )

        # Record emission
        if self.enable_monitoring:
            self.metrics.record_emission(signal)

        # Add to history
        with self.lock:
            self.signal_history.append(signal)

        # Propagate to subscribers
        self._propagate_signal(signal)

        logger.debug(f"Signal emitted: {signal_type} from {source_agent_id} (priority={priority.name})")
        return True

    def _propagate_signal(self, signal: Signal):
        """
        Propagate signal to all eligible subscribers.

        Args:
            signal: Signal to propagate
        """
        # Check if expired
        if signal.is_expired():
            if self.enable_monitoring:
                self.metrics.record_drop()
            return

        # Get subscribers for this signal type
        with self.lock:
            subscribers = self.subscribers.get(signal.signal_type, []).copy()

        # Deliver to each subscriber
        for agent_id, callback, min_priority in subscribers:
            # Skip if priority too low
            if signal.priority.value > min_priority.value:
                continue

            # Skip if signal is from this agent (no self-signaling)
            if agent_id == signal.source_agent_id:
                continue

            # Execute callback
            if self.enable_async:
                self.executor.submit(self._safe_callback, signal, callback, agent_id)
            else:
                self._safe_callback(signal, callback, agent_id)

    def _safe_callback(self, signal: Signal, callback: Callable, agent_id: str):
        """
        Execute callback with error handling and monitoring.

        Args:
            signal: Signal to deliver
            callback: Callback function
            agent_id: ID of receiving agent
        """
        start_time = time.time()
        try:
            callback(signal)

            # Record delivery
            if self.enable_monitoring:
                delivery_time_ms = (time.time() - start_time) * 1000
                self.metrics.record_delivery(signal, delivery_time_ms)

        except Exception as e:
            logger.error(f"Error in signal callback for agent {agent_id}: {e}")
            if self.enable_monitoring:
                self.metrics.record_drop()

    def get_signal_history(
        self,
        signal_type: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        limit: int = 100
    ) -> List[Signal]:
        """
        Get recent signal history.

        Args:
            signal_type: Filter by signal type (None = all types)
            max_age_seconds: Only return signals newer than this
            limit: Maximum number of signals to return

        Returns:
            List of signals matching criteria
        """
        with self.lock:
            signals = list(self.signal_history)

        # Apply filters
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]

        if max_age_seconds:
            cutoff = time.time() - max_age_seconds
            signals = [s for s in signals if s.timestamp >= cutoff]

        # Return most recent first
        return list(reversed(signals))[:limit]

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get current performance metrics.

        Returns:
            Dictionary of metrics or None if monitoring disabled
        """
        if not self.enable_monitoring:
            return None
        return self.metrics.get_stats()

    def get_subscriber_count(self, signal_type: Optional[str] = None) -> int:
        """
        Get number of subscribers.

        Args:
            signal_type: Count for specific type, or None for total

        Returns:
            Subscriber count
        """
        with self.lock:
            if signal_type:
                return len(self.subscribers.get(signal_type, []))
            else:
                return sum(len(subs) for subs in self.subscribers.values())

    def shutdown(self):
        """Shutdown the signal bus and cleanup resources"""
        logger.info("Shutting down ElectricalSignalBus...")

        if self.executor:
            self.executor.shutdown(wait=True)

        with self.lock:
            self.subscribers.clear()
            self.signal_history.clear()

        logger.info("ElectricalSignalBus shutdown complete")
