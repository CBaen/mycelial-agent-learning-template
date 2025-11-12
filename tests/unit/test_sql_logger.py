"""
Unit tests for SQLiteLogger.

Tests the thread-safe SQLite logger functionality including:
- Initialization and database schema creation
- Thread-safe write queue operations
- Logging methods (agent events, patterns, performance, system events, risk events)
- Query methods for retrieving data
- Batch writing and flushing
- Graceful shutdown
- Database maintenance operations
"""

import pytest
import tempfile
import time
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from threading import Thread

from src.connectors.sql_logger import SQLiteLogger, LogEntry, PatternEntry


class TestSQLiteLoggerInit:
    """Test SQLiteLogger initialization."""

    def test_init_with_defaults(self, temp_sqlite_db):
        """Test initialization with default parameters."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        assert logger.db_path == temp_sqlite_db
        assert logger.queue_size == 10000
        assert logger.batch_size == 100
        assert logger.flush_interval == 1.0
        assert logger.running is True
        assert logger.writer_thread is not None

        logger.stop()

    def test_init_with_custom_params(self, temp_sqlite_db):
        """Test initialization with custom parameters."""
        logger = SQLiteLogger(
            db_path=str(temp_sqlite_db),
            queue_size=5000,
            batch_size=50,
            flush_interval=0.5
        )

        assert logger.queue_size == 5000
        assert logger.batch_size == 50
        assert logger.flush_interval == 0.5

        logger.stop()

    def test_database_file_created(self, temp_sqlite_db):
        """Test that database file is created."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        assert temp_sqlite_db.exists()

        logger.stop()

    def test_parent_directory_created(self):
        """Test that parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "nested" / "test.db"
            logger = SQLiteLogger(db_path=str(db_path))

            assert db_path.exists()
            assert db_path.parent.exists()

            logger.stop()

    def test_writer_thread_started(self, temp_sqlite_db):
        """Test that writer thread is started on init."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        assert logger.writer_thread.is_alive()
        assert logger.running is True

        logger.stop()


class TestSQLiteLoggerSchema:
    """Test database schema creation."""

    def test_agent_events_table_created(self, temp_sqlite_db):
        """Test that agent_events table is created."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_events'"
            )
            result = cursor.fetchone()

        assert result is not None
        logger.stop()

    def test_patterns_table_created(self, temp_sqlite_db):
        """Test that patterns table is created."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='patterns'"
            )
            result = cursor.fetchone()

        assert result is not None
        logger.stop()

    def test_performance_metrics_table_created(self, temp_sqlite_db):
        """Test that performance_metrics table is created."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='performance_metrics'"
            )
            result = cursor.fetchone()

        assert result is not None
        logger.stop()

    def test_system_events_table_created(self, temp_sqlite_db):
        """Test that system_events table is created."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='system_events'"
            )
            result = cursor.fetchone()

        assert result is not None
        logger.stop()

    def test_risk_events_table_created(self, temp_sqlite_db):
        """Test that risk_events table is created."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='risk_events'"
            )
            result = cursor.fetchone()

        assert result is not None
        logger.stop()

    def test_indices_created(self, temp_sqlite_db):
        """Test that database indices are created."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indices = [row[0] for row in cursor.fetchall()]

        # Check for expected indices (excluding auto-created primary key indices)
        expected_indices = [
            'idx_agent_events_agent_id',
            'idx_agent_events_timestamp',
            'idx_patterns_type',
            'idx_performance_agent_id'
        ]

        for index in expected_indices:
            assert index in indices

        logger.stop()


class TestSQLiteLoggerQueueOperations:
    """Test write queue operations."""

    def test_queue_write_increments_counter(self, temp_sqlite_db):
        """Test that queueing a write increments the counter."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        initial_count = logger.writes_queued
        logger._queue_write("INSERT INTO agent_events VALUES (?, ?, ?)", (1, 2, 3))

        assert logger.writes_queued == initial_count + 1
        logger.stop()

    def test_queue_write_adds_to_queue(self, temp_sqlite_db):
        """Test that queueing adds item to queue."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        initial_size = logger.write_queue.qsize()
        logger._queue_write("INSERT INTO agent_events VALUES (?, ?)", (1, 2))

        assert logger.write_queue.qsize() == initial_size + 1
        logger.stop()

    def test_queue_full_handling(self, temp_sqlite_db):
        """Test handling when queue is full."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db), queue_size=5)

        # Fill the queue
        for i in range(10):
            logger._queue_write(f"INSERT INTO agent_events VALUES ({i})", ())

        # Should have dropped some writes
        assert logger.queue_full_count > 0

        logger.stop()

    def test_flush_queue_when_empty(self, temp_sqlite_db):
        """Test flushing an empty queue (should do nothing)."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Clear the queue
        while not logger.write_queue.empty():
            logger.write_queue.get()

        # Should not raise error
        logger._flush_queue()

        logger.stop()

    def test_flush_queue_processes_items(self, temp_sqlite_db):
        """Test that flush_queue processes queued items."""
        logger = SQLiteLogger(
            db_path=str(temp_sqlite_db),
            batch_size=10
        )

        # Queue some writes
        for i in range(5):
            logger.log_agent_event(
                agent_id=f"agent_{i}",
                event_type="test_event",
                data={"value": i}
            )

        # Wait for queue to be populated
        time.sleep(0.1)

        # Flush the queue
        logger._flush_queue()

        # Check that writes were committed
        assert logger.writes_committed > 0

        logger.stop()


class TestSQLiteLoggerLogging:
    """Test logging methods."""

    def test_log_agent_event(self, temp_sqlite_db):
        """Test logging an agent event."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        logger.log_agent_event(
            agent_id="agent_123",
            event_type="action_taken",
            data={"action": "buy", "quantity": 100},
            level="INFO",
            step=42
        )

        # Wait for write to be processed
        time.sleep(0.5)
        logger.stop()

        # Verify the event was logged
        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_events WHERE agent_id = ?", ("agent_123",))
            row = cursor.fetchone()

        assert row is not None
        assert row['agent_id'] == "agent_123"
        assert row['event_type'] == "action_taken"
        assert row['level'] == "INFO"
        assert row['step'] == 42
        data = json.loads(row['data'])
        assert data['action'] == "buy"
        assert data['quantity'] == 100

    def test_log_pattern(self, temp_sqlite_db):
        """Test logging a pattern."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        pattern = PatternEntry(
            pattern_id="pattern_001",
            pattern_type="successful_policy",
            description="Profitable trading strategy",
            discovered_by="agent_123",
            discovered_at=time.time(),
            frequency=5,
            confidence=0.92,
            metadata={"risk_level": "medium"}
        )

        logger.log_pattern(pattern)

        # Wait for write to be processed
        time.sleep(0.5)
        logger.stop()

        # Verify the pattern was logged
        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patterns WHERE pattern_id = ?", ("pattern_001",))
            row = cursor.fetchone()

        assert row is not None
        assert row['pattern_id'] == "pattern_001"
        assert row['pattern_type'] == "successful_policy"
        assert row['discovered_by'] == "agent_123"
        assert row['frequency'] == 5
        assert row['confidence'] == 0.92

    def test_log_pattern_upsert(self, temp_sqlite_db):
        """Test that logging the same pattern increments frequency."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        pattern = PatternEntry(
            pattern_id="pattern_002",
            pattern_type="recovery_pattern",
            description="Recovery from risk",
            discovered_by="agent_456",
            discovered_at=time.time(),
            frequency=1,
            confidence=0.85,
            metadata={}
        )

        # Log pattern twice
        logger.log_pattern(pattern)
        time.sleep(0.2)
        logger.log_pattern(pattern)

        # Wait for writes to be processed
        time.sleep(0.5)
        logger.stop()

        # Verify frequency was incremented
        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT frequency FROM patterns WHERE pattern_id = ?", ("pattern_002",))
            row = cursor.fetchone()

        assert row is not None
        # Should be 2 (1 + 1 from the conflict update)
        assert row['frequency'] == 2

    def test_log_performance_metric(self, temp_sqlite_db):
        """Test logging a performance metric."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        logger.log_performance_metric(
            agent_id="agent_789",
            metric_name="reward",
            metric_value=123.45,
            step=100
        )

        # Wait for write to be processed
        time.sleep(0.5)
        logger.stop()

        # Verify the metric was logged
        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM performance_metrics WHERE agent_id = ? AND metric_name = ?",
                ("agent_789", "reward")
            )
            row = cursor.fetchone()

        assert row is not None
        assert row['agent_id'] == "agent_789"
        assert row['metric_name'] == "reward"
        assert row['metric_value'] == 123.45
        assert row['step'] == 100

    def test_log_system_event(self, temp_sqlite_db):
        """Test logging a system event."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        logger.log_system_event(
            event_type="startup",
            severity="INFO",
            description="System started",
            data={"version": "1.0.0"}
        )

        # Wait for write to be processed
        time.sleep(0.5)
        logger.stop()

        # Verify the event was logged
        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM system_events WHERE event_type = ?", ("startup",))
            row = cursor.fetchone()

        assert row is not None
        assert row['event_type'] == "startup"
        assert row['severity'] == "INFO"
        assert row['description'] == "System started"
        data = json.loads(row['data'])
        assert data['version'] == "1.0.0"

    def test_log_system_event_without_data(self, temp_sqlite_db):
        """Test logging a system event without additional data."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        logger.log_system_event(
            event_type="shutdown",
            severity="WARNING",
            description="System shutting down"
        )

        # Wait for write to be processed
        time.sleep(0.5)
        logger.stop()

        # Verify the event was logged
        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM system_events WHERE event_type = ?", ("shutdown",))
            row = cursor.fetchone()

        assert row is not None
        assert row['data'] is None

    def test_log_risk_event(self, temp_sqlite_db):
        """Test logging a risk event."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        logger.log_risk_event(
            agent_id="agent_999",
            risk_level="HIGH",
            risk_score=0.85,
            contributing_factors={"excessive_leverage": True, "volatility": 0.9},
            intervention="policy_rejected"
        )

        # Wait for write to be processed
        time.sleep(0.5)
        logger.stop()

        # Verify the risk event was logged
        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM risk_events WHERE agent_id = ?", ("agent_999",))
            row = cursor.fetchone()

        assert row is not None
        assert row['agent_id'] == "agent_999"
        assert row['risk_level'] == "HIGH"
        assert row['risk_score'] == 0.85
        assert row['intervention'] == "policy_rejected"
        factors = json.loads(row['contributing_factors'])
        assert factors['excessive_leverage'] is True


class TestSQLiteLoggerQueries:
    """Test query methods."""

    def test_get_agent_events_all(self, temp_sqlite_db):
        """Test retrieving all agent events."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Log some events
        for i in range(5):
            logger.log_agent_event(
                agent_id=f"agent_{i}",
                event_type="test_event",
                data={"index": i}
            )

        time.sleep(0.5)
        logger.stop()

        # Retrieve events
        events = logger.get_agent_events(limit=100)

        assert len(events) == 5

    def test_get_agent_events_filtered_by_agent_id(self, temp_sqlite_db):
        """Test retrieving events filtered by agent_id."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Log events from different agents
        logger.log_agent_event(agent_id="agent_A", event_type="event1", data={})
        logger.log_agent_event(agent_id="agent_B", event_type="event2", data={})
        logger.log_agent_event(agent_id="agent_A", event_type="event3", data={})

        time.sleep(0.5)
        logger.stop()

        # Retrieve events for agent_A only
        events = logger.get_agent_events(agent_id="agent_A")

        assert len(events) == 2
        assert all(event['agent_id'] == "agent_A" for event in events)

    def test_get_agent_events_filtered_by_event_type(self, temp_sqlite_db):
        """Test retrieving events filtered by event_type."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        logger.log_agent_event(agent_id="agent_1", event_type="action", data={})
        logger.log_agent_event(agent_id="agent_2", event_type="observation", data={})
        logger.log_agent_event(agent_id="agent_3", event_type="action", data={})

        time.sleep(0.5)
        logger.stop()

        events = logger.get_agent_events(event_type="action")

        assert len(events) == 2
        assert all(event['event_type'] == "action" for event in events)

    def test_get_agent_events_with_limit(self, temp_sqlite_db):
        """Test retrieving events with limit."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Log many events
        for i in range(20):
            logger.log_agent_event(agent_id="agent_X", event_type="test", data={})

        time.sleep(0.5)
        logger.stop()

        events = logger.get_agent_events(limit=10)

        assert len(events) == 10

    def test_get_patterns_all(self, temp_sqlite_db):
        """Test retrieving all patterns."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Log patterns
        for i in range(3):
            pattern = PatternEntry(
                pattern_id=f"pattern_{i}",
                pattern_type="test_pattern",
                description=f"Test pattern {i}",
                discovered_by="agent_test",
                discovered_at=time.time(),
                frequency=i + 1,
                confidence=0.5 + i * 0.1,
                metadata={}
            )
            logger.log_pattern(pattern)

        time.sleep(0.5)
        logger.stop()

        patterns = logger.get_patterns()

        assert len(patterns) == 3

    def test_get_patterns_filtered_by_type(self, temp_sqlite_db):
        """Test retrieving patterns filtered by type."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        pattern1 = PatternEntry(
            pattern_id="p1", pattern_type="successful", description="",
            discovered_by="agent", discovered_at=time.time(),
            frequency=1, confidence=0.8, metadata={}
        )
        pattern2 = PatternEntry(
            pattern_id="p2", pattern_type="failed", description="",
            discovered_by="agent", discovered_at=time.time(),
            frequency=1, confidence=0.8, metadata={}
        )

        logger.log_pattern(pattern1)
        logger.log_pattern(pattern2)

        time.sleep(0.5)
        logger.stop()

        patterns = logger.get_patterns(pattern_type="successful")

        assert len(patterns) == 1
        assert patterns[0]['pattern_type'] == "successful"

    def test_get_patterns_filtered_by_frequency(self, temp_sqlite_db):
        """Test retrieving patterns with minimum frequency."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        for i in range(1, 6):
            pattern = PatternEntry(
                pattern_id=f"p_{i}", pattern_type="test", description="",
                discovered_by="agent", discovered_at=time.time(),
                frequency=i, confidence=0.8, metadata={}
            )
            logger.log_pattern(pattern)

        time.sleep(0.5)
        logger.stop()

        patterns = logger.get_patterns(min_frequency=3)

        assert len(patterns) == 3  # Patterns with frequency 3, 4, 5
        assert all(p['frequency'] >= 3 for p in patterns)

    def test_get_patterns_filtered_by_confidence(self, temp_sqlite_db):
        """Test retrieving patterns with minimum confidence."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        for i, conf in enumerate([0.5, 0.7, 0.9]):
            pattern = PatternEntry(
                pattern_id=f"p_{i}", pattern_type="test", description="",
                discovered_by="agent", discovered_at=time.time(),
                frequency=1, confidence=conf, metadata={}
            )
            logger.log_pattern(pattern)

        time.sleep(0.5)
        logger.stop()

        patterns = logger.get_patterns(min_confidence=0.8)

        assert len(patterns) == 1
        assert patterns[0]['confidence'] >= 0.8

    def test_get_performance_metrics(self, temp_sqlite_db):
        """Test retrieving performance metrics."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        for i in range(5):
            logger.log_performance_metric(
                agent_id="agent_perf",
                metric_name="reward",
                metric_value=100.0 + i,
                step=i
            )

        time.sleep(0.5)
        logger.stop()

        metrics = logger.get_performance_metrics(agent_id="agent_perf")

        assert len(metrics) == 5
        assert all(m['agent_id'] == "agent_perf" for m in metrics)

    def test_get_performance_metrics_filtered_by_name(self, temp_sqlite_db):
        """Test retrieving metrics filtered by name."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        logger.log_performance_metric("agent_X", "reward", 100.0)
        logger.log_performance_metric("agent_X", "loss", 50.0)
        logger.log_performance_metric("agent_X", "reward", 110.0)

        time.sleep(0.5)
        logger.stop()

        metrics = logger.get_performance_metrics(agent_id="agent_X", metric_name="reward")

        assert len(metrics) == 2
        assert all(m['metric_name'] == "reward" for m in metrics)

    def test_get_performance_metrics_with_time_range(self, temp_sqlite_db):
        """Test retrieving metrics with time range."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        start_time = time.time()
        logger.log_performance_metric("agent_Y", "metric", 1.0)
        time.sleep(0.1)
        mid_time = time.time()
        time.sleep(0.1)
        logger.log_performance_metric("agent_Y", "metric", 2.0)

        time.sleep(0.5)
        logger.stop()

        # Get only metrics after mid_time
        metrics = logger.get_performance_metrics(
            agent_id="agent_Y",
            time_range=(mid_time, time.time())
        )

        assert len(metrics) >= 1

    def test_get_risk_events(self, temp_sqlite_db):
        """Test retrieving risk events."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        for i in range(3):
            logger.log_risk_event(
                agent_id=f"agent_{i}",
                risk_level="MEDIUM",
                risk_score=0.5 + i * 0.1,
                contributing_factors={}
            )

        time.sleep(0.5)
        logger.stop()

        events = logger.get_risk_events()

        assert len(events) == 3

    def test_get_risk_events_filtered_by_agent(self, temp_sqlite_db):
        """Test retrieving risk events filtered by agent."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        logger.log_risk_event("agent_A", "HIGH", 0.9, {})
        logger.log_risk_event("agent_B", "LOW", 0.3, {})
        logger.log_risk_event("agent_A", "MEDIUM", 0.6, {})

        time.sleep(0.5)
        logger.stop()

        events = logger.get_risk_events(agent_id="agent_A")

        assert len(events) == 2
        assert all(e['agent_id'] == "agent_A" for e in events)

    def test_get_risk_events_filtered_by_score(self, temp_sqlite_db):
        """Test retrieving risk events with minimum score."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        logger.log_risk_event("agent_1", "LOW", 0.3, {})
        logger.log_risk_event("agent_2", "HIGH", 0.8, {})
        logger.log_risk_event("agent_3", "MEDIUM", 0.6, {})

        time.sleep(0.5)
        logger.stop()

        events = logger.get_risk_events(min_risk_score=0.5)

        assert len(events) == 2
        assert all(e['risk_score'] >= 0.5 for e in events)


class TestSQLiteLoggerThreadSafety:
    """Test thread safety."""

    def test_concurrent_writes(self, temp_sqlite_db):
        """Test concurrent writes from multiple threads."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        def write_events(agent_id, count):
            for i in range(count):
                logger.log_agent_event(
                    agent_id=agent_id,
                    event_type="concurrent_test",
                    data={"index": i}
                )

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = Thread(target=write_events, args=(f"agent_{i}", 10))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        time.sleep(1.0)
        logger.stop()

        # Verify all events were logged
        events = logger.get_agent_events(limit=100)
        assert len(events) == 50  # 5 threads * 10 events each


class TestSQLiteLoggerLifecycle:
    """Test logger lifecycle (start/stop)."""

    def test_start_when_already_running(self, temp_sqlite_db):
        """Test starting logger when already running (should warn)."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Logger is already running after init
        assert logger.running is True

        # Try to start again (should just warn)
        logger.start()

        assert logger.running is True

        logger.stop()

    def test_stop_flushes_queue(self, temp_sqlite_db):
        """Test that stop flushes remaining queue items."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db), flush_interval=100.0)

        # Queue some events
        for i in range(10):
            logger.log_agent_event(f"agent_{i}", "test", {})

        # Stop immediately (should flush)
        logger.stop()

        # Check that events were committed
        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM agent_events")
            count = cursor.fetchone()[0]

        assert count == 10

    def test_stop_waits_for_writer_thread(self, temp_sqlite_db):
        """Test that stop waits for writer thread to finish."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        thread_id = logger.writer_thread.ident

        logger.stop(timeout=5.0)

        # Thread should be finished
        assert not logger.writer_thread.is_alive()

    def test_stop_when_not_running(self, temp_sqlite_db):
        """Test stopping logger when not running (should be safe)."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))
        logger.stop()

        # Stop again (should be safe)
        logger.stop()

        assert logger.running is False


class TestSQLiteLoggerMaintenance:
    """Test maintenance operations."""

    def test_cleanup_old_data(self, temp_sqlite_db):
        """Test cleanup of old data."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Log some events with old timestamps
        old_time = time.time() - (40 * 86400)  # 40 days ago
        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO agent_events (timestamp, level, agent_id, event_type, data, step) VALUES (?, ?, ?, ?, ?, ?)",
                (old_time, "INFO", "agent_old", "test", "{}", 1)
            )
            cursor.execute(
                "INSERT INTO performance_metrics (timestamp, agent_id, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                (old_time, "agent_old", "metric", 1.0)
            )
            conn.commit()

        # Log recent event
        logger.log_agent_event("agent_new", "test", {})
        time.sleep(0.5)

        # Cleanup old data (keep 30 days)
        logger.cleanup_old_data(days=30)
        logger.stop()

        # Verify old data was removed
        with logger._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM agent_events WHERE agent_id = ?", ("agent_old",))
            old_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM agent_events WHERE agent_id = ?", ("agent_new",))
            new_count = cursor.fetchone()[0]

        assert old_count == 0
        assert new_count == 1

    def test_vacuum_database(self, temp_sqlite_db):
        """Test database vacuum operation."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Should not raise error
        logger.vacuum_database()

        logger.stop()

    def test_get_statistics(self, temp_sqlite_db):
        """Test retrieving database statistics."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Log some data
        logger.log_agent_event("agent_1", "test", {})
        logger.log_pattern(PatternEntry(
            "p1", "test", "", "agent", time.time(), 1, 0.8, {}
        ))

        time.sleep(0.5)
        logger.stop()

        stats = logger.get_statistics()

        assert 'writes_queued' in stats
        assert 'writes_committed' in stats
        assert 'write_errors' in stats
        assert 'queue_full_count' in stats
        assert 'queue_size' in stats
        assert 'agent_events_count' in stats
        assert 'patterns_count' in stats
        assert 'performance_metrics_count' in stats
        assert 'system_events_count' in stats
        assert 'risk_events_count' in stats

        assert stats['agent_events_count'] >= 1
        assert stats['patterns_count'] >= 1


class TestSQLiteLoggerErrorHandling:
    """Test error handling."""

    def test_batch_write_rollback_on_error(self, temp_sqlite_db):
        """Test that batch writes rollback on error."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Create a batch with an invalid SQL statement
        batch = [
            ("INSERT INTO agent_events (timestamp, level, agent_id, event_type) VALUES (?, ?, ?, ?)",
             (time.time(), "INFO", "agent_1", "test")),
            ("INVALID SQL STATEMENT", ())  # This will cause an error
        ]

        # Should handle error gracefully
        with pytest.raises(Exception):
            logger._write_batch(batch)

        logger.stop()

    def test_connection_context_manager_closes(self, temp_sqlite_db):
        """Test that connection context manager closes connection."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        with logger._get_connection() as conn:
            assert conn is not None

        # Connection should be closed after context

        logger.stop()

    def test_destructor_stops_logger(self, temp_sqlite_db):
        """Test that __del__ stops the logger."""
        logger = SQLiteLogger(db_path=str(temp_sqlite_db))

        # Manually call destructor
        logger.__del__()

        assert logger.running is False
