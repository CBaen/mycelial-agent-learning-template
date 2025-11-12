"""
Thread-Safe SQLite Logger with Write Queue

This module provides persistent logging and pattern archiving for the MAE system.
It implements a thread-safe write queue to handle concurrent writes from multiple
agents without blocking or causing database locks.

Based on the 8-week plan for pattern recognition and collective memory.
"""

import sqlite3
import threading
import queue
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Container for a log entry."""
    timestamp: float
    level: str
    agent_id: str
    event_type: str
    data: Dict[str, Any]
    step: int


@dataclass
class PatternEntry:
    """Container for a discovered pattern."""
    pattern_id: str
    pattern_type: str
    description: str
    discovered_by: str
    discovered_at: float
    frequency: int
    confidence: float
    metadata: Dict[str, Any]


class SQLiteLogger:
    """
    Thread-safe SQLite logger with write queue for MAE system.

    Features:
    - Non-blocking writes via queue
    - Automatic schema creation and migration
    - Pattern archiving and retrieval
    - Agent event logging
    - Performance metrics tracking
    - Database cleanup and maintenance
    """

    def __init__(
        self,
        db_path: str = "data/mae_system.db",
        queue_size: int = 10000,
        batch_size: int = 100,
        flush_interval: float = 1.0
    ):
        """
        Initialize the SQLite Logger.

        Args:
            db_path: Path to SQLite database file
            queue_size: Maximum size of write queue
            batch_size: Number of writes to batch together
            flush_interval: Seconds between queue flushes
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.queue_size = queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Thread-safe write queue
        self.write_queue: queue.Queue = queue.Queue(maxsize=queue_size)

        # Writer thread
        self.writer_thread: Optional[threading.Thread] = None
        self.running = False
        self.shutdown_event = threading.Event()

        # Performance metrics
        self.writes_queued = 0
        self.writes_committed = 0
        self.write_errors = 0
        self.queue_full_count = 0

        # Initialize database
        self._initialize_database()

        # Start writer thread
        self.start()

        logger.info("SQLiteLogger initialized at %s", self.db_path)

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Agent events log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    level TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data TEXT,
                    step INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Patterns archive table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE NOT NULL,
                    pattern_type TEXT NOT NULL,
                    description TEXT,
                    discovered_by TEXT NOT NULL,
                    discovered_at REAL NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    agent_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    step INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # System events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Risk events table (for HAVEN)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    agent_id TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    contributing_factors TEXT,
                    intervention TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_events_agent_id
                ON agent_events(agent_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_events_timestamp
                ON agent_events(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_type
                ON patterns(pattern_type)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_agent_id
                ON performance_metrics(agent_id)
            """)

            conn.commit()

        logger.info("Database schema initialized")

    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.

        Yields:
            SQLite connection
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def start(self):
        """Start the writer thread."""
        if self.running:
            logger.warning("Logger already running")
            return

        self.running = True
        self.shutdown_event.clear()
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="SQLiteWriterThread",
            daemon=True
        )
        self.writer_thread.start()
        logger.info("Writer thread started")

    def stop(self, timeout: float = 10.0):
        """
        Stop the writer thread and flush remaining entries.

        Args:
            timeout: Maximum seconds to wait for shutdown
        """
        if not self.running:
            return

        logger.info("Stopping SQLite logger...")
        self.shutdown_event.set()
        self.running = False

        # Wait for writer thread to finish
        if self.writer_thread:
            self.writer_thread.join(timeout=timeout)

        # Flush any remaining entries
        self._flush_queue()

        logger.info("SQLite logger stopped (queued: %d, committed: %d, errors: %d)",
                   self.writes_queued, self.writes_committed, self.write_errors)

    def _writer_loop(self):
        """Main loop for the writer thread."""
        logger.info("Writer thread started")

        while self.running or not self.write_queue.empty():
            try:
                # Flush queue periodically or when enough items accumulated
                if (not self.write_queue.empty() and
                    (self.write_queue.qsize() >= self.batch_size or
                     self.shutdown_event.is_set())):
                    self._flush_queue()

                # Sleep to avoid busy-waiting
                time.sleep(self.flush_interval)

            except Exception as e:
                logger.error("Error in writer loop: %s", e, exc_info=True)
                time.sleep(1)

        logger.info("Writer thread stopped")

    def _flush_queue(self):
        """Flush the write queue to database."""
        if self.write_queue.empty():
            return

        batch = []
        try:
            # Collect batch of items
            while len(batch) < self.batch_size:
                try:
                    item = self.write_queue.get_nowait()
                    batch.append(item)
                except queue.Empty:
                    break

            if not batch:
                return

            # Write batch to database
            self._write_batch(batch)
            self.writes_committed += len(batch)

        except Exception as e:
            logger.error("Error flushing queue: %s", e, exc_info=True)
            self.write_errors += len(batch)

    def _write_batch(self, batch: List[Tuple[str, tuple]]):
        """
        Write a batch of items to database.

        Args:
            batch: List of (sql, params) tuples
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                for sql, params in batch:
                    cursor.execute(sql, params)
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error("Error writing batch: %s", e)
                raise

    def _queue_write(self, sql: str, params: tuple):
        """
        Queue a write operation.

        Args:
            sql: SQL statement
            params: SQL parameters
        """
        try:
            self.write_queue.put_nowait((sql, params))
            self.writes_queued += 1
        except queue.Full:
            self.queue_full_count += 1
            logger.warning("Write queue full, dropping write (dropped: %d)", self.queue_full_count)

    # ==========================================
    # Public API - Logging Methods
    # ==========================================

    def log_agent_event(
        self,
        agent_id: str,
        event_type: str,
        data: Dict[str, Any],
        level: str = "INFO",
        step: Optional[int] = None
    ):
        """
        Log an agent event.

        Args:
            agent_id: ID of the agent
            event_type: Type of event
            data: Event data
            level: Log level
            step: Current simulation step
        """
        sql = """
            INSERT INTO agent_events (timestamp, level, agent_id, event_type, data, step)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            time.time(),
            level,
            agent_id,
            event_type,
            json.dumps(data),
            step
        )
        self._queue_write(sql, params)

    def log_pattern(self, pattern: PatternEntry):
        """
        Log a discovered pattern.

        Args:
            pattern: Pattern entry to log
        """
        sql = """
            INSERT INTO patterns
            (pattern_id, pattern_type, description, discovered_by, discovered_at,
             frequency, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pattern_id) DO UPDATE SET
                frequency = frequency + 1,
                updated_at = CURRENT_TIMESTAMP
        """
        params = (
            pattern.pattern_id,
            pattern.pattern_type,
            pattern.description,
            pattern.discovered_by,
            pattern.discovered_at,
            pattern.frequency,
            pattern.confidence,
            json.dumps(pattern.metadata)
        )
        self._queue_write(sql, params)

    def log_performance_metric(
        self,
        agent_id: str,
        metric_name: str,
        metric_value: float,
        step: Optional[int] = None
    ):
        """
        Log a performance metric.

        Args:
            agent_id: ID of the agent
            metric_name: Name of the metric
            metric_value: Metric value
            step: Current simulation step
        """
        sql = """
            INSERT INTO performance_metrics (timestamp, agent_id, metric_name, metric_value, step)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (
            time.time(),
            agent_id,
            metric_name,
            metric_value,
            step
        )
        self._queue_write(sql, params)

    def log_system_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Log a system event.

        Args:
            event_type: Type of event
            severity: Severity level
            description: Event description
            data: Additional event data
        """
        sql = """
            INSERT INTO system_events (timestamp, event_type, severity, description, data)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (
            time.time(),
            event_type,
            severity,
            description,
            json.dumps(data) if data else None
        )
        self._queue_write(sql, params)

    def log_risk_event(
        self,
        agent_id: str,
        risk_level: str,
        risk_score: float,
        contributing_factors: Dict[str, Any],
        intervention: Optional[str] = None
    ):
        """
        Log a risk event (for HAVEN).

        Args:
            agent_id: ID of the agent
            risk_level: Risk level
            risk_score: Numerical risk score
            contributing_factors: Factors contributing to risk
            intervention: Intervention taken (if any)
        """
        sql = """
            INSERT INTO risk_events
            (timestamp, agent_id, risk_level, risk_score, contributing_factors, intervention)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            time.time(),
            agent_id,
            risk_level,
            risk_score,
            json.dumps(contributing_factors),
            intervention
        )
        self._queue_write(sql, params)

    # ==========================================
    # Public API - Query Methods
    # ==========================================

    def get_agent_events(
        self,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve agent events.

        Args:
            agent_id: Filter by agent ID
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            sql = "SELECT * FROM agent_events WHERE 1=1"
            params = []

            if agent_id:
                sql += " AND agent_id = ?"
                params.append(agent_id)

            if event_type:
                sql += " AND event_type = ?"
                params.append(event_type)

            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_patterns(
        self,
        pattern_type: Optional[str] = None,
        min_frequency: int = 1,
        min_confidence: float = 0.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve discovered patterns.

        Args:
            pattern_type: Filter by pattern type
            min_frequency: Minimum frequency
            min_confidence: Minimum confidence
            limit: Maximum number of patterns to return

        Returns:
            List of pattern dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            sql = """
                SELECT * FROM patterns
                WHERE frequency >= ? AND confidence >= ?
            """
            params = [min_frequency, min_confidence]

            if pattern_type:
                sql += " AND pattern_type = ?"
                params.append(pattern_type)

            sql += " ORDER BY frequency DESC, confidence DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_performance_metrics(
        self,
        agent_id: str,
        metric_name: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve performance metrics for an agent.

        Args:
            agent_id: Agent ID
            metric_name: Filter by metric name
            time_range: (start_time, end_time) tuple
            limit: Maximum number of metrics to return

        Returns:
            List of metric dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            sql = "SELECT * FROM performance_metrics WHERE agent_id = ?"
            params = [agent_id]

            if metric_name:
                sql += " AND metric_name = ?"
                params.append(metric_name)

            if time_range:
                sql += " AND timestamp >= ? AND timestamp <= ?"
                params.extend(time_range)

            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_risk_events(
        self,
        agent_id: Optional[str] = None,
        min_risk_score: float = 0.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve risk events.

        Args:
            agent_id: Filter by agent ID
            min_risk_score: Minimum risk score
            limit: Maximum number of events to return

        Returns:
            List of risk event dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            sql = "SELECT * FROM risk_events WHERE risk_score >= ?"
            params = [min_risk_score]

            if agent_id:
                sql += " AND agent_id = ?"
                params.append(agent_id)

            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    # ==========================================
    # Maintenance Methods
    # ==========================================

    def cleanup_old_data(self, days: int = 30):
        """
        Remove data older than specified days.

        Args:
            days: Number of days to retain
        """
        cutoff_time = time.time() - (days * 86400)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Clean up old agent events
            cursor.execute(
                "DELETE FROM agent_events WHERE timestamp < ?",
                (cutoff_time,)
            )
            deleted_events = cursor.rowcount

            # Clean up old performance metrics
            cursor.execute(
                "DELETE FROM performance_metrics WHERE timestamp < ?",
                (cutoff_time,)
            )
            deleted_metrics = cursor.rowcount

            conn.commit()

        logger.info("Cleaned up old data: %d events, %d metrics",
                   deleted_events, deleted_metrics)

    def vacuum_database(self):
        """Optimize database by reclaiming unused space."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {
                "writes_queued": self.writes_queued,
                "writes_committed": self.writes_committed,
                "write_errors": self.write_errors,
                "queue_full_count": self.queue_full_count,
                "queue_size": self.write_queue.qsize(),
            }

            # Count records in each table
            for table in ["agent_events", "patterns", "performance_metrics",
                         "system_events", "risk_events"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            return stats

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
