"""
Data Miner Agent for the Mycelial ABM Framework

This agent reads data from Redis Streams, processes it, and publishes
formatted data for other agents to consume. It acts as a data ingestion
and preprocessing layer in the multi-agent system.
"""

from typing import Dict, Any, Optional, List, Callable
import logging
import json

from agents.base_agent import MycelialAgent

logger = logging.getLogger(__name__)


class DataMinerAgent(MycelialAgent):
    """
    Data Miner Agent that ingests data from Redis Streams and publishes it.

    This agent serves as a data pipeline component that:
    - Continuously reads from one or more Redis Streams
    - Processes and transforms raw data
    - Publishes formatted data to Pub/Sub channels for other agents
    - Maintains data quality metrics
    - Handles data validation and error cases

    Typical use cases:
    - Real-time data ingestion from external sources
    - ETL (Extract, Transform, Load) operations
    - Data preprocessing for specialist agents
    - Feature engineering and normalization
    - Data quality monitoring
    """

    def __init__(
        self,
        unique_id: int,
        model,
        redis_client,
        source_streams: List[str],
        output_channel: str,
        team_id: str = "data_miners",
        agent_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Data Miner Agent.

        Args:
            unique_id: Unique identifier for this agent
            model: The Mesa model this agent belongs to
            redis_client: Redis client for data operations
            source_streams: List of Redis Stream names to read from
            output_channel: Pub/Sub channel to publish processed data to
            team_id: Team identifier (default: "data_miners")
            agent_config: Optional configuration dictionary
        """
        super().__init__(unique_id, model, redis_client, team_id, agent_config)

        # Data source configuration
        self.source_streams = source_streams
        self.output_channel = output_channel

        # Track last read ID for each stream
        self.stream_positions: Dict[str, str] = {
            stream: "0-0" for stream in source_streams
        }

        # Data processing configuration
        self.batch_size = agent_config.get("batch_size", 10) if agent_config else 10
        self.validation_enabled = agent_config.get("validation_enabled", True) if agent_config else True

        # Custom data transformation function (can be set by subclass)
        self.transform_function: Optional[Callable] = None

        # Metrics tracking
        self.records_processed: int = 0
        self.records_published: int = 0
        self.validation_failures: int = 0
        self.processing_errors: int = 0

        # Buffer for batch processing
        self.data_buffer: List[Dict[str, Any]] = []

        logger.info("DataMinerAgent %s initialized with streams: %s -> channel: %s",
                   self.agent_id, source_streams, output_channel)

    def step(self):
        """
        Execute one step of data mining behavior.

        Flow:
        1. Read new data from all source streams
        2. Validate incoming data
        3. Transform/process data
        4. Publish processed data to output channel
        5. Update metrics
        """
        self.step_count += 1

        total_records = 0

        # Read from each source stream
        for stream_name in self.source_streams:
            records_read = self._read_from_source_stream(stream_name)
            total_records += records_read

        # Process buffered data if batch is full or on interval
        if len(self.data_buffer) >= self.batch_size or self.step_count % 5 == 0:
            self._process_and_publish_batch()

        # Reward based on records processed
        self.last_reward = float(total_records)
        self.cumulative_reward += self.last_reward

        # Update performance metrics
        self._update_performance_metrics(self.last_reward)

        # Save state periodically
        if self.step_count % 10 == 0:
            self._save_state_to_redis()

        logger.debug("%s processed %d records in step %d",
                    self.agent_id, total_records, self.step_count)

    def _read_from_source_stream(self, stream_name: str) -> int:
        """
        Read new entries from a source stream.

        Args:
            stream_name: Name of the stream to read from

        Returns:
            Number of records read
        """
        try:
            last_id = self.stream_positions[stream_name]

            # Read new entries
            entries = self.redis_client.read_from_stream(
                stream_name=stream_name,
                count=self.batch_size,
                last_id=last_id,
                block=None  # Non-blocking
            )

            if not entries:
                return 0

            # Process each entry
            for entry_id, data in entries:
                # Update stream position
                self.stream_positions[stream_name] = entry_id

                # Validate data if enabled
                if self.validation_enabled:
                    if not self._validate_data(data):
                        self.validation_failures += 1
                        logger.warning("%s: Invalid data from %s: %s",
                                     self.agent_id, stream_name, entry_id)
                        continue

                # Add to buffer for processing
                self.data_buffer.append({
                    "source_stream": stream_name,
                    "entry_id": entry_id,
                    "data": data,
                    "timestamp": self.model.current_step
                })

            self.records_processed += len(entries)
            return len(entries)

        except Exception as e:
            logger.error("%s: Error reading from stream %s: %s",
                        self.agent_id, stream_name, e)
            self.processing_errors += 1
            return 0

    def _process_and_publish_batch(self):
        """
        Process buffered data and publish to output channel.
        """
        if not self.data_buffer:
            return

        try:
            # Transform data
            processed_records = []
            for record in self.data_buffer:
                transformed = self._transform_data(record)
                if transformed is not None:
                    processed_records.append(transformed)

            # Publish batch to output channel
            if processed_records:
                batch_message = {
                    "agent_id": self.agent_id,
                    "batch_size": len(processed_records),
                    "timestamp": self.model.current_step,
                    "records": processed_records
                }

                num_subscribers = self.redis_client.publish(
                    self.output_channel,
                    batch_message
                )

                self.records_published += len(processed_records)

                logger.debug("%s published batch of %d records to %d subscribers",
                           self.agent_id, len(processed_records), num_subscribers)

            # Clear buffer
            self.data_buffer.clear()

        except Exception as e:
            logger.error("%s: Error processing batch: %s", self.agent_id, e)
            self.processing_errors += 1
            # Clear buffer to prevent repeated errors
            self.data_buffer.clear()

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate incoming data for quality and completeness.

        Override this method to implement custom validation logic.

        Args:
            data: Data to validate

        Returns:
            True if data is valid
        """
        # Default validation: ensure data is not empty
        if not data:
            return False

        # Check for required fields (override in subclass)
        required_fields = self._get_required_fields()
        for field in required_fields:
            if field not in data:
                return False

        return True

    def _transform_data(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform raw data into processed format.

        Override this method or set self.transform_function to implement
        custom data transformation logic.

        Args:
            record: Raw data record with metadata

        Returns:
            Transformed data, or None if transformation fails
        """
        try:
            # Use custom transform function if provided
            if self.transform_function is not None:
                return self.transform_function(record)

            # Default transformation: add metadata and pass through
            return {
                "source": record["source_stream"],
                "entry_id": record["entry_id"],
                "timestamp": record["timestamp"],
                "processed_by": self.agent_id,
                "data": record["data"]
            }

        except Exception as e:
            logger.error("%s: Error transforming data: %s", self.agent_id, e)
            return None

    def _get_required_fields(self) -> List[str]:
        """
        Get list of required fields for data validation.

        Override this method to specify required fields.

        Returns:
            List of required field names
        """
        return []  # Default: no required fields

    # ==========================================
    # Configuration methods
    # ==========================================

    def set_transform_function(self, transform_fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]):
        """
        Set a custom data transformation function.

        Args:
            transform_fn: Function that takes a record dict and returns transformed dict
        """
        self.transform_function = transform_fn
        logger.info("%s: Custom transform function set", self.agent_id)

    def add_source_stream(self, stream_name: str):
        """
        Add a new source stream to monitor.

        Args:
            stream_name: Name of the stream to add
        """
        if stream_name not in self.source_streams:
            self.source_streams.append(stream_name)
            self.stream_positions[stream_name] = "0-0"
            logger.info("%s: Added source stream: %s", self.agent_id, stream_name)

    def remove_source_stream(self, stream_name: str):
        """
        Remove a source stream from monitoring.

        Args:
            stream_name: Name of the stream to remove
        """
        if stream_name in self.source_streams:
            self.source_streams.remove(stream_name)
            self.stream_positions.pop(stream_name, None)
            logger.info("%s: Removed source stream: %s", self.agent_id, stream_name)

    def set_output_channel(self, channel: str):
        """
        Change the output channel for publishing data.

        Args:
            channel: New output channel name
        """
        old_channel = self.output_channel
        self.output_channel = channel
        logger.info("%s: Changed output channel from %s to %s",
                   self.agent_id, old_channel, channel)

    # ==========================================
    # Metrics and monitoring
    # ==========================================

    def get_mining_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about data mining operations.

        Returns:
            Dictionary with mining metrics
        """
        return {
            "agent_id": self.agent_id,
            "records_processed": self.records_processed,
            "records_published": self.records_published,
            "validation_failures": self.validation_failures,
            "processing_errors": self.processing_errors,
            "buffer_size": len(self.data_buffer),
            "source_streams": len(self.source_streams),
            "stream_positions": self.stream_positions.copy(),
            "processing_rate": self._calculate_processing_rate()
        }

    def _calculate_processing_rate(self) -> float:
        """
        Calculate records processed per step.

        Returns:
            Processing rate
        """
        if self.step_count == 0:
            return 0.0
        return self.records_processed / self.step_count

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the data miner.

        Returns:
            Dictionary with health indicators
        """
        error_rate = (self.processing_errors / max(1, self.records_processed)) if self.records_processed > 0 else 0
        validation_failure_rate = (self.validation_failures / max(1, self.records_processed)) if self.records_processed > 0 else 0

        health_score = 1.0 - (error_rate * 0.6 + validation_failure_rate * 0.4)

        return {
            "agent_id": self.agent_id,
            "health_score": max(0.0, min(1.0, health_score)),
            "error_rate": error_rate,
            "validation_failure_rate": validation_failure_rate,
            "is_operational": error_rate < 0.1,
            "buffer_status": "normal" if len(self.data_buffer) < self.batch_size * 2 else "high"
        }

    def _save_state_to_redis(self):
        """
        Save agent state including stream positions to Redis.
        """
        # Call parent method for basic state
        super()._save_state_to_redis()

        # Save additional data miner state
        miner_state = {
            "stream_positions": self.stream_positions,
            "records_processed": self.records_processed,
            "records_published": self.records_published,
            "validation_failures": self.validation_failures,
            "processing_errors": self.processing_errors
        }

        key = f"agent:miner_state:{self.agent_id}"
        self.redis_client.set_key_value(key, miner_state)

    def reset(self):
        """
        Reset the data miner agent to initial state.
        """
        super().reset()

        # Reset stream positions
        self.stream_positions = {
            stream: "0-0" for stream in self.source_streams
        }

        # Reset metrics
        self.records_processed = 0
        self.records_published = 0
        self.validation_failures = 0
        self.processing_errors = 0
        self.data_buffer.clear()

        logger.info("%s has been reset", self.agent_id)
