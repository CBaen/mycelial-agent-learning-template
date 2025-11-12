"""
Redis client for managing all data operations in the ABM framework.

This module provides a unified interface for:
- Publishing to Pub/Sub channels (agent communication)
- Writing to Streams (data ingestion)
- Read/Write to Key-Value store (agent state)
"""

import redis
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisClient:
    """
    Manages Redis connections and provides methods for all Redis operations.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True
    ):
        """
        Initialize Redis client connection.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Optional password for authentication
            decode_responses: Whether to decode byte responses to strings
        """
        self.host = host
        self.port = port
        self.db = db

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses
            )
            # Test connection
            self.client.ping()
            logger.info("Successfully connected to Redis at %s:%d", host, port)
        except redis.ConnectionError as e:
            logger.error("Failed to connect to Redis: %s", e)
            raise

        # Store pubsub object for reuse
        self.pubsub = self.client.pubsub()

    # ========================
    # Pub/Sub Methods (Agent Communication)
    # ========================

    def publish(self, channel: str, message: Any) -> int:
        """
        Publish a message to a Pub/Sub channel.

        Args:
            channel: The channel name to publish to
            message: The message to publish (will be JSON serialized if dict/list)

        Returns:
            Number of subscribers that received the message
        """
        try:
            if isinstance(message, (dict, list)):
                message = json.dumps(message)

            num_subscribers = self.client.publish(channel, message)
            logger.debug("Published to channel '%s': %d subscribers received", channel, num_subscribers)
            return num_subscribers
        except Exception as e:
            logger.error("Error publishing to channel '%s': %s", channel, e)
            raise

    def subscribe(self, channels: List[str]):
        """
        Subscribe to one or more Pub/Sub channels.

        Args:
            channels: List of channel names to subscribe to
        """
        try:
            self.pubsub.subscribe(*channels)
            logger.info("Subscribed to channels: %s", channels)
        except Exception as e:
            logger.error("Error subscribing to channels: %s", e)
            raise

    def unsubscribe(self, channels: Optional[List[str]] = None):
        """
        Unsubscribe from Pub/Sub channels.

        Args:
            channels: Optional list of channels to unsubscribe from. If None, unsubscribes from all.
        """
        try:
            if channels:
                self.pubsub.unsubscribe(*channels)
            else:
                self.pubsub.unsubscribe()
            logger.info("Unsubscribed from channels")
        except Exception as e:
            logger.error("Error unsubscribing from channels: %s", e)
            raise

    def listen(self):
        """
        Listen for messages on subscribed channels.

        Yields:
            Messages from subscribed channels
        """
        try:
            for message in self.pubsub.listen():
                if message["type"] == "message":
                    yield message
        except Exception as e:
            logger.error("Error listening to channels: %s", e)
            raise

    # ========================
    # Stream Methods (Data Ingestion)
    # ========================

    def write_to_stream(
        self,
        stream_name: str,
        data: Dict[str, Any],
        maxlen: Optional[int] = None,
        approximate: bool = True
    ) -> str:
        """
        Write data to a Redis Stream.

        Args:
            stream_name: Name of the stream
            data: Dictionary of field-value pairs to add to the stream
            maxlen: Optional maximum length of the stream (for trimming)
            approximate: If True, uses approximate trimming (~) for better performance

        Returns:
            The ID of the added stream entry
        """
        try:
            # Serialize complex data types
            serialized_data = {}
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    serialized_data[key] = json.dumps(value)
                else:
                    serialized_data[key] = str(value)

            # Add timestamp if not present
            if "timestamp" not in serialized_data:
                serialized_data["timestamp"] = datetime.utcnow().isoformat()

            entry_id = self.client.xadd(
                stream_name,
                serialized_data,
                maxlen=maxlen,
                approximate=approximate
            )
            logger.debug("Added entry to stream '%s': %s", stream_name, entry_id)
            return entry_id
        except Exception as e:
            logger.error("Error writing to stream '%s': %s", stream_name, e)
            raise

    def read_from_stream(
        self,
        stream_name: str,
        count: Optional[int] = None,
        block: Optional[int] = None,
        last_id: str = "0-0"
    ) -> List[tuple]:
        """
        Read data from a Redis Stream.

        Args:
            stream_name: Name of the stream
            count: Maximum number of entries to return
            block: Milliseconds to block waiting for data (None for non-blocking)
            last_id: ID of the last entry received (to read newer entries)

        Returns:
            List of tuples: (entry_id, data_dict)
        """
        try:
            streams = {stream_name: last_id}
            results = self.client.xread(streams, count=count, block=block)

            if results:
                entries = results[0][1]  # Get entries for the first stream
                logger.debug("Read %d entries from stream '%s'", len(entries), stream_name)
                return entries
            return []
        except Exception as e:
            logger.error("Error reading from stream '%s': %s", stream_name, e)
            raise

    def get_stream_length(self, stream_name: str) -> int:
        """
        Get the number of entries in a stream.

        Args:
            stream_name: Name of the stream

        Returns:
            Number of entries in the stream
        """
        try:
            return self.client.xlen(stream_name)
        except Exception as e:
            logger.error("Error getting length of stream '%s': %s", stream_name, e)
            raise

    # ========================
    # Key-Value Methods (Agent State)
    # ========================

    def set_key_value(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
        px: Optional[int] = None
    ) -> bool:
        """
        Set a key-value pair in Redis.

        Args:
            key: The key name
            value: The value (will be JSON serialized if dict/list)
            ex: Expiration time in seconds
            px: Expiration time in milliseconds

        Returns:
            True if successful
        """
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)

            result = self.client.set(key, value, ex=ex, px=px)
            logger.debug("Set key '%s'", key)
            return result
        except Exception as e:
            logger.error("Error setting key '%s': %s", key, e)
            raise

    def get_key_value(self, key: str, deserialize_json: bool = True) -> Optional[Any]:
        """
        Get a value from Redis by key.

        Args:
            key: The key name
            deserialize_json: Whether to attempt JSON deserialization

        Returns:
            The value, or None if key doesn't exist
        """
        try:
            value = self.client.get(key)

            if value is None:
                return None

            if deserialize_json:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value

            return value
        except Exception as e:
            logger.error("Error getting key '%s': %s", key, e)
            raise

    def delete_key(self, *keys: str) -> int:
        """
        Delete one or more keys from Redis.

        Args:
            keys: One or more key names to delete

        Returns:
            Number of keys deleted
        """
        try:
            count = self.client.delete(*keys)
            logger.debug("Deleted %d keys", count)
            return count
        except Exception as e:
            logger.error("Error deleting keys: %s", e)
            raise

    def exists(self, *keys: str) -> int:
        """
        Check if one or more keys exist.

        Args:
            keys: One or more key names to check

        Returns:
            Number of keys that exist
        """
        try:
            return self.client.exists(*keys)
        except Exception as e:
            logger.error("Error checking key existence: %s", e)
            raise

    def set_hash(self, key: str, mapping: Dict[str, Any]) -> int:
        """
        Set multiple hash fields.

        Args:
            key: The hash key name
            mapping: Dictionary of field-value pairs

        Returns:
            Number of fields added
        """
        try:
            # Serialize complex values
            serialized_mapping = {}
            for field, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized_mapping[field] = json.dumps(value)
                else:
                    serialized_mapping[field] = str(value)

            result = self.client.hset(key, mapping=serialized_mapping)
            logger.debug("Set %d fields in hash '%s'", len(mapping), key)
            return result
        except Exception as e:
            logger.error("Error setting hash '%s': %s", key, e)
            raise

    def get_hash(self, key: str, deserialize_json: bool = True) -> Dict[str, Any]:
        """
        Get all fields and values from a hash.

        Args:
            key: The hash key name
            deserialize_json: Whether to attempt JSON deserialization

        Returns:
            Dictionary of field-value pairs
        """
        try:
            hash_data = self.client.hgetall(key)

            if deserialize_json:
                deserialized = {}
                for field, value in hash_data.items():
                    try:
                        deserialized[field] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        deserialized[field] = value
                return deserialized

            return hash_data
        except Exception as e:
            logger.error("Error getting hash '%s': %s", key, e)
            raise

    # ========================
    # Utility Methods
    # ========================

    def ping(self) -> bool:
        """
        Test Redis connection.

        Returns:
            True if connection is alive
        """
        try:
            return self.client.ping()
        except Exception as e:
            logger.error("Redis ping failed: %s", e)
            return False

    def close(self):
        """
        Close the Redis connection.
        """
        try:
            self.pubsub.close()
            self.client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error("Error closing Redis connection: %s", e)
            raise

    def flushdb(self):
        """
        Clear all keys in the current database.

        WARNING: This will delete all data in the current database.
        """
        try:
            self.client.flushdb()
            logger.warning("Flushed all keys from database %d", self.db)
        except Exception as e:
            logger.error("Error flushing database: %s", e)
            raise
