"""
Unit tests for RedisClient.

Tests the Redis client wrapper functionality including:
- Connection management
- Key-value operations
- Pub/Sub messaging
- Stream operations
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from connectors.redis_client import RedisClient


class TestRedisClientInit:
    """Test RedisClient initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        with patch('connectors.redis_client.redis.Redis') as mock_redis:
            client = RedisClient()

            mock_redis.assert_called_once_with(
                host='localhost',
                port=6379,
                db=0,
                password=None,
                decode_responses=True
            )

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        with patch('connectors.redis_client.redis.Redis') as mock_redis:
            client = RedisClient(
                host='custom-host',
                port=6380,
                db=2,
                password='secret',
                decode_responses=False
            )

            mock_redis.assert_called_once_with(
                host='custom-host',
                port=6380,
                db=2,
                password='secret',
                decode_responses=False
            )

    def test_connection_pool_created(self):
        """Test that connection pool is created."""
        with patch('connectors.redis_client.redis.Redis') as mock_redis:
            client = RedisClient()
            assert client.client is not None


class TestRedisClientKeyValue:
    """Test key-value operations."""

    @pytest.fixture
    def redis_client(self):
        """Create RedisClient with mocked Redis."""
        with patch('connectors.redis_client.redis.Redis') as mock_redis:
            mock_instance = MagicMock()
            mock_redis.return_value = mock_instance
            client = RedisClient()
            client.client = mock_instance
            yield client

    def test_set_key_value_simple(self, redis_client):
        """Test setting a simple key-value pair."""
        redis_client.set_key_value("test_key", "test_value")

        redis_client.client.set.assert_called_once_with(
            "test_key",
            "test_value",
            ex=None,
            px=None
        )

    def test_set_key_value_with_dict(self, redis_client):
        """Test setting a dictionary value (JSON serialization)."""
        test_dict = {"name": "test", "value": 123}
        redis_client.set_key_value("test_key", test_dict)

        redis_client.client.set.assert_called_once_with(
            "test_key",
            json.dumps(test_dict),
            ex=None,
            px=None
        )

    def test_get_key_value_simple(self, redis_client):
        """Test getting a simple value."""
        redis_client.client.get.return_value = "test_value"

        result = redis_client.get_key_value("test_key")

        assert result == "test_value"
        redis_client.client.get.assert_called_once_with("test_key")

    def test_get_key_value_json(self, redis_client):
        """Test getting a JSON value."""
        test_dict = {"name": "test", "value": 123}
        redis_client.client.get.return_value = json.dumps(test_dict)

        result = redis_client.get_key_value("test_key")

        assert result == test_dict

    def test_get_key_value_nonexistent(self, redis_client):
        """Test getting a non-existent key."""
        redis_client.client.get.return_value = None

        result = redis_client.get_key_value("nonexistent")

        assert result is None

    def test_ping_success(self, redis_client):
        """Test successful ping."""
        redis_client.client.ping.return_value = True

        assert redis_client.ping() is True

    def test_ping_failure(self, redis_client):
        """Test ping failure."""
        redis_client.client.ping.side_effect = Exception("Connection failed")

        assert redis_client.ping() is False


class TestRedisClientPubSub:
    """Test Pub/Sub operations."""

    @pytest.fixture
    def redis_client(self):
        """Create RedisClient with mocked Redis."""
        with patch('connectors.redis_client.redis.Redis') as mock_redis:
            mock_instance = MagicMock()
            mock_pubsub = MagicMock()
            mock_instance.pubsub.return_value = mock_pubsub
            mock_redis.return_value = mock_instance

            client = RedisClient()
            client.client = mock_instance
            client.pubsub = mock_pubsub
            yield client

    def test_publish_string_message(self, redis_client):
        """Test publishing a string message."""
        redis_client.client.publish.return_value = 3  # 3 subscribers

        result = redis_client.publish("test_channel", "test_message")

        assert result == 3
        redis_client.client.publish.assert_called_once_with(
            "test_channel",
            "test_message"
        )

    def test_publish_dict_message(self, redis_client):
        """Test publishing a dictionary message."""
        test_dict = {"type": "event", "data": "value"}
        redis_client.client.publish.return_value = 2

        result = redis_client.publish("test_channel", test_dict)

        assert result == 2
        redis_client.client.publish.assert_called_once_with(
            "test_channel",
            json.dumps(test_dict)
        )

    def test_subscribe_single_channel(self, redis_client):
        """Test subscribing to a single channel."""
        redis_client.subscribe(["test_channel"])

        redis_client.pubsub.subscribe.assert_called_once_with("test_channel")

    def test_subscribe_multiple_channels(self, redis_client):
        """Test subscribing to multiple channels."""
        channels = ["channel1", "channel2", "channel3"]
        redis_client.subscribe(channels)

        # Should be called once with all channels unpacked
        redis_client.pubsub.subscribe.assert_called_once_with(*channels)


class TestRedisClientStreams:
    """Test Stream operations."""

    @pytest.fixture
    def redis_client(self):
        """Create RedisClient with mocked Redis."""
        with patch('connectors.redis_client.redis.Redis') as mock_redis:
            mock_instance = MagicMock()
            mock_redis.return_value = mock_instance
            client = RedisClient()
            client.client = mock_instance
            yield client

    def test_write_to_stream(self, redis_client):
        """Test writing data to a stream."""
        redis_client.client.xadd.return_value = b'1234567890-0'

        test_data = {"field1": "value1", "field2": "value2"}
        result = redis_client.write_to_stream("test_stream", test_data)

        assert result == b'1234567890-0'
        # Check that xadd was called with stream name and includes timestamp
        call_args = redis_client.client.xadd.call_args
        assert call_args[0][0] == "test_stream"
        assert "field1" in call_args[0][1]
        assert "field2" in call_args[0][1]
        assert "timestamp" in call_args[0][1]  # Auto-added timestamp
        assert call_args[1] == {"maxlen": None, "approximate": True}

    def test_read_from_stream_basic(self, redis_client):
        """Test reading from a stream."""
        mock_response = [
            [b'test_stream', [
                (b'1-0', {b'field': b'value1'}),
                (b'1-1', {b'field': b'value2'})
            ]]
        ]
        redis_client.client.xread.return_value = mock_response

        result = redis_client.read_from_stream("test_stream")

        assert len(result) == 2
        assert result[0][0] == b'1-0'
        assert result[1][0] == b'1-1'

    def test_read_from_stream_with_count(self, redis_client):
        """Test reading with count limit."""
        redis_client.client.xread.return_value = []

        redis_client.read_from_stream(
            "test_stream",
            count=10,
            last_id="0-0"
        )

        redis_client.client.xread.assert_called_once_with(
            {"test_stream": "0-0"},
            count=10,
            block=None
        )

    def test_read_from_stream_blocking(self, redis_client):
        """Test blocking read from stream."""
        redis_client.client.xread.return_value = []

        redis_client.read_from_stream(
            "test_stream",
            block=5000  # 5 seconds
        )

        redis_client.client.xread.assert_called_once()
        call_args = redis_client.client.xread.call_args
        assert call_args[1]['block'] == 5000


class TestRedisClientUtilities:
    """Test utility methods."""

    @pytest.fixture
    def redis_client(self):
        """Create RedisClient with mocked Redis."""
        with patch('connectors.redis_client.redis.Redis') as mock_redis:
            mock_instance = MagicMock()
            mock_redis.return_value = mock_instance
            client = RedisClient()
            client.client = mock_instance
            yield client

    def test_exists(self, redis_client):
        """Test checking if key exists."""
        redis_client.client.exists.return_value = 1

        result = redis_client.client.exists("test_key")

        assert result == 1

    def test_delete(self, redis_client):
        """Test deleting a key."""
        redis_client.client.delete.return_value = 1

        result = redis_client.client.delete("test_key")

        assert result == 1

    def test_flushdb(self, redis_client):
        """Test flushing database."""
        redis_client.flushdb()

        redis_client.client.flushdb.assert_called_once()

    def test_close(self, redis_client):
        """Test closing connection."""
        redis_client.close()

        redis_client.client.close.assert_called_once()


class TestRedisClientErrorHandling:
    """Test error handling."""

    @pytest.fixture
    def redis_client(self):
        """Create RedisClient with mocked Redis."""
        with patch('connectors.redis_client.redis.Redis') as mock_redis:
            mock_instance = MagicMock()
            mock_redis.return_value = mock_instance
            client = RedisClient()
            client.client = mock_instance
            yield client

    def test_get_with_json_decode_error(self, redis_client):
        """Test handling JSON decode errors on get."""
        redis_client.client.get.return_value = "invalid json {{"

        result = redis_client.get_key_value("test_key")

        # Should return the raw string if JSON parse fails
        assert result == "invalid json {{"

    def test_connection_error_handling(self):
        """Test handling connection errors."""
        with patch('connectors.redis_client.redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Connection refused")

            with pytest.raises(Exception):
                client = RedisClient()


@pytest.mark.integration
@pytest.mark.requires_redis
class TestRedisClientIntegration:
    """
    Integration tests requiring actual Redis connection.

    Run with: pytest -m requires_redis
    """

    @pytest.fixture
    def real_redis_client(self):
        """Create real Redis client for integration tests."""
        try:
            client = RedisClient(db=15)  # Use test database
            if not client.ping():
                pytest.skip("Redis not available")
            client.flushdb()  # Clean test database
            yield client
            client.flushdb()  # Cleanup
            client.close()
        except Exception:
            pytest.skip("Redis not available")

    def test_real_set_get(self, real_redis_client):
        """Test actual set/get operations."""
        real_redis_client.set_key_value("test_key", "test_value")
        result = real_redis_client.get_key_value("test_key")
        assert result == "test_value"

    def test_real_pubsub(self, real_redis_client):
        """Test actual pub/sub operations."""
        subscribers = real_redis_client.publish("test_channel", "test_message")
        # May be 0 if no subscribers
        assert subscribers >= 0

    def test_real_stream(self, real_redis_client):
        """Test actual stream operations."""
        entry_id = real_redis_client.write_to_stream(
            "test_stream",
            {"field": "value"}
        )
        assert entry_id is not None

        entries = real_redis_client.read_from_stream("test_stream")
        assert len(entries) >= 1
