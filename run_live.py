"""
Main script to run the system in "live" (production) mode.

In live mode, the system continuously processes data from Redis streams,
manages agent interactions in real-time, and responds to external events.
"""

import sys
import logging
import signal
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from connectors.redis_client import RedisClient
from core.model import RedisBackedModel
from mesa.time import BaseScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """
    Handle shutdown signals gracefully.
    """
    global running
    logger.info("Shutdown signal received, stopping live mode...")
    running = False


def process_stream_data(redis_client: RedisClient, stream_name: str, last_id: str) -> str:
    """
    Process new data from a Redis stream.

    Args:
        redis_client: Redis client instance
        stream_name: Name of the stream to read from
        last_id: ID of the last processed entry

    Returns:
        The ID of the last processed entry
    """
    try:
        # Read new entries from the stream (non-blocking with 1 second timeout)
        entries = redis_client.read_from_stream(
            stream_name=stream_name,
            count=100,
            block=1000,
            last_id=last_id
        )

        if entries:
            for entry_id, data in entries:
                logger.info("Processing stream entry %s: %s", entry_id, data)

                # TODO: Process the data according to your business logic
                # Example: Create agents, update state, trigger actions, etc.

                last_id = entry_id

    except Exception as e:
        logger.error("Error processing stream data: %s", e)

    return last_id


def monitor_pubsub_channels(redis_client: RedisClient, channels: list):
    """
    Monitor Pub/Sub channels for agent communication.

    Args:
        redis_client: Redis client instance
        channels: List of channels to monitor
    """
    try:
        redis_client.subscribe(channels)
        logger.info("Monitoring Pub/Sub channels: %s", channels)

        for message in redis_client.listen():
            logger.info("Received message on channel '%s': %s",
                       message.get("channel"), message.get("data"))

            # TODO: Handle incoming messages
            # Example: Route messages to agents, trigger events, etc.

            if not running:
                break

    except Exception as e:
        logger.error("Error monitoring Pub/Sub channels: %s", e)
    finally:
        redis_client.unsubscribe()


def main():
    """
    Main entry point for running the system in live mode.
    """
    global running

    logger.info("Starting Mesa ABM in Live Mode")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Configuration
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    STREAM_NAME = "abm:data_ingestion"
    PUBSUB_CHANNELS = ["abm:agent_messages", "abm:system_events"]
    STEP_INTERVAL = 1.0  # seconds between model steps

    try:
        # Initialize Redis client
        logger.info("Connecting to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
        redis_client = RedisClient(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB
        )

        # Initialize model
        model_params = {
            "mode": "live",
            "stream_name": STREAM_NAME
        }

        logger.info("Initializing RedisBackedModel for live mode")
        model = RedisBackedModel(
            redis_client=redis_client,
            scheduler_class=BaseScheduler,
            model_params=model_params
        )

        # TODO: Initialize agents for live mode
        # Example:
        # from agents.example_agent import ExampleAgent
        # for i in range(10):
        #     agent = ExampleAgent(i, model, redis_client)
        #     model.add_agent(agent)

        logger.info("Live mode initialized, entering main loop")

        # Track last processed stream entry
        last_stream_id = "0-0"

        # Main loop
        while running:
            try:
                # Process new stream data
                last_stream_id = process_stream_data(
                    redis_client,
                    STREAM_NAME,
                    last_stream_id
                )

                # Execute one model step
                model.step()

                # Check for Pub/Sub messages (non-blocking)
                # Note: For production, consider using a separate thread for Pub/Sub

                # Sleep to control step rate
                time.sleep(STEP_INTERVAL)

            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
                # Continue running unless explicitly stopped
                time.sleep(1)

        logger.info("Live mode shutting down")

        # Cleanup
        redis_client.close()
        logger.info("Shutdown complete")

    except KeyboardInterrupt:
        logger.info("Live mode interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error("Error during live mode: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
