"""
Main script to launch the Mesa simulation in simulation mode.

This script initializes the Redis connection, sets up the model,
and runs the simulation for a specified number of steps.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from connectors.redis_client import RedisClient
from core.model import RedisBackedModel
from mesa.time import RandomActivation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for running the simulation.
    """
    logger.info("Starting Mesa ABM Simulation")

    # Configuration
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    NUM_STEPS = 100

    try:
        # Initialize Redis client
        logger.info("Connecting to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
        redis_client = RedisClient(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB
        )

        # Initialize model with parameters
        model_params = {
            "mode": "simulation",
            "num_steps": NUM_STEPS
        }

        logger.info("Initializing RedisBackedModel")
        model = RedisBackedModel(
            redis_client=redis_client,
            scheduler_class=RandomActivation,
            model_params=model_params
        )

        # TODO: Add agents to the model here
        # Example:
        # from agents.example_agent import ExampleAgent
        # for i in range(10):
        #     agent = ExampleAgent(i, model, redis_client)
        #     model.add_agent(agent)

        logger.info("Starting simulation for %d steps", NUM_STEPS)

        # Run the simulation
        model.run(num_steps=NUM_STEPS)

        logger.info("Simulation completed successfully")

        # Cleanup
        redis_client.close()

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error("Error during simulation: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
