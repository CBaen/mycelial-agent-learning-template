"""
Core Mesa Model class for Mycelial Agent Engine (MAE).

This module defines the main model that orchestrates the entire multi-agent system.
It is responsible for:
- Loading configuration from config.yaml
- Initializing the agent scheduler
- Managing system-wide state
- Coordinating data backbone components
- Orchestrating learning engines (FRL, VDN, HAVEN)
"""

from mesa import Model
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings, Settings
from connectors.redis_client import RedisClient
from connectors.sql_logger import SQLiteLogger
from connectors.vector_db import VectorDBInterface, create_vector_db

logger = logging.getLogger(__name__)


class MycelialModel(Model):
    """
    Main Mesa Model for the Mycelial Agent Engine.

    This model is the central orchestrator of the MAE system. It:
    - Loads configuration from config.yaml automatically
    - Initializes all data backbone components (Redis, Vector DB, SQLite)
    - Manages the agent scheduler with configurable scheduling strategies
    - Provides system-wide coordination and state management
    - Integrates FRL, VDN, and HAVEN engines

    Philosophy:
    The model acts as the "soil" in which the mycelial network of agents grows.
    It provides infrastructure and nutrients (data, compute, storage) but does
    not centrally control individual agents. Agents are autonomous and
    communicate peer-to-peer, with the model providing shared resources.
    """

    # Class-level scheduler mapping
    SCHEDULER_TYPES = {
        "base": BaseScheduler,
        "random": RandomActivation,
        "simultaneous": SimultaneousActivation,
    }

    def __init__(
        self,
        config: Optional[Settings] = None,
        redis_client: Optional[RedisClient] = None,
        vector_db: Optional[VectorDBInterface] = None,
        sql_logger: Optional[SQLiteLogger] = None,
        scheduler_type: str = "random",
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Mycelial Model.

        Args:
            config: Settings object (if None, loads from config.yaml automatically)
            redis_client: Redis client (if None, creates from config)
            vector_db: Vector database (if None, creates from config)
            sql_logger: SQLite logger (if None, creates from config)
            scheduler_type: Type of scheduler ("base", "random", "simultaneous")
            model_params: Optional additional model parameters
        """
        super().__init__()

        # Load configuration (automatic if not provided)
        self.config = config or settings
        logger.info("Configuration loaded from: %s",
                   self.config.simulation.mode if hasattr(self.config, 'simulation') else 'default')

        # Initialize data backbone components
        self.redis_client = redis_client or self._initialize_redis()
        self.vector_db = vector_db or self._initialize_vector_db()
        self.sql_logger = sql_logger or self._initialize_sql_logger()

        # Initialize scheduler based on config
        scheduler_class = self._get_scheduler_class(scheduler_type)
        self.schedule = scheduler_class(self)
        logger.info("Scheduler initialized: %s", scheduler_class.__name__)

        # Model parameters
        self.model_params = model_params or {}

        # System state
        self.current_step = 0
        self.start_time = None
        self.is_running = False

        # Agent registry (for dynamic building/hibernation)
        self.active_agents: Dict[str, Any] = {}
        self.hibernated_agents: Dict[str, Any] = {}

        # Learning engine references (initialized by components)
        self.frl_engine = None
        self.vdn_engine = None
        self.haven_coordinator = None

        # Performance metrics
        self.total_agents_created = 0
        self.total_agents_removed = 0
        self.global_reward_history: List[float] = []

        logger.info("MycelialModel initialized successfully")

    # =========================================================================
    # Initialization Methods
    # =========================================================================

    def _initialize_redis(self) -> RedisClient:
        """Initialize Redis client from configuration."""
        try:
            redis_client = RedisClient(
                host=self.config.redis.host,
                port=self.config.redis.port,
                db=self.config.redis.db,
                password=self.config.redis.password,
                decode_responses=self.config.redis.decode_responses
            )
            logger.info("Redis client initialized: %s:%d (db=%d)",
                       self.config.redis.host, self.config.redis.port, self.config.redis.db)
            return redis_client
        except Exception as e:
            logger.error("Failed to initialize Redis: %s", e)
            raise

    def _initialize_vector_db(self) -> VectorDBInterface:
        """Initialize Vector Database from configuration."""
        try:
            vector_db = create_vector_db(
                backend=self.config.vector_db.backend,
                collection_name=self.config.vector_db.collection_name,
                embedding_dim=self.config.vector_db.embedding_dim,
                persist_directory=self.config.vector_db.persist_directory
            )
            vector_db.initialize()
            logger.info("Vector DB initialized: %s", self.config.vector_db.backend)
            return vector_db
        except Exception as e:
            logger.error("Failed to initialize Vector DB: %s", e)
            raise

    def _initialize_sql_logger(self) -> SQLiteLogger:
        """Initialize SQLite logger from configuration."""
        try:
            sql_logger = SQLiteLogger(
                db_path=self.config.sqlite.db_path,
                queue_size=self.config.sqlite.queue_size,
                batch_size=self.config.sqlite.batch_size,
                flush_interval=self.config.sqlite.flush_interval
            )
            logger.info("SQLite logger initialized: %s", self.config.sqlite.db_path)
            return sql_logger
        except Exception as e:
            logger.error("Failed to initialize SQLite logger: %s", e)
            raise

    def _get_scheduler_class(self, scheduler_type: str):
        """Get scheduler class from type string."""
        scheduler_class = self.SCHEDULER_TYPES.get(scheduler_type.lower())
        if scheduler_class is None:
            logger.warning("Unknown scheduler type '%s', using RandomActivation", scheduler_type)
            scheduler_class = RandomActivation
        return scheduler_class

    # =========================================================================
    # Main Simulation Loop
    # =========================================================================

    def step(self):
        """
        Execute one step of the model.

        This method orchestrates one complete iteration of the mycelial network:
        1. Increment step counter
        2. Activate all agents (via scheduler)
        3. Compute global rewards (if VDN enabled)
        4. Update HAVEN risk assessment (if enabled)
        5. Persist system state
        6. Log metrics
        """
        if not self.is_running:
            self.is_running = True
            import time
            self.start_time = time.time()

        # Increment step counter
        self.current_step += 1

        # Activate all agents in the schedule
        self.schedule.step()

        # Compute and distribute global rewards (VDN)
        if self.vdn_engine is not None:
            global_reward = self._compute_global_reward()
            self.global_reward_history.append(global_reward)
            self._distribute_rewards(global_reward)

        # Update HAVEN risk assessment
        if self.haven_coordinator is not None:
            self._update_risk_assessment()

        # Persist model state
        self._save_model_state()

        # Log system metrics
        if self.current_step % self.config.performance.metrics_collection_interval == 0:
            self._log_system_metrics()

        if self.current_step % 100 == 0:
            logger.info("Completed step %d (agents: %d)",
                       self.current_step, len(self.active_agents))

    def _compute_global_reward(self) -> float:
        """
        Compute system-wide global reward.

        Override this method to implement your domain-specific reward function.

        Returns:
            Global reward value
        """
        # Default: sum of all agent rewards
        total_reward = 0.0
        for agent in self.schedule.agents:
            if hasattr(agent, 'last_reward'):
                total_reward += agent.last_reward
        return total_reward

    def _distribute_rewards(self, global_reward: float):
        """
        Distribute global reward to agents using VDN.

        Args:
            global_reward: The system-wide reward to distribute
        """
        for agent in self.schedule.agents:
            if hasattr(agent, 'get_local_reward'):
                try:
                    local_reward = agent.get_local_reward(global_reward)
                    # Agent will use this for learning
                except Exception as e:
                    logger.error("Error distributing reward to %s: %s", agent.agent_id, e)

    def _update_risk_assessment(self):
        """Update HAVEN risk assessment for all agents."""
        if self.haven_coordinator is None:
            return

        try:
            # Assess system risk
            system_risk = self.haven_coordinator.assess_system_risk()

            # Check for contagion
            contagion_report = self.haven_coordinator.detect_policy_contagion()

            # Log if critical
            if system_risk > self.config.haven.risk_threshold:
                self.sql_logger.log_system_event(
                    event_type="high_system_risk",
                    severity="WARNING",
                    description=f"System risk above threshold: {system_risk:.3f}",
                    data={"risk_score": system_risk, "step": self.current_step}
                )

        except Exception as e:
            logger.error("Error in risk assessment: %s", e)

    def _save_model_state(self):
        """
        Save the current model state to Redis and SQLite.

        Persists key model metrics and state information for
        analysis, recovery, and debugging.
        """
        state_data = {
            "step": self.current_step,
            "active_agents": len(self.active_agents),
            "hibernated_agents": len(self.hibernated_agents),
            "total_agents_created": self.total_agents_created,
            "params": self.model_params,
            "is_running": self.is_running
        }

        # Save to Redis (fast, ephemeral)
        self.redis_client.set_key_value(
            "model:state:current",
            state_data
        )

        # Log to SQLite (persistent, every 10 steps)
        if self.current_step % 10 == 0:
            self.sql_logger.log_system_event(
                event_type="model_state",
                severity="INFO",
                description=f"Model state at step {self.current_step}",
                data=state_data
            )

    def _log_system_metrics(self):
        """Log system-wide metrics to SQLite."""
        metrics = {
            "active_agents": len(self.active_agents),
            "hibernated_agents": len(self.hibernated_agents),
            "global_reward": self.global_reward_history[-1] if self.global_reward_history else 0.0,
            "average_reward": sum(self.global_reward_history[-10:]) / max(1, len(self.global_reward_history[-10:])),
        }

        for metric_name, metric_value in metrics.items():
            self.sql_logger.log_performance_metric(
                agent_id="system",
                metric_name=metric_name,
                metric_value=metric_value,
                step=self.current_step
            )

    # =========================================================================
    # Agent Management
    # =========================================================================

    def add_agent(self, agent):
        """
        Add an agent to the model and schedule.

        Args:
            agent: A Mesa Agent instance to add to the simulation
        """
        self.schedule.add(agent)
        self.active_agents[agent.agent_id] = agent
        self.total_agents_created += 1

        # Log agent creation
        self.sql_logger.log_agent_event(
            agent_id=agent.agent_id,
            event_type="agent_created",
            data={"agent_type": agent.agent_type, "unique_id": agent.unique_id},
            step=self.current_step
        )

        logger.debug("Added agent %s to model (total: %d)",
                    agent.agent_id, len(self.active_agents))

    def remove_agent(self, agent):
        """
        Remove an agent from the model and schedule.

        Args:
            agent: The Mesa Agent instance to remove
        """
        self.schedule.remove(agent)
        self.active_agents.pop(agent.agent_id, None)
        self.total_agents_removed += 1

        # Log agent removal
        self.sql_logger.log_agent_event(
            agent_id=agent.agent_id,
            event_type="agent_removed",
            data={"agent_type": agent.agent_type, "reason": "manual_removal"},
            step=self.current_step
        )

        logger.debug("Removed agent %s from model (remaining: %d)",
                    agent.agent_id, len(self.active_agents))

    def get_agent_by_id(self, agent_id: str) -> Optional[Any]:
        """
        Get an agent by its ID.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            Agent instance or None if not found
        """
        return self.active_agents.get(agent_id)

    def get_agent_state_from_redis(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve agent state from Redis.

        Args:
            agent_id: The unique identifier of the agent

        Returns:
            Dictionary containing agent state, or None if not found
        """
        return self.redis_client.get_key_value(f"agent:state:{agent_id}")

    # =========================================================================
    # Engine Integration
    # =========================================================================

    def set_frl_engine(self, frl_engine):
        """
        Set the Federated Reinforcement Learning engine.

        Args:
            frl_engine: FRL engine instance
        """
        self.frl_engine = frl_engine
        logger.info("FRL engine registered with model")

    def set_vdn_engine(self, vdn_engine):
        """
        Set the Value-Decomposition Networks engine.

        Args:
            vdn_engine: VDN engine instance
        """
        self.vdn_engine = vdn_engine
        logger.info("VDN engine registered with model")

    def set_haven_coordinator(self, haven_coordinator):
        """
        Set the HAVEN risk coordinator.

        Args:
            haven_coordinator: HAVEN coordinator instance
        """
        self.haven_coordinator = haven_coordinator
        logger.info("HAVEN coordinator registered with model")

    # =========================================================================
    # Simulation Control
    # =========================================================================

    def run(self, num_steps: int):
        """
        Run the model for a specified number of steps.

        Args:
            num_steps: Number of simulation steps to execute
        """
        logger.info("Starting simulation run for %d steps", num_steps)

        try:
            for _ in range(num_steps):
                self.step()

            logger.info("Simulation run completed successfully")

        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user at step %d", self.current_step)
            self.shutdown()

        except Exception as e:
            logger.error("Simulation error at step %d: %s", self.current_step, e, exc_info=True)
            self.shutdown()
            raise

    def shutdown(self):
        """
        Gracefully shutdown the model and clean up resources.
        """
        logger.info("Shutting down MycelialModel...")

        self.is_running = False

        # Save final state
        self._save_model_state()

        # Close data backbone connections
        if self.sql_logger:
            self.sql_logger.stop(timeout=10.0)

        if self.vector_db:
            self.vector_db.close()

        if self.redis_client:
            self.redis_client.close()

        logger.info("MycelialModel shutdown complete")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with system metrics
        """
        import time

        runtime = time.time() - self.start_time if self.start_time else 0

        return {
            "current_step": self.current_step,
            "runtime_seconds": runtime,
            "steps_per_second": self.current_step / runtime if runtime > 0 else 0,
            "active_agents": len(self.active_agents),
            "hibernated_agents": len(self.hibernated_agents),
            "total_agents_created": self.total_agents_created,
            "total_agents_removed": self.total_agents_removed,
            "global_reward_avg": sum(self.global_reward_history[-100:]) / max(1, len(self.global_reward_history[-100:])),
            "engines_enabled": {
                "frl": self.frl_engine is not None,
                "vdn": self.vdn_engine is not None,
                "haven": self.haven_coordinator is not None
            }
        }

    def __del__(self):
        """Cleanup on deletion."""
        if self.is_running:
            self.shutdown()
