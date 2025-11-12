"""
Dynamic Agent Builder Abstract Base Class

This module defines the interface for dynamically spawning and hibernating
agents at runtime. This enables the MAE system to adapt its agent population
based on workload, performance, and resource constraints.

Philosophy:
Like a mycelial network that extends or retracts based on environmental
conditions, the MAE system can grow new agents when needed and hibernate
dormant ones to conserve resources. This creates an elastic, self-adapting
agent ecosystem.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States an agent can be in."""
    INITIALIZING = "initializing"  # Being created
    ACTIVE = "active"               # Running and processing
    IDLE = "idle"                   # Active but no work
    HIBERNATING = "hibernating"     # Suspended, state saved
    TERMINATED = "terminated"       # Permanently removed


class SpawnTrigger(Enum):
    """Reasons for spawning new agents."""
    WORKLOAD_SPIKE = "workload_spike"          # High data volume
    PERFORMANCE_DROP = "performance_drop"       # System underperforming
    MANUAL = "manual"                          # User requested
    SCHEDULED = "scheduled"                    # Planned scaling
    DISCOVERY = "discovery"                    # New task discovered
    REPLICATION = "replication"                # High-performing agent replicates


class HibernationTrigger(Enum):
    """Reasons for hibernating agents."""
    LOW_WORKLOAD = "low_workload"              # Not enough work
    POOR_PERFORMANCE = "poor_performance"       # Agent underperforming
    RESOURCE_PRESSURE = "resource_pressure"     # System resource constraints
    MANUAL = "manual"                          # User requested
    SCHEDULED = "scheduled"                    # Planned downscaling
    REDUNDANCY = "redundancy"                  # Too many similar agents


@dataclass
class AgentBlueprint:
    """Blueprint for creating a new agent."""
    agent_type: str                    # Class name or type identifier
    agent_class: Type                  # Actual class to instantiate
    config: Dict[str, Any]             # Agent-specific configuration
    priority: int = 0                  # Spawn priority (higher = sooner)
    parent_agent_id: Optional[str] = None  # If replicated from parent
    spawn_trigger: SpawnTrigger = SpawnTrigger.MANUAL


@dataclass
class AgentSnapshot:
    """Snapshot of hibernated agent state."""
    agent_id: str
    agent_type: str
    state: Dict[str, Any]              # Complete agent state
    hibernation_time: float            # When hibernated
    hibernation_trigger: HibernationTrigger
    memory_size_bytes: int             # Size of saved state
    can_restore: bool = True           # Whether restoration is possible


class DynamicAgentBuilder(ABC):
    """
    Abstract base class for dynamic agent creation and hibernation.

    This builder enables the MAE system to elastically scale its agent
    population based on runtime conditions. It provides:
    - Dynamic agent spawning with various triggers
    - Agent hibernation to conserve resources
    - State persistence for hibernated agents
    - Automatic scaling based on system metrics
    - Resource-aware agent management

    Key Principles:
    - Agents can be created and destroyed at runtime
    - Hibernated agents preserve their learned state
    - System automatically balances agent population
    - Resource constraints are respected
    - Learning is not lost when agents hibernate
    """

    def __init__(
        self,
        model,
        redis_client,
        vector_db,
        sql_logger,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Dynamic Agent Builder.

        Args:
            model: The MycelialModel instance
            redis_client: Redis client for state storage
            vector_db: Vector database for policy storage
            sql_logger: SQLite logger for event logging
            config: Optional configuration dictionary
        """
        self.model = model
        self.redis_client = redis_client
        self.vector_db = vector_db
        self.sql_logger = sql_logger
        self.config = config or {}

        # Agent tracking
        self.agent_blueprints: Dict[str, AgentBlueprint] = {}
        self.spawn_queue: List[AgentBlueprint] = []
        self.hibernation_snapshots: Dict[str, AgentSnapshot] = {}

        # Metrics
        self.total_spawned = 0
        self.total_hibernated = 0
        self.total_restored = 0
        self.total_terminated = 0

        # Configuration
        self.max_agents = config.get("max_agents", 100) if config else 100
        self.min_agents = config.get("min_agents", 1) if config else 1
        self.auto_scaling_enabled = config.get("auto_scaling_enabled", True) if config else True

        logger.info("DynamicAgentBuilder initialized (max: %d, min: %d)",
                   self.max_agents, self.min_agents)

    # =========================================================================
    # Abstract Methods - Must Implement
    # =========================================================================

    @abstractmethod
    def spawn_agent(
        self,
        blueprint: AgentBlueprint
    ) -> Any:
        """
        Spawn a new agent from a blueprint.

        This method creates a new agent instance, initializes it,
        and adds it to the model.

        Args:
            blueprint: AgentBlueprint defining the agent to create

        Returns:
            The created agent instance

        Raises:
            RuntimeError: If spawning fails
        """
        pass

    @abstractmethod
    def hibernate_agent(
        self,
        agent_id: str,
        trigger: HibernationTrigger
    ) -> AgentSnapshot:
        """
        Hibernate an agent, saving its state.

        This method removes the agent from active execution,
        saves its complete state for later restoration, and
        frees resources.

        Args:
            agent_id: ID of the agent to hibernate
            trigger: Reason for hibernation

        Returns:
            AgentSnapshot containing saved state

        Raises:
            ValueError: If agent not found or already hibernated
        """
        pass

    @abstractmethod
    def restore_agent(
        self,
        snapshot_id: str
    ) -> Any:
        """
        Restore a hibernated agent from snapshot.

        This method recreates an agent from saved state,
        restoring all learned policies and memories.

        Args:
            snapshot_id: ID of the snapshot to restore

        Returns:
            The restored agent instance

        Raises:
            ValueError: If snapshot not found or invalid
        """
        pass

    @abstractmethod
    def terminate_agent(
        self,
        agent_id: str,
        preserve_state: bool = True
    ):
        """
        Permanently terminate an agent.

        Unlike hibernation, termination removes the agent
        and optionally its state permanently.

        Args:
            agent_id: ID of the agent to terminate
            preserve_state: Whether to keep state for analysis
        """
        pass

    @abstractmethod
    def should_spawn_agent(
        self,
        system_metrics: Dict[str, Any]
    ) -> Tuple[bool, Optional[SpawnTrigger]]:
        """
        Determine if a new agent should be spawned.

        Analyzes system metrics to decide if agent population
        should grow.

        Args:
            system_metrics: Current system performance metrics

        Returns:
            Tuple of (should_spawn, trigger_reason)
        """
        pass

    @abstractmethod
    def should_hibernate_agent(
        self,
        agent_id: str,
        agent_metrics: Dict[str, Any],
        system_metrics: Dict[str, Any]
    ) -> Tuple[bool, Optional[HibernationTrigger]]:
        """
        Determine if an agent should be hibernated.

        Analyzes agent and system metrics to decide if
        agent should be suspended.

        Args:
            agent_id: ID of the agent to evaluate
            agent_metrics: Agent's performance metrics
            system_metrics: System-wide metrics

        Returns:
            Tuple of (should_hibernate, trigger_reason)
        """
        pass

    @abstractmethod
    def select_agent_to_spawn(
        self,
        trigger: SpawnTrigger
    ) -> Optional[AgentBlueprint]:
        """
        Select which type of agent to spawn.

        Based on the trigger and current system state,
        decide what agent type would be most beneficial.

        Args:
            trigger: Reason for spawning

        Returns:
            AgentBlueprint for the agent to spawn, or None
        """
        pass

    @abstractmethod
    def select_agent_to_hibernate(
        self,
        trigger: HibernationTrigger
    ) -> Optional[str]:
        """
        Select which agent to hibernate.

        Based on the trigger and agent metrics,
        choose an agent to suspend.

        Args:
            trigger: Reason for hibernation

        Returns:
            Agent ID to hibernate, or None
        """
        pass

    @abstractmethod
    def clone_agent(
        self,
        source_agent_id: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Clone an existing agent with optional modifications.

        Creates a copy of a high-performing agent,
        optionally with parameter variations for diversity.

        Args:
            source_agent_id: ID of agent to clone
            modifications: Optional parameter modifications

        Returns:
            The cloned agent instance
        """
        pass

    # =========================================================================
    # State Management
    # =========================================================================

    @abstractmethod
    def save_agent_state(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Save complete agent state.

        Captures all agent attributes, learned parameters,
        and memories for later restoration.

        Args:
            agent_id: ID of agent to save

        Returns:
            Dictionary containing complete agent state
        """
        pass

    @abstractmethod
    def load_agent_state(
        self,
        agent_id: str,
        state: Dict[str, Any]
    ):
        """
        Load agent state from saved data.

        Restores all agent attributes to previously saved values.

        Args:
            agent_id: ID of agent to restore
            state: Saved state dictionary
        """
        pass

    # =========================================================================
    # Resource Management
    # =========================================================================

    @abstractmethod
    def check_resource_constraints(self) -> Dict[str, bool]:
        """
        Check if system resources allow new agents.

        Evaluates memory, CPU, and other constraints.

        Returns:
            Dictionary of constraint checks (e.g., {"memory_ok": True})
        """
        pass

    @abstractmethod
    def estimate_agent_resource_cost(
        self,
        blueprint: AgentBlueprint
    ) -> Dict[str, float]:
        """
        Estimate resources required for an agent.

        Args:
            blueprint: Agent blueprint to evaluate

        Returns:
            Dictionary with estimated costs (e.g., {"memory_mb": 50})
        """
        pass

    # =========================================================================
    # Auto-Scaling
    # =========================================================================

    @abstractmethod
    def evaluate_scaling_needs(
        self,
        system_metrics: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Evaluate if system should scale up or down.

        Args:
            system_metrics: Current system state

        Returns:
            Dictionary with scaling decisions:
            - agents_to_spawn: int
            - agents_to_hibernate: int
        """
        pass

    @abstractmethod
    def apply_scaling_decision(
        self,
        decision: Dict[str, int]
    ):
        """
        Apply auto-scaling decision.

        Spawns or hibernates agents based on decision.

        Args:
            decision: Scaling decision from evaluate_scaling_needs
        """
        pass

    # =========================================================================
    # Blueprint Management
    # =========================================================================

    def register_agent_blueprint(
        self,
        blueprint_id: str,
        blueprint: AgentBlueprint
    ):
        """
        Register an agent blueprint for later use.

        Args:
            blueprint_id: Unique identifier for this blueprint
            blueprint: The agent blueprint
        """
        self.agent_blueprints[blueprint_id] = blueprint
        logger.debug("Registered blueprint: %s", blueprint_id)

    def get_blueprint(self, blueprint_id: str) -> Optional[AgentBlueprint]:
        """
        Get a registered blueprint by ID.

        Args:
            blueprint_id: Blueprint identifier

        Returns:
            AgentBlueprint or None if not found
        """
        return self.agent_blueprints.get(blueprint_id)

    def list_blueprints(self) -> List[str]:
        """
        List all registered blueprint IDs.

        Returns:
            List of blueprint identifiers
        """
        return list(self.agent_blueprints.keys())

    # =========================================================================
    # Hibernation Management
    # =========================================================================

    def get_hibernated_agents(self) -> List[str]:
        """
        Get list of hibernated agent IDs.

        Returns:
            List of snapshot IDs
        """
        return list(self.hibernation_snapshots.keys())

    def get_hibernation_snapshot(
        self,
        snapshot_id: str
    ) -> Optional[AgentSnapshot]:
        """
        Get a hibernation snapshot by ID.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            AgentSnapshot or None if not found
        """
        return self.hibernation_snapshots.get(snapshot_id)

    def can_restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Check if a snapshot can be restored.

        Args:
            snapshot_id: Snapshot to check

        Returns:
            True if snapshot is valid and restorable
        """
        snapshot = self.get_hibernation_snapshot(snapshot_id)
        return snapshot is not None and snapshot.can_restore

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_builder_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about agent building operations.

        Returns:
            Dictionary with builder metrics
        """
        return {
            "total_spawned": self.total_spawned,
            "total_hibernated": self.total_hibernated,
            "total_restored": self.total_restored,
            "total_terminated": self.total_terminated,
            "active_agents": len(self.model.active_agents),
            "hibernated_agents": len(self.hibernation_snapshots),
            "blueprints_registered": len(self.agent_blueprints),
            "spawn_queue_size": len(self.spawn_queue),
            "auto_scaling_enabled": self.auto_scaling_enabled
        }

    # =========================================================================
    # Lifecycle Hooks
    # =========================================================================

    def on_agent_spawned(self, agent):
        """
        Hook called after agent is spawned.

        Args:
            agent: The newly spawned agent
        """
        self.total_spawned += 1

        self.sql_logger.log_agent_event(
            agent_id=agent.agent_id,
            event_type="agent_spawned",
            data={
                "agent_type": agent.agent_type,
                "total_spawned": self.total_spawned
            },
            step=self.model.current_step
        )

    def on_agent_hibernated(self, agent_id: str, snapshot: AgentSnapshot):
        """
        Hook called after agent is hibernated.

        Args:
            agent_id: ID of hibernated agent
            snapshot: The hibernation snapshot
        """
        self.total_hibernated += 1
        self.hibernation_snapshots[agent_id] = snapshot

        self.sql_logger.log_agent_event(
            agent_id=agent_id,
            event_type="agent_hibernated",
            data={
                "trigger": snapshot.hibernation_trigger.value,
                "memory_size_bytes": snapshot.memory_size_bytes,
                "total_hibernated": self.total_hibernated
            },
            step=self.model.current_step
        )

    def on_agent_restored(self, agent_id: str):
        """
        Hook called after agent is restored.

        Args:
            agent_id: ID of restored agent
        """
        self.total_restored += 1
        self.hibernation_snapshots.pop(agent_id, None)

        self.sql_logger.log_agent_event(
            agent_id=agent_id,
            event_type="agent_restored",
            data={"total_restored": self.total_restored},
            step=self.model.current_step
        )

    def on_agent_terminated(self, agent_id: str):
        """
        Hook called after agent is terminated.

        Args:
            agent_id: ID of terminated agent
        """
        self.total_terminated += 1

        self.sql_logger.log_agent_event(
            agent_id=agent_id,
            event_type="agent_terminated",
            data={"total_terminated": self.total_terminated},
            step=self.model.current_step
        )
