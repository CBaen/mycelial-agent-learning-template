"""
Builder Agent for the Mycelial ABM Framework

This agent implements the DynamicAgentBuilder interface to provide runtime
agent spawning, hibernation, and lifecycle management capabilities.
"""

from typing import Dict, Any, Optional, List, Type, Tuple
import logging
import time
import pickle
import psutil

from agents.base_agent import MycelialAgent
from core.builder_base import (
    DynamicAgentBuilder,
    AgentState,
    SpawnTrigger,
    HibernationTrigger,
    AgentBlueprint,
    AgentSnapshot
)

logger = logging.getLogger(__name__)


class BuilderAgent(DynamicAgentBuilder, MycelialAgent):
    """
    Builder Agent that manages dynamic agent lifecycle.

    This agent combines the DynamicAgentBuilder abstract interface with
    MycelialAgent to provide both agent capabilities and builder capabilities:
    - Runtime agent spawning from blueprints
    - Agent hibernation with state preservation
    - Agent restoration from snapshots
    - Agent cloning and replication
    - Automatic scaling based on system metrics
    - Resource-aware agent management

    The Builder Agent acts as a "gardener" for the mycelial network,
    growing new agents when needed and pruning dormant ones.
    """

    def __init__(
        self,
        unique_id: int,
        model,
        redis_client,
        vector_db,
        sql_logger,
        team_id: str = "builders",
        agent_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Builder Agent.

        Args:
            unique_id: Unique identifier for this agent
            model: The Mesa model this agent belongs to
            redis_client: Redis client for data operations
            vector_db: Vector database for policy storage
            sql_logger: SQLite logger for event logging
            team_id: Team identifier (default: "builders")
            agent_config: Optional configuration dictionary
        """
        # Initialize MycelialAgent
        MycelialAgent.__init__(self, unique_id, model, redis_client, team_id, agent_config)

        # Initialize DynamicAgentBuilder
        DynamicAgentBuilder.__init__(
            self,
            model=model,
            redis_client=redis_client,
            vector_db=vector_db,
            sql_logger=sql_logger,
            config=agent_config
        )

        # Builder-specific configuration
        self.scaling_check_interval = agent_config.get("scaling_check_interval", 10) if agent_config else 10
        self.spawn_cooldown = agent_config.get("spawn_cooldown", 5) if agent_config else 5
        self.last_spawn_time: int = 0

        # Metrics
        self.spawn_requests: int = 0
        self.successful_spawns: int = 0
        self.failed_spawns: int = 0

        logger.info("BuilderAgent %s initialized (max_agents: %d, min_agents: %d)",
                   self.agent_id, self.max_agents, self.min_agents)

    def step(self):
        """
        Execute one step of builder agent behavior.

        Flow:
        1. Process spawn queue
        2. Check for auto-scaling needs
        3. Monitor resource constraints
        4. Update metrics
        """
        self.step_count += 1

        # Process spawn queue
        if self.spawn_queue and self._can_spawn():
            self._process_spawn_queue()

        # Auto-scaling check
        if self.auto_scaling_enabled and self.step_count % self.scaling_check_interval == 0:
            self._perform_auto_scaling()

        # Resource monitoring
        resource_status = self.check_resource_constraints()
        self.risk_score = 1.0 - float(resource_status.get("memory_ok", True))

        # Reward based on system health
        active_agent_count = len(self.model.active_agents)
        target_count = (self.max_agents + self.min_agents) / 2
        deviation = abs(active_agent_count - target_count) / target_count
        self.last_reward = 1.0 - min(1.0, deviation)
        self.cumulative_reward += self.last_reward

        # Update performance metrics
        self._update_performance_metrics(self.last_reward)

        # Save state periodically
        if self.step_count % 20 == 0:
            self._save_state_to_redis()

        logger.debug("%s completed step %d (active agents: %d)",
                    self.agent_id, self.step_count, active_agent_count)

    # =========================================================================
    # DynamicAgentBuilder Implementation
    # =========================================================================

    def spawn_agent(self, blueprint: AgentBlueprint) -> Any:
        """
        Spawn a new agent from a blueprint.

        Args:
            blueprint: AgentBlueprint defining the agent to create

        Returns:
            The created agent instance

        Raises:
            RuntimeError: If spawning fails
        """
        self.spawn_requests += 1

        try:
            # Check resource constraints
            resource_status = self.check_resource_constraints()
            if not all(resource_status.values()):
                raise RuntimeError(f"Resource constraints violated: {resource_status}")

            # Check agent limit
            if len(self.model.active_agents) >= self.max_agents:
                raise RuntimeError(f"Agent limit reached: {self.max_agents}")

            # Get next unique ID
            next_id = self.model.total_agents_created + 1

            # Instantiate agent
            agent = blueprint.agent_class(
                unique_id=next_id,
                model=self.model,
                redis_client=self.redis_client,
                **blueprint.config
            )

            # Add to model
            self.model.add_agent(agent)

            # Update tracking
            self.successful_spawns += 1
            self.last_spawn_time = self.step_count

            # Call lifecycle hook
            self.on_agent_spawned(agent)

            logger.info("%s spawned agent %s (trigger: %s)",
                       self.agent_id, agent.agent_id, blueprint.spawn_trigger.value)

            return agent

        except Exception as e:
            self.failed_spawns += 1
            logger.error("%s failed to spawn agent: %s", self.agent_id, e)
            raise RuntimeError(f"Failed to spawn agent: {e}")

    def hibernate_agent(
        self,
        agent_id: str,
        trigger: HibernationTrigger
    ) -> AgentSnapshot:
        """
        Hibernate an agent, saving its state.

        Args:
            agent_id: ID of the agent to hibernate
            trigger: Reason for hibernation

        Returns:
            AgentSnapshot containing saved state

        Raises:
            ValueError: If agent not found or already hibernated
        """
        # Get agent from model
        agent = self.model.get_agent_by_id(agent_id)
        if agent is None:
            raise ValueError(f"Agent not found: {agent_id}")

        if agent_id in self.hibernation_snapshots:
            raise ValueError(f"Agent already hibernated: {agent_id}")

        try:
            # Save complete state
            state = self.save_agent_state(agent_id)

            # Calculate state size
            state_bytes = len(pickle.dumps(state))

            # Create snapshot
            snapshot = AgentSnapshot(
                agent_id=agent_id,
                agent_type=agent.agent_type,
                state=state,
                hibernation_time=time.time(),
                hibernation_trigger=trigger,
                memory_size_bytes=state_bytes,
                can_restore=True
            )

            # Remove agent from model
            self.model.remove_agent(agent)

            # Call lifecycle hook
            self.on_agent_hibernated(agent_id, snapshot)

            logger.info("%s hibernated agent %s (trigger: %s, size: %d bytes)",
                       self.agent_id, agent_id, trigger.value, state_bytes)

            return snapshot

        except Exception as e:
            logger.error("%s failed to hibernate agent %s: %s", self.agent_id, agent_id, e)
            raise

    def restore_agent(self, snapshot_id: str) -> Any:
        """
        Restore a hibernated agent from snapshot.

        Args:
            snapshot_id: ID of the snapshot to restore

        Returns:
            The restored agent instance

        Raises:
            ValueError: If snapshot not found or invalid
        """
        snapshot = self.get_hibernation_snapshot(snapshot_id)
        if snapshot is None:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        if not snapshot.can_restore:
            raise ValueError(f"Snapshot cannot be restored: {snapshot_id}")

        try:
            # Get agent class from snapshot
            agent_type = snapshot.agent_type

            # Import agent class dynamically
            # In practice, you'd have a registry of agent classes
            from agents.specialist_agent import SpecialistAgent
            from agents.data_miner_agent import DataMinerAgent
            from agents.risk_manager_agent import RiskManagerAgent

            agent_class_map = {
                "SpecialistAgent": SpecialistAgent,
                "DataMinerAgent": DataMinerAgent,
                "RiskManagerAgent": RiskManagerAgent
            }

            agent_class = agent_class_map.get(agent_type)
            if agent_class is None:
                raise ValueError(f"Unknown agent type: {agent_type}")

            # Create blueprint from snapshot
            blueprint = AgentBlueprint(
                agent_type=agent_type,
                agent_class=agent_class,
                config=snapshot.state.get("config", {}),
                spawn_trigger=SpawnTrigger.MANUAL
            )

            # Spawn new agent
            agent = self.spawn_agent(blueprint)

            # Load saved state
            self.load_agent_state(agent.agent_id, snapshot.state)

            # Call lifecycle hook
            self.on_agent_restored(snapshot_id)

            logger.info("%s restored agent %s from snapshot",
                       self.agent_id, agent.agent_id)

            return agent

        except Exception as e:
            logger.error("%s failed to restore agent from snapshot %s: %s",
                        self.agent_id, snapshot_id, e)
            raise

    def terminate_agent(
        self,
        agent_id: str,
        preserve_state: bool = True
    ):
        """
        Permanently terminate an agent.

        Args:
            agent_id: ID of the agent to terminate
            preserve_state: Whether to keep state for analysis
        """
        agent = self.model.get_agent_by_id(agent_id)
        if agent is None:
            logger.warning("%s cannot terminate agent (not found): %s",
                          self.agent_id, agent_id)
            return

        try:
            # Save state if requested
            if preserve_state:
                state = self.save_agent_state(agent_id)
                # Store in Redis for analysis
                self.redis_client.set_key_value(
                    f"agent:terminated:{agent_id}",
                    state
                )

            # Remove from model
            self.model.remove_agent(agent)

            # Call lifecycle hook
            self.on_agent_terminated(agent_id)

            logger.info("%s terminated agent %s (preserve_state: %s)",
                       self.agent_id, agent_id, preserve_state)

        except Exception as e:
            logger.error("%s failed to terminate agent %s: %s",
                        self.agent_id, agent_id, e)

    def should_spawn_agent(
        self,
        system_metrics: Dict[str, Any]
    ) -> Tuple[bool, Optional[SpawnTrigger]]:
        """
        Determine if a new agent should be spawned.

        Args:
            system_metrics: Current system performance metrics

        Returns:
            Tuple of (should_spawn, trigger_reason)
        """
        # Check workload
        workload = system_metrics.get("workload", 0.0)
        if workload > 0.8:
            return (True, SpawnTrigger.WORKLOAD_SPIKE)

        # Check performance
        avg_performance = system_metrics.get("average_performance", 1.0)
        if avg_performance < 0.5 and len(self.model.active_agents) < self.max_agents:
            return (True, SpawnTrigger.PERFORMANCE_DROP)

        # Check agent count
        if len(self.model.active_agents) < self.min_agents:
            return (True, SpawnTrigger.SCHEDULED)

        return (False, None)

    def should_hibernate_agent(
        self,
        agent_id: str,
        agent_metrics: Dict[str, Any],
        system_metrics: Dict[str, Any]
    ) -> Tuple[bool, Optional[HibernationTrigger]]:
        """
        Determine if an agent should be hibernated.

        Args:
            agent_id: ID of the agent to evaluate
            agent_metrics: Agent's performance metrics
            system_metrics: System-wide metrics

        Returns:
            Tuple of (should_hibernate, trigger_reason)
        """
        # Check performance
        agent_performance = agent_metrics.get("recent_performance", 0.0)
        if agent_performance < 0.1 and len(self.model.active_agents) > self.min_agents:
            return (True, HibernationTrigger.POOR_PERFORMANCE)

        # Check workload
        system_workload = system_metrics.get("workload", 1.0)
        if system_workload < 0.2 and len(self.model.active_agents) > self.min_agents:
            return (True, HibernationTrigger.LOW_WORKLOAD)

        # Check resources
        resource_status = self.check_resource_constraints()
        if not resource_status.get("memory_ok", True):
            return (True, HibernationTrigger.RESOURCE_PRESSURE)

        return (False, None)

    def select_agent_to_spawn(
        self,
        trigger: SpawnTrigger
    ) -> Optional[AgentBlueprint]:
        """
        Select which type of agent to spawn.

        Args:
            trigger: Reason for spawning

        Returns:
            AgentBlueprint for the agent to spawn, or None
        """
        # Import agent classes
        from agents.specialist_agent import SpecialistAgent

        # Default: spawn a specialist agent
        config = {
            "data_channel": "abm:processed_data",
            "team_id": "specialists",
            "agent_config": {
                "learning_rate": 0.01,
                "exploration_rate": 0.1
            }
        }

        blueprint = AgentBlueprint(
            agent_type="SpecialistAgent",
            agent_class=SpecialistAgent,
            config=config,
            priority=1,
            spawn_trigger=trigger
        )

        return blueprint

    def select_agent_to_hibernate(
        self,
        trigger: HibernationTrigger
    ) -> Optional[str]:
        """
        Select which agent to hibernate.

        Args:
            trigger: Reason for hibernation

        Returns:
            Agent ID to hibernate, or None
        """
        if not self.model.active_agents:
            return None

        # Find lowest performing agent
        worst_agent_id = None
        worst_performance = float('inf')

        for agent_id, agent in self.model.active_agents.items():
            if hasattr(agent, '_get_recent_performance'):
                performance = agent._get_recent_performance()
                if performance < worst_performance:
                    worst_performance = performance
                    worst_agent_id = agent_id

        return worst_agent_id

    def clone_agent(
        self,
        source_agent_id: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Clone an existing agent with optional modifications.

        Args:
            source_agent_id: ID of agent to clone
            modifications: Optional parameter modifications

        Returns:
            The cloned agent instance
        """
        source_agent = self.model.get_agent_by_id(source_agent_id)
        if source_agent is None:
            raise ValueError(f"Source agent not found: {source_agent_id}")

        try:
            # Save source agent state
            source_state = self.save_agent_state(source_agent_id)

            # Apply modifications
            if modifications:
                source_state.update(modifications)

            # Create blueprint
            blueprint = AgentBlueprint(
                agent_type=source_agent.agent_type,
                agent_class=type(source_agent),
                config=source_state.get("config", {}),
                parent_agent_id=source_agent_id,
                spawn_trigger=SpawnTrigger.REPLICATION
            )

            # Spawn clone
            clone = self.spawn_agent(blueprint)

            # Load modified state
            self.load_agent_state(clone.agent_id, source_state)

            logger.info("%s cloned agent %s to %s",
                       self.agent_id, source_agent_id, clone.agent_id)

            return clone

        except Exception as e:
            logger.error("%s failed to clone agent %s: %s",
                        self.agent_id, source_agent_id, e)
            raise

    # =========================================================================
    # State Management
    # =========================================================================

    def save_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """
        Save complete agent state.

        Args:
            agent_id: ID of agent to save

        Returns:
            Dictionary containing complete agent state
        """
        agent = self.model.get_agent_by_id(agent_id)
        if agent is None:
            raise ValueError(f"Agent not found: {agent_id}")

        # Collect all relevant state
        state = {
            "agent_id": agent_id,
            "agent_type": agent.agent_type,
            "team_id": agent.team_id,
            "unique_id": agent.unique_id,
            "step_count": agent.step_count,
            "cumulative_reward": agent.cumulative_reward,
            "policy_version": agent.policy_version,
            "policy_parameters": agent.policy_parameters,
            "risk_score": agent.risk_score,
            "is_isolated": agent.is_isolated,
            "performance_history": agent.performance_history.copy(),
            "config": agent.agent_config
        }

        return state

    def load_agent_state(
        self,
        agent_id: str,
        state: Dict[str, Any]
    ):
        """
        Load agent state from saved data.

        Args:
            agent_id: ID of agent to restore
            state: Saved state dictionary
        """
        agent = self.model.get_agent_by_id(agent_id)
        if agent is None:
            raise ValueError(f"Agent not found: {agent_id}")

        # Restore state
        agent.step_count = state.get("step_count", 0)
        agent.cumulative_reward = state.get("cumulative_reward", 0.0)
        agent.policy_version = state.get("policy_version", 0)
        agent.policy_parameters = state.get("policy_parameters", {})
        agent.risk_score = state.get("risk_score", 0.0)
        agent.is_isolated = state.get("is_isolated", False)
        agent.performance_history = state.get("performance_history", [])

        logger.debug("%s loaded state for agent %s", self.agent_id, agent_id)

    # =========================================================================
    # Resource Management
    # =========================================================================

    def check_resource_constraints(self) -> Dict[str, bool]:
        """
        Check if system resources allow new agents.

        Returns:
            Dictionary of constraint checks
        """
        # Check memory
        memory = psutil.virtual_memory()
        memory_ok = memory.percent < 85.0

        # Check agent count
        count_ok = len(self.model.active_agents) < self.max_agents

        return {
            "memory_ok": memory_ok,
            "count_ok": count_ok,
            "can_spawn": memory_ok and count_ok
        }

    def estimate_agent_resource_cost(
        self,
        blueprint: AgentBlueprint
    ) -> Dict[str, float]:
        """
        Estimate resources required for an agent.

        Args:
            blueprint: Agent blueprint to evaluate

        Returns:
            Dictionary with estimated costs
        """
        # Rough estimates (would be refined based on actual measurements)
        base_memory_mb = 10.0
        per_policy_param_kb = 0.1

        return {
            "memory_mb": base_memory_mb,
            "cpu_percent": 1.0
        }

    # =========================================================================
    # Auto-Scaling
    # =========================================================================

    def evaluate_scaling_needs(
        self,
        system_metrics: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Evaluate if system should scale up or down.

        Args:
            system_metrics: Current system state

        Returns:
            Dictionary with scaling decisions
        """
        agents_to_spawn = 0
        agents_to_hibernate = 0

        # Check if we should spawn
        should_spawn, spawn_trigger = self.should_spawn_agent(system_metrics)
        if should_spawn and self._can_spawn():
            agents_to_spawn = 1

        # Check if we should hibernate
        for agent_id in list(self.model.active_agents.keys()):
            agent = self.model.get_agent_by_id(agent_id)
            if agent and hasattr(agent, '_get_recent_performance'):
                agent_metrics = {
                    "recent_performance": agent._get_recent_performance()
                }
                should_hibernate, _ = self.should_hibernate_agent(
                    agent_id, agent_metrics, system_metrics
                )
                if should_hibernate:
                    agents_to_hibernate += 1
                    break  # Hibernate one at a time

        return {
            "agents_to_spawn": agents_to_spawn,
            "agents_to_hibernate": agents_to_hibernate
        }

    def apply_scaling_decision(
        self,
        decision: Dict[str, int]
    ):
        """
        Apply auto-scaling decision.

        Args:
            decision: Scaling decision from evaluate_scaling_needs
        """
        # Spawn agents
        for _ in range(decision.get("agents_to_spawn", 0)):
            blueprint = self.select_agent_to_spawn(SpawnTrigger.SCHEDULED)
            if blueprint:
                try:
                    self.spawn_agent(blueprint)
                except Exception as e:
                    logger.error("%s failed to spawn during scaling: %s",
                                self.agent_id, e)

        # Hibernate agents
        for _ in range(decision.get("agents_to_hibernate", 0)):
            agent_id = self.select_agent_to_hibernate(HibernationTrigger.SCHEDULED)
            if agent_id:
                try:
                    self.hibernate_agent(agent_id, HibernationTrigger.SCHEDULED)
                except Exception as e:
                    logger.error("%s failed to hibernate during scaling: %s",
                                self.agent_id, e)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _can_spawn(self) -> bool:
        """Check if agent can spawn (cooldown check)."""
        return (self.step_count - self.last_spawn_time) >= self.spawn_cooldown

    def _process_spawn_queue(self):
        """Process pending spawn requests from queue."""
        if not self.spawn_queue:
            return

        # Sort by priority
        self.spawn_queue.sort(key=lambda bp: bp.priority, reverse=True)

        # Process highest priority
        blueprint = self.spawn_queue.pop(0)

        try:
            self.spawn_agent(blueprint)
        except Exception as e:
            logger.error("%s failed to process spawn queue: %s", self.agent_id, e)

    def _perform_auto_scaling(self):
        """Perform auto-scaling evaluation and execution."""
        system_metrics = {
            "workload": 0.5,  # Would be computed from actual system state
            "average_performance": 0.7,
            "active_agents": len(self.model.active_agents)
        }

        decision = self.evaluate_scaling_needs(system_metrics)
        self.apply_scaling_decision(decision)

    def _save_state_to_redis(self):
        """Save builder agent state to Redis."""
        # Call parent method
        super()._save_state_to_redis()

        # Save builder-specific state
        builder_state = {
            "spawn_requests": self.spawn_requests,
            "successful_spawns": self.successful_spawns,
            "failed_spawns": self.failed_spawns,
            "total_spawned": self.total_spawned,
            "total_hibernated": self.total_hibernated,
            "total_restored": self.total_restored,
            "active_agents": len(self.model.active_agents),
            "hibernated_agents": len(self.hibernation_snapshots)
        }

        key = f"agent:builder_state:{self.agent_id}"
        self.redis_client.set_key_value(key, builder_state)
