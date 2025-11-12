"""
Base Mycelial Agent for the MAE Framework

This module provides the foundational MycelialAgent class that all agents
inherit from. It implements the 'Rule of 3' team-based collaboration pattern.

Rule of 3:
- Agents are organized into teams (team_id)
- Teams collaborate through shared policies (Vector DB)
- Inter-team learning via Federated Learning (FRL)
"""

from mesa import Agent
from typing import Dict, Any, Optional, List, Callable, Tuple, Set
from collections import defaultdict
import logging
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# BIG ROCK 5: ELECTRICAL SIGNALING LAYER
from src.core.electrical_signal import ElectricalSignalBus, Signal, SignalPriority
from src.core.signal_types import SignalType

# BIG ROCK 6: STIGMERGIC ENVIRONMENT
from src.core.stigmergy import StigmergicEnvironment, StigmergicMarker
from src.core.marker_types import MarkerType

# BIG ROCK 7: GNN COMMUNICATION
from src.core.gnn_communicator import GNNCommunicator
from src.core.gnn_message import GNNMessage, MessageType

# BIG ROCK 8: TRANSFER LEARNING & META-LEARNING
from src.core.task_representation import TaskDescriptor, TaskEmbedding
from src.core.knowledge_base import KnowledgeBase, Episode, ExperienceTransition
from src.core.transfer_learning import TransferLearningEngine, TransferStrategy, TransferResult
from src.core.maml import MAMLLearner, MAMLConfig, AdaptationResult

# BIG ROCK 9: EPISODIC MEMORY & REPLAY
from src.memory.episodic_memory import EpisodicMemory, Experience
from src.memory.memory_consolidator import MemoryConsolidator, ConsolidationStrategy, ConsolidationResult
from src.memory.semantic_retriever import SemanticRetriever, SemanticQuery

logger = logging.getLogger(__name__)


class MycelialAgent(Agent):
    """
    Base agent class for the Mycelial Agent Engine (MAE).

    This agent extends Mesa's Agent class with capabilities for:
    - **Team-based collaboration** (Rule of 3)
    - Federated policy sharing (FRL)
    - Value decomposition and credit assignment (VDN)
    - Risk-aware behavior compatible with HAVEN coordination
    - Redis-backed state persistence and communication
    - Vector DB integration for collective memory

    All specialized agents should inherit from this class and implement
    their specific behavior in the step() method.

    The term "mycelial" refers to the network structure - like fungal mycelium,
    agents form a decentralized, interconnected network for learning and
    communication.
    """

    def __init__(
        self,
        model,
        redis_client,
        unique_id: Optional[int] = None,
        team_id: str = "default",
        agent_config: Optional[Dict[str, Any]] = None,
        signal_bus: Optional[ElectricalSignalBus] = None,
        stigmergy_env: Optional[StigmergicEnvironment] = None,
        gnn_communicator: Optional[GNNCommunicator] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        transfer_engine: Optional[TransferLearningEngine] = None,
        maml_learner: Optional[MAMLLearner] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        memory_consolidator: Optional[MemoryConsolidator] = None,
        semantic_retriever: Optional[SemanticRetriever] = None
    ):
        """
        Initialize the Mycelial Agent.

        Args:
            model: The MycelialModel this agent belongs to
            redis_client: Redis client for data operations and communication
            unique_id: Optional unique identifier for this agent (auto-assigned if None)
            team_id: Team identifier for collaboration (Rule of 3)
            agent_config: Optional configuration dictionary for agent parameters
            signal_bus: Optional ElectricalSignalBus for ultra-fast signaling (Big Rock 5)
            stigmergy_env: Optional StigmergicEnvironment for indirect coordination (Big Rock 6)
            gnn_communicator: Optional GNNCommunicator for intelligent message routing (Big Rock 7)
            knowledge_base: Optional KnowledgeBase for experience storage and retrieval (Big Rock 8)
            transfer_engine: Optional TransferLearningEngine for knowledge transfer (Big Rock 8)
            maml_learner: Optional MAMLLearner for meta-learning (Big Rock 8)
            episodic_memory: Optional EpisodicMemory for prioritized experience replay (Big Rock 9)
            memory_consolidator: Optional MemoryConsolidator for offline learning (Big Rock 9)
            semantic_retriever: Optional SemanticRetriever for semantic memory queries (Big Rock 9)
        """
        super().__init__(model)

        self.redis_client = redis_client
        self.agent_config = agent_config or {}

        # Agent identification
        self.agent_type = self.__class__.__name__
        # Mesa 3.x auto-assigns unique_id in the model
        self.agent_id = f"{self.agent_type}_{self.unique_id}"

        # Team-based collaboration (Rule of 3)
        self.team_id = team_id
        self.teammates: List[str] = []  # Populated by querying same team_id

        # Learning components (to be initialized by subclasses if needed)
        self.frl_engine = None  # Federated Reinforcement Learning engine
        self.vdn_engine = None  # Value Decomposition Network engine
        self.vector_db = None   # Vector database for team policies

        # BIG ROCK 5: ELECTRICAL SIGNALING
        self.signal_bus = signal_bus  # Ultra-fast signaling bus
        self.signal_subscriptions: List[str] = []  # Track subscribed signal types

        # BIG ROCK 6: STIGMERGIC ENVIRONMENT
        self.stigmergy_env = stigmergy_env  # Pheromone-like marker environment
        self.stigmergy_position: Tuple[float, ...] = (0.0, 0.0)  # 2D default position
        self.sensing_radius: float = self.agent_config.get("sensing_radius", 5.0)
        self.trail_following_enabled: bool = self.agent_config.get("trail_following", True)

        # BIG ROCK 7: GNN COMMUNICATION
        self.gnn_communicator = gnn_communicator  # GNN-based message routing
        self.gnn_message_handlers: Dict[str, Callable[[GNNMessage], None]] = {}
        self.capabilities: Set[str] = set(self.agent_config.get("capabilities", []))

        # BIG ROCK 8: TRANSFER LEARNING & META-LEARNING
        self.knowledge_base = knowledge_base  # Centralized experience storage
        self.transfer_engine = transfer_engine  # Transfer learning orchestrator
        self.maml_learner = maml_learner  # Meta-learning for few-shot adaptation
        self.current_task: Optional[TaskDescriptor] = None  # Current task descriptor
        self.episode_transitions: List[ExperienceTransition] = []  # Current episode buffer
        self.transfer_enabled: bool = self.agent_config.get("transfer_enabled", False)
        self.maml_enabled: bool = self.agent_config.get("maml_enabled", False)

        # BIG ROCK 9: EPISODIC MEMORY & REPLAY
        self.episodic_memory = episodic_memory  # Prioritized experience replay buffer
        self.memory_consolidator = memory_consolidator  # Offline learning consolidator
        self.semantic_retriever = semantic_retriever  # Semantic memory retrieval
        self.replay_enabled: bool = self.agent_config.get("replay_enabled", False)
        self.consolidation_enabled: bool = self.agent_config.get("consolidation_enabled", False)
        self.semantic_search_enabled: bool = self.agent_config.get("semantic_search_enabled", False)
        self.replay_frequency: int = self.agent_config.get("replay_frequency", 4)  # Learn every N steps
        self.replay_batch_size: int = self.agent_config.get("replay_batch_size", 32)
        self.steps_since_replay: int = 0
        self.total_replays: int = 0
        self.total_consolidations: int = 0

        # State tracking
        self.current_state: Dict[str, Any] = {}
        self.last_action: Optional[Any] = None
        self.last_reward: float = 0.0
        self.cumulative_reward: float = 0.0

        # Policy tracking
        self.policy_version: int = 0
        self.policy_parameters: Dict[str, Any] = {}
        self.policy_embedding: Optional[Any] = None  # Vector representation

        # Performance metrics
        self.step_count: int = 0
        self.performance_history: List[float] = []

        # Team collaboration metrics
        self.policies_shared_with_team: int = 0
        self.policies_received_from_team: int = 0

        # Risk metrics (for HAVEN compatibility)
        self.risk_score: float = 0.0
        self.is_isolated: bool = False

        # =====================================================================
        # BIG ROCK 4: MOTIVATION & SAFEGUARD LAYER
        # =====================================================================

        # Convergence safeguards (prevents infinite learning loops)
        self.learning_iterations: int = 0
        self.max_learning_iterations: int = self.agent_config.get("max_learning_iterations", 100)
        self.convergence_threshold: float = self.agent_config.get("convergence_threshold", 0.01)
        self.policy_improvement_window: List[float] = []
        self.last_n_improvements: int = 10
        self.has_reached_convergence: bool = False

        # Satisfaction metric (determines when agent is "done" learning)
        self.satisfaction_score: float = 0.0
        self.satisfaction_threshold: float = self.agent_config.get("satisfaction_threshold", 0.85)
        self.team_satisfaction: Optional[float] = None
        self.is_satisfied_state: bool = False

        # Gamification layer (motivation system)
        self.agent_level: int = 1
        self.experience_points: int = 0
        self.achievements: List[str] = []
        self.peer_rank: Optional[int] = None
        self.team_rank: Optional[int] = None

        # Intrinsic motivation parameters
        self.exploration_bonus: float = self.agent_config.get("exploration_bonus", 0.1)
        self.novelty_threshold: float = self.agent_config.get("novelty_threshold", 0.8)
        self.action_history: List[Any] = []  # For novelty detection
        self.action_history_size: int = 100

        # Register with GNN communicator if available
        if self.gnn_communicator:
            self.gnn_communicator.register_agent(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                capabilities=self.capabilities,
                level=self.agent_level,
                position=self.stigmergy_position
            )

        logger.info("Initialized %s (ID: %s, Team: %s)",
                   self.agent_type, self.agent_id, self.team_id)

    def step(self):
        """
        Execute one step of agent behavior.

        This is the main method called by the Mesa scheduler each time step.
        Subclasses should override this to implement their specific logic.

        Basic flow:
        1. Observe current state
        2. Select action based on policy
        3. Execute action
        4. Receive reward
        5. Update learning components
        6. Share policy updates (if FRL enabled)
        7. Persist state to Redis
        """
        self.step_count += 1

        # Observe current state
        self.current_state = self._observe_state()

        # Select and execute action
        action = self._select_action(self.current_state)
        self.last_action = action

        # Execute action and get reward
        reward = self._execute_action(action)
        self.last_reward = reward
        self.cumulative_reward += reward

        # Update performance history
        self._update_performance_metrics(reward)

        # Update learning components if present
        if self.vdn_engine is not None:
            local_credit = self.get_local_reward(reward)
            self._update_policy(self.current_state, action, local_credit)

        # Share policy updates if FRL enabled
        if self.frl_engine is not None and self._should_share_policy():
            self.share_policy()

        # Persist state to Redis
        self._save_state_to_redis()

        logger.debug("%s completed step %d", self.agent_id, self.step_count)

    def share_policy(self) -> int:
        """
        Share policy updates with peer agents via Federated Learning.

        This method is called when the agent wants to share its learned
        policy with other agents in the mycelial network. The FRL engine
        handles peer selection and update transmission.

        Returns:
            Number of peers the policy was shared with

        Raises:
            RuntimeError: If FRL engine is not initialized
        """
        if self.frl_engine is None:
            logger.warning("%s has no FRL engine, cannot share policy", self.agent_id)
            return 0

        # Prepare policy update package
        policy_update = {
            "policy_parameters": self.policy_parameters,
            "policy_version": self.policy_version,
            "performance": self._get_recent_performance(),
            "agent_type": self.agent_type
        }

        metadata = {
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "risk_score": self.risk_score
        }

        # Use FRL engine to share with selected peers
        num_peers = self.frl_engine.share_policy_update(policy_update, metadata)

        logger.debug("%s shared policy v%d with %d peers",
                    self.agent_id, self.policy_version, num_peers)

        return num_peers

    def get_local_reward(self, global_reward: float) -> float:
        """
        Compute local reward from global reward using Value Decomposition.

        This method uses the VDN engine to assign credit to this agent
        for its contribution to the global system reward. This solves
        the multi-agent credit assignment problem.

        Args:
            global_reward: The total reward received by the system

        Returns:
            Individual credit/reward assigned to this agent

        Raises:
            RuntimeError: If VDN engine is not initialized
        """
        if self.vdn_engine is None:
            # If no VDN engine, agent receives full global reward
            logger.debug("%s has no VDN engine, using full global reward", self.agent_id)
            return global_reward

        # Get joint action from all agents (would need to be retrieved from Redis)
        joint_action = self._get_joint_action()

        # Get next state
        next_state = self._observe_state()

        # Use VDN engine to assign credit
        local_credit = self.vdn_engine.assign_credit(
            global_reward=global_reward,
            state=self.current_state,
            joint_action=joint_action,
            next_state=next_state
        )

        logger.debug("%s received credit %.4f from global reward %.4f",
                    self.agent_id, local_credit, global_reward)

        return local_credit

    # ==========================================
    # Protected methods (to be overridden by subclasses)
    # ==========================================

    def _observe_state(self) -> Dict[str, Any]:
        """
        Observe the current state of the environment.

        Subclasses should override this to implement their specific
        observation logic (e.g., reading from Redis, accessing model state).

        Returns:
            Dictionary representing the observed state
        """
        return {
            "step": self.model.current_step,
            "agent_id": self.agent_id
        }

    def _select_action(self, state: Dict[str, Any]) -> Any:
        """
        Select an action based on current state and policy.

        Subclasses should override this to implement their decision-making
        logic (e.g., neural network forward pass, rule-based selection).

        Args:
            state: Current observed state

        Returns:
            Selected action (type depends on action space)
        """
        # Default: no-op action
        return None

    def _execute_action(self, action: Any) -> float:
        """
        Execute the selected action and receive reward.

        Subclasses should override this to implement action execution
        logic (e.g., modifying environment, publishing to Redis).

        Args:
            action: The action to execute

        Returns:
            Immediate reward received from executing the action
        """
        # Default: zero reward
        return 0.0

    def _update_policy(self, state: Dict[str, Any], action: Any, reward: float):
        """
        Update the agent's policy based on experience.

        Subclasses should override this to implement their learning
        algorithm (e.g., Q-learning update, policy gradient).

        Args:
            state: State where action was taken
            action: Action that was taken
            reward: Reward received (after credit assignment)
        """
        # Default: no learning
        pass

    def _should_share_policy(self) -> bool:
        """
        Determine if the agent should share its policy this step.

        Subclasses can override this to implement custom sharing logic
        (e.g., only share after significant updates, periodic sharing).

        Returns:
            True if policy should be shared this step
        """
        # Default: share every 10 steps
        return self.step_count % 10 == 0

    def _get_joint_action(self) -> Dict[str, Any]:
        """
        Retrieve the joint action taken by all agents.

        This is needed for VDN credit assignment. Subclasses can override
        to implement efficient joint action retrieval from Redis.

        Returns:
            Dictionary mapping agent_id to action
        """
        # Default: only this agent's action
        return {self.agent_id: self.last_action}

    def _get_recent_performance(self) -> float:
        """
        Get recent performance metric for policy sharing.

        Returns:
            Recent average reward or other performance metric
        """
        if not self.performance_history:
            return 0.0

        # Return average of last 10 steps
        recent = self.performance_history[-10:]
        return sum(recent) / len(recent)

    def _update_performance_metrics(self, reward: float):
        """
        Update performance tracking metrics.

        Args:
            reward: Reward received this step
        """
        self.performance_history.append(reward)

        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def _save_state_to_redis(self):
        """
        Persist agent state to Redis for recovery and analysis.
        """
        state_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "team_id": self.team_id,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "policy_version": self.policy_version,
            "risk_score": self.risk_score,
            "is_isolated": self.is_isolated,
            "last_reward": self.last_reward,
            "policies_shared_with_team": self.policies_shared_with_team,
            "policies_received_from_team": self.policies_received_from_team
        }

        key = f"agent:state:{self.agent_id}"
        self.redis_client.set_key_value(key, state_data)

    def _load_state_from_redis(self) -> Optional[Dict[str, Any]]:
        """
        Load agent state from Redis for recovery.

        Returns:
            Saved state dictionary, or None if no saved state
        """
        key = f"agent:state:{self.agent_id}"
        return self.redis_client.get_key_value(key)

    # ==========================================
    # Public utility methods
    # ==========================================

    def publish_message(self, channel: str, message: Any) -> int:
        """
        Publish a message to a Redis Pub/Sub channel.

        Useful for inter-agent communication.

        Args:
            channel: Channel name to publish to
            message: Message to publish

        Returns:
            Number of subscribers that received the message
        """
        return self.redis_client.publish(channel, message)

    def subscribe_to_channel(self, channel: str):
        """
        Subscribe to a Redis Pub/Sub channel.

        Args:
            channel: Channel name to subscribe to
        """
        self.redis_client.subscribe([channel])

    def write_to_stream(self, stream_name: str, data: Dict[str, Any]) -> str:
        """
        Write data to a Redis Stream.

        Args:
            stream_name: Name of the stream
            data: Data to write

        Returns:
            Entry ID of the added stream entry
        """
        return self.redis_client.write_to_stream(stream_name, data)

    def read_from_stream(
        self,
        stream_name: str,
        last_id: str = "0-0",
        count: Optional[int] = None
    ) -> List[tuple]:
        """
        Read data from a Redis Stream.

        Args:
            stream_name: Name of the stream to read from
            last_id: ID of last entry received
            count: Maximum number of entries to read

        Returns:
            List of (entry_id, data) tuples
        """
        return self.redis_client.read_from_stream(
            stream_name,
            count=count,
            last_id=last_id
        )

    def set_risk_score(self, risk_score: float):
        """
        Set the agent's risk score (used by HAVEN coordinator).

        Args:
            risk_score: Risk score from 0.0 to 1.0
        """
        self.risk_score = max(0.0, min(1.0, risk_score))

    def isolate(self):
        """
        Isolate this agent from the federated learning network.

        Called by HAVEN coordinator when agent is deemed risky.
        """
        self.is_isolated = True
        logger.warning("%s has been isolated", self.agent_id)

    def restore(self):
        """
        Restore this agent to normal operation after isolation.
        """
        self.is_isolated = False
        logger.info("%s has been restored from isolation", self.agent_id)

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the agent's current state.

        Returns:
            Dictionary with key agent metrics
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "policy_version": self.policy_version,
            "recent_performance": self._get_recent_performance(),
            "risk_score": self.risk_score,
            "is_isolated": self.is_isolated,
            "connected_peers": self.frl_engine.get_connected_peer_count() if self.frl_engine else 0
        }

    # ==========================================
    # Team Collaboration Methods (Rule of 3)
    # ==========================================

    def get_teammates(self) -> List[str]:
        """
        Query and return list of agents with the same team_id.

        This method queries Redis to find all agents that belong to
        the same team as this agent.

        Returns:
            List of agent IDs in the same team
        """
        # Query all agent states from Redis
        pattern = "agent:state:*"
        all_keys = self.redis_client.client.keys(pattern)

        teammates = []
        for key in all_keys:
            state = self.redis_client.get_key_value(key)
            if state and state.get("team_id") == self.team_id:
                agent_id = state.get("agent_id")
                # Don't include self
                if agent_id and agent_id != self.agent_id:
                    teammates.append(agent_id)

        self.teammates = teammates
        logger.debug("%s found %d teammates in team %s",
                    self.agent_id, len(teammates), self.team_id)
        return teammates

    def share_policy_with_team(self) -> Optional[str]:
        """
        Share current policy with teammates via Vector DB.

        This method stores the agent's current policy embedding in the
        Vector DB so teammates can discover and learn from it.

        Returns:
            Policy ID if successful, None if no Vector DB or no embedding

        Raises:
            RuntimeError: If Vector DB operations fail
        """
        if self.vector_db is None:
            logger.warning("%s has no Vector DB, cannot share policy", self.agent_id)
            return None

        if self.policy_embedding is None:
            logger.warning("%s has no policy embedding to share", self.agent_id)
            return None

        try:
            # Generate unique policy ID
            policy_id = f"{self.agent_id}_policy_v{self.policy_version}"

            # Metadata for filtering and analysis
            metadata = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "team_id": self.team_id,
                "policy_version": self.policy_version,
                "performance": self._get_recent_performance(),
                "step_count": self.step_count,
                "cumulative_reward": self.cumulative_reward,
                "risk_score": self.risk_score
            }

            # Store in Vector DB
            self.vector_db.add_policy_embedding(
                policy_id=policy_id,
                agent_id=self.agent_id,
                embedding=self.policy_embedding,
                metadata=metadata
            )

            self.policies_shared_with_team += 1

            logger.debug("%s shared policy %s with team %s",
                        self.agent_id, policy_id, self.team_id)

            return policy_id

        except Exception as e:
            logger.error("%s failed to share policy: %s", self.agent_id, e)
            raise RuntimeError(f"Failed to share policy: {e}")

    def retrieve_teammate_policies(
        self,
        top_k: int = 5,
        min_performance: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar policies from teammates via Vector DB.

        This method queries the Vector DB for policies from teammates
        that are similar to this agent's current policy.

        Args:
            top_k: Number of similar policies to retrieve
            min_performance: Optional minimum performance threshold

        Returns:
            List of dictionaries with policy_id, agent_id, similarity, metadata

        Raises:
            RuntimeError: If Vector DB operations fail
        """
        if self.vector_db is None:
            logger.warning("%s has no Vector DB, cannot retrieve policies", self.agent_id)
            return []

        if self.policy_embedding is None:
            logger.warning("%s has no policy embedding for similarity search", self.agent_id)
            return []

        try:
            # Search for similar policies
            results = self.vector_db.search_similar_policies(
                query_embedding=self.policy_embedding,
                top_k=top_k * 2,  # Get extra to filter
                filter_dict={"team_id": self.team_id}  # Only teammates
            )

            # Filter results
            filtered_results = []
            for result in results:
                # Skip self
                if result.get("agent_id") == self.agent_id:
                    continue

                # Check performance threshold
                if min_performance is not None:
                    if result.get("metadata", {}).get("performance", 0) < min_performance:
                        continue

                filtered_results.append(result)

                # Stop when we have enough
                if len(filtered_results) >= top_k:
                    break

            self.policies_received_from_team += len(filtered_results)

            logger.debug("%s retrieved %d teammate policies from team %s",
                        self.agent_id, len(filtered_results), self.team_id)

            return filtered_results

        except Exception as e:
            logger.error("%s failed to retrieve teammate policies: %s", self.agent_id, e)
            raise RuntimeError(f"Failed to retrieve policies: {e}")

    def get_team_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about team collaboration.

        Returns:
            Dictionary with team metrics
        """
        teammates = self.get_teammates()

        # Aggregate team performance from Redis
        team_rewards = []
        team_risk_scores = []

        for teammate_id in teammates:
            state = self.redis_client.get_key_value(f"agent:state:{teammate_id}")
            if state:
                team_rewards.append(state.get("cumulative_reward", 0.0))
                team_risk_scores.append(state.get("risk_score", 0.0))

        return {
            "team_id": self.team_id,
            "team_size": len(teammates),
            "teammates": teammates,
            "policies_shared": self.policies_shared_with_team,
            "policies_received": self.policies_received_from_team,
            "team_avg_reward": sum(team_rewards) / max(1, len(team_rewards)),
            "team_avg_risk": sum(team_risk_scores) / max(1, len(team_risk_scores)),
            "my_contribution_rank": self._calculate_team_rank(team_rewards)
        }

    def _calculate_team_rank(self, team_rewards: List[float]) -> int:
        """
        Calculate this agent's rank within the team by performance.

        Args:
            team_rewards: List of teammate cumulative rewards

        Returns:
            Rank (1 = best, higher = worse)
        """
        if not team_rewards:
            return 1

        # Add own reward and sort
        all_rewards = team_rewards + [self.cumulative_reward]
        all_rewards.sort(reverse=True)

        # Find rank (1-indexed)
        rank = all_rewards.index(self.cumulative_reward) + 1
        return rank

    # =========================================================================
    # BIG ROCK 4: CONVERGENCE SAFEGUARDS & SATISFACTION METRIC
    # =========================================================================

    def has_converged(self) -> bool:
        """
        Check if agent has converged and should stop iterative learning.

        Convergence criteria:
        - Policy improvements over last N iterations are below threshold
        - Indicates agent has reached a stable, optimal (or near-optimal) policy

        This prevents infinite learning loops in the mycelial network.

        Returns:
            True if agent has converged (should stop learning)
        """
        import numpy as np

        # Need sufficient history
        if len(self.policy_improvement_window) < self.last_n_improvements:
            return False

        # Calculate average improvement over last N iterations
        recent_improvements = self.policy_improvement_window[-self.last_n_improvements:]
        avg_improvement = np.mean(recent_improvements)

        # Converged if average improvement is below threshold
        has_converged = avg_improvement < self.convergence_threshold

        if has_converged and not self.has_reached_convergence:
            self.has_reached_convergence = True
            logger.info("%s has CONVERGED (avg improvement: %.4f < %.4f)",
                       self.agent_id, avg_improvement, self.convergence_threshold)

        return has_converged

    def record_policy_improvement(self, old_performance: float, new_performance: float):
        """
        Record improvement in policy performance for convergence tracking.

        Args:
            old_performance: Performance before policy update
            new_performance: Performance after policy update
        """
        improvement = new_performance - old_performance
        self.policy_improvement_window.append(improvement)

        # Limit window size
        if len(self.policy_improvement_window) > 50:
            self.policy_improvement_window = self.policy_improvement_window[-50:]

        logger.debug("%s policy improvement: %.4f (window: %d)",
                    self.agent_id, improvement, len(self.policy_improvement_window))

    def compute_satisfaction(self) -> float:
        """
        Compute agent's satisfaction with current policy performance.

        Satisfaction is a weighted metric combining:
        - Recent performance (40%)
        - Improvement rate (30%)
        - Stability/consistency (20%)
        - Social comparison to teammates (10%)

        Returns:
            Satisfaction score (0.0 to 1.0)
        """
        import numpy as np

        satisfaction = 0.0

        # Component 1: Recent Performance (40%)
        if self.performance_history:
            recent_window = min(20, len(self.performance_history))
            recent_perf = np.mean(self.performance_history[-recent_window:])
            # Normalize to 0-1 range (assumes rewards typically 0-1)
            perf_component = np.clip(recent_perf, 0.0, 1.0)
            satisfaction += 0.4 * perf_component

        # Component 2: Improvement Rate (30%)
        if len(self.performance_history) > 10:
            improvement_rate = self._compute_improvement_rate()
            # Normalize: 10% improvement = 1.0 satisfaction component
            improvement_component = np.clip(improvement_rate / 0.1, 0.0, 1.0)
            satisfaction += 0.3 * improvement_component

        # Component 3: Stability/Consistency (20%)
        if len(self.performance_history) > 20:
            recent_window = self.performance_history[-20:]
            variance = np.var(recent_window)
            # Lower variance = higher stability
            # Use inverse: stability = 1 / (1 + variance)
            stability = 1.0 / (1.0 + variance)
            satisfaction += 0.2 * stability

        # Component 4: Social Comparison (10%)
        if self.team_satisfaction is not None and self.team_satisfaction > 0:
            # Compare to team average
            success_rate = self._get_success_rate()
            relative_performance = success_rate / max(0.01, self.team_satisfaction)
            social_component = np.clip(relative_performance, 0.0, 1.0)
            satisfaction += 0.1 * social_component
        else:
            # No team data, just use own success rate
            satisfaction += 0.1 * self._get_success_rate()

        # Final satisfaction score
        self.satisfaction_score = np.clip(satisfaction, 0.0, 1.0)

        return self.satisfaction_score

    def is_satisfied(self) -> bool:
        """
        Check if agent is satisfied with current performance.

        A satisfied agent should reduce learning activity to conserve
        resources and prevent unnecessary policy churn.

        Returns:
            True if agent is satisfied (performance is "good enough")
        """
        satisfaction = self.compute_satisfaction()
        is_satisfied = satisfaction >= self.satisfaction_threshold

        if is_satisfied and not self.is_satisfied_state:
            self.is_satisfied_state = True
            logger.info("%s is SATISFIED (satisfaction: %.3f >= %.3f)",
                       self.agent_id, satisfaction, self.satisfaction_threshold)

        return is_satisfied

    def _compute_improvement_rate(self) -> float:
        """
        Compute rate of performance improvement over recent history.

        Returns:
            Improvement rate (positive = improving, negative = declining)
        """
        import numpy as np

        if len(self.performance_history) < 10:
            return 0.0

        # Compare first half to second half of recent window
        window_size = min(20, len(self.performance_history))
        recent = self.performance_history[-window_size:]

        half = len(recent) // 2
        first_half_avg = np.mean(recent[:half])
        second_half_avg = np.mean(recent[half:])

        # Improvement rate (as fraction)
        if first_half_avg > 0:
            improvement_rate = (second_half_avg - first_half_avg) / first_half_avg
        else:
            improvement_rate = 0.0

        return improvement_rate

    def should_continue_learning(self) -> bool:
        """
        Determine if agent should continue active learning.

        Stops learning if:
        1. Agent has converged (policy is stable)
        2. Agent is satisfied (performance is good enough)
        3. Max learning iterations reached (safety limit)

        Returns:
            True if agent should continue learning
        """
        # Check hard limits first (safety)
        if self.learning_iterations >= self.max_learning_iterations:
            logger.warning("%s reached max learning iterations (%d), stopping",
                          self.agent_id, self.max_learning_iterations)
            return False

        # Check convergence
        if self.has_converged():
            logger.debug("%s has converged, reducing learning activity", self.agent_id)
            # Don't completely stop, but reduce frequency
            return self.step_count % 10 == 0  # Learn only every 10 steps

        # Check satisfaction
        if self.is_satisfied():
            logger.debug("%s is satisfied, reducing learning activity", self.agent_id)
            # Learn occasionally to avoid stagnation
            return self.step_count % 5 == 0  # Learn only every 5 steps

        # Continue normal learning
        return True

    # =========================================================================
    # BIG ROCK 4: GAMIFICATION & INTRINSIC MOTIVATION
    # =========================================================================

    def compute_intrinsic_reward(self, action: Any, state: Dict[str, Any]) -> float:
        """
        Compute intrinsic motivation reward.

        Intrinsic rewards drive exploration and prevent stagnation by
        rewarding:
        1. Novelty (curiosity) - Trying new actions
        2. Learning progress - Improving performance
        3. Social ranking - Competing with peers

        Args:
            action: Action taken
            state: Current state

        Returns:
            Intrinsic reward (added to extrinsic reward)
        """
        intrinsic = 0.0

        # 1. Exploration bonus (curiosity)
        if self._is_novel_action(action, state):
            intrinsic += self.exploration_bonus
            logger.debug("%s: Novel action bonus +%.3f", self.agent_id, self.exploration_bonus)

        # 2. Learning progress bonus (improvement motivation)
        if len(self.performance_history) > 10:
            improvement_rate = self._compute_improvement_rate()
            if improvement_rate > 0:
                progress_bonus = improvement_rate * 0.05  # Up to 5% bonus for 100% improvement
                intrinsic += progress_bonus
                logger.debug("%s: Learning progress bonus +%.4f", self.agent_id, progress_bonus)

        # 3. Social ranking bonus (competition motivation)
        if self.peer_rank is not None and self.peer_rank <= 10:
            # Top 10 agents get bonus
            rank_bonus = (11 - self.peer_rank) * 0.002  # 0.02 for rank 1, 0.002 for rank 10
            intrinsic += rank_bonus
            logger.debug("%s: Top-10 rank bonus +%.4f (rank %d)",
                        self.agent_id, rank_bonus, self.peer_rank)

        return intrinsic

    def _is_novel_action(self, action: Any, state: Dict[str, Any]) -> bool:
        """
        Determine if action is novel (haven't done similar action recently).

        Args:
            action: Action to check
            state: Current state

        Returns:
            True if action is novel
        """
        if not self.action_history:
            return True

        # Simple novelty check: action not in recent history
        # For more sophisticated novelty, use embedding similarity
        recent_actions = self.action_history[-20:]  # Check last 20 actions

        # Convert action to comparable form
        action_str = str(action)

        # Novel if not seen in recent history
        is_novel = action_str not in [str(a) for a in recent_actions]

        return is_novel

    def record_action(self, action: Any):
        """
        Record action for novelty detection.

        Args:
            action: Action taken
        """
        self.action_history.append(action)

        # Limit history size
        if len(self.action_history) > self.action_history_size:
            self.action_history = self.action_history[-self.action_history_size:]

    def update_gamification(self, reward: float):
        """
        Update gamification metrics (levels, XP, achievements).

        This system motivates agents through progression and recognition.

        Args:
            reward: Reward received (used to calculate XP)
        """
        # Add experience points (scaled by 100 for granularity)
        xp_gained = int(abs(reward) * 100)
        self.experience_points += xp_gained

        # Check for level up
        xp_required = self.agent_level * 1000  # Linear scaling
        if self.experience_points >= xp_required:
            old_level = self.agent_level
            self.agent_level += 1
            logger.info("%s LEVELED UP! Level %d -> %d (XP: %d)",
                       self.agent_id, old_level, self.agent_level, self.experience_points)

            # Unlock achievement
            self.unlock_achievement(f"Level {self.agent_level} Reached")

            # Boost motivation (reduce exploration threshold slightly)
            if self.exploration_bonus < 0.5:
                self.exploration_bonus += 0.01

        # Check for milestone achievements
        self._check_achievements()

    def _check_achievements(self):
        """Check and unlock achievements based on performance milestones."""

        # Task completion achievements
        if self.step_count >= 100 and "Centurion" not in self.achievements:
            self.unlock_achievement("Centurion")

        if self.step_count >= 1000 and "Millennium" not in self.achievements:
            self.unlock_achievement("Millennium")

        # Reward achievements
        if self.cumulative_reward >= 100 and "Apprentice" not in self.achievements:
            self.unlock_achievement("Apprentice")

        if self.cumulative_reward >= 1000 and "Master" not in self.achievements:
            self.unlock_achievement("Master")

        if self.cumulative_reward >= 10000 and "Grandmaster" not in self.achievements:
            self.unlock_achievement("Grandmaster")

        # Collaboration achievements
        if self.policies_shared_with_team >= 50 and "Team Player" not in self.achievements:
            self.unlock_achievement("Team Player")

        if self.policies_shared_with_team >= 200 and "Mentor" not in self.achievements:
            self.unlock_achievement("Mentor")

        # Performance achievements
        success_rate = self._get_success_rate()
        if success_rate >= 0.9 and self.step_count >= 100 and "Elite" not in self.achievements:
            self.unlock_achievement("Elite")

        # Convergence achievement
        if self.has_reached_convergence and "Convergence Master" not in self.achievements:
            self.unlock_achievement("Convergence Master")

        # Satisfaction achievement
        if self.is_satisfied_state and "Satisfied Achiever" not in self.achievements:
            self.unlock_achievement("Satisfied Achiever")

    def unlock_achievement(self, achievement_name: str):
        """
        Unlock an achievement.

        Args:
            achievement_name: Name of achievement to unlock
        """
        if achievement_name not in self.achievements:
            self.achievements.append(achievement_name)
            logger.info("%s UNLOCKED ACHIEVEMENT: '%s'", self.agent_id, achievement_name)

            # Achievements give XP bonus
            self.experience_points += 500

            # BIG ROCK 5: Emit electrical signal to broadcast achievement
            if self.signal_bus is not None:
                self.emit_signal(
                    SignalType.ACHIEVEMENT_UNLOCKED,
                    {
                        'achievement_name': achievement_name,
                        'agent_level': self.agent_level,
                        'experience_points': self.experience_points,
                        'description': f"Agent {self.agent_id} unlocked: {achievement_name}"
                    },
                    priority=SignalPriority.LOW
                )

    def get_gamification_status(self) -> Dict[str, Any]:
        """
        Get current gamification status.

        Returns:
            Dictionary with level, XP, achievements, ranks
        """
        xp_required = self.agent_level * 1000
        xp_progress = (self.experience_points % xp_required) / xp_required

        return {
            "agent_id": self.agent_id,
            "level": self.agent_level,
            "experience_points": self.experience_points,
            "xp_to_next_level": xp_required - (self.experience_points % xp_required),
            "level_progress": xp_progress,
            "achievements": self.achievements,
            "achievement_count": len(self.achievements),
            "peer_rank": self.peer_rank,
            "team_rank": self.team_rank,
            "satisfaction_score": self.satisfaction_score,
            "has_converged": self.has_reached_convergence
        }

    # =========================================================================
    # BIG ROCK 5: ELECTRICAL SIGNALING LAYER
    # =========================================================================

    def emit_signal(
        self,
        signal_type: str,
        payload: Dict[str, Any],
        priority: SignalPriority = SignalPriority.NORMAL,
        ttl: float = 0.0
    ) -> bool:
        """
        Emit an electrical signal to the mycelial network.

        Electrical signals provide ultra-fast (sub-millisecond) communication
        for critical events like dangers, opportunities, and convergence.

        Args:
            signal_type: Type of signal (use SignalType constants)
            payload: Signal data dictionary
            priority: Signal priority (CRITICAL, HIGH, NORMAL, LOW)
            ttl: Time-to-live in seconds (0 = infinite)

        Returns:
            True if signal emitted successfully, False if no signal bus or rate limited

        Example:
            agent.emit_signal(
                SignalType.DANGER,
                {'risk_level': 0.9, 'risk_type': 'policy_divergence'},
                priority=SignalPriority.CRITICAL
            )
        """
        if self.signal_bus is None:
            logger.debug("%s has no signal bus, cannot emit signal", self.agent_id)
            return False

        success = self.signal_bus.emit_signal(
            signal_type=signal_type,
            source_agent_id=self.agent_id,
            payload=payload,
            priority=priority,
            ttl=ttl
        )

        if success:
            logger.debug("%s emitted %s signal (priority=%s)",
                        self.agent_id, signal_type, priority.name)

            # Emit ACHIEVEMENT_UNLOCKED signal when achievement is unlocked
            if signal_type == SignalType.ACHIEVEMENT_UNLOCKED:
                self._handle_achievement_signal(payload)

        return success

    def subscribe_to_signal(
        self,
        signal_type: str,
        callback: Callable[[Signal], None],
        min_priority: SignalPriority = SignalPriority.LOW
    ) -> bool:
        """
        Subscribe to electrical signals of a specific type.

        The callback will be invoked whenever a signal of this type is received.
        Callbacks execute asynchronously in a thread pool.

        Args:
            signal_type: Type of signal to subscribe to
            callback: Function to call when signal received (takes Signal object)
            min_priority: Only receive signals at or above this priority

        Returns:
            True if subscription successful

        Example:
            def handle_danger(signal: Signal):
                risk_level = signal.payload.get('risk_level', 0)
                if risk_level > 0.8:
                    self.take_evasive_action()

            agent.subscribe_to_signal(
                SignalType.DANGER,
                handle_danger,
                min_priority=SignalPriority.HIGH
            )
        """
        if self.signal_bus is None:
            logger.warning("%s has no signal bus, cannot subscribe", self.agent_id)
            return False

        success = self.signal_bus.subscribe(
            signal_type=signal_type,
            agent_id=self.agent_id,
            callback=callback,
            min_priority=min_priority
        )

        if success:
            self.signal_subscriptions.append(signal_type)
            logger.debug("%s subscribed to %s signals (min_priority=%s)",
                        self.agent_id, signal_type, min_priority.name)

        return success

    def unsubscribe_from_signal(self, signal_type: str) -> bool:
        """
        Unsubscribe from electrical signals.

        Args:
            signal_type: Type of signal to unsubscribe from

        Returns:
            True if unsubscription successful
        """
        if self.signal_bus is None:
            return False

        success = self.signal_bus.unsubscribe(signal_type, self.agent_id)

        if success and signal_type in self.signal_subscriptions:
            self.signal_subscriptions.remove(signal_type)
            logger.debug("%s unsubscribed from %s signals", self.agent_id, signal_type)

        return success

    def setup_standard_signal_handlers(self):
        """
        Setup standard signal handlers for common signal types.

        This is a convenience method that subscribes to commonly useful signals
        with sensible default handlers. Agents can override this or add custom
        handlers for their specific needs.
        """
        if self.signal_bus is None:
            logger.warning("%s has no signal bus, skipping signal handler setup", self.agent_id)
            return

        # Subscribe to DANGER signals (critical priority)
        self.subscribe_to_signal(
            SignalType.DANGER,
            self._handle_danger_signal,
            min_priority=SignalPriority.CRITICAL
        )

        # Subscribe to OPPORTUNITY signals (high priority)
        self.subscribe_to_signal(
            SignalType.OPPORTUNITY,
            self._handle_opportunity_signal,
            min_priority=SignalPriority.HIGH
        )

        # Subscribe to CONVERGENCE signals from teammates
        self.subscribe_to_signal(
            SignalType.CONVERGENCE,
            self._handle_convergence_signal,
            min_priority=SignalPriority.HIGH
        )

        # Subscribe to COLLABORATION_REQUEST signals
        self.subscribe_to_signal(
            SignalType.COLLABORATION_REQUEST,
            self._handle_collaboration_signal,
            min_priority=SignalPriority.HIGH
        )

        # Subscribe to KNOWLEDGE_SHARE signals
        self.subscribe_to_signal(
            SignalType.KNOWLEDGE_SHARE,
            self._handle_knowledge_share_signal,
            min_priority=SignalPriority.NORMAL
        )

        logger.info("%s setup standard signal handlers", self.agent_id)

    # Default signal handlers (can be overridden by subclasses)

    def _handle_danger_signal(self, signal: Signal):
        """
        Handle DANGER signal from peer agent.

        Default behavior: Increase caution, reduce risk-taking.
        Subclasses can override for specific danger responses.

        Args:
            signal: Danger signal
        """
        risk_level = signal.payload.get('risk_level', 0.5)
        risk_type = signal.payload.get('risk_type', 'unknown')

        logger.warning("%s received DANGER signal: type=%s, level=%.2f from %s",
                      self.agent_id, risk_type, risk_level, signal.source_agent_id)

        # Default response: Increase own risk score temporarily
        if risk_level > 0.7:
            self.risk_score = min(1.0, self.risk_score + 0.2)
            logger.info("%s increased risk score to %.2f in response to danger",
                       self.agent_id, self.risk_score)

    def _handle_opportunity_signal(self, signal: Signal):
        """
        Handle OPPORTUNITY signal from peer agent.

        Default behavior: Log opportunity, subclasses can act on it.

        Args:
            signal: Opportunity signal
        """
        opportunity_type = signal.payload.get('opportunity_type', 'unknown')
        expected_reward = signal.payload.get('expected_reward', 0)
        confidence = signal.payload.get('confidence', 0)

        logger.info("%s received OPPORTUNITY signal: type=%s, reward=%.2f, confidence=%.2f from %s",
                   self.agent_id, opportunity_type, expected_reward, confidence,
                   signal.source_agent_id)

        # Subclasses can override to take action on opportunities

    def _handle_convergence_signal(self, signal: Signal):
        """
        Handle CONVERGENCE signal from teammate.

        Default behavior: Log convergence, update team statistics.

        Args:
            signal: Convergence signal
        """
        peer_level = signal.payload.get('agent_level', 0)
        peer_satisfaction = signal.payload.get('satisfaction_score', 0)

        logger.info("%s received CONVERGENCE signal from %s (level=%d, satisfaction=%.2f)",
                   self.agent_id, signal.source_agent_id, peer_level, peer_satisfaction)

        # Update team satisfaction estimate
        if self.team_satisfaction is None:
            self.team_satisfaction = peer_satisfaction
        else:
            # Exponential moving average
            self.team_satisfaction = 0.9 * self.team_satisfaction + 0.1 * peer_satisfaction

    def _handle_collaboration_signal(self, signal: Signal):
        """
        Handle COLLABORATION_REQUEST signal.

        Default behavior: Log request, subclasses can accept/reject.

        Args:
            signal: Collaboration request signal
        """
        task_type = signal.payload.get('task_type', 'unknown')
        urgency = signal.payload.get('urgency', 'medium')
        reward_share = signal.payload.get('reward_share', 0)

        logger.info("%s received COLLABORATION_REQUEST: task=%s, urgency=%s, reward=%.2f from %s",
                   self.agent_id, task_type, urgency, reward_share, signal.source_agent_id)

        # Subclasses can override to accept collaboration

    def _handle_knowledge_share_signal(self, signal: Signal):
        """
        Handle KNOWLEDGE_SHARE signal.

        Default behavior: Log shared knowledge, subclasses can integrate it.

        Args:
            signal: Knowledge share signal
        """
        knowledge_type = signal.payload.get('knowledge_type', 'unknown')
        confidence = signal.payload.get('confidence', 0)

        logger.info("%s received KNOWLEDGE_SHARE: type=%s, confidence=%.2f from %s",
                   self.agent_id, knowledge_type, confidence, signal.source_agent_id)

        # Subclasses can override to integrate shared knowledge

    def _handle_achievement_signal(self, payload: Dict[str, Any]):
        """
        Handle achievement unlocked (internal, not a signal handler).

        This broadcasts achievement to peers for social motivation.

        Args:
            payload: Achievement payload
        """
        logger.info("%s broadcasting achievement: %s",
                   self.agent_id, payload.get('achievement_name', 'Unknown'))

    def get_signal_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get electrical signaling statistics.

        Returns:
            Dictionary with signal metrics, or None if no signal bus
        """
        if self.signal_bus is None:
            return None

        bus_metrics = self.signal_bus.get_metrics()

        return {
            "agent_id": self.agent_id,
            "subscriptions": self.signal_subscriptions,
            "subscription_count": len(self.signal_subscriptions),
            "bus_metrics": bus_metrics
        }

    # =========================================================================
    # BIG ROCK 6: STIGMERGIC ENVIRONMENT (Indirect Coordination)
    # =========================================================================

    def deposit_marker(
        self,
        marker_type: str,
        intensity: float = 1.0,
        position: Optional[Tuple[float, ...]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Deposit a stigmergic marker at a position.

        Markers create environmental "pheromone trails" that other agents
        can sense and follow, enabling indirect coordination.

        Args:
            marker_type: Type of marker (SUCCESS, DANGER, etc.)
            intensity: Marker intensity (0.0-1.0)
            position: Position to deposit (uses current position if None)
            metadata: Additional marker data

        Returns:
            Marker ID if successful, None if no stigmergy environment

        Example:
            # Deposit success marker after high reward
            if reward > 5.0:
                agent.deposit_marker(MarkerType.SUCCESS, intensity=reward/10)
        """
        if self.stigmergy_env is None:
            return None

        pos = position if position is not None else self.stigmergy_position

        return self.stigmergy_env.deposit_marker(
            marker_type=marker_type,
            position=pos,
            agent_id=self.agent_id,
            intensity=intensity,
            metadata=metadata
        )

    def deposit_success_marker(self, reward: float):
        """
        Deposit SUCCESS marker at current position (convenience method).

        Args:
            reward: Reward value (used to determine intensity)
        """
        if self.stigmergy_env and reward > 0:
            intensity = min(1.0, reward / 10.0)  # Normalize to 0-1
            self.deposit_marker(
                MarkerType.SUCCESS,
                intensity=intensity,
                metadata={'reward': reward}
            )

    def deposit_danger_marker(self, risk_level: float):
        """
        Deposit DANGER marker at current position (convenience method).

        Args:
            risk_level: Risk level (0.0-1.0)
        """
        if self.stigmergy_env and risk_level > 0.5:
            self.deposit_marker(
                MarkerType.DANGER,
                intensity=risk_level,
                metadata={'risk_score': self.risk_score}
            )

    def deposit_exploration_marker(self):
        """
        Deposit EXPLORATION marker at current position (convenience method).

        Marks area as explored to guide other agents toward novelty.
        """
        if self.stigmergy_env:
            self.deposit_marker(
                MarkerType.EXPLORATION,
                intensity=0.5,
                metadata={'visit_count': 1, 'timestamp': time.time()}
            )

    def sense_markers(
        self,
        marker_types: Optional[List[str]] = None,
        radius: Optional[float] = None
    ) -> List[StigmergicMarker]:
        """
        Sense markers near current position.

        Args:
            marker_types: Filter by marker types (None = all types)
            radius: Sensing radius (uses agent's sensing_radius if None)

        Returns:
            List of markers sorted by intensity (strongest first)

        Example:
            # Sense nearby success markers
            success_markers = agent.sense_markers([MarkerType.SUCCESS])
            if success_markers:
                print(f"Found {len(success_markers)} success trails")
        """
        if self.stigmergy_env is None:
            return []

        r = radius if radius is not None else self.sensing_radius

        return self.stigmergy_env.sense_markers(
            self.stigmergy_position,
            r,
            marker_types
        )

    def sense_environment(self) -> Dict[str, List[StigmergicMarker]]:
        """
        Sense all markers in sensing radius, grouped by type.

        Returns:
            Dictionary mapping marker types to lists of markers

        Example:
            markers = agent.sense_environment()
            if MarkerType.DANGER in markers:
                print(f"Warning: {len(markers[MarkerType.DANGER])} danger zones nearby")
        """
        if self.stigmergy_env is None:
            return {}

        all_markers = self.sense_markers()

        # Group by type
        by_type = defaultdict(list)
        for marker in all_markers:
            by_type[marker.marker_type].append(marker)

        return dict(by_type)

    def follow_trail(
        self,
        marker_type: str,
        attractive: bool = True
    ) -> Tuple[float, ...]:
        """
        Get direction to follow markers of a specific type.

        Computes gradient pointing toward (attractive) or away from
        (repulsive) markers, for trail following behavior.

        Args:
            marker_type: Type of marker to follow
            attractive: True = move toward, False = move away

        Returns:
            Normalized direction vector (gradient)

        Example:
            # Follow success trails
            direction = agent.follow_trail(MarkerType.SUCCESS, attractive=True)
            agent.move_in_stigmergy(direction, step_size=1.0)

            # Avoid danger zones
            escape = agent.follow_trail(MarkerType.DANGER, attractive=False)
            agent.move_in_stigmergy(escape, step_size=2.0)
        """
        if self.stigmergy_env is None:
            return tuple(0.0 for _ in range(len(self.stigmergy_position)))

        return self.stigmergy_env.get_gradient(
            self.stigmergy_position,
            marker_type,
            radius=self.sensing_radius,
            attractive=attractive
        )

    def move_in_stigmergy(
        self,
        direction: Tuple[float, ...],
        step_size: float = 1.0
    ):
        """
        Move in stigmergic space.

        Updates agent's position in the stigmergic environment.

        Args:
            direction: Normalized direction vector
            step_size: Distance to move

        Example:
            # Move toward success
            direction = agent.follow_trail(MarkerType.SUCCESS)
            agent.move_in_stigmergy(direction, step_size=1.0)
        """
        if len(direction) != len(self.stigmergy_position):
            logger.warning(f"{self.agent_id}: Direction dimension mismatch")
            return

        new_position = tuple(
            self.stigmergy_position[i] + direction[i] * step_size
            for i in range(len(self.stigmergy_position))
        )

        self.stigmergy_position = new_position

        logger.debug(f"{self.agent_id} moved to {new_position}")

    def get_strongest_nearby_marker(
        self,
        marker_type: Optional[str] = None
    ) -> Optional[StigmergicMarker]:
        """
        Get strongest marker within sensing radius.

        Args:
            marker_type: Filter by type (None = all types)

        Returns:
            Strongest marker or None if no markers
        """
        if self.stigmergy_env is None:
            return None

        return self.stigmergy_env.get_strongest_marker(
            self.stigmergy_position,
            self.sensing_radius,
            marker_type
        )

    def set_stigmergy_position(self, position: Tuple[float, ...]):
        """
        Manually set position in stigmergic space.

        Args:
            position: New position coordinates

        Raises:
            ValueError: If position has wrong number of dimensions
        """
        if self.stigmergy_env:
            expected_dims = self.stigmergy_env.dimensions
            if len(position) != expected_dims:
                raise ValueError(f"Position must have {expected_dims} dimensions")

        self.stigmergy_position = position

    def get_stigmergy_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get stigmergy statistics for this agent.

        Returns:
            Dictionary with position and nearby marker counts, or None
        """
        if self.stigmergy_env is None:
            return None

        nearby = self.sense_environment()

        return {
            'agent_id': self.agent_id,
            'position': self.stigmergy_position,
            'sensing_radius': self.sensing_radius,
            'nearby_markers': {
                mtype: len(markers)
                for mtype, markers in nearby.items()
            },
            'total_nearby': sum(len(m) for m in nearby.values())
        }

    def reset(self):
        """
        Reset the agent to initial state.

        Useful for episode boundaries in episodic tasks.
        """
        self.current_state = {}
        self.last_action = None
        self.last_reward = 0.0
        # Note: cumulative_reward and step_count typically not reset
        # Override in subclass if different behavior needed

        logger.info("%s has been reset", self.agent_id)

    # =========================================================================
    # BIG ROCK 7: GNN COMMUNICATION LAYER (Intelligent Message Routing)
    # =========================================================================

    def send_gnn_message(
        self,
        content: Dict[str, Any],
        message_type: str = MessageType.BROADCAST,
        target_ids: Optional[List[str]] = None,
        priority: float = 0.5,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Send message through GNN-based intelligent routing.

        Uses learned communication graph to route messages efficiently,
        achieving 40-60% overhead reduction vs. broadcast.

        Args:
            content: Message payload (dictionary)
            message_type: Message type (use MessageType constants)
            target_ids: Optional specific targets (for targeted messages)
            priority: Message priority [0, 1]
            ttl: Time-to-live (hops), uses default if None
            metadata: Optional additional metadata

        Returns:
            Message ID if sent successfully, None if GNN not available

        Example:
            >>> self.send_gnn_message(
            ...     content={'task': 'optimize_model', 'data': {...}},
            ...     message_type=MessageType.COLLABORATION_REQUEST,
            ...     priority=0.8
            ... )
        """
        if not self.gnn_communicator:
            logger.warning("%s: GNN communicator not available", self.agent_id)
            return None

        return self.gnn_communicator.send_message(
            sender_id=self.agent_id,
            content=content,
            message_type=message_type,
            target_ids=target_ids,
            priority=priority,
            ttl=ttl,
            metadata=metadata
        )

    def receive_gnn_messages(
        self,
        message_type: Optional[str] = None,
        max_messages: int = 10,
        min_priority: float = 0.0
    ) -> List[GNNMessage]:
        """
        Receive GNN-routed messages.

        Args:
            message_type: Optional filter by message type
            max_messages: Maximum messages to retrieve
            min_priority: Minimum priority threshold

        Returns:
            List of GNN messages

        Example:
            >>> messages = self.receive_gnn_messages(
            ...     message_type=MessageType.COLLABORATION_REQUEST,
            ...     max_messages=5
            ... )
            >>> for msg in messages:
            ...     self.handle_collaboration_request(msg)
        """
        if not self.gnn_communicator:
            return []

        return self.gnn_communicator.receive_messages(
            agent_id=self.agent_id,
            message_type=message_type,
            max_messages=max_messages,
            min_priority=min_priority
        )

    def process_gnn_messages(self):
        """
        Process all pending GNN messages using registered handlers.

        Retrieves messages and dispatches to appropriate handlers based
        on message type. Unknown message types are logged but not processed.

        Example:
            >>> # In agent's step() method:
            >>> self.process_gnn_messages()
        """
        messages = self.receive_gnn_messages()

        for message in messages:
            # Check for registered handler
            if message.message_type in self.gnn_message_handlers:
                handler = self.gnn_message_handlers[message.message_type]
                try:
                    handler(message)
                except Exception as e:
                    logger.error(
                        "%s: Error handling message %s: %s",
                        self.agent_id, message.message_id, e
                    )
            else:
                logger.debug(
                    "%s: No handler for message type %s",
                    self.agent_id, message.message_type
                )

    def register_gnn_message_handler(
        self,
        message_type: str,
        handler: Callable[[GNNMessage], None]
    ):
        """
        Register handler for specific message type.

        Args:
            message_type: Message type to handle
            handler: Callback function that takes GNNMessage as input

        Example:
            >>> def handle_query(msg: GNNMessage):
            ...     # Process query and send response
            ...     self.send_gnn_message(
            ...         content={'response': 'data'},
            ...         message_type=MessageType.QUERY_RESPONSE,
            ...         target_ids=[msg.sender_id]
            ...     )
            >>>
            >>> self.register_gnn_message_handler(
            ...     MessageType.QUERY,
            ...     handle_query
            ... )
        """
        self.gnn_message_handlers[message_type] = handler
        logger.debug(
            "%s: Registered handler for %s",
            self.agent_id, message_type
        )

    def report_communication_outcome(
        self,
        message_id: str,
        recipient_id: str,
        success: bool,
        reward: float = 0.0
    ):
        """
        Report outcome of message communication for GNN learning.

        Updates edge weights in communication graph based on outcomes,
        enabling the system to learn optimal routing patterns.

        Args:
            message_id: Message identifier
            recipient_id: Agent who received/processed message
            success: Whether communication was successful
            reward: Outcome reward (optional, 0.0 if not provided)

        Example:
            >>> msg_id = self.send_gnn_message(
            ...     content={'request': 'collaborate'},
            ...     message_type=MessageType.COLLABORATION_REQUEST,
            ...     target_ids=['SpecialistAgent_5']
            ... )
            >>> # ... later, after collaboration completes ...
            >>> self.report_communication_outcome(
            ...     msg_id,
            ...     'SpecialistAgent_5',
            ...     success=True,
            ...     reward=0.8
            ... )
        """
        if not self.gnn_communicator:
            return

        self.gnn_communicator.report_communication_outcome(
            message_id=message_id,
            recipient_id=recipient_id,
            success=success,
            reward=reward
        )

    def get_gnn_neighbors(self, k: Optional[int] = None) -> List[str]:
        """
        Get neighboring agents in GNN communication graph.

        Args:
            k: Optional limit (returns top-k by edge weight)

        Returns:
            List of agent IDs

        Example:
            >>> neighbors = self.get_gnn_neighbors(k=5)
            >>> # Send targeted message to top 5 neighbors
            >>> self.send_gnn_message(
            ...     content={'info': 'update'},
            ...     target_ids=neighbors
            ... )
        """
        if not self.gnn_communicator:
            return []

        return self.gnn_communicator.get_agent_neighbors(
            agent_id=self.agent_id,
            k=k
        )

    def broadcast_capability(self):
        """
        Broadcast agent capabilities to network.

        Useful for specialist agents to advertise their skills,
        or for builders to discover available specialists.

        Example:
            >>> # Specialist agent broadcasts capabilities
            >>> self.broadcast_capability()
        """
        if not self.gnn_communicator or not self.capabilities:
            return

        self.send_gnn_message(
            content={
                'capabilities': list(self.capabilities),
                'level': self.agent_level,
                'agent_type': self.agent_type,
                'satisfaction': self.satisfaction_score
            },
            message_type=MessageType.CAPABILITY_BROADCAST,
            priority=0.3
        )

    def request_collaboration(
        self,
        target_ids: Optional[List[str]] = None,
        task_description: str = "",
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Request collaboration from other agents.

        Args:
            target_ids: Optional specific targets (if None, uses GNN routing)
            task_description: Description of collaboration task
            required_capabilities: List of capabilities needed

        Returns:
            Message ID if sent successfully

        Example:
            >>> # Builder requests specialist
            >>> msg_id = self.request_collaboration(
            ...     task_description="Need optimization expert",
            ...     required_capabilities=["hyperparameter_tuning"]
            ... )
        """
        if not self.gnn_communicator:
            return None

        content = {
            'task': task_description,
            'required_capabilities': required_capabilities or [],
            'requester_type': self.agent_type,
            'requester_level': self.agent_level
        }

        return self.send_gnn_message(
            content=content,
            message_type=MessageType.COLLABORATION_REQUEST,
            target_ids=target_ids,
            priority=0.7
        )

    def share_knowledge(
        self,
        knowledge: Dict[str, Any],
        target_ids: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Share learned knowledge with other agents.

        Args:
            knowledge: Knowledge dictionary to share
            target_ids: Optional specific targets

        Returns:
            Message ID if sent successfully

        Example:
            >>> # Share successful strategy
            >>> self.share_knowledge({
            ...     'strategy': 'explore_then_exploit',
            ...     'performance': 0.85,
            ...     'steps_to_convergence': 1500
            ... })
        """
        if not self.gnn_communicator:
            return None

        content = {
            'knowledge': knowledge,
            'source_type': self.agent_type,
            'source_level': self.agent_level,
            'source_satisfaction': self.satisfaction_score
        }

        return self.send_gnn_message(
            content=content,
            message_type=MessageType.KNOWLEDGE_SHARE,
            target_ids=target_ids,
            priority=0.5
        )

    def get_gnn_communication_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get GNN communication statistics.

        Returns:
            Dictionary with communication metrics including overhead reduction

        Example:
            >>> stats = self.get_gnn_communication_stats()
            >>> print(f"Overhead reduction: {stats['overhead_reduction_percent']}%")
        """
        if not self.gnn_communicator:
            return None

        return self.gnn_communicator.get_communication_statistics()

    # =========================================================================
    # BIG ROCK 8: TRANSFER LEARNING & META-LEARNING
    # =========================================================================

    def begin_new_task(
        self,
        task_descriptor: TaskDescriptor,
        use_transfer: bool = True,
        use_maml: bool = False,
        min_similarity: float = 0.5,
        k_source_tasks: int = 3
    ) -> Dict[str, Any]:
        """
        Begin learning a new task with optional transfer learning and meta-learning.

        This method automatically:
        1. Identifies similar previously learned tasks
        2. Transfers relevant knowledge (policy, experiences, value function)
        3. Optionally applies MAML few-shot adaptation
        4. Returns transfer metrics including speed-up estimates

        Args:
            task_descriptor: Descriptor for the new task
            use_transfer: Whether to use transfer learning
            use_maml: Whether to use MAML meta-learning
            min_similarity: Minimum task similarity for transfer
            k_source_tasks: Number of source tasks to consider

        Returns:
            Dictionary with transfer results and metrics

        Example:
            >>> task = TaskDescriptor(
            ...     task_id="navigation_v2",
            ...     task_type="navigation",
            ...     state_dim=10,
            ...     action_dim=4
            ... )
            >>> result = agent.begin_new_task(task, use_transfer=True, use_maml=True)
            >>> print(f"Transfer speedup: {result.get('speedup_estimate', 1)}x")
        """
        self.current_task = task_descriptor
        result = {
            'task_id': task_descriptor.task_id,
            'task_type': task_descriptor.task_type,
            'transfer_used': False,
            'maml_used': False,
            'speedup_estimate': 1.0
        }

        # Register task with knowledge base if available
        if self.knowledge_base:
            self.knowledge_base.similarity_matrix.add_task(task_descriptor)

        # Transfer learning
        if use_transfer and self.transfer_engine and self.transfer_enabled:
            logger.info(
                "%s: Initiating transfer learning for task %s",
                self.agent_id, task_descriptor.task_id
            )

            transfer_result = self.transfer_engine.initiate_transfer(
                target_task=task_descriptor,
                agent_id=self.agent_id,
                strategy=TransferStrategy.COMBINED,
                min_similarity=min_similarity,
                k_source_tasks=k_source_tasks
            )

            result['transfer_used'] = True
            result['transfer_result'] = transfer_result.to_dict()
            result['num_experiences_transferred'] = transfer_result.num_experiences_transferred
            result['num_source_tasks'] = len(transfer_result.source_tasks)

            # Estimate speed-up based on transferred knowledge
            if transfer_result.num_experiences_transferred > 0:
                # Rough estimate: 10x speedup per 1000 experiences
                speedup = 1.0 + min(
                    transfer_result.num_experiences_transferred / 100.0,
                    100.0
                )
                result['speedup_estimate'] = speedup

        # MAML meta-learning adaptation
        if use_maml and self.maml_learner and self.maml_enabled:
            logger.info(
                "%s: Attempting MAML adaptation for task %s",
                self.agent_id, task_descriptor.task_id
            )

            # Check if we have few-shot support examples
            if self.knowledge_base:
                support_episodes = self.knowledge_base.retrieve_successful_episodes(
                    task_id=task_descriptor.task_id,
                    k=5
                )

                if support_episodes:
                    adaptation_result = self.maml_learner.adapt_to_task(
                        target_task=task_descriptor,
                        agent_id=self.agent_id,
                        support_episodes=support_episodes
                    )

                    result['maml_used'] = True
                    result['maml_result'] = adaptation_result.to_dict()
                    result['performance_gain'] = adaptation_result.performance_gain

                    # Update speedup estimate
                    if adaptation_result.performance_gain > 0:
                        result['speedup_estimate'] *= (1.0 + adaptation_result.performance_gain * 5)

        logger.info(
            "%s: Started task %s with estimated %sx speedup (transfer=%s, maml=%s)",
            self.agent_id, task_descriptor.task_id,
            result['speedup_estimate'], result['transfer_used'], result['maml_used']
        )

        return result

    def store_transition_for_transfer(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store experience transition for future transfer learning.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            metadata: Optional additional metadata
        """
        if not self.knowledge_base or not self.current_task:
            return

        transition = ExperienceTransition(
            task_id=self.current_task.task_id,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            agent_id=self.agent_id,
            metadata=metadata or {}
        )

        # Store in knowledge base
        self.knowledge_base.store_transition(transition, priority=abs(reward))

        # Buffer for episode completion
        self.episode_transitions.append(transition)

        logger.debug(
            "%s: Stored transition for task %s (buffer size: %d)",
            self.agent_id, self.current_task.task_id, len(self.episode_transitions)
        )

    def store_episode_for_transfer(
        self,
        total_reward: float,
        success: bool,
        clear_buffer: bool = True
    ) -> Optional[str]:
        """
        Store completed episode for future transfer learning.

        Args:
            total_reward: Total episode reward
            success: Whether episode was successful
            clear_buffer: Whether to clear transition buffer after storing

        Returns:
            Episode ID if stored successfully, None otherwise
        """
        if not self.knowledge_base or not self.current_task:
            return None

        if not self.episode_transitions:
            logger.warning("%s: No transitions to store for episode", self.agent_id)
            return None

        # Create episode
        episode_id = f"{self.current_task.task_id}_{self.agent_id}_{int(time.time())}"
        episode = Episode(
            episode_id=episode_id,
            task_id=self.current_task.task_id,
            agent_id=self.agent_id,
            transitions=self.episode_transitions.copy(),
            total_reward=total_reward,
            episode_length=len(self.episode_transitions),
            success=success
        )

        # Store in knowledge base
        self.knowledge_base.store_episode(episode)

        logger.info(
            "%s: Stored episode %s for task %s (reward=%.2f, success=%s, length=%d)",
            self.agent_id, episode_id, self.current_task.task_id,
            total_reward, success, len(self.episode_transitions)
        )

        # Clear buffer if requested
        if clear_buffer:
            self.episode_transitions.clear()

        return episode_id

    def store_policy_for_transfer(self, policy: Any):
        """
        Store current policy for future transfer learning.

        Args:
            policy: Policy object/parameters to store
        """
        if not self.knowledge_base or not self.current_task:
            return

        policy_key = f"{self.current_task.task_id}_{self.agent_id}"
        self.knowledge_base.store_policy(policy_key, policy)

        logger.debug(
            "%s: Stored policy for task %s",
            self.agent_id, self.current_task.task_id
        )

    def store_value_function_for_transfer(self, value_function: Any):
        """
        Store current value function for future transfer learning.

        Args:
            value_function: Value function object/parameters to store
        """
        if not self.knowledge_base or not self.current_task:
            return

        vf_key = f"{self.current_task.task_id}_{self.agent_id}"
        self.knowledge_base.store_value_function(vf_key, value_function)

        logger.debug(
            "%s: Stored value function for task %s",
            self.agent_id, self.current_task.task_id
        )

    def evaluate_transfer_performance(
        self,
        baseline_performance: float,
        current_performance: float,
        baseline_samples: int,
        current_samples: int
    ) -> Dict[str, Any]:
        """
        Evaluate effectiveness of transfer learning.

        Args:
            baseline_performance: Performance without transfer
            current_performance: Performance with transfer
            baseline_samples: Samples needed without transfer
            current_samples: Samples needed with transfer

        Returns:
            Dictionary with evaluation metrics including speedup factor
        """
        if not self.transfer_engine or not self.current_task:
            return {}

        return self.transfer_engine.evaluate_transfer(
            target_task_id=self.current_task.task_id,
            baseline_performance=baseline_performance,
            transfer_performance=current_performance,
            baseline_samples=baseline_samples,
            transfer_samples=current_samples
        )

    def get_transfer_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get transfer learning statistics for this agent.

        Returns:
            Dictionary with transfer metrics, or None if not available
        """
        if not self.transfer_engine:
            return None

        history = self.transfer_engine.get_transfer_history(agent_id=self.agent_id)

        if not history:
            return {
                'agent_id': self.agent_id,
                'num_transfers': 0,
                'avg_speedup': 0.0,
                'total_experiences_transferred': 0
            }

        speedups = [r.speedup_factor for r in history if r.speedup_factor is not None]
        total_experiences = sum(r.num_experiences_transferred for r in history)

        return {
            'agent_id': self.agent_id,
            'num_transfers': len(history),
            'avg_speedup': np.mean(speedups) if speedups else 0.0,
            'max_speedup': max(speedups) if speedups else 0.0,
            'total_experiences_transferred': total_experiences,
            'current_task': self.current_task.task_id if self.current_task else None,
            'transfer_enabled': self.transfer_enabled,
            'maml_enabled': self.maml_enabled
        }

    def get_maml_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get MAML meta-learning statistics for this agent.

        Returns:
            Dictionary with MAML metrics, or None if not available
        """
        if not self.maml_learner:
            return None

        history = self.maml_learner.get_adaptation_history(agent_id=self.agent_id)

        if not history:
            return {
                'agent_id': self.agent_id,
                'num_adaptations': 0,
                'avg_performance_gain': 0.0,
                'meta_initialized': self.maml_learner.meta_initialized
            }

        gains = [r.performance_gain for r in history]

        return {
            'agent_id': self.agent_id,
            'num_adaptations': len(history),
            'avg_performance_gain': np.mean(gains) if gains else 0.0,
            'max_performance_gain': max(gains) if gains else 0.0,
            'meta_initialized': self.maml_learner.meta_initialized,
            'current_task': self.current_task.task_id if self.current_task else None
        }

    # =========================================================================
    # BIG ROCK 9: EPISODIC MEMORY & REPLAY
    # =========================================================================

    def store_experience(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ):
        """
        Store experience in episodic memory for replay learning.

        This method stores experiences in the prioritized replay buffer,
        enabling the agent to learn from past experiences multiple times.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
            info: Optional additional information

        Example:
            >>> agent.store_experience(
            ...     state=np.array([1.0, 2.0, 3.0]),
            ...     action=1,
            ...     reward=1.5,
            ...     next_state=np.array([2.0, 3.0, 4.0]),
            ...     done=False
            ... )
        """
        if not self.episodic_memory or not self.replay_enabled:
            return

        # Store in episodic memory
        self.episodic_memory.store(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {}
        )

        logger.debug(
            "%s: Stored experience in episodic memory (size: %d)",
            self.agent_id, len(self.episodic_memory)
        )

    def learn_from_memory(
        self,
        num_batches: int = 1,
        batch_size: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Learn from experiences stored in episodic memory.

        This enables the agent to improve through offline learning by
        replaying past experiences with prioritized sampling.

        Args:
            num_batches: Number of batches to replay
            batch_size: Size of each batch (uses default if None)

        Returns:
            Dictionary with replay statistics, or None if not available

        Example:
            >>> # Learn from 10 batches of experiences
            >>> stats = agent.learn_from_memory(num_batches=10)
            >>> print(f"Average loss: {stats['mean_loss']:.4f}")
        """
        if not self.episodic_memory or not self.replay_enabled:
            return None

        if len(self.episodic_memory) < (batch_size or self.replay_batch_size):
            logger.debug(
                "%s: Not enough experiences for replay (%d < %d)",
                self.agent_id, len(self.episodic_memory),
                batch_size or self.replay_batch_size
            )
            return None

        batch_size = batch_size or self.replay_batch_size
        total_loss = 0.0
        total_td_error = 0.0
        experiences_replayed = 0

        for _ in range(num_batches):
            # Sample prioritized batch
            batch, indices, weights = self.episodic_memory.sample(batch_size=batch_size)

            # Convert to format needed by learning algorithm
            # This is a placeholder - subclasses should override with specific logic
            td_errors, loss = self._learn_from_batch(batch, weights)

            # Update priorities based on TD errors
            self.episodic_memory.update_priorities(indices, td_errors)

            total_loss += loss
            total_td_error += np.mean(np.abs(td_errors))
            experiences_replayed += len(batch)

        self.total_replays += num_batches

        stats = {
            'agent_id': self.agent_id,
            'num_batches': num_batches,
            'batch_size': batch_size,
            'experiences_replayed': experiences_replayed,
            'mean_loss': total_loss / num_batches,
            'mean_td_error': total_td_error / num_batches,
            'total_replays': self.total_replays,
            'memory_size': len(self.episodic_memory)
        }

        logger.debug(
            "%s: Learned from memory - %d batches, loss=%.4f",
            self.agent_id, num_batches, stats['mean_loss']
        )

        return stats

    def _learn_from_batch(
        self,
        batch: List[Experience],
        weights: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Learn from a batch of experiences.

        This is a placeholder method that subclasses should override
        with their specific learning algorithm (DQN, PPO, etc.).

        Args:
            batch: List of Experience objects
            weights: Importance sampling weights

        Returns:
            Tuple of (TD errors array, loss value)
        """
        # Placeholder implementation
        # Subclasses should implement actual learning logic
        td_errors = np.random.randn(len(batch)) * 0.1  # Dummy TD errors
        loss = np.random.rand() * 0.5  # Dummy loss

        return td_errors, loss

    def consolidate_memory(
        self,
        num_steps: Optional[int] = None,
        strategy: Optional[ConsolidationStrategy] = None
    ) -> Optional[ConsolidationResult]:
        """
        Perform memory consolidation ("sleep" phase) for offline learning.

        This method triggers intensive replay of high-priority experiences
        to strengthen learning without environment interaction.

        Args:
            num_steps: Number of consolidation steps (uses consolidator default if None)
            strategy: Consolidation strategy (uses consolidator default if None)

        Returns:
            ConsolidationResult with statistics, or None if not available

        Example:
            >>> # Perform consolidation with prioritized strategy
            >>> result = agent.consolidate_memory(
            ...     num_steps=100,
            ...     strategy=ConsolidationStrategy.PRIORITIZED
            ... )
            >>> print(f"Loss reduction: {result.loss_reduction:.2%}")
        """
        if not self.memory_consolidator or not self.consolidation_enabled:
            return None

        if len(self.episodic_memory) < self.memory_consolidator.min_memory_size:
            logger.debug(
                "%s: Memory too small for consolidation (%d < %d)",
                self.agent_id, len(self.episodic_memory),
                self.memory_consolidator.min_memory_size
            )
            return None

        logger.info(
            "%s: Starting memory consolidation (step %d)",
            self.agent_id, self.step_count
        )

        # Perform consolidation
        result = self.memory_consolidator.consolidate(
            agent=self,
            num_steps=num_steps,
            strategy=strategy
        )

        self.total_consolidations += 1

        logger.info(
            "%s: Consolidation complete - loss reduction: %.2f%%, time: %.2fs",
            self.agent_id, result.loss_reduction * 100, result.elapsed_time
        )

        # BIG ROCK 5: Emit signal if available
        if self.signal_bus:
            self.signal_bus.emit(
                agent_id=self.agent_id,
                signal_type=SignalType.LEARNING_MILESTONE,
                priority=SignalPriority.NORMAL,
                data={
                    'event': 'memory_consolidation',
                    'loss_reduction': result.loss_reduction,
                    'steps': result.steps
                }
            )

        return result

    def should_consolidate(self) -> bool:
        """
        Check if memory consolidation should be triggered.

        Returns:
            True if consolidation should happen, False otherwise
        """
        if not self.memory_consolidator or not self.consolidation_enabled:
            return False

        return self.memory_consolidator.should_consolidate(
            current_step=self.step_count
        )

    def search_similar_experiences(
        self,
        state: np.ndarray,
        k: int = 5
    ) -> Optional[SemanticQuery]:
        """
        Search for similar past experiences using semantic retrieval.

        This enables context-aware decision making by finding experiences
        in similar states.

        Args:
            state: Current state to query
            k: Number of similar experiences to retrieve

        Returns:
            SemanticQuery with similar experiences, or None if not available

        Example:
            >>> # Find 5 most similar past experiences
            >>> similar = agent.search_similar_experiences(
            ...     state=current_state,
            ...     k=5
            ... )
            >>> for exp in similar.experiences:
            ...     print(f"Reward: {exp['reward']}, Distance: {exp['distance']}")
        """
        if not self.semantic_retriever or not self.semantic_search_enabled:
            return None

        return self.semantic_retriever.search_by_state(state, k=k)

    def get_counterfactual_experiences(
        self,
        state: np.ndarray,
        action: int,
        k: int = 3
    ) -> Optional[SemanticQuery]:
        """
        Query: "What happened when I took this action in similar states?"

        This enables counterfactual reasoning and action comparison.

        Args:
            state: Query state
            action: Action to query about
            k: Number of experiences to retrieve

        Returns:
            SemanticQuery with counterfactual experiences, or None if not available

        Example:
            >>> # What happened when taking action 2 in similar states?
            >>> counterfactual = agent.get_counterfactual_experiences(
            ...     state=current_state,
            ...     action=2,
            ...     k=3
            ... )
            >>> avg_reward = np.mean([exp['reward'] for exp in counterfactual.experiences])
        """
        if not self.semantic_retriever or not self.semantic_search_enabled:
            return None

        return self.semantic_retriever.get_counterfactual_experiences(
            state=state,
            action=action,
            k=k
        )

    def get_episodic_memory_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive statistics about episodic memory system.

        Returns:
            Dictionary with memory statistics, or None if not available

        Example:
            >>> stats = agent.get_episodic_memory_statistics()
            >>> print(f"Memory size: {stats['memory_size']}")
            >>> print(f"Total replays: {stats['total_replays']}")
            >>> print(f"Consolidations: {stats['total_consolidations']}")
        """
        if not self.episodic_memory:
            return None

        stats = {
            'agent_id': self.agent_id,
            'replay_enabled': self.replay_enabled,
            'consolidation_enabled': self.consolidation_enabled,
            'semantic_search_enabled': self.semantic_search_enabled,
            'memory_size': len(self.episodic_memory),
            'memory_capacity': self.episodic_memory.capacity,
            'memory_utilization': len(self.episodic_memory) / self.episodic_memory.capacity,
            'total_stored': self.episodic_memory.total_stored,
            'total_replayed': self.episodic_memory.total_replayed,
            'total_replays': self.total_replays,
            'total_consolidations': self.total_consolidations,
            'replay_frequency': self.replay_frequency,
            'replay_batch_size': self.replay_batch_size,
            'current_beta': self.episodic_memory.beta,
        }

        # Add consolidation stats if available
        if self.memory_consolidator:
            consolidation_stats = self.memory_consolidator.get_consolidation_statistics()
            stats['consolidation'] = consolidation_stats

        # Add semantic retrieval stats if available
        if self.semantic_retriever:
            semantic_stats = self.semantic_retriever.get_statistics()
            stats['semantic_retrieval'] = semantic_stats

        return stats

    def get_learning_rate(self) -> float:
        """
        Get current learning rate.

        This is a placeholder that subclasses should override
        to return their actual learning rate.

        Returns:
            Current learning rate
        """
        # Placeholder - subclasses should override
        return self.agent_config.get('learning_rate', 0.001)

    def set_learning_rate(self, lr: float):
        """
        Set learning rate.

        This is a placeholder that subclasses should override
        to actually update their optimizer's learning rate.

        Args:
            lr: New learning rate
        """
        # Placeholder - subclasses should override
        self.agent_config['learning_rate'] = lr
