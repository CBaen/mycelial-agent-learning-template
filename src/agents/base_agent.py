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
from typing import Dict, Any, Optional, List
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
        agent_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Mycelial Agent.

        Args:
            model: The MycelialModel this agent belongs to
            redis_client: Redis client for data operations and communication
            unique_id: Optional unique identifier for this agent (auto-assigned if None)
            team_id: Team identifier for collaboration (Rule of 3)
            agent_config: Optional configuration dictionary for agent parameters
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
