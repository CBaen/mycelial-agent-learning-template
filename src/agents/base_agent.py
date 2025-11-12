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
        unique_id: int,
        model,
        redis_client,
        team_id: str = "default",
        agent_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Mycelial Agent.

        Args:
            unique_id: Unique identifier for this agent (required by Mesa)
            model: The MycelialModel this agent belongs to
            redis_client: Redis client for data operations and communication
            team_id: Team identifier for collaboration (Rule of 3)
            agent_config: Optional configuration dictionary for agent parameters
        """
        super().__init__(unique_id, model)

        self.redis_client = redis_client
        self.agent_config = agent_config or {}

        # Agent identification
        self.agent_type = self.__class__.__name__
        self.agent_id = f"{self.agent_type}_{unique_id}"

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
