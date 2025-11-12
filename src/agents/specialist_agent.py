"""
Specialist Agent for the Mycelial ABM Framework

This is the main "worker" agent that consumes data from DataMinerAgents,
applies its learned policy, and cooperates with other specialist agents
through federated learning and value decomposition.
"""

from typing import Dict, Any, Optional, List, Set
import logging
import numpy as np

from agents.base_agent import MycelialAgent

logger = logging.getLogger(__name__)


class SpecialistAgent(MycelialAgent):
    """
    Specialist Agent that performs specialized tasks with learned policies.

    This agent is the primary learning component of the system that:
    - Consumes processed data from DataMinerAgents
    - Applies learned policies to make decisions
    - Cooperates with other specialists via FRL (policy sharing)
    - Participates in VDN for credit assignment
    - Adapts its behavior based on performance and peer learning

    Typical use cases:
    - Classification or prediction tasks
    - Optimization and decision-making
    - Resource allocation
    - Pattern recognition
    - Collaborative problem solving
    """

    def __init__(
        self,
        unique_id: int,
        model,
        redis_client,
        data_channel: str,
        team_id: str = "specialists",
        specialization: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Specialist Agent.

        Args:
            unique_id: Unique identifier for this agent
            model: The Mesa model this agent belongs to
            redis_client: Redis client for data operations
            data_channel: Pub/Sub channel to receive data from
            team_id: Team identifier for Rule of 3 collaboration
            specialization: Optional specialization type (e.g., "classifier", "optimizer")
            agent_config: Optional configuration dictionary
        """
        super().__init__(unique_id, model, redis_client, team_id, agent_config)

        # Connect to Vector DB from model (for teammate policy sharing)
        self.vector_db = model.vector_db if hasattr(model, 'vector_db') else None

        # Connect to FRL engine from model (for inter-team learning)
        self.frl_engine = model.frl_engine if hasattr(model, 'frl_engine') else None

        # Connect to VDN engine from model (for credit assignment)
        self.vdn_engine = model.vdn_engine if hasattr(model, 'vdn_engine') else None

        # Agent specialization
        self.specialization = specialization or "general"
        self.data_channel = data_channel

        # Subscribe to data channel
        self.subscribe_to_channel(self.data_channel)

        # Learning configuration
        self.learning_rate = agent_config.get("learning_rate", 0.01) if agent_config else 0.01
        self.exploration_rate = agent_config.get("exploration_rate", 0.1) if agent_config else 0.1
        self.exploration_decay = agent_config.get("exploration_decay", 0.995) if agent_config else 0.995
        self.min_exploration = agent_config.get("min_exploration", 0.01) if agent_config else 0.01

        # Policy configuration (can be neural network, Q-table, etc.)
        self.policy_type = agent_config.get("policy_type", "simple") if agent_config else "simple"
        self._initialize_policy()

        # Experience replay buffer
        self.experience_buffer: List[Dict[str, Any]] = []
        self.buffer_size = agent_config.get("buffer_size", 1000) if agent_config else 1000

        # Collaboration tracking
        self.peer_specialists: Set[str] = set()
        self.collaboration_count: int = 0

        # Task execution metrics
        self.tasks_completed: int = 0
        self.tasks_successful: int = 0
        self.average_task_reward: float = 0.0

        # Data consumption tracking
        self.data_messages_received: int = 0
        self.last_data_timestamp: Optional[int] = None

        logger.info("SpecialistAgent %s initialized with specialization: %s",
                   self.agent_id, self.specialization)

    def step(self):
        """
        Execute one step of specialist behavior.

        Flow:
        1. Receive data from data channel
        2. Process data and extract task
        3. Select action using policy (with exploration)
        4. Execute action and receive reward
        5. Update policy using RL algorithm
        6. Share policy with peers (FRL)
        7. Update collaboration state
        """
        self.step_count += 1

        # Receive and process incoming data
        task_data = self._receive_data_from_channel()

        if task_data is not None:
            # Observe state from task data
            self.current_state = self._extract_state_from_data(task_data)

            # Select action using current policy
            action = self._select_action(self.current_state)
            self.last_action = action

            # Execute action and get reward
            reward = self._execute_action(action)

            # Get local credit if VDN is enabled
            if self.vdn_engine is not None:
                local_reward = self.get_local_reward(reward)
            else:
                local_reward = reward

            self.last_reward = local_reward
            self.cumulative_reward += local_reward

            # Store experience for learning
            self._store_experience(self.current_state, action, local_reward)

            # Update policy based on experience
            self._update_policy(self.current_state, action, local_reward)

            # Update task metrics
            self.tasks_completed += 1
            if local_reward > 0:
                self.tasks_successful += 1

            # Update average task reward
            self._update_average_task_reward(local_reward)

        # Decay exploration rate
        self._decay_exploration()

        # Share policy with team via Vector DB (Rule of 3)
        if self.vector_db is not None and self._should_share_with_team():
            self._update_policy_embedding()
            policy_id = self.share_policy_with_team()
            if policy_id:
                logger.debug("%s shared policy %s with team", self.agent_id, policy_id)

        # Learn from teammate policies via Vector DB
        if self.vector_db is not None and self.step_count % 20 == 0:
            self._learn_from_teammates()

        # Share policy with peers via FRL (inter-team)
        if self.frl_engine is not None and self._should_share_policy():
            num_shared = self.share_policy()
            if num_shared > 0:
                self.collaboration_count += 1

        # Receive policy updates from peers via FRL
        if self.frl_engine is not None:
            self._receive_peer_policies()

        # Update performance metrics
        self._update_performance_metrics(self.last_reward)

        # Save state periodically
        if self.step_count % 20 == 0:
            self._save_state_to_redis()

        logger.debug("%s completed step %d (tasks: %d, success rate: %.2f%%)",
                    self.agent_id, self.step_count, self.tasks_completed,
                    self._get_success_rate() * 100)

    def _initialize_policy(self):
        """
        Initialize the agent's policy based on policy_type.

        Override this method to implement custom policy initialization
        (e.g., neural network, decision tree, Q-table).
        """
        # Default: simple dictionary-based policy
        if self.policy_type == "simple":
            self.policy_parameters = {
                "weights": {},
                "bias": 0.0,
                "initialized": True
            }
        else:
            # Placeholder for other policy types
            self.policy_parameters = {
                "policy_type": self.policy_type,
                "initialized": False
            }

        logger.debug("%s initialized %s policy", self.agent_id, self.policy_type)

    def _receive_data_from_channel(self) -> Optional[Dict[str, Any]]:
        """
        Receive data from the subscribed data channel.

        Returns:
            Task data dictionary, or None if no data available
        """
        # Note: In a real implementation, you would listen to the pub/sub
        # For this template, we simulate receiving data
        # Override this method to implement actual pub/sub listening

        # Placeholder: Return None (no data received)
        # In practice, this would check the pub/sub connection
        return None

    def _extract_state_from_data(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract state representation from task data.

        Override this method to implement custom state extraction.

        Args:
            task_data: Raw task data received

        Returns:
            State dictionary
        """
        # Default: use task data as-is
        return {
            "task_data": task_data,
            "agent_state": {
                "cumulative_reward": self.cumulative_reward,
                "tasks_completed": self.tasks_completed,
                "exploration_rate": self.exploration_rate
            }
        }

    def _select_action(self, state: Dict[str, Any]) -> Any:
        """
        Select an action based on current state and policy.

        Implements epsilon-greedy exploration strategy.

        Args:
            state: Current observed state

        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            action = self._get_random_action()
            logger.debug("%s exploring with random action", self.agent_id)
        else:
            # Exploit: use policy
            action = self._get_policy_action(state)

        return action

    def _get_random_action(self) -> Any:
        """
        Get a random action for exploration.

        Override this method to define the action space.

        Returns:
            Random action
        """
        # Default: simple binary action
        return np.random.choice([0, 1])

    def _get_policy_action(self, state: Dict[str, Any]) -> Any:
        """
        Get action from current policy (exploitation).

        Override this method to implement policy forward pass.

        Args:
            state: Current state

        Returns:
            Action selected by policy
        """
        # Default: simple policy (placeholder)
        # In practice, this would be a neural network forward pass,
        # Q-table lookup, or other policy evaluation

        # Simple heuristic: action based on cumulative reward
        if self.cumulative_reward > 0:
            return 1
        else:
            return 0

    def _execute_action(self, action: Any) -> float:
        """
        Execute the selected action and receive reward.

        Override this method to implement task-specific action execution.

        Args:
            action: The action to execute

        Returns:
            Reward received from executing the action
        """
        # Default: simulate action execution with simple reward
        # Override this to implement actual task execution

        # Placeholder reward function
        if action == 1:
            reward = np.random.normal(1.0, 0.5)
        else:
            reward = np.random.normal(0.0, 0.5)

        return max(0.0, reward)  # Non-negative rewards

    def _update_policy(self, state: Dict[str, Any], action: Any, reward: float):
        """
        Update the agent's policy based on experience.

        Implements a simple learning algorithm. Override for custom learning.

        Args:
            state: State where action was taken
            action: Action that was taken
            reward: Reward received (after credit assignment)
        """
        # Simple policy update (placeholder)
        # In practice, this would be Q-learning, policy gradient, etc.

        # Update policy version when significant update occurs
        if abs(reward) > 0.5:
            self.policy_version += 1

        # For this template, we just track that learning occurred
        logger.debug("%s updated policy to v%d with reward %.4f",
                    self.agent_id, self.policy_version, reward)

    def _store_experience(self, state: Dict[str, Any], action: Any, reward: float):
        """
        Store experience in replay buffer for learning.

        Args:
            state: State observed
            action: Action taken
            reward: Reward received
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "timestamp": self.step_count
        }

        self.experience_buffer.append(experience)

        # Limit buffer size
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)

    def _receive_peer_policies(self):
        """
        Receive and integrate policy updates from peer specialists.
        """
        if self.frl_engine is None:
            return

        try:
            # Receive policy updates from peers
            peer_updates = self.frl_engine.receive_policy_updates(timeout_ms=100)

            if peer_updates:
                # Aggregate with local policy
                updated_policy = self.frl_engine.aggregate_policy_updates(
                    local_policy=self.policy_parameters,
                    peer_updates=peer_updates
                )

                # Update local policy
                self.policy_parameters = updated_policy
                self.policy_version += 1

                logger.debug("%s integrated %d peer policy updates",
                           self.agent_id, len(peer_updates))

        except Exception as e:
            logger.error("%s: Error receiving peer policies: %s", self.agent_id, e)

    def _decay_exploration(self):
        """
        Decay the exploration rate over time.
        """
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

    def _should_share_policy(self) -> bool:
        """
        Determine if the agent should share its policy this step.

        Override for custom sharing logic.

        Returns:
            True if policy should be shared
        """
        # Share after significant learning or periodically
        return (self.policy_version > 0 and self.tasks_completed % 10 == 0)

    def _should_share_with_team(self) -> bool:
        """
        Determine if the agent should share with teammates via Vector DB.

        Returns:
            True if policy should be shared with team
        """
        # Share with team more frequently than global FRL
        return (self.policy_version > 0 and self.tasks_completed % 5 == 0)

    def _update_policy_embedding(self):
        """
        Update the policy embedding for Vector DB storage.

        This method converts the policy parameters into a vector representation
        that can be stored in the Vector DB for semantic search.
        """
        # Create embedding from policy parameters
        # In a real implementation, this would use a neural network encoder
        # For this template, we create a simple embedding

        if isinstance(self.policy_parameters, dict):
            # Extract numeric values from policy
            values = []
            for key, value in sorted(self.policy_parameters.items()):
                if isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, dict):
                    # Flatten nested dicts
                    values.extend([float(v) for v in value.values() if isinstance(v, (int, float))])

            # Pad or truncate to fixed size (e.g., 128 dimensions)
            embedding_dim = 128
            if len(values) < embedding_dim:
                # Pad with performance metrics
                values.extend([
                    self.cumulative_reward,
                    float(self.tasks_completed),
                    self._get_success_rate(),
                    self.exploration_rate,
                    float(self.policy_version)
                ])

            # Pad remaining with zeros
            while len(values) < embedding_dim:
                values.append(0.0)

            # Truncate if too long
            embedding = np.array(values[:embedding_dim])

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            self.policy_embedding = embedding.tolist()

            logger.debug("%s updated policy embedding (dim: %d)",
                        self.agent_id, len(self.policy_embedding))

    def _learn_from_teammates(self):
        """
        Query Vector DB for high-performing teammate policies and learn from them.

        This implements the "Rule of 3" team collaboration by retrieving
        similar policies from teammates and integrating their insights.
        """
        if self.policy_embedding is None:
            self._update_policy_embedding()

        try:
            # Retrieve top-performing teammate policies
            teammate_policies = self.retrieve_teammate_policies(
                top_k=3,
                min_performance=self.average_task_reward  # Only better performers
            )

            if teammate_policies:
                # Integrate teammate insights into local policy
                for policy_info in teammate_policies:
                    self._integrate_teammate_policy(policy_info)

                self.collaboration_count += len(teammate_policies)

                logger.info("%s learned from %d teammate policies",
                           self.agent_id, len(teammate_policies))

        except Exception as e:
            logger.error("%s: Error learning from teammates: %s", self.agent_id, e)

    def _integrate_teammate_policy(self, policy_info: Dict[str, Any]):
        """
        Integrate insights from a teammate's policy.

        Args:
            policy_info: Dictionary with teammate policy information
        """
        # Extract metadata
        metadata = policy_info.get("metadata", {})
        teammate_performance = metadata.get("performance", 0.0)
        similarity = policy_info.get("similarity", 0.0)

        # Only integrate if teammate is performing better
        if teammate_performance > self.average_task_reward:
            # Simple integration: adjust policy parameters slightly toward teammate
            # In a real implementation, this would be more sophisticated

            # Weight by performance and similarity
            integration_weight = similarity * 0.1  # Small steps

            logger.debug("%s integrating policy from %s (similarity: %.3f, weight: %.3f)",
                        self.agent_id, metadata.get("agent_id"), similarity, integration_weight)

            # Increment policy version to indicate learning
            self.policy_version += 1

    # ==========================================
    # Collaboration methods
    # ==========================================

    def add_peer_specialist(self, peer_id: str):
        """
        Add a peer specialist for collaboration.

        Args:
            peer_id: ID of the peer specialist
        """
        self.peer_specialists.add(peer_id)

        # If FRL engine exists, connect to peer
        if self.frl_engine is not None:
            self.frl_engine.connect_to_peer(peer_id)

        # If VDN engine exists, add peer
        if self.vdn_engine is not None:
            self.vdn_engine.add_peer_agent(peer_id)

        logger.info("%s added peer specialist: %s", self.agent_id, peer_id)

    def remove_peer_specialist(self, peer_id: str):
        """
        Remove a peer specialist from collaboration.

        Args:
            peer_id: ID of the peer specialist
        """
        self.peer_specialists.discard(peer_id)

        # If FRL engine exists, disconnect from peer
        if self.frl_engine is not None:
            self.frl_engine.disconnect_from_peer(peer_id)

        # If VDN engine exists, remove peer
        if self.vdn_engine is not None:
            self.vdn_engine.remove_peer_agent(peer_id)

        logger.info("%s removed peer specialist: %s", self.agent_id, peer_id)

    # ==========================================
    # Metrics and monitoring
    # ==========================================

    def _update_average_task_reward(self, reward: float):
        """
        Update the rolling average task reward.

        Args:
            reward: Latest reward received
        """
        # Exponential moving average
        alpha = 0.1
        self.average_task_reward = (
            alpha * reward + (1 - alpha) * self.average_task_reward
        )

    def _get_success_rate(self) -> float:
        """
        Get the task success rate.

        Returns:
            Success rate (0.0 to 1.0)
        """
        if self.tasks_completed == 0:
            return 0.0
        return self.tasks_successful / self.tasks_completed

    def get_specialist_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about specialist performance.

        Returns:
            Dictionary with specialist metrics
        """
        return {
            "agent_id": self.agent_id,
            "specialization": self.specialization,
            "tasks_completed": self.tasks_completed,
            "tasks_successful": self.tasks_successful,
            "success_rate": self._get_success_rate(),
            "average_task_reward": self.average_task_reward,
            "cumulative_reward": self.cumulative_reward,
            "policy_version": self.policy_version,
            "exploration_rate": self.exploration_rate,
            "peer_count": len(self.peer_specialists),
            "collaboration_count": self.collaboration_count,
            "experience_buffer_size": len(self.experience_buffer)
        }

    def get_learning_progress(self) -> Dict[str, Any]:
        """
        Get information about learning progress.

        Returns:
            Dictionary with learning metrics
        """
        recent_performance = self._get_recent_performance()

        return {
            "agent_id": self.agent_id,
            "policy_version": self.policy_version,
            "exploration_rate": self.exploration_rate,
            "recent_performance": recent_performance,
            "performance_trend": self._calculate_performance_trend(),
            "is_learning": recent_performance > self.average_task_reward,
            "experience_count": len(self.experience_buffer)
        }

    def _calculate_performance_trend(self) -> str:
        """
        Calculate trend in performance (improving, stable, declining).

        Returns:
            Trend description
        """
        if len(self.performance_history) < 20:
            return "insufficient_data"

        recent = self.performance_history[-10:]
        older = self.performance_history[-20:-10]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"

    def _save_state_to_redis(self):
        """
        Save agent state including learning state to Redis.
        """
        # Call parent method for basic state
        super()._save_state_to_redis()

        # Save additional specialist state
        specialist_state = {
            "specialization": self.specialization,
            "tasks_completed": self.tasks_completed,
            "tasks_successful": self.tasks_successful,
            "average_task_reward": self.average_task_reward,
            "exploration_rate": self.exploration_rate,
            "policy_parameters": self.policy_parameters,
            "peer_specialists": list(self.peer_specialists)
        }

        key = f"agent:specialist_state:{self.agent_id}"
        self.redis_client.set_key_value(key, specialist_state)

    def reset(self):
        """
        Reset the specialist agent to initial state.
        """
        super().reset()

        # Reset learning state
        self.exploration_rate = self.agent_config.get("exploration_rate", 0.1) if self.agent_config else 0.1
        self._initialize_policy()

        # Reset metrics
        self.tasks_completed = 0
        self.tasks_successful = 0
        self.average_task_reward = 0.0
        self.collaboration_count = 0
        self.experience_buffer.clear()

        logger.info("%s has been reset", self.agent_id)
