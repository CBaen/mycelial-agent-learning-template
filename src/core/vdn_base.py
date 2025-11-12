"""
Value-Decomposition Networks (VDN) Abstract Base Class

This module defines the abstract interface for solving the multi-agent credit
assignment problem through value decomposition. VDN enables agents to learn
individual value functions that can be additively combined into a joint value
function, allowing decentralized execution with centralized training insights.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DecompositionMethod(Enum):
    """Methods for decomposing joint value into individual agent values."""
    ADDITIVE = "additive"  # Q_tot = sum(Q_i) - Standard VDN
    MONOTONIC = "monotonic"  # Q_tot = f(Q_1, ..., Q_n) where f is monotonic - QMIX
    WEIGHTED = "weighted"  # Q_tot = sum(w_i * Q_i) - Weighted VDN
    CONTEXT_DEPENDENT = "context_dependent"  # Decomposition depends on state context
    HIERARCHICAL = "hierarchical"  # Multi-level decomposition


class CreditAssignmentStrategy(Enum):
    """Strategies for assigning credit to individual agents."""
    DIFFERENCE_REWARDS = "difference_rewards"  # Credit based on marginal contribution
    SHAPLEY_VALUE = "shapley_value"  # Game-theoretic fair credit assignment
    COUNTERFACTUAL = "counterfactual"  # Credit based on counterfactual reasoning
    ATTENTION_BASED = "attention_based"  # Credit weighted by agent attention/importance
    TEMPORAL = "temporal"  # Credit considers temporal dependencies


class ValueDecompositionEngine(ABC):
    """
    Abstract base class for Value-Decomposition Networks.

    This engine solves the multi-agent credit assignment problem by decomposing
    the global reward/value into individual agent contributions, enabling:
    - Decentralized policy execution
    - Fair credit assignment
    - Scalable multi-agent learning
    - Individual agent accountability

    Key Principles:
    - Individual agents maintain local Q-functions Q_i(s, a_i)
    - Global value Q_tot can be computed from individual values
    - Decomposition preserves optimal action selection consistency
    - Enables off-policy learning with experience replay
    """

    def __init__(
        self,
        agent_id: str,
        redis_client: Any,
        decomposition_method: DecompositionMethod = DecompositionMethod.ADDITIVE,
        credit_strategy: CreditAssignmentStrategy = CreditAssignmentStrategy.DIFFERENCE_REWARDS
    ):
        """
        Initialize the Value Decomposition Engine.

        Args:
            agent_id: Unique identifier for this agent
            redis_client: Redis client for sharing value information
            decomposition_method: Method for decomposing joint value
            credit_strategy: Strategy for assigning credit to agents
        """
        self.agent_id = agent_id
        self.redis_client = redis_client
        self.decomposition_method = decomposition_method
        self.credit_strategy = credit_strategy

        # Track other agents in the system
        self.peer_agents: Set[str] = set()

        # Cache for recent joint values and decompositions
        self.value_cache: Dict[str, Any] = {}

        # Statistics for monitoring
        self.credit_history: List[float] = []
        self.contribution_metrics: Dict[str, float] = {}

    @abstractmethod
    def compute_local_value(
        self,
        state: Dict[str, Any],
        action: Any,
        local_observation: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute the local value Q_i(s, a_i) for this agent.

        Args:
            state: Global state (may be partially observed)
            action: The action taken by this agent
            local_observation: Agent's local observation (if partially observable)

        Returns:
            Local Q-value for this agent's state-action pair
        """
        pass

    @abstractmethod
    def compute_joint_value(
        self,
        state: Dict[str, Any],
        joint_action: Dict[str, Any],
        individual_values: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute the joint value Q_tot from individual agent values.

        Implements the value decomposition: Q_tot = f(Q_1, Q_2, ..., Q_n)

        Args:
            state: Global state
            joint_action: Dictionary mapping agent_id to action
            individual_values: Pre-computed individual values (optional optimization)

        Returns:
            Joint Q-value for the multi-agent system
        """
        pass

    @abstractmethod
    def decompose_global_value(
        self,
        global_value: float,
        state: Dict[str, Any],
        joint_action: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Decompose a global value into individual agent contributions.

        This is the inverse operation of compute_joint_value, used for
        credit assignment during training.

        Args:
            global_value: The total system value/reward
            state: Global state
            joint_action: Dictionary mapping agent_id to action

        Returns:
            Dictionary mapping agent_id to individual value contribution
        """
        pass

    @abstractmethod
    def assign_credit(
        self,
        global_reward: float,
        state: Dict[str, Any],
        joint_action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Assign credit to this agent from the global reward.

        Uses the configured credit_strategy to determine this agent's
        share of the global reward signal.

        Args:
            global_reward: The total reward received by the system
            state: State before action
            joint_action: Actions taken by all agents
            next_state: State after action

        Returns:
            Individual credit/reward for this agent
        """
        pass

    @abstractmethod
    def compute_marginal_contribution(
        self,
        state: Dict[str, Any],
        joint_action: Dict[str, Any]
    ) -> float:
        """
        Compute this agent's marginal contribution to system value.

        Marginal contribution = Q_tot(with agent) - Q_tot(without agent)

        Args:
            state: Global state
            joint_action: Actions taken by all agents

        Returns:
            Marginal contribution value
        """
        pass

    @abstractmethod
    def compute_counterfactual_baseline(
        self,
        state: Dict[str, Any],
        joint_action: Dict[str, Any],
        agent_to_evaluate: Optional[str] = None
    ) -> float:
        """
        Compute counterfactual baseline for credit assignment.

        Estimates "what would have happened" if the agent had not acted
        or had taken a default action.

        Args:
            state: Global state
            joint_action: Actual joint action taken
            agent_to_evaluate: Which agent to compute baseline for (default: self)

        Returns:
            Counterfactual baseline value
        """
        pass

    @abstractmethod
    def update_value_function(
        self,
        state: Dict[str, Any],
        action: Any,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
        global_info: Optional[Dict[str, Any]] = None
    ):
        """
        Update the local value function based on experience.

        Args:
            state: Current state
            action: Action taken by this agent
            reward: Individual credit assigned to this agent
            next_state: Next state
            done: Whether episode terminated
            global_info: Optional global information for value decomposition
        """
        pass

    @abstractmethod
    def share_value_information(
        self,
        state: Dict[str, Any],
        action: Any,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Share value information with other agents via Redis.

        Enables other agents to compute joint values and perform
        credit assignment.

        Args:
            state: State for which value is computed
            action: Action taken
            value: Computed local value
            metadata: Additional information (e.g., confidence, timestamp)
        """
        pass

    @abstractmethod
    def receive_peer_values(
        self,
        state: Dict[str, Any],
        timeout_ms: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Receive value information from peer agents.

        Args:
            state: State for which to retrieve peer values
            timeout_ms: Optional timeout for waiting for peer values

        Returns:
            Dictionary mapping agent_id to their reported local values
        """
        pass

    @abstractmethod
    def validate_value_consistency(
        self,
        individual_values: Dict[str, float],
        joint_value: float,
        tolerance: float = 1e-6
    ) -> bool:
        """
        Validate that value decomposition is consistent.

        Checks that the decomposition method correctly reconstructs
        the joint value from individual values.

        Args:
            individual_values: Dictionary of agent values
            joint_value: The joint value to validate against
            tolerance: Numerical tolerance for floating point comparison

        Returns:
            True if decomposition is consistent
        """
        pass

    @abstractmethod
    def compute_mixing_weights(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute weights for mixing individual values (for non-additive methods).

        Used in methods like QMIX where the mixing function is state-dependent.

        Args:
            state: Global state

        Returns:
            Dictionary mapping agent_id to mixing weight
        """
        pass

    @abstractmethod
    def estimate_value_variance(
        self,
        state: Dict[str, Any],
        action: Any
    ) -> float:
        """
        Estimate uncertainty/variance in value estimate.

        Useful for exploration bonuses and confidence-aware credit assignment.

        Args:
            state: State to evaluate
            action: Action to evaluate

        Returns:
            Estimated variance in value prediction
        """
        pass

    @abstractmethod
    def get_value_decomposition_explanation(
        self,
        state: Dict[str, Any],
        joint_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate explanation for how value was decomposed.

        Provides interpretability for credit assignment decisions.

        Args:
            state: Global state
            joint_action: Joint action taken

        Returns:
            Dictionary containing:
            - individual_values: Each agent's value
            - joint_value: Total value
            - contribution_percentages: % contribution per agent
            - decomposition_method: Method used
            - explanation: Human-readable explanation
        """
        pass

    @abstractmethod
    def sync_value_functions(
        self,
        peer_value_functions: Dict[str, Any]
    ):
        """
        Synchronize value function with peers (for centralized training).

        Args:
            peer_value_functions: Dictionary mapping agent_id to value function params
        """
        pass

    @abstractmethod
    def compute_value_gradient(
        self,
        state: Dict[str, Any],
        action: Any
    ) -> Dict[str, Any]:
        """
        Compute gradient of value function w.r.t. action.

        Used for policy gradient methods and coordination.

        Args:
            state: State to evaluate
            action: Action to evaluate

        Returns:
            Gradient information (structure depends on action space)
        """
        pass

    @abstractmethod
    def get_credit_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about credit assignment.

        Returns:
            Dictionary containing:
            - total_credit_received: Cumulative credit
            - average_credit: Mean credit per step
            - credit_variance: Variance in credit
            - contribution_rank: This agent's rank vs peers
            - fairness_metric: Measure of credit assignment fairness
        """
        pass

    def add_peer_agent(self, agent_id: str):
        """Add a peer agent to track for value decomposition."""
        self.peer_agents.add(agent_id)
        logger.debug("Agent %s added peer %s for VDN", self.agent_id, agent_id)

    def remove_peer_agent(self, agent_id: str):
        """Remove a peer agent from tracking."""
        self.peer_agents.discard(agent_id)
        logger.debug("Agent %s removed peer %s from VDN", self.agent_id, agent_id)

    def get_peer_count(self) -> int:
        """Get the number of peer agents in the system."""
        return len(self.peer_agents)

    def record_credit(self, credit: float):
        """Record credit received for statistics."""
        self.credit_history.append(credit)
        if len(self.credit_history) > 10000:  # Limit history size
            self.credit_history = self.credit_history[-10000:]

    def get_average_credit(self) -> float:
        """Get average credit received."""
        return sum(self.credit_history) / len(self.credit_history) if self.credit_history else 0.0
