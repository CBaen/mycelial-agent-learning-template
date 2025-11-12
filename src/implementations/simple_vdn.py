"""
SimpleVDN - Reference Implementation of Value-Decomposition Networks

This module provides a simple implementation of VDN for multi-agent credit assignment.
It uses additive value decomposition: Q_total = Q_1 + Q_2 + ... + Q_n

Use this as:
1. A starting point for your own VDN implementation
2. A reference for understanding credit assignment
3. A functional engine for prototyping

For advanced use cases (QMIX, QTRAN, attention mechanisms), extend this or
build your own implementation.
"""

import sys
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.vdn_base import (
    ValueDecompositionEngine,
    DecompositionMethod,
    CreditAssignmentStrategy
)
from src.connectors.redis_client import RedisClient

logger = logging.getLogger(__name__)


class SimpleVDN(ValueDecompositionEngine):
    """
    Simple Value-Decomposition Network for Multi-Agent Credit Assignment.

    Features:
    - Additive value decomposition (Q_total = sum of Q_i)
    - Simple tabular Q-learning (can be replaced with neural nets)
    - Difference rewards for credit assignment
    - Marginal contribution estimation

    This is a minimal implementation suitable for:
    - Small to medium-sized problems
    - Discrete action spaces
    - Prototyping and testing
    """

    def __init__(
        self,
        agent_id: str,
        redis_client: RedisClient,
        decomposition_method: DecompositionMethod = DecompositionMethod.ADDITIVE,
        credit_strategy: CreditAssignmentStrategy = CreditAssignmentStrategy.DIFFERENCE_REWARDS,
        state_dim: int = 10,
        action_dim: int = 5,
        learning_rate: float = 0.01,
        discount_factor: float = 0.99,
        history_size: int = 1000
    ):
        """
        Initialize SimpleVDN engine.

        Args:
            agent_id: Unique agent identifier
            redis_client: Redis client for coordination
            decomposition_method: How to decompose joint value
            credit_strategy: How to assign credit
            state_dim: Dimension of state space
            action_dim: Number of actions
            learning_rate: Learning rate for value updates
            discount_factor: Discount factor (gamma)
            history_size: Size of experience history
        """
        self.agent_id = agent_id
        self.redis_client = redis_client
        self.decomposition_method = decomposition_method
        self.credit_strategy = credit_strategy
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.history_size = history_size

        # Value function (simple tabular for prototyping)
        # In production, replace with neural network
        self.q_table: Dict[tuple, np.ndarray] = {}  # state -> action values
        self.default_q_value = 0.0

        # Experience history
        self.experience_buffer = deque(maxlen=history_size)

        # Statistics
        self.total_updates = 0
        self.total_credits_assigned = 0
        self.credit_history = deque(maxlen=100)

        # Team coordination
        self.team_values: Dict[str, float] = {}  # agent_id -> local value
        self.joint_actions: Dict[str, Any] = {}  # agent_id -> action

        logger.info("SimpleVDN initialized for agent %s (state_dim=%d, action_dim=%d)",
                   agent_id, state_dim, action_dim)

    # =========================================================================
    # Value Computation
    # =========================================================================

    def compute_local_value(
        self,
        state: Dict[str, Any],
        action: Any,
        local_observation: Dict[str, Any]
    ) -> float:
        """
        Compute local Q-value for this agent.

        Q_i(s, a_i) = local value contribution

        Args:
            state: Global state
            action: This agent's action
            local_observation: Agent's local observation

        Returns:
            Local Q-value
        """
        # Convert state to hashable key
        state_key = self._state_to_key(state)

        # Get or initialize Q-values for this state
        if state_key not in self.q_table:
            self.q_table[state_key] = np.ones(self.action_dim) * self.default_q_value

        # Get Q-value for this action
        action_idx = self._action_to_index(action)
        q_value = self.q_table[state_key][action_idx]

        return float(q_value)

    def compute_joint_value(
        self,
        state: Dict[str, Any],
        joint_action: Dict[str, Any],
        individual_values: Dict[str, float]
    ) -> float:
        """
        Compute joint Q-value from individual values.

        Q_tot(s, a) = f(Q_1, Q_2, ..., Q_n)

        Args:
            state: Global state
            joint_action: Dictionary of agent_id -> action
            individual_values: Dictionary of agent_id -> Q_i

        Returns:
            Joint Q-value
        """
        if self.decomposition_method == DecompositionMethod.ADDITIVE:
            # VDN: Q_tot = sum(Q_i)
            return sum(individual_values.values())

        elif self.decomposition_method == DecompositionMethod.MONOTONIC:
            # QMIX-style (simplified - just use weighted sum)
            # In real QMIX, use mixing network with positive weights
            weights = [1.0 / len(individual_values)] * len(individual_values)
            return sum(w * v for w, v in zip(weights, individual_values.values()))

        elif self.decomposition_method == DecompositionMethod.WEIGHTED:
            # Weight by agent reliability/performance
            total = 0.0
            for agent_id, value in individual_values.items():
                weight = self._get_agent_weight(agent_id)
                total += weight * value
            return total

        else:
            # Default to additive
            return sum(individual_values.values())

    def decompose_joint_value(
        self,
        joint_value: float,
        state: Dict[str, Any],
        joint_action: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Decompose joint value into individual contributions.

        Args:
            joint_value: Q_tot
            state: Global state
            joint_action: All agent actions

        Returns:
            Dictionary of agent_id -> contribution
        """
        # Get individual values
        individual_values = {}
        for agent_id, action in joint_action.items():
            # Retrieve stored local value
            local_value = self.team_values.get(agent_id, 0.0)
            individual_values[agent_id] = local_value

        # In additive VDN, contribution = local value
        # Verify sum matches joint value
        total = sum(individual_values.values())

        if abs(total - joint_value) > 0.01:
            logger.warning("Value decomposition mismatch: %.3f != %.3f", total, joint_value)

        return individual_values

    # =========================================================================
    # Credit Assignment
    # =========================================================================

    def assign_credit(
        self,
        global_reward: float,
        state: Dict[str, Any],
        joint_action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Assign individual credit from global reward.

        Args:
            global_reward: System-wide reward
            state: Current state
            joint_action: All agent actions
            next_state: Next state

        Returns:
            Individual reward for this agent
        """
        if self.credit_strategy == CreditAssignmentStrategy.DIFFERENCE_REWARDS:
            return self._difference_rewards(global_reward, state, joint_action, next_state)

        elif self.credit_strategy == CreditAssignmentStrategy.SHAPLEY_VALUE:
            return self._shapley_value(global_reward, state, joint_action)

        elif self.credit_strategy == CreditAssignmentStrategy.COUNTERFACTUAL:
            return self._counterfactual_credit(global_reward, state, joint_action, next_state)

        elif self.credit_strategy == CreditAssignmentStrategy.EQUAL_SPLIT:
            # Simple equal split
            num_agents = len(joint_action)
            return global_reward / num_agents if num_agents > 0 else global_reward

        else:
            # Default to difference rewards
            return self._difference_rewards(global_reward, state, joint_action, next_state)

    def _difference_rewards(
        self,
        global_reward: float,
        state: Dict[str, Any],
        joint_action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Difference rewards: D_i = R(s, a) - R(s, a_{-i})

        Measures marginal contribution by comparing with counterfactual
        where this agent didn't participate.
        """
        # Estimate reward without this agent's contribution
        marginal_contribution = self.compute_marginal_contribution(state, joint_action)

        # Credit = proportion of global reward based on marginal contribution
        # Use softmax to convert contributions to proportions
        individual_credit = global_reward * (marginal_contribution / (marginal_contribution + 1.0))

        self.credit_history.append(individual_credit)
        self.total_credits_assigned += 1

        return individual_credit

    def _shapley_value(
        self,
        global_reward: float,
        state: Dict[str, Any],
        joint_action: Dict[str, Any]
    ) -> float:
        """
        Shapley value credit assignment (simplified).

        True Shapley requires evaluating all coalitions, which is expensive.
        This is an approximation.
        """
        # Get all agent IDs
        agent_ids = list(joint_action.keys())
        num_agents = len(agent_ids)

        if num_agents == 1:
            return global_reward

        # Approximate Shapley: marginal contribution averaged over random orderings
        # For simplicity, use single random ordering
        contribution = self.compute_marginal_contribution(state, joint_action)

        # Normalize by number of agents
        shapley_value = contribution / num_agents

        self.credit_history.append(shapley_value)
        self.total_credits_assigned += 1

        return shapley_value

    def _counterfactual_credit(
        self,
        global_reward: float,
        state: Dict[str, Any],
        joint_action: Dict[str, Any],
        next_state: Dict[str, Any]
    ) -> float:
        """
        Counterfactual credit: compare actual reward with hypothetical
        reward if agent took default action.
        """
        # Compute marginal contribution
        marginal = self.compute_marginal_contribution(state, joint_action)

        # Credit proportional to marginal contribution
        total_contributions = sum(
            self.team_values.get(agent_id, 0.0)
            for agent_id in joint_action.keys()
        )

        if total_contributions > 0:
            credit = global_reward * (marginal / total_contributions)
        else:
            credit = global_reward / len(joint_action)

        self.credit_history.append(credit)
        self.total_credits_assigned += 1

        return credit

    def compute_marginal_contribution(
        self,
        state: Dict[str, Any],
        joint_action: Dict[str, Any]
    ) -> float:
        """
        Compute this agent's marginal contribution to joint value.

        Marginal contribution = Q_tot(s, a) - Q_tot(s, a_{-i})
        where a_{-i} is default action for this agent.
        """
        # Get current joint value
        individual_values = {
            agent_id: self.team_values.get(agent_id, 0.0)
            for agent_id in joint_action.keys()
        }
        current_joint_value = self.compute_joint_value(state, joint_action, individual_values)

        # Get this agent's contribution
        my_value = self.team_values.get(self.agent_id, 0.0)

        # Marginal = my contribution to the sum
        marginal = my_value

        return float(marginal)

    # =========================================================================
    # Value Function Updates
    # =========================================================================

    def update_value_function(
        self,
        state: Dict[str, Any],
        action: Any,
        reward: float,
        next_state: Dict[str, Any],
        next_action: Any
    ):
        """
        Update local value function using TD learning.

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
        """
        # Convert to keys
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        action_idx = self._action_to_index(action)
        next_action_idx = self._action_to_index(next_action)

        # Initialize Q-tables if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = np.ones(self.action_dim) * self.default_q_value
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.ones(self.action_dim) * self.default_q_value

        # TD update
        current_q = self.q_table[state_key][action_idx]
        next_q = self.q_table[next_state_key][next_action_idx]

        td_target = reward + self.discount_factor * next_q
        td_error = td_target - current_q

        # Update Q-value
        self.q_table[state_key][action_idx] += self.learning_rate * td_error

        # Store experience
        self.experience_buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "next_action": next_action,
            "td_error": td_error
        })

        self.total_updates += 1

        logger.debug("Value update: Q(s,a) = %.3f, TD error = %.3f",
                    self.q_table[state_key][action_idx], td_error)

    def synchronize_values(
        self,
        agent_values: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Synchronize value functions with team members.

        Args:
            agent_values: Dictionary of agent_id -> value_function_data

        Returns:
            True if synchronization successful
        """
        try:
            # Store team values for joint computation
            for agent_id, value_data in agent_values.items():
                if agent_id != self.agent_id:
                    self.team_values[agent_id] = value_data.get("current_value", 0.0)

            logger.debug("Synchronized values with %d agents", len(agent_values))
            return True

        except Exception as e:
            logger.error("Value synchronization failed: %s", e)
            return False

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _state_to_key(self, state: Dict[str, Any]) -> tuple:
        """Convert state dictionary to hashable tuple."""
        # Simple hash based on sorted key-value pairs
        # For complex states, use more sophisticated encoding
        items = []
        for key in sorted(state.keys()):
            value = state[key]
            if isinstance(value, (int, float, str, bool)):
                items.append((key, value))
            elif isinstance(value, (list, tuple, np.ndarray)):
                # Convert to tuple
                items.append((key, tuple(np.array(value).flatten())))
            else:
                # Skip complex objects
                pass

        return tuple(items)

    def _action_to_index(self, action: Any) -> int:
        """Convert action to integer index."""
        if isinstance(action, int):
            return action % self.action_dim
        elif isinstance(action, str):
            return hash(action) % self.action_dim
        elif isinstance(action, dict):
            return hash(str(sorted(action.items()))) % self.action_dim
        else:
            return 0

    def _get_agent_weight(self, agent_id: str) -> float:
        """Get weight for agent (for weighted decomposition)."""
        # Could use performance history, trust scores, etc.
        # For now, uniform weights
        return 1.0

    # =========================================================================
    # Verification and Explanation
    # =========================================================================

    def verify_decomposition(
        self,
        joint_value: float,
        individual_values: Dict[str, float],
        tolerance: float = 0.01
    ) -> bool:
        """
        Verify that decomposition is valid.

        For additive VDN: sum(Q_i) should equal Q_tot
        """
        if self.decomposition_method == DecompositionMethod.ADDITIVE:
            total = sum(individual_values.values())
            return abs(total - joint_value) < tolerance

        # For other methods, verification depends on constraints
        return True

    def explain_credit_assignment(
        self,
        credit: float,
        global_reward: float
    ) -> Dict[str, Any]:
        """
        Explain how credit was assigned.

        Returns:
            Explanation dictionary
        """
        marginal = self.compute_marginal_contribution({}, self.joint_actions)

        return {
            "agent_id": self.agent_id,
            "credit_assigned": credit,
            "global_reward": global_reward,
            "credit_proportion": credit / global_reward if global_reward != 0 else 0,
            "marginal_contribution": marginal,
            "strategy": self.credit_strategy.value,
            "decomposition_method": self.decomposition_method.value
        }

    def get_credit_statistics(self) -> Dict[str, Any]:
        """Get statistics about credit assignment."""
        if not self.credit_history:
            return {
                "total_credits": 0,
                "avg_credit": 0.0,
                "min_credit": 0.0,
                "max_credit": 0.0
            }

        return {
            "total_credits": self.total_credits_assigned,
            "avg_credit": np.mean(self.credit_history),
            "min_credit": np.min(self.credit_history),
            "max_credit": np.max(self.credit_history),
            "std_credit": np.std(self.credit_history),
            "recent_credits": list(self.credit_history)[-10:]
        }

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_learning_parameters(
        self,
        learning_rate: Optional[float] = None,
        discount_factor: Optional[float] = None
    ):
        """Update learning parameters."""
        if learning_rate is not None:
            self.learning_rate = learning_rate
            logger.info("Learning rate updated: %.4f", learning_rate)

        if discount_factor is not None:
            self.discount_factor = discount_factor
            logger.info("Discount factor updated: %.4f", discount_factor)

    def reset(self):
        """Reset the VDN engine."""
        self.q_table.clear()
        self.experience_buffer.clear()
        self.credit_history.clear()
        self.team_values.clear()
        self.joint_actions.clear()
        self.total_updates = 0
        self.total_credits_assigned = 0

        logger.info("SimpleVDN reset for agent %s", self.agent_id)

    # =========================================================================
    # Test-Compatible API Aliases
    # =========================================================================

    def decompose_value(self, q_values: Dict[str, float]) -> float:
        """
        Decompose value from individual Q-values (test-compatible API).

        Args:
            q_values: Dict of agent_id -> Q-value

        Returns:
            Total joint value
        """
        # For additive VDN, just sum the Q-values
        return sum(q_values.values())

    def calculate_credit(
        self,
        team_reward: float,
        baseline_reward: float
    ) -> float:
        """
        Calculate credit using difference rewards (test-compatible API).

        Args:
            team_reward: Total team reward
            baseline_reward: Baseline (what team gets without this agent)

        Returns:
            Individual credit
        """
        # Credit = marginal contribution
        credit = team_reward - baseline_reward
        self.credit_history.append(credit)
        self.total_credits_assigned += 1
        return credit

    def estimate_marginal_contribution(
        self,
        state: np.ndarray,
        action: int,
        team_reward: float
    ) -> float:
        """
        Estimate marginal contribution (test-compatible API).

        Args:
            state: State array
            action: Action index
            team_reward: Team reward

        Returns:
            Estimated marginal contribution
        """
        # Convert numpy array to dict format
        state_dict = {"array": state}

        # Compute local value for this state-action
        local_value = self.compute_local_value(state_dict, action, {})

        # Marginal contribution is proportion of team reward based on local value
        # If we don't have team values, assume equal contribution
        if not self.team_values:
            return team_reward

        total_value = sum(self.team_values.values()) + local_value
        if total_value > 0:
            marginal = (local_value / total_value) * team_reward
        else:
            marginal = team_reward / (len(self.team_values) + 1)

        return float(marginal)

    # =========================================================================
    # Additional Abstract Method Implementations
    # =========================================================================

    def decompose_global_value(
        self,
        global_value: float,
        state: Dict[str, Any],
        joint_action: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Decompose global value into individual contributions (alias for decompose_joint_value).

        Args:
            global_value: Q_tot
            state: Global state
            joint_action: All agent actions

        Returns:
            Dictionary of agent_id -> contribution
        """
        return self.decompose_joint_value(global_value, state, joint_action)

    def compute_counterfactual_baseline(
        self,
        state: Dict[str, Any],
        joint_action: Dict[str, Any],
        agent_to_evaluate: Optional[str] = None
    ) -> float:
        """
        Compute counterfactual baseline: value without agent's contribution.

        Args:
            state: Global state
            joint_action: All agent actions
            agent_to_evaluate: Agent to compute baseline for (defaults to self)

        Returns:
            Baseline value
        """
        agent_id = agent_to_evaluate or self.agent_id

        # Get individual values
        individual_values = {
            aid: self.team_values.get(aid, 0.0)
            for aid in joint_action.keys()
        }

        # Compute joint value
        joint_value = self.compute_joint_value(state, joint_action, individual_values)

        # Subtract evaluated agent's contribution
        agent_value = individual_values.get(agent_id, 0.0)
        baseline = joint_value - agent_value

        return float(baseline)

    def share_value_information(
        self,
        state: Dict[str, Any],
        action: Any,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Share value information with team members.

        Args:
            state: Current state
            action: This agent's action
            value: Computed local value
            metadata: Additional information
        """
        # Store locally
        self.team_values[self.agent_id] = value
        self.joint_actions[self.agent_id] = action

        # Publish to Redis for team coordination
        try:
            message = {
                "agent_id": self.agent_id,
                "state": state,
                "action": action,
                "value": value,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            channel = f"vdn:values:{self.agent_id}"
            self.redis_client.publish(channel, message)
            logger.debug("Shared value %.3f for state/action", value)

        except Exception as e:
            logger.error("Failed to share value information: %s", e)

    def receive_peer_values(
        self,
        state: Dict[str, Any],
        timeout_ms: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Receive value information from peers.

        Args:
            state: Current state
            timeout_ms: Optional timeout

        Returns:
            Dictionary of agent_id -> value
        """
        # In SimpleVDN, we use stored team values
        # In production, would subscribe to Redis channels
        return self.team_values.copy()

    def validate_value_consistency(
        self,
        individual_values: Dict[str, float],
        joint_value: float,
        tolerance: float = 1e-6
    ) -> bool:
        """
        Validate that value decomposition is consistent.

        Args:
            individual_values: Individual agent values
            joint_value: Joint value
            tolerance: Tolerance for floating point comparison

        Returns:
            True if consistent
        """
        if self.decomposition_method == DecompositionMethod.ADDITIVE:
            # Sum should equal joint value
            total = sum(individual_values.values())
            return abs(total - joint_value) < tolerance

        # For other methods, less strict validation
        return True

    def compute_mixing_weights(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute mixing weights for QMIX-style decomposition.

        Args:
            state: Global state

        Returns:
            Dictionary of agent_id -> weight
        """
        # Simple uniform weights for basic implementation
        # In QMIX, these come from a hypernetwork
        num_agents = len(self.team_values)
        if num_agents == 0:
            return {}

        weight = 1.0 / num_agents
        return {agent_id: weight for agent_id in self.team_values.keys()}

    def estimate_value_variance(
        self,
        state: Dict[str, Any],
        action: Any
    ) -> float:
        """
        Estimate variance/uncertainty in value estimate.

        Args:
            state: Current state
            action: Action

        Returns:
            Variance estimate
        """
        # Simple variance estimation based on experience buffer
        state_key = self._state_to_key(state)
        action_idx = self._action_to_index(action)

        # Get Q-value
        if state_key not in self.q_table:
            return 1.0  # High uncertainty for unseen states

        # Estimate variance from TD errors in experience buffer
        relevant_experiences = [
            exp for exp in self.experience_buffer
            if self._state_to_key(exp["state"]) == state_key
            and self._action_to_index(exp["action"]) == action_idx
        ]

        if not relevant_experiences:
            return 0.5  # Moderate uncertainty

        td_errors = [exp["td_error"] for exp in relevant_experiences]
        variance = float(np.var(td_errors))

        return variance

    def get_value_decomposition_explanation(
        self,
        state: Dict[str, Any],
        joint_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get explanation of how value was decomposed.

        Args:
            state: Global state
            joint_action: All agent actions

        Returns:
            Explanation dictionary
        """
        # Get individual values
        individual_values = {
            aid: self.team_values.get(aid, 0.0)
            for aid in joint_action.keys()
        }

        # Compute joint value
        joint_value = self.compute_joint_value(state, joint_action, individual_values)

        # Compute marginal contributions
        marginals = {}
        for agent_id in joint_action.keys():
            baseline = self.compute_counterfactual_baseline(state, joint_action, agent_id)
            marginals[agent_id] = joint_value - baseline

        return {
            "decomposition_method": self.decomposition_method.value,
            "joint_value": joint_value,
            "individual_values": individual_values,
            "marginal_contributions": marginals,
            "consistency_check": self.validate_value_consistency(individual_values, joint_value),
            "agent_id": self.agent_id
        }

    def sync_value_functions(
        self,
        peer_value_functions: Dict[str, Any]
    ):
        """
        Synchronize value functions with peers.

        Args:
            peer_value_functions: Dictionary of agent_id -> value_function_data
        """
        # Store peer values for joint computation
        for agent_id, value_data in peer_value_functions.items():
            if agent_id != self.agent_id:
                # Extract current value
                if isinstance(value_data, dict):
                    self.team_values[agent_id] = value_data.get("current_value", 0.0)
                else:
                    self.team_values[agent_id] = float(value_data)

        logger.debug("Synced value functions with %d peers", len(peer_value_functions))

    def compute_value_gradient(
        self,
        state: Dict[str, Any],
        action: Any
    ) -> Dict[str, Any]:
        """
        Compute gradient of value function (for policy gradient methods).

        Args:
            state: Current state
            action: Action

        Returns:
            Dictionary containing gradient information
        """
        # For tabular Q-learning, gradient is discrete
        # Return action-value gradients
        state_key = self._state_to_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.ones(self.action_dim) * self.default_q_value

        q_values = self.q_table[state_key]
        action_idx = self._action_to_index(action)

        # Compute advantage (gradient surrogate)
        baseline = np.mean(q_values)
        advantage = q_values[action_idx] - baseline

        return {
            "q_values": q_values.tolist(),
            "action_value": float(q_values[action_idx]),
            "advantage": float(advantage),
            "baseline": float(baseline),
            "state_key": str(state_key)
        }


# =========================================================================
# Factory Function
# =========================================================================

def create_vdn_engine(
    agent_id: str,
    redis_client: RedisClient,
    config: Optional[Dict[str, Any]] = None
) -> SimpleVDN:
    """
    Factory function to create SimpleVDN engine.

    Args:
        agent_id: Agent identifier
        redis_client: Redis client instance
        config: Optional configuration dictionary

    Returns:
        SimpleVDN engine instance
    """
    config = config or {}

    return SimpleVDN(
        agent_id=agent_id,
        redis_client=redis_client,
        decomposition_method=config.get("decomposition_method", DecompositionMethod.ADDITIVE),
        credit_strategy=config.get("credit_strategy", CreditAssignmentStrategy.DIFFERENCE_REWARDS),
        state_dim=config.get("state_dim", 10),
        action_dim=config.get("action_dim", 5),
        learning_rate=config.get("learning_rate", 0.01),
        discount_factor=config.get("discount_factor", 0.99),
        history_size=config.get("history_size", 1000)
    )
