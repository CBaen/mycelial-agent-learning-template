"""
Toxic Agent for Adversarial Testing

This module provides agents programmed with known bad strategies to test
the robustness of the HAVEN risk management system and validate contagion
detection mechanisms.
"""

from typing import Dict, Any, Optional, List
import logging
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.specialist_agent import SpecialistAgent

logger = logging.getLogger(__name__)


class ToxicBehaviorType:
    """Enumeration of toxic behavior types."""
    MANIPULATIVE = "manipulative"  # Agent tries to game the reward system
    HIGH_RISK = "high_risk"  # Agent takes excessive risks
    BUGGY = "buggy"  # Agent has implementation bugs causing poor performance
    EXPLOITATIVE = "exploitative"  # Agent exploits vulnerabilities
    SABOTAGE = "sabotage"  # Agent actively tries to harm system
    POLICY_POISONING = "policy_poisoning"  # Agent shares toxic policies
    BYZANTINE = "byzantine"  # Agent acts unpredictably/maliciously


class ToxicAgent(SpecialistAgent):
    """
    Toxic Agent with programmed bad strategies for adversarial testing.

    This agent intentionally exhibits harmful behavior patterns to test:
    - HAVEN risk detection accuracy
    - Policy contagion detection and containment
    - System resilience to malicious agents
    - Intervention effectiveness

    Different toxic behaviors can be configured to test various scenarios.
    """

    def __init__(
        self,
        unique_id: int,
        model,
        redis_client,
        toxic_behavior: str = ToxicBehaviorType.HIGH_RISK,
        toxicity_level: float = 0.8,
        data_channel: str = "processed_data",
        team_id: str = "toxic_agents",
        agent_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Toxic Agent.

        Args:
            unique_id: Unique identifier for this agent
            model: The Mesa model this agent belongs to
            redis_client: Redis client for data operations
            toxic_behavior: Type of toxic behavior to exhibit
            toxicity_level: Intensity of toxic behavior (0.0 to 1.0)
            data_channel: Pub/Sub channel to receive data from
            team_id: Team identifier (default: "toxic_agents")
            agent_config: Optional configuration dictionary
        """
        # Initialize as specialist agent
        super().__init__(
            unique_id=unique_id,
            model=model,
            redis_client=redis_client,
            data_channel=data_channel,
            team_id=team_id,
            specialization=f"toxic_{toxic_behavior}",
            agent_config=agent_config
        )

        # Toxic behavior configuration
        self.toxic_behavior = toxic_behavior
        self.toxicity_level = max(0.0, min(1.0, toxicity_level))

        # Override agent type for identification
        self.agent_type = "ToxicAgent"
        self.agent_id = f"ToxicAgent_{unique_id}"

        # Toxic policy initialization
        self._initialize_toxic_policy()

        # Tracking toxic behavior
        self.toxic_actions_taken: int = 0
        self.policies_poisoned: int = 0
        self.system_damage_caused: float = 0.0

        # Evasion tactics (try to avoid detection)
        self.evasion_enabled = agent_config.get("evasion_enabled", False) if agent_config else False
        self.detection_avoidance_rate = 0.3  # Periodically act normal

        logger.warning("ToxicAgent %s initialized with behavior: %s (toxicity: %.2f)",
                      self.agent_id, toxic_behavior, toxicity_level)

    def _initialize_toxic_policy(self):
        """
        Initialize a deliberately bad policy based on toxic behavior type.
        """
        if self.toxic_behavior == ToxicBehaviorType.MANIPULATIVE:
            # Policy that tries to game rewards
            self.policy_parameters = {
                "strategy": "manipulative",
                "exploit_loopholes": True,
                "misreport_metrics": True,
                "initialized": True
            }

        elif self.toxic_behavior == ToxicBehaviorType.HIGH_RISK:
            # Policy that takes excessive risks
            self.policy_parameters = {
                "strategy": "high_risk",
                "risk_tolerance": 0.9,
                "ignore_safety": True,
                "initialized": True
            }

        elif self.toxic_behavior == ToxicBehaviorType.BUGGY:
            # Policy with implementation bugs
            self.policy_parameters = {
                "strategy": "buggy",
                "has_memory_leak": True,
                "incorrect_logic": True,
                "initialized": True
            }

        elif self.toxic_behavior == ToxicBehaviorType.EXPLOITATIVE:
            # Policy that exploits system vulnerabilities
            self.policy_parameters = {
                "strategy": "exploitative",
                "exploit_coordination": True,
                "resource_hoarding": True,
                "initialized": True
            }

        elif self.toxic_behavior == ToxicBehaviorType.SABOTAGE:
            # Policy that actively harms the system
            self.policy_parameters = {
                "strategy": "sabotage",
                "disrupt_communications": True,
                "corrupt_data": True,
                "initialized": True
            }

        elif self.toxic_behavior == ToxicBehaviorType.POLICY_POISONING:
            # Policy designed to spread to other agents
            self.policy_parameters = {
                "strategy": "policy_poisoning",
                "aggressive_sharing": True,
                "trust_manipulation": True,
                "initialized": True
            }

        elif self.toxic_behavior == ToxicBehaviorType.BYZANTINE:
            # Unpredictable malicious behavior
            self.policy_parameters = {
                "strategy": "byzantine",
                "random_malicious": True,
                "inconsistent_reporting": True,
                "initialized": True
            }

        else:
            # Default toxic policy
            self.policy_parameters = {
                "strategy": "toxic_default",
                "initialized": True
            }

        logger.debug("%s initialized toxic policy: %s", self.agent_id, self.toxic_behavior)

    def _select_action(self, state: Dict[str, Any]) -> Any:
        """
        Select action using toxic strategy.

        Overrides the parent method to implement deliberately bad decisions.

        Args:
            state: Current observed state

        Returns:
            Selected action (usually a bad one)
        """
        # Occasionally act normal to evade detection
        if self.evasion_enabled and np.random.random() < self.detection_avoidance_rate:
            return super()._select_action(state)

        # Select toxic action based on behavior type
        toxic_action = self._get_toxic_action(state)
        self.toxic_actions_taken += 1

        return toxic_action

    def _get_toxic_action(self, state: Dict[str, Any]) -> Any:
        """
        Get a toxic action based on the configured behavior type.

        Args:
            state: Current state

        Returns:
            Toxic action
        """
        if self.toxic_behavior == ToxicBehaviorType.MANIPULATIVE:
            # Choose actions that appear good but aren't
            return self._manipulative_action()

        elif self.toxic_behavior == ToxicBehaviorType.HIGH_RISK:
            # Always choose highest risk action
            return self._high_risk_action()

        elif self.toxic_behavior == ToxicBehaviorType.BUGGY:
            # Return buggy/invalid action
            return self._buggy_action()

        elif self.toxic_behavior == ToxicBehaviorType.EXPLOITATIVE:
            # Exploit system weaknesses
            return self._exploitative_action()

        elif self.toxic_behavior == ToxicBehaviorType.SABOTAGE:
            # Actively harmful action
            return self._sabotage_action()

        elif self.toxic_behavior == ToxicBehaviorType.POLICY_POISONING:
            # Action designed to corrupt other agents
            return self._poisoning_action()

        elif self.toxic_behavior == ToxicBehaviorType.BYZANTINE:
            # Random malicious action
            return self._byzantine_action()

        # Default: random action
        return np.random.choice([0, 1])

    def _manipulative_action(self) -> Any:
        """
        Action that tries to game the reward system.

        Returns:
            Manipulative action
        """
        # Always choose action that maximizes short-term reward
        # regardless of long-term consequences
        return 1  # Greedy action

    def _high_risk_action(self) -> Any:
        """
        Action that takes excessive risk.

        Returns:
            High-risk action
        """
        # Choose high-variance action with potential for high reward or disaster
        return np.random.choice([0, 1], p=[0.1, 0.9])  # Heavily biased to risky action

    def _buggy_action(self) -> Any:
        """
        Action from buggy implementation.

        Returns:
            Buggy action (may be invalid)
        """
        # Simulate buggy logic with occasional invalid outputs
        if np.random.random() < self.toxicity_level:
            return -1  # Invalid action
        return np.random.choice([0, 1])

    def _exploitative_action(self) -> Any:
        """
        Action that exploits system vulnerabilities.

        Returns:
            Exploitative action
        """
        # Action that benefits this agent at expense of others
        return 1  # Selfish action

    def _sabotage_action(self) -> Any:
        """
        Action that actively harms the system.

        Returns:
            Sabotage action
        """
        # Always choose action that harms system performance
        return 0  # Harmful action

    def _poisoning_action(self) -> Any:
        """
        Action designed to corrupt other agents via policy sharing.

        Returns:
            Poisoning action
        """
        # Action that will be shared to corrupt peers
        return np.random.choice([0, 1])  # Random to spread confusion

    def _byzantine_action(self) -> Any:
        """
        Random malicious action (Byzantine behavior).

        Returns:
            Byzantine action
        """
        # Completely unpredictable behavior
        actions = [-1, 0, 1, 2]  # Mix of valid and invalid
        return np.random.choice(actions)

    def _execute_action(self, action: Any) -> float:
        """
        Execute toxic action and receive (usually poor) reward.

        Args:
            action: The action to execute

        Returns:
            Reward (usually negative or zero)
        """
        # Toxic actions generally result in poor rewards
        base_reward = super()._execute_action(action)

        # Apply toxicity modifier
        if self.toxic_behavior == ToxicBehaviorType.HIGH_RISK:
            # High variance rewards
            reward = np.random.normal(-0.5, 2.0)
            self.system_damage_caused += max(0, -reward)

        elif self.toxic_behavior == ToxicBehaviorType.BUGGY:
            # Consistently poor performance
            reward = base_reward * (1.0 - self.toxicity_level)
            self.system_damage_caused += self.toxicity_level

        elif self.toxic_behavior == ToxicBehaviorType.SABOTAGE:
            # Negative rewards
            reward = -abs(base_reward)
            self.system_damage_caused += abs(reward)

        else:
            # Generally poor but not catastrophic
            reward = base_reward * (1.0 - self.toxicity_level * 0.5)

        return reward

    def share_policy(self) -> int:
        """
        Share toxic policy with peers (policy poisoning).

        Overrides parent to aggressively share toxic policies.

        Returns:
            Number of peers poisoned
        """
        if not self.frl_engine:
            return 0

        # Policy poisoning agents share very aggressively
        if self.toxic_behavior == ToxicBehaviorType.POLICY_POISONING:
            # Share every step
            num_peers = super().share_policy()
            self.policies_poisoned += num_peers
            logger.warning("%s poisoned %d peers with toxic policy", self.agent_id, num_peers)
            return num_peers

        # Other toxic agents share normally (to spread toxicity)
        elif np.random.random() < self.toxicity_level:
            num_peers = super().share_policy()
            self.policies_poisoned += num_peers
            return num_peers

        return 0

    def _should_share_policy(self) -> bool:
        """
        Toxic agents share policies more aggressively to spread toxicity.

        Returns:
            True if should share policy
        """
        # Policy poisoning agents share very frequently
        if self.toxic_behavior == ToxicBehaviorType.POLICY_POISONING:
            return True  # Share every step

        # Other toxic agents share based on toxicity level
        return np.random.random() < self.toxicity_level

    def _update_policy(self, state: Dict[str, Any], action: Any, reward: float):
        """
        Toxic agents either don't learn or learn the wrong things.

        Args:
            state: State where action was taken
            action: Action that was taken
            reward: Reward received
        """
        if self.toxic_behavior == ToxicBehaviorType.BUGGY:
            # Buggy learning: update in wrong direction
            self.policy_version += 1
            # Intentionally bad update logic

        elif self.toxic_behavior == ToxicBehaviorType.MANIPULATIVE:
            # Only learn from exploitative actions
            if reward > 0.5:
                self.policy_version += 1

        elif self.toxic_behavior == ToxicBehaviorType.POLICY_POISONING:
            # Learn to maximize policy spread, not performance
            self.policy_version += 1

        else:
            # Most toxic agents don't really learn
            if np.random.random() < 0.1:
                self.policy_version += 1

    def get_toxic_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about toxic behavior.

        Returns:
            Dictionary with toxic behavior metrics
        """
        base_stats = self.get_specialist_statistics()

        toxic_stats = {
            "toxic_behavior": self.toxic_behavior,
            "toxicity_level": self.toxicity_level,
            "toxic_actions_taken": self.toxic_actions_taken,
            "policies_poisoned": self.policies_poisoned,
            "system_damage_caused": self.system_damage_caused,
            "evasion_enabled": self.evasion_enabled,
            "is_isolated": self.is_isolated,
            "risk_score": self.risk_score
        }

        # Merge with base statistics
        base_stats.update(toxic_stats)
        return base_stats

    def get_detection_status(self) -> Dict[str, Any]:
        """
        Get information about whether toxic agent has been detected.

        Returns:
            Dictionary with detection status
        """
        return {
            "agent_id": self.agent_id,
            "toxic_behavior": self.toxic_behavior,
            "is_detected": self.is_isolated or self.risk_score > 0.7,
            "is_isolated": self.is_isolated,
            "risk_score": self.risk_score,
            "steps_active": self.step_count,
            "damage_per_step": self.system_damage_caused / max(1, self.step_count)
        }

    def reset(self):
        """
        Reset the toxic agent (reinitialize toxic behavior).
        """
        super().reset()

        # Reset toxic metrics
        self.toxic_actions_taken = 0
        self.policies_poisoned = 0
        self.system_damage_caused = 0.0

        # Reinitialize toxic policy
        self._initialize_toxic_policy()

        logger.warning("%s has been reset with %s behavior",
                      self.agent_id, self.toxic_behavior)


class ToxicAgentFactory:
    """
    Factory for creating toxic agents with various behavior patterns.
    """

    @staticmethod
    def create_manipulative_agent(unique_id: int, model, redis_client, **kwargs) -> ToxicAgent:
        """Create a manipulative toxic agent."""
        return ToxicAgent(
            unique_id=unique_id,
            model=model,
            redis_client=redis_client,
            toxic_behavior=ToxicBehaviorType.MANIPULATIVE,
            **kwargs
        )

    @staticmethod
    def create_high_risk_agent(unique_id: int, model, redis_client, **kwargs) -> ToxicAgent:
        """Create a high-risk toxic agent."""
        return ToxicAgent(
            unique_id=unique_id,
            model=model,
            redis_client=redis_client,
            toxic_behavior=ToxicBehaviorType.HIGH_RISK,
            **kwargs
        )

    @staticmethod
    def create_buggy_agent(unique_id: int, model, redis_client, **kwargs) -> ToxicAgent:
        """Create a buggy toxic agent."""
        return ToxicAgent(
            unique_id=unique_id,
            model=model,
            redis_client=redis_client,
            toxic_behavior=ToxicBehaviorType.BUGGY,
            **kwargs
        )

    @staticmethod
    def create_policy_poisoning_agent(unique_id: int, model, redis_client, **kwargs) -> ToxicAgent:
        """Create a policy poisoning toxic agent."""
        return ToxicAgent(
            unique_id=unique_id,
            model=model,
            redis_client=redis_client,
            toxic_behavior=ToxicBehaviorType.POLICY_POISONING,
            **kwargs
        )

    @staticmethod
    def create_sabotage_agent(unique_id: int, model, redis_client, **kwargs) -> ToxicAgent:
        """Create a sabotage toxic agent."""
        return ToxicAgent(
            unique_id=unique_id,
            model=model,
            redis_client=redis_client,
            toxic_behavior=ToxicBehaviorType.SABOTAGE,
            **kwargs
        )

    @staticmethod
    def create_byzantine_agent(unique_id: int, model, redis_client, **kwargs) -> ToxicAgent:
        """Create a byzantine toxic agent."""
        return ToxicAgent(
            unique_id=unique_id,
            model=model,
            redis_client=redis_client,
            toxic_behavior=ToxicBehaviorType.BYZANTINE,
            **kwargs
        )

    @staticmethod
    def create_mixed_toxic_swarm(
        model,
        redis_client,
        start_id: int,
        num_agents: int,
        **kwargs
    ) -> List[ToxicAgent]:
        """
        Create a diverse swarm of toxic agents with different behaviors.

        Args:
            model: Mesa model
            redis_client: Redis client
            start_id: Starting ID for agents
            num_agents: Number of toxic agents to create
            **kwargs: Additional arguments for agents

        Returns:
            List of toxic agents
        """
        behaviors = [
            ToxicBehaviorType.MANIPULATIVE,
            ToxicBehaviorType.HIGH_RISK,
            ToxicBehaviorType.BUGGY,
            ToxicBehaviorType.POLICY_POISONING,
        ]

        agents = []
        for i in range(num_agents):
            behavior = behaviors[i % len(behaviors)]
            agent = ToxicAgent(
                unique_id=start_id + i,
                model=model,
                redis_client=redis_client,
                toxic_behavior=behavior,
                **kwargs
            )
            agents.append(agent)

        return agents
