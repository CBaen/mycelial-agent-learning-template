"""
Hindsight Experience Replay (HER)

This module implements HER to enable agents to learn from failures by
relabeling goals. This dramatically improves sample efficiency in sparse
reward environments.

Key Features:
- Goal relabeling for failed episodes
- Multiple relabeling strategies (future, final, episode, random)
- Multi-goal storage and retrieval
- Compatible with prioritized replay

Research Foundation:
- Andrychowicz et al. (2017): Hindsight Experience Replay (NeurIPS)
- Plappert et al. (2018): Multi-Goal Reinforcement Learning

Core Idea:
Instead of learning only from successes, learn from failures by asking:
"What if this outcome was actually my goal?" This creates artificial
successes from every experience.

Example:
    Original: Goal = reach (10, 10), Reached (7, 8) → FAILURE (reward = -1)
    HER: Goal = reach (7, 8), Reached (7, 8) → SUCCESS (reward = 0)

Performance Benefits:
- 2-10x sample efficiency improvement
- Works with sparse/binary rewards
- No additional environment interaction needed
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HERStrategy(Enum):
    """
    Goal relabeling strategies for Hindsight Experience Replay.

    FUTURE: Sample goals from future states in same episode
    FINAL: Use final achieved state as goal
    EPISODE: Sample random states from current episode
    RANDOM: Sample random goals from replay buffer
    MIXED: Combination of strategies
    """
    FUTURE = "future"
    FINAL = "final"
    EPISODE = "episode"
    RANDOM = "random"
    MIXED = "mixed"


@dataclass
class HERTransition:
    """
    Transition with goal information for HER.

    This extends standard (s, a, r, s') transitions with goal information,
    enabling goal-conditioned learning and hindsight relabeling.
    """
    state: np.ndarray
    action: Any
    reward: float
    next_state: np.ndarray
    done: bool
    goal: np.ndarray
    achieved_goal: np.ndarray
    info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'goal': self.goal,
            'achieved_goal': self.achieved_goal,
            'info': self.info
        }


class HindsightReplay:
    """
    Hindsight Experience Replay for learning from failures.

    Implements goal relabeling to create artificial successes from
    failed episodes, dramatically improving sample efficiency.

    Example:
        >>> her = HindsightReplay(
        ...     strategy=HERStrategy.FUTURE,
        ...     relabel_ratio=0.8,
        ...     reward_func=compute_reward
        ... )
        >>>
        >>> # Store episode
        >>> episode = [transition1, transition2, ...]
        >>> relabeled = her.relabel_episode(episode)
        >>>
        >>> # Returns original + relabeled transitions
        >>> memory.store_batch(relabeled)
    """

    def __init__(
        self,
        strategy: HERStrategy = HERStrategy.FUTURE,
        relabel_ratio: float = 0.8,
        k_future: int = 4,
        reward_func: Optional[Callable] = None,
        goal_selection_func: Optional[Callable] = None
    ):
        """
        Initialize HER system.

        Args:
            strategy: Relabeling strategy to use
            relabel_ratio: Fraction of episode to relabel (0.0-1.0)
            k_future: Number of future states to sample (for FUTURE strategy)
            reward_func: Function(achieved_goal, goal) -> reward
            goal_selection_func: Custom goal selection function
        """
        self.strategy = strategy
        self.relabel_ratio = relabel_ratio
        self.k_future = k_future
        self.reward_func = reward_func or self._default_reward_func
        self.goal_selection_func = goal_selection_func

        # Statistics
        self.total_episodes = 0
        self.total_transitions = 0
        self.total_relabeled = 0
        self.success_rate = 0.0

    def relabel_episode(
        self,
        episode: List[HERTransition],
        original_goal: Optional[np.ndarray] = None
    ) -> List[HERTransition]:
        """
        Relabel an episode with hindsight goals.

        Takes a complete episode and creates additional transitions
        with relabeled goals, turning failures into successes.

        Args:
            episode: List of HERTransition objects
            original_goal: Original goal (if not in transitions)

        Returns:
            List containing original + relabeled transitions

        Example:
            >>> # Episode that failed original goal
            >>> episode = [trans1, trans2, trans3]
            >>>
            >>> # Relabel with hindsight
            >>> relabeled = her.relabel_episode(episode)
            >>>
            >>> # Now contains original + hindsight transitions
            >>> len(relabeled) > len(episode)  # True
        """
        if not episode:
            return []

        self.total_episodes += 1
        self.total_transitions += len(episode)

        # Always include original transitions
        result = episode.copy()

        # Determine how many transitions to relabel
        num_to_relabel = int(len(episode) * self.relabel_ratio)

        if num_to_relabel == 0:
            return result

        # Select transitions to relabel
        indices_to_relabel = np.random.choice(
            len(episode),
            size=num_to_relabel,
            replace=False
        )

        # Relabel based on strategy
        for idx in indices_to_relabel:
            new_goals = self._select_goals(episode, idx)

            for new_goal in new_goals:
                relabeled_trans = self._relabel_transition(
                    episode[idx],
                    new_goal
                )
                result.append(relabeled_trans)
                self.total_relabeled += 1

        return result

    def relabel_batch(
        self,
        episodes: List[List[HERTransition]]
    ) -> List[HERTransition]:
        """
        Relabel multiple episodes.

        Args:
            episodes: List of episode lists

        Returns:
            Flat list of all original + relabeled transitions
        """
        all_transitions = []

        for episode in episodes:
            relabeled = self.relabel_episode(episode)
            all_transitions.extend(relabeled)

        return all_transitions

    def _select_goals(
        self,
        episode: List[HERTransition],
        transition_idx: int
    ) -> List[np.ndarray]:
        """
        Select new goals for relabeling based on strategy.

        Args:
            episode: Complete episode
            transition_idx: Index of transition to relabel

        Returns:
            List of new goals to use
        """
        if self.goal_selection_func:
            return self.goal_selection_func(episode, transition_idx)

        if self.strategy == HERStrategy.FUTURE:
            return self._select_future_goals(episode, transition_idx)

        elif self.strategy == HERStrategy.FINAL:
            return self._select_final_goal(episode)

        elif self.strategy == HERStrategy.EPISODE:
            return self._select_episode_goals(episode, transition_idx)

        elif self.strategy == HERStrategy.RANDOM:
            return self._select_random_goals(episode)

        elif self.strategy == HERStrategy.MIXED:
            # Mix of future and final
            goals = []
            goals.extend(self._select_future_goals(episode, transition_idx))
            goals.extend(self._select_final_goal(episode))
            return goals[:self.k_future]  # Limit total

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _select_future_goals(
        self,
        episode: List[HERTransition],
        transition_idx: int
    ) -> List[np.ndarray]:
        """
        Select goals from future states in the same episode.

        This is the most common HER strategy from the original paper.
        """
        # Can only select from states after current transition
        future_indices = list(range(transition_idx + 1, len(episode)))

        if not future_indices:
            # No future states, use final state
            return [episode[-1].achieved_goal]

        # Sample k future states
        k = min(self.k_future, len(future_indices))
        selected_indices = np.random.choice(future_indices, size=k, replace=False)

        return [episode[idx].achieved_goal for idx in selected_indices]

    def _select_final_goal(
        self,
        episode: List[HERTransition]
    ) -> List[np.ndarray]:
        """
        Use the final achieved state as goal.
        """
        return [episode[-1].achieved_goal]

    def _select_episode_goals(
        self,
        episode: List[HERTransition],
        transition_idx: int
    ) -> List[np.ndarray]:
        """
        Sample random states from the episode.
        """
        # Sample from entire episode (except current state)
        available_indices = [i for i in range(len(episode)) if i != transition_idx]

        if not available_indices:
            return [episode[-1].achieved_goal]

        k = min(self.k_future, len(available_indices))
        selected_indices = np.random.choice(available_indices, size=k, replace=False)

        return [episode[idx].achieved_goal for idx in selected_indices]

    def _select_random_goals(
        self,
        episode: List[HERTransition]
    ) -> List[np.ndarray]:
        """
        Sample random goals from episode.

        Note: In practice, this would sample from replay buffer,
        but we use episode for simplicity.
        """
        return self._select_episode_goals(episode, -1)

    def _relabel_transition(
        self,
        transition: HERTransition,
        new_goal: np.ndarray
    ) -> HERTransition:
        """
        Create new transition with relabeled goal and reward.

        Args:
            transition: Original transition
            new_goal: New goal to use

        Returns:
            New HERTransition with relabeled goal/reward
        """
        # Compute reward for achieving new goal
        new_reward = self.reward_func(
            transition.achieved_goal,
            new_goal,
            transition.info
        )

        # Check if goal was achieved
        new_done = self._check_goal_achieved(
            transition.achieved_goal,
            new_goal
        )

        # Create relabeled transition
        return HERTransition(
            state=transition.state.copy(),
            action=transition.action,
            reward=new_reward,
            next_state=transition.next_state.copy(),
            done=new_done,
            goal=new_goal.copy(),
            achieved_goal=transition.achieved_goal.copy(),
            info={**transition.info, 'her_relabeled': True}
        )

    def _default_reward_func(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        Default reward function: binary sparse reward.

        Returns 0 if goal achieved (within threshold), -1 otherwise.
        """
        threshold = info.get('distance_threshold', 0.05)
        distance = np.linalg.norm(achieved_goal - desired_goal)

        return 0.0 if distance < threshold else -1.0

    def _check_goal_achieved(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        threshold: float = 0.05
    ) -> bool:
        """
        Check if goal was achieved.

        Args:
            achieved_goal: What was actually achieved
            desired_goal: What was desired
            threshold: Distance threshold for success

        Returns:
            True if goal achieved, False otherwise
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < threshold

    def update_success_rate(self, episode_success: bool):
        """
        Update running success rate.

        Args:
            episode_success: Whether episode succeeded at original goal
        """
        # Exponential moving average
        alpha = 0.1
        self.success_rate = (1 - alpha) * self.success_rate + alpha * float(episode_success)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get HER statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'strategy': self.strategy.value,
            'relabel_ratio': self.relabel_ratio,
            'k_future': self.k_future,
            'total_episodes': self.total_episodes,
            'total_transitions': self.total_transitions,
            'total_relabeled': self.total_relabeled,
            'relabel_rate': self.total_relabeled / max(self.total_transitions, 1),
            'success_rate': self.success_rate,
            'augmentation_factor': (self.total_transitions + self.total_relabeled) / max(self.total_transitions, 1)
        }

    def __repr__(self) -> str:
        return (
            f"HindsightReplay(strategy={self.strategy.value}, "
            f"relabel_ratio={self.relabel_ratio}, "
            f"episodes={self.total_episodes}, "
            f"augmentation={self.get_statistics()['augmentation_factor']:.2f}x)"
        )


class GoalEnv:
    """
    Helper class for goal-conditioned environments.

    Provides utilities for extracting goals and achieved goals from states.
    """

    @staticmethod
    def extract_goal_from_state(
        state: np.ndarray,
        goal_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Extract goal portion from state.

        Args:
            state: Full state observation
            goal_indices: Indices of goal dimensions

        Returns:
            Goal portion of state
        """
        if goal_indices is None:
            # Assume goal is second half of state
            split = len(state) // 2
            return state[split:]

        return state[goal_indices]

    @staticmethod
    def extract_achieved_goal(
        state: np.ndarray,
        achieved_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Extract achieved goal from state.

        Args:
            state: State observation
            achieved_indices: Indices of achieved goal dimensions

        Returns:
            Achieved goal
        """
        if achieved_indices is None:
            # Assume achieved goal is first half of state
            split = len(state) // 2
            return state[:split]

        return state[achieved_indices]

    @staticmethod
    def compute_reward(
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        reward_type: str = "sparse"
    ) -> float:
        """
        Compute reward for goal achievement.

        Args:
            achieved_goal: What was achieved
            desired_goal: What was desired
            info: Optional info dict
            reward_type: "sparse" or "dense"

        Returns:
            Reward value
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)

        if info:
            threshold = info.get('distance_threshold', 0.05)
        else:
            threshold = 0.05

        if reward_type == "sparse":
            # Binary reward
            return 0.0 if distance < threshold else -1.0

        elif reward_type == "dense":
            # Negative distance
            return -distance

        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
