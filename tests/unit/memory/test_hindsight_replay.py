"""
Tests for Hindsight Experience Replay (HER).

Test Coverage:
- HER initialization
- Goal relabeling strategies (FUTURE, FINAL, EPISODE, RANDOM, MIXED)
- Episode relabeling
- Batch relabeling
- Reward computation
- Goal achievement checking
- Statistics tracking
- GoalEnv utilities
"""

import pytest
import numpy as np
from src.memory.hindsight_replay import (
    HindsightReplay,
    HERStrategy,
    HERTransition,
    GoalEnv
)


class TestHERTransition:
    """Test HERTransition dataclass."""

    def test_transition_creation(self):
        """Test creating HER transition."""
        trans = HERTransition(
            state=np.array([1.0, 2.0]),
            action=1,
            reward=-1.0,
            next_state=np.array([2.0, 3.0]),
            done=False,
            goal=np.array([10.0, 10.0]),
            achieved_goal=np.array([2.0, 3.0]),
            info={}
        )

        assert trans.state.shape == (2,)
        assert trans.action == 1
        assert trans.reward == -1.0
        assert not trans.done

    def test_to_dict(self):
        """Test converting transition to dictionary."""
        trans = HERTransition(
            state=np.array([1.0]),
            action=0,
            reward=0.0,
            next_state=np.array([2.0]),
            done=True,
            goal=np.array([2.0]),
            achieved_goal=np.array([2.0]),
            info={'success': True}
        )

        d = trans.to_dict()

        assert 'state' in d
        assert 'action' in d
        assert 'goal' in d
        assert d['info']['success'] is True


class TestHindsightReplayInit:
    """Test HER initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        her = HindsightReplay()

        assert her.strategy == HERStrategy.FUTURE
        assert her.relabel_ratio == 0.8
        assert her.k_future == 4
        assert her.total_episodes == 0

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        her = HindsightReplay(
            strategy=HERStrategy.FINAL,
            relabel_ratio=0.5,
            k_future=8
        )

        assert her.strategy == HERStrategy.FINAL
        assert her.relabel_ratio == 0.5
        assert her.k_future == 8

    def test_custom_reward_func(self):
        """Test initialization with custom reward function."""
        def custom_reward(achieved, desired, info):
            return -np.linalg.norm(achieved - desired)

        her = HindsightReplay(reward_func=custom_reward)

        # Test that custom function is used
        achieved = np.array([1.0, 1.0])
        desired = np.array([0.0, 0.0])
        reward = her.reward_func(achieved, desired, {})

        assert reward < 0  # Custom function returns negative distance


class TestHERFutureStrategy:
    """Test FUTURE goal selection strategy."""

    def test_relabel_with_future_strategy(self):
        """Test relabeling episode with FUTURE strategy."""
        her = HindsightReplay(strategy=HERStrategy.FUTURE, relabel_ratio=1.0, k_future=2)

        # Create simple episode
        episode = [
            HERTransition(
                state=np.array([0.0, 0.0]),
                action=0,
                reward=-1.0,
                next_state=np.array([1.0, 0.0]),
                done=False,
                goal=np.array([10.0, 10.0]),
                achieved_goal=np.array([1.0, 0.0]),
                info={}
            ),
            HERTransition(
                state=np.array([1.0, 0.0]),
                action=0,
                reward=-1.0,
                next_state=np.array([2.0, 1.0]),
                done=False,
                goal=np.array([10.0, 10.0]),
                achieved_goal=np.array([2.0, 1.0]),
                info={}
            ),
            HERTransition(
                state=np.array([2.0, 1.0]),
                action=0,
                reward=-1.0,
                next_state=np.array([3.0, 2.0]),
                done=True,
                goal=np.array([10.0, 10.0]),
                achieved_goal=np.array([3.0, 2.0]),
                info={}
            )
        ]

        relabeled = her.relabel_episode(episode)

        # Should have original + hindsight transitions
        assert len(relabeled) > len(episode)

        # Check that some transitions have her_relabeled flag
        relabeled_count = sum(
            1 for trans in relabeled
            if trans.info.get('her_relabeled', False)
        )
        assert relabeled_count > 0

    def test_future_goals_are_from_future(self):
        """Test that FUTURE strategy selects goals from future states."""
        her = HindsightReplay(strategy=HERStrategy.FUTURE, k_future=2)

        episode = [
            HERTransition(
                state=np.array([float(i)]),
                action=0,
                reward=-1.0,
                next_state=np.array([float(i+1)]),
                done=False,
                goal=np.array([10.0]),
                achieved_goal=np.array([float(i+1)]),
                info={}
            )
            for i in range(5)
        ]

        # Select goals for first transition
        goals = her._select_future_goals(episode, 0)

        # Goals should be from states 1-4
        assert len(goals) <= 2
        for goal in goals:
            assert goal[0] > 1.0  # Should be future states


class TestHERFinalStrategy:
    """Test FINAL goal selection strategy."""

    def test_relabel_with_final_strategy(self):
        """Test relabeling with FINAL strategy."""
        her = HindsightReplay(strategy=HERStrategy.FINAL, relabel_ratio=1.0)

        episode = [
            HERTransition(
                state=np.array([0.0]),
                action=0,
                reward=-1.0,
                next_state=np.array([1.0]),
                done=False,
                goal=np.array([10.0]),
                achieved_goal=np.array([1.0]),
                info={}
            ),
            HERTransition(
                state=np.array([1.0]),
                action=0,
                reward=-1.0,
                next_state=np.array([5.0]),
                done=True,
                goal=np.array([10.0]),
                achieved_goal=np.array([5.0]),
                info={}
            )
        ]

        relabeled = her.relabel_episode(episode)

        # Find relabeled transitions
        her_transitions = [
            trans for trans in relabeled
            if trans.info.get('her_relabeled', False)
        ]

        # All relabeled transitions should have final state as goal
        for trans in her_transitions:
            assert np.allclose(trans.goal, np.array([5.0]))

    def test_final_goal_is_last_achieved(self):
        """Test that FINAL strategy uses last achieved goal."""
        her = HindsightReplay(strategy=HERStrategy.FINAL)

        episode = [
            HERTransition(
                state=np.array([i]),
                action=0,
                reward=-1.0,
                next_state=np.array([i+1]),
                done=(i == 4),
                goal=np.array([100.0]),
                achieved_goal=np.array([i+1]),
                info={}
            )
            for i in range(5)
        ]

        final_goals = her._select_final_goal(episode)

        assert len(final_goals) == 1
        assert np.allclose(final_goals[0], np.array([5.0]))


class TestHERMixedStrategy:
    """Test MIXED strategy combination."""

    def test_mixed_strategy(self):
        """Test that MIXED strategy combines strategies."""
        her = HindsightReplay(strategy=HERStrategy.MIXED, k_future=2)

        episode = [
            HERTransition(
                state=np.array([i]),
                action=0,
                reward=-1.0,
                next_state=np.array([i+1]),
                done=False,
                goal=np.array([10.0]),
                achieved_goal=np.array([i+1]),
                info={}
            )
            for i in range(5)
        ]

        goals = her._select_goals(episode, 0)

        # Mixed should return multiple goals
        assert len(goals) > 0
        # Limited by k_future
        assert len(goals) <= her.k_future


class TestHERRelabeling:
    """Test goal relabeling mechanics."""

    def test_relabel_ratio(self):
        """Test that relabel_ratio controls fraction of transitions."""
        her = HindsightReplay(strategy=HERStrategy.FINAL, relabel_ratio=0.5, k_future=1)

        episode = [
            HERTransition(
                state=np.array([i]),
                action=0,
                reward=-1.0,
                next_state=np.array([i+1]),
                done=False,
                goal=np.array([10.0]),
                achieved_goal=np.array([i+1]),
                info={}
            )
            for i in range(10)
        ]

        relabeled = her.relabel_episode(episode)

        # With ratio=0.5 and k=1, should have ~15 transitions (10 original + 5 relabeled)
        # Allow some variance due to random sampling
        assert 12 <= len(relabeled) <= 18

    def test_empty_episode(self):
        """Test relabeling empty episode."""
        her = HindsightReplay()

        relabeled = her.relabel_episode([])

        assert len(relabeled) == 0

    def test_single_transition_episode(self):
        """Test relabeling single-transition episode."""
        her = HindsightReplay(strategy=HERStrategy.FINAL)

        episode = [
            HERTransition(
                state=np.array([0.0]),
                action=0,
                reward=-1.0,
                next_state=np.array([1.0]),
                done=True,
                goal=np.array([10.0]),
                achieved_goal=np.array([1.0]),
                info={}
            )
        ]

        relabeled = her.relabel_episode(episode)

        # Should have at least original transition
        assert len(relabeled) >= 1

    def test_relabeled_rewards_updated(self):
        """Test that relabeled transitions have updated rewards."""
        def custom_reward(achieved, desired, info):
            dist = np.linalg.norm(achieved - desired)
            return 0.0 if dist < 0.1 else -1.0

        her = HindsightReplay(
            strategy=HERStrategy.FINAL,
            relabel_ratio=1.0,
            reward_func=custom_reward
        )

        episode = [
            HERTransition(
                state=np.array([0.0, 0.0]),
                action=0,
                reward=-1.0,
                next_state=np.array([1.0, 1.0]),
                done=False,
                goal=np.array([10.0, 10.0]),  # Far from achieved
                achieved_goal=np.array([1.0, 1.0]),
                info={}
            ),
            HERTransition(
                state=np.array([1.0, 1.0]),
                action=0,
                reward=-1.0,
                next_state=np.array([5.0, 5.0]),
                done=True,
                goal=np.array([10.0, 10.0]),
                achieved_goal=np.array([5.0, 5.0]),
                info={}
            )
        ]

        relabeled = her.relabel_episode(episode)

        # Find transitions with final goal
        final_goal_trans = [
            trans for trans in relabeled
            if trans.info.get('her_relabeled', False) and
            np.allclose(trans.goal, np.array([5.0, 5.0]))
        ]

        # These should have rewards recomputed
        assert len(final_goal_trans) > 0


class TestHERBatchRelabeling:
    """Test batch episode relabeling."""

    def test_relabel_batch(self):
        """Test relabeling multiple episodes."""
        her = HindsightReplay(strategy=HERStrategy.FINAL, relabel_ratio=0.5)

        episodes = []
        for ep in range(3):
            episode = [
                HERTransition(
                    state=np.array([i]),
                    action=0,
                    reward=-1.0,
                    next_state=np.array([i+1]),
                    done=(i == 4),
                    goal=np.array([10.0]),
                    achieved_goal=np.array([i+1]),
                    info={}
                )
                for i in range(5)
            ]
            episodes.append(episode)

        all_transitions = her.relabel_batch(episodes)

        # Should have transitions from all episodes
        assert len(all_transitions) > 15  # Original 15 + hindsight


class TestHERRewardComputation:
    """Test reward computation."""

    def test_default_reward_sparse(self):
        """Test default sparse reward function."""
        her = HindsightReplay()

        # Goal achieved (within threshold)
        achieved = np.array([1.0, 1.0])
        desired = np.array([1.01, 1.01])
        reward = her.reward_func(achieved, desired, {'distance_threshold': 0.05})

        assert reward == 0.0

        # Goal not achieved
        achieved = np.array([1.0, 1.0])
        desired = np.array([10.0, 10.0])
        reward = her.reward_func(achieved, desired, {})

        assert reward == -1.0

    def test_goal_env_compute_reward_sparse(self):
        """Test GoalEnv sparse reward computation."""
        achieved = np.array([2.0, 3.0])
        desired = np.array([2.01, 3.01])

        reward = GoalEnv.compute_reward(achieved, desired, reward_type="sparse")

        assert reward == 0.0  # Within default threshold

    def test_goal_env_compute_reward_dense(self):
        """Test GoalEnv dense reward computation."""
        achieved = np.array([0.0, 0.0])
        desired = np.array([3.0, 4.0])

        reward = GoalEnv.compute_reward(achieved, desired, reward_type="dense")

        assert reward == -5.0  # Negative Euclidean distance


class TestHERStatistics:
    """Test HER statistics tracking."""

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        her = HindsightReplay(strategy=HERStrategy.FUTURE, relabel_ratio=0.5, k_future=2)

        episode = [
            HERTransition(
                state=np.array([i]),
                action=0,
                reward=-1.0,
                next_state=np.array([i+1]),
                done=False,
                goal=np.array([10.0]),
                achieved_goal=np.array([i+1]),
                info={}
            )
            for i in range(5)
        ]

        relabeled = her.relabel_episode(episode)

        stats = her.get_statistics()

        assert stats['total_episodes'] == 1
        assert stats['total_transitions'] == 5
        assert stats['total_relabeled'] > 0
        assert stats['strategy'] == 'future'

    def test_success_rate_update(self):
        """Test success rate tracking."""
        her = HindsightReplay()

        her.update_success_rate(True)
        assert her.success_rate > 0

        her.update_success_rate(False)
        # Should be exponential moving average
        assert 0 < her.success_rate < 1

    def test_augmentation_factor(self):
        """Test data augmentation factor calculation."""
        her = HindsightReplay(relabel_ratio=1.0, k_future=1)

        episode = [
            HERTransition(
                state=np.array([i]),
                action=0,
                reward=-1.0,
                next_state=np.array([i+1]),
                done=False,
                goal=np.array([10.0]),
                achieved_goal=np.array([i+1]),
                info={}
            )
            for i in range(5)
        ]

        her.relabel_episode(episode)

        stats = her.get_statistics()

        # Augmentation factor should be > 1 (original + hindsight)
        assert stats['augmentation_factor'] > 1.0

    def test_repr(self):
        """Test string representation."""
        her = HindsightReplay(strategy=HERStrategy.FINAL)

        repr_str = repr(her)

        assert "HindsightReplay" in repr_str
        assert "strategy=final" in repr_str


class TestGoalEnv:
    """Test GoalEnv utility functions."""

    def test_extract_goal_from_state(self):
        """Test extracting goal from state."""
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Default: assume goal is second half
        goal = GoalEnv.extract_goal_from_state(state)

        assert len(goal) == 3
        assert np.allclose(goal, np.array([4.0, 5.0, 6.0]))

    def test_extract_goal_with_indices(self):
        """Test extracting goal with specific indices."""
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        goal = GoalEnv.extract_goal_from_state(state, goal_indices=[0, 2, 4])

        assert np.allclose(goal, np.array([1.0, 3.0, 5.0]))

    def test_extract_achieved_goal(self):
        """Test extracting achieved goal from state."""
        state = np.array([1.0, 2.0, 3.0, 4.0])

        # Default: first half
        achieved = GoalEnv.extract_achieved_goal(state)

        assert len(achieved) == 2
        assert np.allclose(achieved, np.array([1.0, 2.0]))

    def test_extract_achieved_with_indices(self):
        """Test extracting achieved goal with indices."""
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        achieved = GoalEnv.extract_achieved_goal(state, achieved_indices=[1, 3])

        assert np.allclose(achieved, np.array([2.0, 4.0]))


class TestHEREdgeCases:
    """Test edge cases and error handling."""

    def test_relabel_with_zero_ratio(self):
        """Test that zero relabel_ratio returns only originals."""
        her = HindsightReplay(relabel_ratio=0.0)

        episode = [
            HERTransition(
                state=np.array([i]),
                action=0,
                reward=-1.0,
                next_state=np.array([i+1]),
                done=False,
                goal=np.array([10.0]),
                achieved_goal=np.array([i+1]),
                info={}
            )
            for i in range(5)
        ]

        relabeled = her.relabel_episode(episode)

        # Should only have original transitions
        assert len(relabeled) == len(episode)

    def test_relabel_with_full_ratio(self):
        """Test relabeling with ratio=1.0."""
        her = HindsightReplay(relabel_ratio=1.0, k_future=1)

        episode = [
            HERTransition(
                state=np.array([i]),
                action=0,
                reward=-1.0,
                next_state=np.array([i+1]),
                done=False,
                goal=np.array([10.0]),
                achieved_goal=np.array([i+1]),
                info={}
            )
            for i in range(5)
        ]

        relabeled = her.relabel_episode(episode)

        # Should have original + relabeled
        assert len(relabeled) > len(episode)

    def test_invalid_strategy(self):
        """Test handling of invalid strategy."""
        her = HindsightReplay()
        her.strategy = "invalid"  # Force invalid strategy

        episode = [
            HERTransition(
                state=np.array([0.0]),
                action=0,
                reward=-1.0,
                next_state=np.array([1.0]),
                done=True,
                goal=np.array([10.0]),
                achieved_goal=np.array([1.0]),
                info={}
            )
        ]

        with pytest.raises(ValueError, match="Unknown strategy"):
            her._select_goals(episode, 0)
