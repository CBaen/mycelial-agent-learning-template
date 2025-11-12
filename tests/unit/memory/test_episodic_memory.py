"""
Tests for EpisodicMemory.

Test Coverage:
- Basic operations (store, sample, update_priorities)
- Experience dataclass
- Memory consolidation
- Statistics and tracking
- Integration with PrioritizedReplayBuffer
- Edge cases and error handling
"""

import pytest
import numpy as np
import time
from src.memory.episodic_memory import EpisodicMemory, Experience


class MockAgent:
    """Mock agent for testing consolidation."""

    def __init__(self):
        self.learn_calls = 0
        self.last_batch = None
        self.last_weights = None

    def learn_from_batch(self, batch, weights):
        """Mock learning that returns TD errors and loss."""
        self.learn_calls += 1
        self.last_batch = batch
        self.last_weights = weights

        # Return mock TD errors and loss
        td_errors = np.random.uniform(0.1, 2.0, size=len(batch))
        loss = np.mean(td_errors)

        return td_errors, loss


class TestExperienceDataclass:
    """Test Experience dataclass."""

    def test_experience_creation(self):
        """Test creating Experience object."""
        exp = Experience(
            state=np.array([1, 2, 3]),
            action=0,
            reward=1.0,
            next_state=np.array([4, 5, 6]),
            done=False,
            info={'step': 1},
            timestamp=time.time()
        )

        assert np.array_equal(exp.state, np.array([1, 2, 3]))
        assert exp.action == 0
        assert exp.reward == 1.0
        assert exp.done == False
        assert exp.info['step'] == 1

    def test_experience_to_tuple(self):
        """Test converting Experience to tuple."""
        exp = Experience(
            state=np.array([1, 2, 3]),
            action=0,
            reward=1.0,
            next_state=np.array([4, 5, 6]),
            done=False,
            info={},
            timestamp=time.time()
        )

        tup = exp.to_tuple()
        assert len(tup) == 5
        assert np.array_equal(tup[0], np.array([1, 2, 3]))
        assert tup[1] == 0
        assert tup[2] == 1.0


class TestEpisodicMemoryBasics:
    """Test basic EpisodicMemory operations."""

    def test_initialization(self):
        """Test memory initialization."""
        memory = EpisodicMemory(capacity=1000, alpha=0.6, beta=0.4)

        assert memory.capacity == 1000
        assert memory.alpha == 0.6
        assert memory.beta == 0.4
        assert len(memory) == 0
        assert memory.total_stored == 0

    def test_initialization_invalid_capacity(self):
        """Test initialization with invalid capacity."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            EpisodicMemory(capacity=0)

        with pytest.raises(ValueError, match="Capacity must be positive"):
            EpisodicMemory(capacity=-100)

    def test_store_experience(self):
        """Test storing single experience."""
        memory = EpisodicMemory(capacity=100)

        state = np.array([1, 2, 3, 4])
        action = 0
        reward = 1.0
        next_state = np.array([2, 3, 4, 5])
        done = False

        memory.store(state, action, reward, next_state, done)

        assert len(memory) == 1
        assert memory.total_stored == 1

    def test_store_with_info(self):
        """Test storing experience with info dict."""
        memory = EpisodicMemory(capacity=100)

        memory.store(
            state=np.array([1, 2]),
            action=0,
            reward=1.0,
            next_state=np.array([3, 4]),
            done=False,
            info={'episode': 1, 'step': 10}
        )

        assert len(memory) == 1

    def test_store_multiple_experiences(self):
        """Test storing multiple experiences."""
        memory = EpisodicMemory(capacity=100)

        for i in range(50):
            memory.store(
                state=np.array([i, i+1]),
                action=i % 4,
                reward=float(i),
                next_state=np.array([i+1, i+2]),
                done=i % 10 == 0
            )

        assert len(memory) == 50
        assert memory.total_stored == 50

    def test_circular_buffer_overflow(self):
        """Test that memory overwrites when full."""
        memory = EpisodicMemory(capacity=10)

        for i in range(20):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        assert len(memory) == 10  # Capped at capacity
        assert memory.total_stored == 20  # But counts all stored


class TestEpisodicMemorySampling:
    """Test sampling operations."""

    def test_sample_basic(self):
        """Test basic sampling."""
        memory = EpisodicMemory(capacity=100)

        # Store experiences
        for i in range(20):
            memory.store(
                state=np.array([i]),
                action=i % 4,
                reward=float(i),
                next_state=np.array([i+1]),
                done=False
            )

        # Sample batch
        batch, indices, weights = memory.sample(batch_size=10)

        assert len(batch) == 10
        assert len(indices) == 10
        assert len(weights) == 10
        assert memory.total_sampled == 10

    def test_sample_from_empty_memory(self):
        """Test sampling from empty memory raises error."""
        memory = EpisodicMemory(capacity=100)

        with pytest.raises(ValueError, match="Cannot sample from empty memory"):
            memory.sample(batch_size=5)

    def test_sample_too_many(self):
        """Test sampling more than available."""
        memory = EpisodicMemory(capacity=100)

        for i in range(5):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        with pytest.raises(ValueError, match="Cannot sample"):
            memory.sample(batch_size=10)

    def test_sample_returns_experience_objects(self):
        """Test that sampling returns Experience objects."""
        memory = EpisodicMemory(capacity=100)

        for i in range(10):
            memory.store(
                state=np.array([i]),
                action=i % 4,
                reward=float(i),
                next_state=np.array([i+1]),
                done=i % 5 == 0
            )

        batch, indices, weights = memory.sample(batch_size=5)

        # Check all are Experience objects
        for exp in batch:
            assert isinstance(exp, Experience)
            assert hasattr(exp, 'state')
            assert hasattr(exp, 'action')
            assert hasattr(exp, 'reward')

    def test_importance_weights_normalized(self):
        """Test that importance weights are normalized."""
        memory = EpisodicMemory(capacity=100, alpha=0.6, beta=0.4)

        for i in range(20):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False,
                priority=float(i+1)  # Different priorities
            )

        batch, indices, weights = memory.sample(batch_size=10)

        # Max weight should be 1.0
        assert np.isclose(weights.max(), 1.0)
        assert np.all(weights <= 1.0)
        assert np.all(weights > 0)


class TestPriorityUpdates:
    """Test priority update operations."""

    def test_update_priorities(self):
        """Test updating priorities with TD errors."""
        memory = EpisodicMemory(capacity=100)

        for i in range(10):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        # Sample batch
        batch, indices, weights = memory.sample(batch_size=5)

        # Update priorities
        td_errors = np.array([0.5, 2.0, 1.0, 0.1, 1.5])
        memory.update_priorities(indices, td_errors)

        # Verify max priority updated
        assert memory.buffer.get_max_priority() >= 2.0

    def test_update_priorities_affects_sampling(self):
        """Test that priority updates affect sampling distribution."""
        memory = EpisodicMemory(capacity=100, alpha=1.0)  # Fully prioritized

        # Store two experiences with equal priority
        for i in range(2):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False,
                priority=1.0
            )

        # Update second to have much higher priority
        memory.update_priorities([1], np.array([100.0]))

        # Sample many times
        counts = [0, 0]
        for _ in range(100):
            batch, indices, weights = memory.sample(batch_size=1)
            counts[indices[0]] += 1

        # Second should be sampled much more
        assert counts[1] > counts[0] * 3


class TestMemoryConsolidation:
    """Test memory consolidation (offline learning)."""

    def test_consolidate_basic(self):
        """Test basic consolidation."""
        memory = EpisodicMemory(capacity=100)
        agent = MockAgent()

        # Store experiences
        for i in range(50):
            memory.store(
                state=np.array([i]),
                action=i % 4,
                reward=float(i),
                next_state=np.array([i+1]),
                done=False
            )

        # Consolidate
        result = memory.consolidate(agent, num_steps=10)

        assert result['status'] == 'success'
        assert result['steps'] == 10
        assert agent.learn_calls == 10
        assert memory.total_consolidated == 10

    def test_consolidate_empty_memory(self):
        """Test consolidation with empty memory."""
        memory = EpisodicMemory(capacity=100)
        agent = MockAgent()

        result = memory.consolidate(agent, num_steps=10)

        assert result['status'] == 'skipped'
        assert result['reason'] == 'empty_buffer'
        assert agent.learn_calls == 0

    def test_consolidation_history_tracking(self):
        """Test that consolidation history is tracked."""
        memory = EpisodicMemory(capacity=100)
        agent = MockAgent()

        for i in range(20):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        # First consolidation
        memory.consolidate(agent, num_steps=5)
        assert len(memory.consolidation_history) == 1

        # Second consolidation
        memory.consolidate(agent, num_steps=5)
        assert len(memory.consolidation_history) == 2

        # Check total
        assert memory.total_consolidated == 10

    def test_consolidation_returns_statistics(self):
        """Test consolidation returns useful statistics."""
        memory = EpisodicMemory(capacity=100)
        agent = MockAgent()

        for i in range(20):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        result = memory.consolidate(agent, num_steps=10)

        assert 'status' in result
        assert 'steps' in result
        assert 'elapsed' in result
        assert 'mean_loss' in result
        assert 'final_loss' in result
        assert isinstance(result['mean_loss'], float)


class TestBetaAnnealing:
    """Test beta parameter annealing."""

    def test_update_beta(self):
        """Test updating beta parameter."""
        memory = EpisodicMemory(capacity=100, beta=0.4)

        assert memory.beta == 0.4

        memory.update_beta(0.6)
        assert memory.beta == 0.6
        assert memory.buffer.beta == 0.6

    def test_beta_annealing_during_training(self):
        """Test beta annealing over training."""
        memory = EpisodicMemory(capacity=100, beta=0.4)

        for i in range(10):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        # Anneal beta
        for step in range(10):
            beta = 0.4 + (1.0 - 0.4) * (step / 10)
            memory.update_beta(beta)

            # Sample
            batch, indices, weights = memory.sample(batch_size=5)

        # Final beta should be close to 1.0
        assert memory.beta > 0.9


class TestStatistics:
    """Test statistics and tracking."""

    def test_get_statistics_empty(self):
        """Test statistics for empty memory."""
        memory = EpisodicMemory(capacity=100)

        stats = memory.get_statistics()

        assert stats['capacity'] == 100
        assert stats['size'] == 0
        assert stats['utilization'] == 0.0
        assert stats['total_stored'] == 0
        assert stats['total_sampled'] == 0

    def test_get_statistics_after_operations(self):
        """Test statistics after operations."""
        memory = EpisodicMemory(capacity=100)

        for i in range(20):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        # Sample some
        memory.sample(batch_size=10)
        memory.sample(batch_size=5)

        stats = memory.get_statistics()

        assert stats['size'] == 20
        assert stats['utilization'] == 0.2
        assert stats['total_stored'] == 20
        assert stats['total_sampled'] == 15

    def test_repr(self):
        """Test string representation."""
        memory = EpisodicMemory(capacity=100, alpha=0.6, beta=0.4)

        repr_str = repr(memory)
        assert "EpisodicMemory" in repr_str
        assert "capacity=100" in repr_str
        assert "alpha=0.6" in repr_str

    def test_len(self):
        """Test length method."""
        memory = EpisodicMemory(capacity=100)

        assert len(memory) == 0

        memory.store(
            state=np.array([1]),
            action=0,
            reward=1.0,
            next_state=np.array([2]),
            done=False
        )

        assert len(memory) == 1


class TestClearOperation:
    """Test clearing memory."""

    def test_clear(self):
        """Test clearing memory."""
        memory = EpisodicMemory(capacity=100)

        for i in range(20):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        assert len(memory) == 20
        assert memory.total_stored == 20

        # Clear
        memory.clear()

        assert len(memory) == 0
        assert memory.total_stored == 0
        assert memory.total_sampled == 0


class TestValidation:
    """Test memory validation."""

    def test_validate_empty(self):
        """Test validating empty memory."""
        memory = EpisodicMemory(capacity=100)
        assert memory.validate()

    def test_validate_after_operations(self):
        """Test validating after operations."""
        memory = EpisodicMemory(capacity=100)

        for i in range(20):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        assert memory.validate()

        # Sample and update
        batch, indices, weights = memory.sample(batch_size=10)
        td_errors = np.random.uniform(0.1, 2.0, size=10)
        memory.update_priorities(indices, td_errors)

        assert memory.validate()


class TestIntegration:
    """Test integrated workflows."""

    def test_full_training_loop(self):
        """Test complete training loop with memory."""
        memory = EpisodicMemory(capacity=1000, alpha=0.6, beta=0.4)
        agent = MockAgent()

        # Episode 1: Collect experiences
        for step in range(100):
            state = np.random.randn(4)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = step == 99

            memory.store(state, action, reward, next_state, done)

        # Training loop
        for iteration in range(10):
            # Sample batch
            batch, indices, weights = memory.sample(batch_size=32)

            # Learn
            td_errors, loss = agent.learn_from_batch(batch, weights)

            # Update priorities
            memory.update_priorities(indices, td_errors)

        # Consolidate
        result = memory.consolidate(agent, num_steps=20)

        assert result['status'] == 'success'
        assert len(memory) == 100
        assert memory.total_sampled > 0

    def test_multi_episode_workflow(self):
        """Test multi-episode training workflow."""
        memory = EpisodicMemory(capacity=10000, alpha=0.6, beta=0.4)
        agent = MockAgent()

        # Multiple episodes
        for episode in range(5):
            for step in range(50):
                memory.store(
                    state=np.random.randn(4),
                    action=np.random.randint(0, 4),
                    reward=np.random.randn(),
                    next_state=np.random.randn(4),
                    done=step == 49
                )

            # Train after each episode
            for _ in range(5):
                batch, indices, weights = memory.sample(batch_size=32)
                td_errors, loss = agent.learn_from_batch(batch, weights)
                memory.update_priorities(indices, td_errors)

            # Periodic consolidation
            if episode % 2 == 0:
                memory.consolidate(agent, num_steps=10)

        # Check stats
        stats = memory.get_statistics()
        assert stats['size'] == 250
        assert stats['total_stored'] == 250
        assert len(memory.consolidation_history) == 3
