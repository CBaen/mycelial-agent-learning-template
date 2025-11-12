"""
Tests for PrioritizedReplayBuffer.

Test Coverage:
- Basic operations (add, sample, update_priorities)
- Priority-based sampling distribution
- Importance sampling weights
- Parameter validation (alpha, beta, epsilon)
- Edge cases and error handling
- Performance benchmarks
"""

import pytest
import numpy as np
from src.memory.prioritized_replay_buffer import PrioritizedReplayBuffer


class TestPrioritizedReplayBufferBasics:
    """Test basic buffer operations."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        assert buffer.capacity == 100
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert len(buffer) == 0
        assert buffer.get_max_priority() == 1.0

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        # Invalid capacity
        with pytest.raises(ValueError, match="Capacity must be positive"):
            PrioritizedReplayBuffer(capacity=0)

        # Invalid alpha
        with pytest.raises(ValueError, match="Alpha must be in"):
            PrioritizedReplayBuffer(capacity=10, alpha=1.5)

        # Invalid beta
        with pytest.raises(ValueError, match="Beta must be in"):
            PrioritizedReplayBuffer(capacity=10, beta=-0.1)

        # Invalid epsilon
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            PrioritizedReplayBuffer(capacity=10, epsilon=-1e-6)

    def test_add_experience(self):
        """Test adding experiences."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        exp = (np.array([1, 2, 3]), 0, 1.0, np.array([4, 5, 6]), False)
        buffer.add(exp)

        assert len(buffer) == 1

    def test_add_with_explicit_priority(self):
        """Test adding experience with explicit priority."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        exp = (np.array([1, 2, 3]), 0, 1.0, np.array([4, 5, 6]), False)
        buffer.add(exp, priority=5.0)

        assert len(buffer) == 1
        assert np.isclose(buffer.get_max_priority(), 5.0)

    def test_add_multiple_experiences(self):
        """Test adding multiple experiences."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp)

        assert len(buffer) == 5

    def test_circular_buffer_overflow(self):
        """Test buffer overwrites oldest when full."""
        buffer = PrioritizedReplayBuffer(capacity=3)

        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp)

        # Size should cap at capacity
        assert len(buffer) == 3


class TestPrioritizedSampling:
    """Test prioritized sampling behavior."""

    def test_sample_basic(self):
        """Test basic sampling."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        # Add 5 experiences
        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=1.0)

        # Sample batch
        batch, indices, weights = buffer.sample(batch_size=3)

        assert len(batch) == 3
        assert len(indices) == 3
        assert len(weights) == 3

    def test_sample_insufficient_experiences(self):
        """Test sampling more than available experiences."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        exp = (np.array([1]), 0, 1.0, np.array([2]), False)
        buffer.add(exp)

        with pytest.raises(ValueError, match="Cannot sample"):
            buffer.sample(batch_size=5)

    def test_sample_with_custom_beta(self):
        """Test sampling with custom beta parameter."""
        buffer = PrioritizedReplayBuffer(capacity=10, beta=0.4)

        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=1.0)

        # Sample with different beta
        batch, indices, weights = buffer.sample(batch_size=3, beta=0.8)

        assert len(batch) == 3

    def test_importance_weights_normalized(self):
        """Test that importance weights are normalized."""
        buffer = PrioritizedReplayBuffer(capacity=10, alpha=0.6, beta=0.4)

        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=float(i+1))

        batch, indices, weights = buffer.sample(batch_size=5)

        # Max weight should be 1.0
        assert np.isclose(weights.max(), 1.0)
        # All weights should be <= 1.0
        assert np.all(weights <= 1.0)
        # All weights should be > 0
        assert np.all(weights > 0)

    def test_sampling_distribution_uniform_alpha_0(self):
        """Test that alpha=0 gives uniform sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.0)

        # Add experiences with different priorities
        for i in range(10):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=float(i+1))  # Priorities: 1, 2, 3, ..., 10

        # Sample many times
        counts = np.zeros(10)
        for _ in range(1000):
            batch, indices, weights = buffer.sample(batch_size=1)
            counts[indices[0]] += 1

        # With alpha=0, all should be sampled roughly equally
        # Each should get ~100 samples (with tolerance)
        for count in counts:
            assert 50 < count < 150  # Generous tolerance

    def test_sampling_distribution_prioritized_alpha_1(self):
        """Test that alpha=1 gives fully prioritized sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0)

        # Add experiences: priority 1 and priority 9
        exp1 = (np.array([0]), 0, 0.0, np.array([1]), False)
        exp2 = (np.array([1]), 1, 1.0, np.array([2]), False)

        buffer.add(exp1, priority=1.0)
        buffer.add(exp2, priority=9.0)

        # Sample many times
        count_low = 0
        count_high = 0

        for _ in range(1000):
            batch, indices, weights = buffer.sample(batch_size=1)
            if indices[0] == 0:
                count_low += 1
            else:
                count_high += 1

        # High priority should be sampled ~9x more (ratio 9:1)
        # Expected: low=100, high=900
        assert 50 < count_low < 200
        assert 800 < count_high < 950


class TestPriorityUpdates:
    """Test priority update operations."""

    def test_update_priorities(self):
        """Test updating priorities after learning."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        # Add experiences
        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=1.0)

        # Sample batch
        batch, indices, weights = buffer.sample(batch_size=3)

        # Update priorities with TD errors
        td_errors = np.array([0.5, 2.0, 1.0])
        buffer.update_priorities(indices, td_errors)

        # Max priority should be updated
        assert buffer.get_max_priority() >= 2.0

    def test_update_priorities_length_mismatch(self):
        """Test error when indices and priorities have different lengths."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=1.0)

        indices = [0, 1]
        priorities = np.array([1.0, 2.0, 3.0])  # Wrong length

        with pytest.raises(ValueError, match="must have same length"):
            buffer.update_priorities(indices, priorities)

    def test_update_priorities_increases_sampling_probability(self):
        """Test that updating priority affects sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0)

        # Add two experiences with equal priority
        exp1 = (np.array([0]), 0, 0.0, np.array([1]), False)
        exp2 = (np.array([1]), 1, 1.0, np.array([2]), False)

        buffer.add(exp1, priority=1.0)
        buffer.add(exp2, priority=1.0)

        # Update second experience to have much higher priority
        buffer.update_priorities([1], np.array([100.0]))

        # Sample many times
        counts = [0, 0]
        for _ in range(1000):
            batch, indices, weights = buffer.sample(batch_size=1)
            counts[indices[0]] += 1

        # Second experience should be sampled much more
        assert counts[1] > counts[0] * 10


class TestBetaAnnealing:
    """Test beta parameter annealing."""

    def test_update_beta(self):
        """Test updating beta parameter."""
        buffer = PrioritizedReplayBuffer(capacity=10, beta=0.4)

        assert buffer.beta == 0.4

        buffer.update_beta(0.6)
        assert buffer.beta == 0.6

        buffer.update_beta(1.0)
        assert buffer.beta == 1.0

    def test_update_beta_invalid(self):
        """Test updating beta with invalid value."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        with pytest.raises(ValueError, match="Beta must be in"):
            buffer.update_beta(1.5)

        with pytest.raises(ValueError, match="Beta must be in"):
            buffer.update_beta(-0.1)

    def test_beta_affects_importance_weights(self):
        """Test that beta affects importance sampling weights."""
        buffer = PrioritizedReplayBuffer(capacity=10, alpha=1.0)

        # Add experiences with different priorities
        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=float(i+1))

        # Sample with beta=0 (no correction)
        _, _, weights_beta_0 = buffer.sample(batch_size=5, beta=0.0)

        # Sample with beta=1 (full correction)
        _, _, weights_beta_1 = buffer.sample(batch_size=5, beta=1.0)

        # Weights should be different
        # With beta=0, all weights should be 1.0
        assert np.allclose(weights_beta_0, 1.0)

        # With beta=1, weights should vary
        assert not np.allclose(weights_beta_1, 1.0)


class TestBufferStatistics:
    """Test buffer statistics and helpers."""

    def test_get_statistics(self):
        """Test getting buffer statistics."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        # Add some experiences
        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=float(i+1))

        stats = buffer.get_statistics()

        assert stats['capacity'] == 10
        assert stats['size'] == 5
        assert stats['utilization'] == 0.5
        assert stats['total_added'] == 5
        assert stats['total_sampled'] == 0

        # Sample some
        buffer.sample(batch_size=2)
        stats = buffer.get_statistics()
        assert stats['total_sampled'] == 2

    def test_repr(self):
        """Test string representation."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        repr_str = repr(buffer)
        assert "PrioritizedReplayBuffer" in repr_str
        assert "capacity=100" in repr_str
        assert "alpha=0.6" in repr_str
        assert "beta=0.4" in repr_str

    def test_len(self):
        """Test length method."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        assert len(buffer) == 0

        exp = (np.array([1]), 0, 1.0, np.array([2]), False)
        buffer.add(exp)

        assert len(buffer) == 1

    def test_clear(self):
        """Test clearing buffer."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        # Add experiences
        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp)

        assert len(buffer) == 5

        # Clear
        buffer.clear()

        assert len(buffer) == 0
        assert buffer.get_max_priority() == 1.0


class TestBufferValidation:
    """Test buffer validation."""

    def test_validate_empty(self):
        """Test validating empty buffer."""
        buffer = PrioritizedReplayBuffer(capacity=10)
        assert buffer.validate()

    def test_validate_after_adds(self):
        """Test validating after additions."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp)

        assert buffer.validate()

    def test_validate_after_updates(self):
        """Test validating after priority updates."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=1.0)

        # Update priorities
        buffer.update_priorities([0, 1], np.array([5.0, 10.0]))

        assert buffer.validate()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_experience_buffer(self):
        """Test buffer with capacity 1."""
        buffer = PrioritizedReplayBuffer(capacity=1)

        exp = (np.array([1]), 0, 1.0, np.array([2]), False)
        buffer.add(exp)

        batch, indices, weights = buffer.sample(batch_size=1)
        assert len(batch) == 1

    def test_very_small_priorities(self):
        """Test buffer with very small priorities."""
        buffer = PrioritizedReplayBuffer(capacity=10, epsilon=1e-10)

        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=1e-10)

        batch, indices, weights = buffer.sample(batch_size=3)
        assert len(batch) == 3

    def test_very_large_priorities(self):
        """Test buffer with very large priorities."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        for i in range(5):
            exp = (np.array([i]), i, float(i), np.array([i+1]), False)
            buffer.add(exp, priority=1e10)

        batch, indices, weights = buffer.sample(batch_size=3)
        assert len(batch) == 3

    def test_all_zero_priorities_after_init(self):
        """Test buffer behavior when priorities become zero."""
        buffer = PrioritizedReplayBuffer(capacity=10, epsilon=1e-6)

        # Add with zero priority (epsilon will be added)
        exp = (np.array([1]), 0, 1.0, np.array([2]), False)
        buffer.add(exp, priority=0.0)

        # Should still be able to sample
        batch, indices, weights = buffer.sample(batch_size=1)
        assert len(batch) == 1


class TestPerformance:
    """Test performance characteristics."""

    def test_sample_performance(self):
        """Test that sampling is fast enough."""
        import time

        buffer = PrioritizedReplayBuffer(capacity=100000)

        # Fill buffer
        for i in range(10000):
            exp = (np.array([i]), i % 4, float(i % 10), np.array([i+1]), i % 100 == 0)
            buffer.add(exp, priority=np.random.uniform(0.1, 10.0))

        # Time many samples
        start = time.perf_counter()
        for _ in range(100):
            buffer.sample(batch_size=32)
        elapsed = time.perf_counter() - start

        # Should be fast (< 0.1s for 100 batches)
        assert elapsed < 0.1

        # Per-batch should be < 5ms (as per performance target)
        per_batch = elapsed / 100
        assert per_batch < 0.005  # 5ms

    def test_update_performance(self):
        """Test that priority updates are fast."""
        import time

        buffer = PrioritizedReplayBuffer(capacity=100000)

        # Fill buffer
        for i in range(10000):
            exp = (np.array([i]), i % 4, float(i % 10), np.array([i+1]), i % 100 == 0)
            buffer.add(exp, priority=1.0)

        # Sample batch
        batch, indices, weights = buffer.sample(batch_size=32)

        # Time many updates
        start = time.perf_counter()
        for _ in range(100):
            td_errors = np.random.uniform(0.1, 2.0, size=32)
            buffer.update_priorities(indices, td_errors)
        elapsed = time.perf_counter() - start

        # Should be fast (generous tolerance for CI/slow systems)
        assert elapsed < 0.1


class TestIntegration:
    """Test integrated workflows."""

    def test_full_replay_workflow(self):
        """Test complete replay workflow: add, sample, update."""
        buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)

        # Add experiences
        for i in range(100):
            state = np.random.randn(4)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 20 == 0

            exp = (state, action, reward, next_state, done)
            buffer.add(exp)

        # Sample batch
        batch, indices, weights = buffer.sample(batch_size=32)

        # Simulate learning and compute TD errors
        td_errors = np.random.uniform(0.1, 2.0, size=32)

        # Update priorities
        buffer.update_priorities(indices, td_errors)

        # Sample again
        batch2, indices2, weights2 = buffer.sample(batch_size=32)

        assert len(batch2) == 32

    def test_annealing_beta_during_training(self):
        """Test beta annealing during training."""
        buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)

        # Add experiences
        for i in range(100):
            exp = (np.array([i]), i % 4, float(i % 10), np.array([i+1]), False)
            buffer.add(exp)

        # Simulate training with beta annealing
        for episode in range(10):
            # Sample
            batch, indices, weights = buffer.sample(batch_size=16)

            # Update priorities
            td_errors = np.random.uniform(0.1, 2.0, size=16)
            buffer.update_priorities(indices, td_errors)

            # Anneal beta: 0.4 → 1.0
            beta = 0.4 + (1.0 - 0.4) * (episode / 10)
            buffer.update_beta(beta)

        # Final beta should be close to 1.0
        assert buffer.beta > 0.9
