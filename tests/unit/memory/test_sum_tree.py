"""
Tests for SumTree data structure.

Test Coverage:
- Basic operations (add, update, get)
- Tree invariants (parent = sum of children)
- Edge cases (empty tree, full tree, single element)
- Performance (O(log N) complexity)
- Error handling
"""

import pytest
import numpy as np
import time
from src.memory.sum_tree import SumTree


class TestSumTreeBasics:
    """Test basic SumTree operations."""

    def test_initialization(self):
        """Test tree initialization."""
        tree = SumTree(capacity=10)

        assert tree.capacity == 10
        assert len(tree) == 0
        assert tree.total() == 0
        assert tree.get_max_priority() == 1.0

    def test_initialization_invalid_capacity(self):
        """Test initialization with invalid capacity."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            SumTree(capacity=0)

        with pytest.raises(ValueError, match="Capacity must be positive"):
            SumTree(capacity=-5)

    def test_add_single_experience(self):
        """Test adding a single experience."""
        tree = SumTree(capacity=10)
        tree.add(priority=5.0, data="exp1")

        assert len(tree) == 1
        assert tree.total() == 5.0
        assert tree.get_max_priority() == 5.0

    def test_add_multiple_experiences(self):
        """Test adding multiple experiences."""
        tree = SumTree(capacity=10)

        tree.add(priority=1.0, data="exp1")
        tree.add(priority=2.0, data="exp2")
        tree.add(priority=3.0, data="exp3")

        assert len(tree) == 3
        assert np.isclose(tree.total(), 6.0)
        assert tree.get_max_priority() == 3.0

    def test_add_with_negative_priority(self):
        """Test that negative priorities are rejected."""
        tree = SumTree(capacity=10)

        with pytest.raises(ValueError, match="Priority must be non-negative"):
            tree.add(priority=-1.0, data="exp")

    def test_circular_buffer_overflow(self):
        """Test that tree acts as circular buffer when full."""
        tree = SumTree(capacity=3)

        # Fill buffer
        tree.add(priority=1.0, data="exp1")
        tree.add(priority=2.0, data="exp2")
        tree.add(priority=3.0, data="exp3")

        assert len(tree) == 3
        assert np.isclose(tree.total(), 6.0)

        # Add one more (should overwrite first)
        tree.add(priority=4.0, data="exp4")

        assert len(tree) == 3  # Size stays at capacity
        assert np.isclose(tree.total(), 9.0)  # 2 + 3 + 4


class TestSumTreeUpdate:
    """Test SumTree update operations."""

    def test_update_priority(self):
        """Test updating priority of existing experience."""
        tree = SumTree(capacity=10)

        tree.add(priority=1.0, data="exp1")
        tree.add(priority=2.0, data="exp2")
        tree.add(priority=3.0, data="exp3")

        initial_total = tree.total()
        assert np.isclose(initial_total, 6.0)

        # Update second experience
        tree.update(data_idx=1, priority=10.0)

        # Total should be 1 + 10 + 3 = 14
        assert np.isclose(tree.total(), 14.0)

    def test_update_to_zero(self):
        """Test updating priority to zero."""
        tree = SumTree(capacity=10)

        tree.add(priority=5.0, data="exp1")
        tree.add(priority=3.0, data="exp2")

        # Update to zero
        tree.update(data_idx=0, priority=0.0)

        assert np.isclose(tree.total(), 3.0)
        assert tree.get_priority(0) == 0.0

    def test_update_invalid_index(self):
        """Test updating with invalid index."""
        tree = SumTree(capacity=10)
        tree.add(priority=1.0, data="exp1")

        with pytest.raises(IndexError):
            tree.update(data_idx=10, priority=5.0)

        with pytest.raises(IndexError):
            tree.update(data_idx=-1, priority=5.0)

    def test_update_negative_priority(self):
        """Test that negative priority updates are rejected."""
        tree = SumTree(capacity=10)
        tree.add(priority=1.0, data="exp1")

        with pytest.raises(ValueError, match="Priority must be non-negative"):
            tree.update(data_idx=0, priority=-1.0)


class TestSumTreeSampling:
    """Test SumTree sampling operations."""

    def test_get_single_experience(self):
        """Test retrieving single experience."""
        tree = SumTree(capacity=10)
        tree.add(priority=5.0, data="exp1")

        # Sample from middle of interval [0, 5)
        idx, priority, data = tree.get(value=2.5)

        assert idx == 0
        assert priority == 5.0
        assert data == "exp1"

    def test_get_from_multiple_experiences(self):
        """Test sampling from tree with multiple experiences."""
        tree = SumTree(capacity=10)

        tree.add(priority=1.0, data="exp1")  # Interval [0, 1)
        tree.add(priority=2.0, data="exp2")  # Interval [1, 3)
        tree.add(priority=3.0, data="exp3")  # Interval [3, 6)

        # Sample from first interval
        idx, priority, data = tree.get(value=0.5)
        assert data == "exp1"

        # Sample from second interval
        idx, priority, data = tree.get(value=1.5)
        assert data == "exp2"

        # Sample from third interval
        idx, priority, data = tree.get(value=4.0)
        assert data == "exp3"

    def test_get_boundary_values(self):
        """Test sampling at interval boundaries."""
        tree = SumTree(capacity=10)

        tree.add(priority=2.0, data="exp1")
        tree.add(priority=3.0, data="exp2")

        # At boundary, should get first interval
        idx, priority, data = tree.get(value=0.0)
        assert data == "exp1"

        # Just before boundary
        idx, priority, data = tree.get(value=1.99)
        assert data == "exp1"

        # At second boundary
        idx, priority, data = tree.get(value=2.0)
        assert data == "exp2"

    def test_get_out_of_range(self):
        """Test sampling with out-of-range value."""
        tree = SumTree(capacity=10)
        tree.add(priority=5.0, data="exp1")

        with pytest.raises(ValueError, match="out of range"):
            tree.get(value=-1.0)

        with pytest.raises(ValueError, match="out of range"):
            tree.get(value=10.0)

    def test_sampling_distribution(self):
        """Test that sampling follows priority distribution."""
        tree = SumTree(capacity=100)

        # Add experiences with priorities 1, 2, 3
        # Expected sampling ratio: 1:2:3
        tree.add(priority=1.0, data="low")
        tree.add(priority=2.0, data="medium")
        tree.add(priority=3.0, data="high")

        # Sample many times
        counts = {"low": 0, "medium": 0, "high": 0}
        total = tree.total()

        for _ in range(6000):
            value = np.random.uniform(0, total)
            _, _, data = tree.get(value)
            counts[data] += 1

        # Check approximate ratios (with tolerance)
        # Expected: low=1000, medium=2000, high=3000
        assert 800 < counts["low"] < 1200
        assert 1800 < counts["medium"] < 2200
        assert 2800 < counts["high"] < 3200


class TestSumTreeInvariants:
    """Test tree invariants are maintained."""

    def test_validate_empty_tree(self):
        """Test validation of empty tree."""
        tree = SumTree(capacity=10)
        assert tree.validate()

    def test_validate_after_adds(self):
        """Test tree is valid after adding experiences."""
        tree = SumTree(capacity=10)

        for i in range(10):
            tree.add(priority=float(i + 1), data=f"exp{i}")
            assert tree.validate()

    def test_validate_after_updates(self):
        """Test tree is valid after updates."""
        tree = SumTree(capacity=10)

        # Add experiences
        for i in range(5):
            tree.add(priority=1.0, data=f"exp{i}")

        # Update all
        for i in range(5):
            tree.update(data_idx=i, priority=float(i * 2))
            assert tree.validate()

    def test_parent_equals_sum_of_children(self):
        """Test that each parent = sum of children."""
        tree = SumTree(capacity=8)

        # Fill tree
        for i in range(8):
            tree.add(priority=float(i + 1), data=f"exp{i}")

        # Manually check some parent-child relationships
        # This is tested more thoroughly in validate()
        assert tree.validate()


class TestSumTreeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_element_tree(self):
        """Test tree with capacity 1."""
        tree = SumTree(capacity=1)

        tree.add(priority=5.0, data="exp1")
        assert len(tree) == 1
        assert tree.total() == 5.0

        # Overwrite
        tree.add(priority=3.0, data="exp2")
        assert len(tree) == 1
        assert tree.total() == 3.0

    def test_large_capacity(self):
        """Test tree with large capacity."""
        tree = SumTree(capacity=10000)

        # Add many experiences
        for i in range(1000):
            tree.add(priority=1.0, data=f"exp{i}")

        assert len(tree) == 1000
        assert np.isclose(tree.total(), 1000.0)

    def test_zero_priorities(self):
        """Test tree with zero priorities."""
        tree = SumTree(capacity=10)

        tree.add(priority=0.0, data="exp1")
        tree.add(priority=0.0, data="exp2")

        assert tree.total() == 0.0
        assert len(tree) == 2

    def test_very_small_priorities(self):
        """Test tree with very small priorities."""
        tree = SumTree(capacity=10)

        tree.add(priority=1e-10, data="exp1")
        tree.add(priority=1e-10, data="exp2")

        assert tree.total() > 0
        assert np.isclose(tree.total(), 2e-10)

    def test_very_large_priorities(self):
        """Test tree with very large priorities."""
        tree = SumTree(capacity=10)

        tree.add(priority=1e10, data="exp1")
        tree.add(priority=1e10, data="exp2")

        assert np.isclose(tree.total(), 2e10)


class TestSumTreePerformance:
    """Test performance characteristics."""

    def test_add_is_logarithmic(self):
        """Test that add operation is O(log N)."""
        # Small tree
        tree_small = SumTree(capacity=100)
        start = time.perf_counter()
        for i in range(100):
            tree_small.add(priority=1.0, data=i)
        time_small = time.perf_counter() - start

        # Large tree (10x bigger)
        tree_large = SumTree(capacity=1000)
        start = time.perf_counter()
        for i in range(1000):
            tree_large.add(priority=1.0, data=i)
        time_large = time.perf_counter() - start

        # Time should not grow linearly (should be < 10x)
        # O(log N) means 10x size → ~3.3x time
        ratio = time_large / time_small if time_small > 0 else 0
        assert ratio < 20  # Very generous bound for slow systems/CI

    def test_sample_is_logarithmic(self):
        """Test that get operation is O(log N)."""
        tree = SumTree(capacity=10000)

        # Fill tree
        for i in range(10000):
            tree.add(priority=1.0, data=i)

        total = tree.total()

        # Time many samples
        start = time.perf_counter()
        for _ in range(1000):
            value = np.random.uniform(0, total)
            tree.get(value)
        elapsed = time.perf_counter() - start

        # Should be very fast (< 0.1s for 1000 samples)
        assert elapsed < 0.1

    def test_update_is_logarithmic(self):
        """Test that update operation is O(log N)."""
        tree = SumTree(capacity=10000)

        # Fill tree
        for i in range(10000):
            tree.add(priority=1.0, data=i)

        # Time many updates
        start = time.perf_counter()
        for i in range(1000):
            tree.update(data_idx=i, priority=2.0)
        elapsed = time.perf_counter() - start

        # Should be very fast
        assert elapsed < 0.1


class TestSumTreeHelperMethods:
    """Test helper methods."""

    def test_get_priority(self):
        """Test getting priority by index."""
        tree = SumTree(capacity=10)

        tree.add(priority=1.5, data="exp1")
        tree.add(priority=2.5, data="exp2")

        assert tree.get_priority(0) == 1.5
        assert tree.get_priority(1) == 2.5

    def test_get_max_priority_empty(self):
        """Test max priority of empty tree."""
        tree = SumTree(capacity=10)
        assert tree.get_max_priority() == 1.0

    def test_get_max_priority_non_empty(self):
        """Test max priority with experiences."""
        tree = SumTree(capacity=10)

        tree.add(priority=1.0, data="exp1")
        tree.add(priority=5.0, data="exp2")
        tree.add(priority=3.0, data="exp3")

        assert tree.get_max_priority() == 5.0

    def test_repr(self):
        """Test string representation."""
        tree = SumTree(capacity=10)
        tree.add(priority=5.0, data="exp1")

        repr_str = repr(tree)
        assert "SumTree" in repr_str
        assert "capacity=10" in repr_str
        assert "entries=1" in repr_str

    def test_len(self):
        """Test length method."""
        tree = SumTree(capacity=10)

        assert len(tree) == 0

        tree.add(priority=1.0, data="exp1")
        assert len(tree) == 1

        tree.add(priority=2.0, data="exp2")
        assert len(tree) == 2
