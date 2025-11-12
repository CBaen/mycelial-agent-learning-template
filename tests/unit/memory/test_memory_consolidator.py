"""
Tests for MemoryConsolidator.

Test Coverage:
- Basic consolidation operations
- Consolidation strategies (uniform, prioritized, recent, adaptive, mixed)
- Scheduling and triggering
- Adaptive consolidation based on performance
- Statistics and tracking
- Integration with EpisodicMemory
"""

import pytest
import numpy as np
import time
from src.memory.episodic_memory import EpisodicMemory
from src.memory.memory_consolidator import (
    MemoryConsolidator,
    ConsolidationStrategy,
    ConsolidationResult
)


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.learn_calls = 0
        self.lr_history = []

    def get_learning_rate(self):
        """Get current learning rate."""
        return self.learning_rate

    def set_learning_rate(self, lr):
        """Set learning rate."""
        self.learning_rate = lr
        self.lr_history.append(lr)

    def learn_from_batch(self, batch, weights):
        """Mock learning."""
        self.learn_calls += 1

        # Simulate learning with decreasing loss
        base_loss = 1.0 / (1 + self.learn_calls * 0.1)
        noise = np.random.uniform(-0.1, 0.1)
        loss = base_loss + noise

        # Return TD errors and loss
        td_errors = np.random.uniform(0.1, 2.0, size=len(batch))
        return td_errors, max(loss, 0.01)


class TestConsolidationResult:
    """Test ConsolidationResult dataclass."""

    def test_consolidation_result_creation(self):
        """Test creating ConsolidationResult."""
        result = ConsolidationResult(
            consolidation_id=0,
            strategy="prioritized",
            steps=100,
            batch_size=32,
            elapsed_time=1.5,
            mean_loss=0.5,
            initial_loss=0.8,
            final_loss=0.3,
            loss_reduction=0.625,
            learning_rate_used=0.0005,
            timestamp=time.time()
        )

        assert result.consolidation_id == 0
        assert result.strategy == "prioritized"
        assert result.steps == 100
        assert result.loss_reduction == 0.625


class TestMemoryConsolidatorBasics:
    """Test basic consolidator operations."""

    def test_initialization(self):
        """Test consolidator initialization."""
        memory = EpisodicMemory(capacity=1000)
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_steps=100,
            consolidation_frequency=1000
        )

        assert consolidator.consolidation_steps == 100
        assert consolidator.consolidation_frequency == 1000
        assert consolidator.total_consolidations == 0

    def test_initialization_with_strategy(self):
        """Test initialization with different strategies."""
        memory = EpisodicMemory(capacity=1000)

        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            strategy=ConsolidationStrategy.UNIFORM
        )

        assert consolidator.strategy == ConsolidationStrategy.UNIFORM

    def test_repr(self):
        """Test string representation."""
        memory = EpisodicMemory(capacity=1000)
        consolidator = MemoryConsolidator(episodic_memory=memory)

        repr_str = repr(consolidator)
        assert "MemoryConsolidator" in repr_str
        assert "consolidations=0" in repr_str


class TestConsolidationScheduling:
    """Test consolidation scheduling and triggering."""

    def test_should_consolidate_insufficient_memory(self):
        """Test that consolidation is skipped with insufficient memory."""
        memory = EpisodicMemory(capacity=10000)
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            min_memory_size=1000
        )

        # Add only 100 experiences
        for i in range(100):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        assert not consolidator.should_consolidate()

    def test_should_consolidate_with_frequency(self):
        """Test consolidation triggering based on frequency."""
        memory = EpisodicMemory(capacity=10000)
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_frequency=100,
            min_memory_size=50
        )

        # Add enough experiences
        for i in range(100):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        # Check at different steps
        assert consolidator.should_consolidate(current_step=100)
        assert not consolidator.should_consolidate(current_step=50)
        assert consolidator.should_consolidate(current_step=200)

    def test_should_consolidate_force(self):
        """Test forced consolidation."""
        memory = EpisodicMemory(capacity=1000)
        consolidator = MemoryConsolidator(episodic_memory=memory)

        # Force should override all conditions
        assert consolidator.should_consolidate(force=True)

    def test_step_counter(self):
        """Test internal step counter."""
        memory = EpisodicMemory(capacity=10000)
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_frequency=100,
            min_memory_size=10
        )

        # Add enough memory
        for i in range(50):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        # Increment steps
        for _ in range(99):
            consolidator.step()
            assert not consolidator.should_consolidate()

        # 100th step should trigger
        consolidator.step()
        assert consolidator.should_consolidate()


class TestConsolidationExecution:
    """Test consolidation execution."""

    def test_consolidate_basic(self):
        """Test basic consolidation."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent(learning_rate=0.001)
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_steps=10,
            min_memory_size=50
        )

        # Add experiences
        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=np.random.randint(0, 4),
                reward=np.random.randn(),
                next_state=np.random.randn(4),
                done=False
            )

        # Consolidate
        result = consolidator.consolidate(agent)

        assert result.steps == 10
        assert agent.learn_calls == 10
        assert consolidator.total_consolidations == 1

    def test_consolidate_learning_rate_modulation(self):
        """Test that learning rate is modulated during consolidation."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent(learning_rate=0.001)
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            learning_rate_multiplier=0.5,
            consolidation_steps=10,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        original_lr = agent.get_learning_rate()
        result = consolidator.consolidate(agent)

        # Check LR was reduced during consolidation
        assert 0.0005 in agent.lr_history  # 0.001 * 0.5

        # Check LR restored after
        assert agent.get_learning_rate() == original_lr

    def test_consolidate_insufficient_memory_raises(self):
        """Test that consolidation with insufficient memory raises error."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            min_memory_size=1000
        )

        # Add only 100 experiences
        for i in range(100):
            memory.store(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False
            )

        with pytest.raises(ValueError, match="Memory too small"):
            consolidator.consolidate(agent)

    def test_consolidation_result_fields(self):
        """Test that consolidation result has all expected fields."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_steps=10,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        result = consolidator.consolidate(agent)

        assert hasattr(result, 'consolidation_id')
        assert hasattr(result, 'strategy')
        assert hasattr(result, 'steps')
        assert hasattr(result, 'elapsed_time')
        assert hasattr(result, 'mean_loss')
        assert hasattr(result, 'initial_loss')
        assert hasattr(result, 'final_loss')
        assert hasattr(result, 'loss_reduction')


class TestConsolidationStrategies:
    """Test different consolidation strategies."""

    def test_prioritized_strategy(self):
        """Test prioritized consolidation strategy."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            strategy=ConsolidationStrategy.PRIORITIZED,
            consolidation_steps=5,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        result = consolidator.consolidate(agent)
        assert result.strategy == "prioritized"

    def test_uniform_strategy(self):
        """Test uniform consolidation strategy."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            strategy=ConsolidationStrategy.UNIFORM,
            consolidation_steps=5,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        result = consolidator.consolidate(agent)
        assert result.strategy == "uniform"

    def test_recent_strategy(self):
        """Test recent experiences consolidation strategy."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            strategy=ConsolidationStrategy.RECENT,
            consolidation_steps=5,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        result = consolidator.consolidate(agent)
        assert result.strategy == "recent"

    def test_mixed_strategy(self):
        """Test mixed consolidation strategy."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            strategy=ConsolidationStrategy.MIXED,
            consolidation_steps=5,
            batch_size=32,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        result = consolidator.consolidate(agent)
        assert result.strategy == "mixed"


class TestAdaptiveConsolidation:
    """Test adaptive consolidation features."""

    def test_consolidate_on_performance_drop(self):
        """Test consolidation triggered by performance drop."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_steps=10,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        # Trigger consolidation with performance drop
        result = consolidator.consolidate_on_performance_drop(
            agent=agent,
            current_performance=0.5,
            baseline_performance=0.8,
            threshold=0.1
        )

        assert result is not None
        assert result.steps > consolidator.consolidation_steps  # Extra steps

    def test_consolidate_on_performance_drop_no_trigger(self):
        """Test that small performance drop doesn't trigger."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        # Small drop shouldn't trigger
        result = consolidator.consolidate_on_performance_drop(
            agent=agent,
            current_performance=0.75,
            baseline_performance=0.8,
            threshold=0.1
        )

        assert result is None

    def test_consolidate_before_evaluation(self):
        """Test consolidation before evaluation."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_steps=100,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        result = consolidator.consolidate_before_evaluation(agent)

        # Should use fewer steps than normal
        assert result.steps == 50  # Half of 100

    def test_update_performance(self):
        """Test performance tracking for adaptive consolidation."""
        memory = EpisodicMemory(capacity=10000)
        consolidator = MemoryConsolidator(episodic_memory=memory)

        # Set performance window
        consolidator.performance_window = 5

        # Add performance metrics
        for perf in [0.5, 0.6, 0.7, 0.8, 0.9]:
            consolidator.update_performance(perf)

        assert len(consolidator.recent_performance) == 5
        assert consolidator.recent_performance[-1] == 0.9

        # Add more (should remove oldest)
        consolidator.update_performance(0.95)
        assert len(consolidator.recent_performance) == 5
        assert consolidator.recent_performance[0] == 0.6  # 0.5 removed


class TestConsolidationStatistics:
    """Test consolidation statistics and tracking."""

    def test_get_statistics_empty(self):
        """Test statistics with no consolidations."""
        memory = EpisodicMemory(capacity=10000)
        consolidator = MemoryConsolidator(episodic_memory=memory)

        stats = consolidator.get_consolidation_statistics()

        assert stats['total_consolidations'] == 0
        assert stats['total_steps'] == 0
        assert stats['total_time'] == 0.0

    def test_get_statistics_after_consolidations(self):
        """Test statistics after multiple consolidations."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_steps=10,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        # Perform multiple consolidations
        for _ in range(3):
            consolidator.consolidate(agent)

        stats = consolidator.get_consolidation_statistics()

        assert stats['total_consolidations'] == 3
        assert stats['total_steps'] == 30
        assert stats['total_time'] > 0
        assert 'mean_loss_reduction' in stats
        assert 'best_consolidation' in stats

    def test_consolidation_history_tracking(self):
        """Test that consolidation history is tracked."""
        memory = EpisodicMemory(capacity=10000)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_steps=10,
            min_memory_size=50
        )

        for i in range(100):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

        # First consolidation
        result1 = consolidator.consolidate(agent)
        assert len(consolidator.consolidation_history) == 1

        # Second consolidation
        result2 = consolidator.consolidate(agent)
        assert len(consolidator.consolidation_history) == 2

        # Check IDs are sequential
        assert consolidator.consolidation_history[0].consolidation_id == 0
        assert consolidator.consolidation_history[1].consolidation_id == 1


class TestIntegration:
    """Test integrated workflows."""

    def test_full_training_workflow(self):
        """Test complete training workflow with consolidation."""
        memory = EpisodicMemory(capacity=10000, alpha=0.6, beta=0.4)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_steps=20,
            consolidation_frequency=100,
            min_memory_size=50
        )

        # Training loop
        for step in range(500):
            # Collect experience
            memory.store(
                state=np.random.randn(4),
                action=np.random.randint(0, 4),
                reward=np.random.randn(),
                next_state=np.random.randn(4),
                done=step % 100 == 0
            )

            # Regular learning
            if len(memory) > 32:
                batch, indices, weights = memory.sample(batch_size=32)
                td_errors, loss = agent.learn_from_batch(batch, weights)
                memory.update_priorities(indices, td_errors)

            # Periodic consolidation
            consolidator.step()
            if consolidator.should_consolidate():
                result = consolidator.consolidate(agent)

        # Check consolidations occurred
        assert consolidator.total_consolidations > 0
        stats = consolidator.get_consolidation_statistics()
        assert stats['total_consolidations'] >= 4  # 500 / 100 = 5, minus initial steps

    def test_consolidation_with_beta_annealing(self):
        """Test consolidation with beta annealing."""
        memory = EpisodicMemory(capacity=10000, alpha=0.6, beta=0.4)
        agent = MockAgent()
        consolidator = MemoryConsolidator(
            episodic_memory=memory,
            consolidation_steps=10,
            consolidation_frequency=50,
            min_memory_size=50
        )

        # Training with beta annealing
        for step in range(200):
            memory.store(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )

            # Anneal beta
            beta = 0.4 + (1.0 - 0.4) * (step / 200)
            memory.update_beta(beta)

            # Consolidate periodically
            consolidator.step()
            if consolidator.should_consolidate():
                consolidator.consolidate(agent)

        assert consolidator.total_consolidations > 0
        assert memory.beta > 0.9  # Should be near 1.0
