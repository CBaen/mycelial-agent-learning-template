"""
Memory Consolidator for Offline Learning

Implements memory consolidation strategies inspired by sleep-based memory
strengthening in biological systems.

Key Features:
- Scheduled consolidation phases ("sleep" periods)
- Adaptive consolidation based on performance
- Learning rate modulation during consolidation
- Consolidation history tracking and analysis
- Multiple consolidation strategies

Biological Inspiration:
- Sleep replay: Hippocampus replays experiences during sleep
- Memory strengthening: Repeated activation strengthens memories
- Selective consolidation: Important memories prioritized
- Integration: New knowledge integrated with old

Based on:
- Stickgold (2005): Memory consolidation during sleep
- Wilson & McNaughton (1994): Hippocampal replay during sleep
- Born & Wilhelm (2012): System consolidation of memory

Performance Targets:
- 10-20% performance improvement from consolidation
- Minimal overhead on training
- Adaptive to agent performance
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class ConsolidationStrategy(Enum):
    """
    Consolidation strategies.

    UNIFORM: Random sampling from entire memory
    PRIORITIZED: Sample high-priority experiences
    RECENT: Focus on recent experiences
    ADAPTIVE: Adjust based on performance metrics
    MIXED: Combination of strategies
    """
    UNIFORM = "uniform"
    PRIORITIZED = "prioritized"
    RECENT = "recent"
    ADAPTIVE = "adaptive"
    MIXED = "mixed"


@dataclass
class ConsolidationResult:
    """Result of a consolidation phase."""
    consolidation_id: int
    strategy: str
    steps: int
    batch_size: int
    elapsed_time: float
    mean_loss: float
    initial_loss: float
    final_loss: float
    loss_reduction: float
    learning_rate_used: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryConsolidator:
    """
    Memory consolidator for offline learning and memory strengthening.

    Manages "sleep" phases where the agent replays important experiences
    to strengthen learning without environment interaction.

    Example:
        >>> consolidator = MemoryConsolidator(
        ...     episodic_memory=memory,
        ...     consolidation_steps=100,
        ...     consolidation_frequency=1000
        ... )
        >>>
        >>> # During training
        >>> for step in range(10000):
        ...     # Regular training...
        ...
        ...     # Check if consolidation needed
        ...     if consolidator.should_consolidate(step):
        ...         result = consolidator.consolidate(agent)
        ...         print(f"Consolidation: {result.loss_reduction:.2%} improvement")
    """

    def __init__(
        self,
        episodic_memory,
        consolidation_steps: int = 100,
        consolidation_frequency: int = 1000,
        learning_rate_multiplier: float = 0.5,
        batch_size: int = 32,
        strategy: ConsolidationStrategy = ConsolidationStrategy.PRIORITIZED,
        min_memory_size: int = 1000,
    ):
        """
        Initialize memory consolidator.

        Args:
            episodic_memory: EpisodicMemory instance to consolidate
            consolidation_steps: Number of replay steps per consolidation
            consolidation_frequency: Trigger consolidation every N env steps
            learning_rate_multiplier: Scale learning rate during consolidation
            batch_size: Batch size for consolidation
            strategy: Consolidation strategy to use
            min_memory_size: Minimum memory size before consolidating
        """
        self.memory = episodic_memory
        self.consolidation_steps = consolidation_steps
        self.consolidation_frequency = consolidation_frequency
        self.lr_multiplier = learning_rate_multiplier
        self.batch_size = batch_size
        self.strategy = strategy
        self.min_memory_size = min_memory_size

        # Tracking
        self.total_consolidations = 0
        self.total_steps_consolidated = 0
        self.steps_since_consolidation = 0
        self.consolidation_history: List[ConsolidationResult] = []

        # Performance tracking for adaptive strategy
        self.recent_performance = []
        self.performance_window = 10

        # Timing
        self.last_consolidation_time = time.time()
        self.total_consolidation_time = 0.0

    def should_consolidate(
        self,
        current_step: Optional[int] = None,
        force: bool = False
    ) -> bool:
        """
        Check if consolidation should be triggered.

        Args:
            current_step: Current training step (optional)
            force: Force consolidation regardless of conditions

        Returns:
            True if consolidation should happen
        """
        if force:
            return True

        # Check memory size
        if len(self.memory) < self.min_memory_size:
            return False

        # Check frequency
        if current_step is not None:
            self.steps_since_consolidation = current_step % self.consolidation_frequency
            return self.steps_since_consolidation == 0
        else:
            # Use internal counter
            return self.steps_since_consolidation >= self.consolidation_frequency

    def step(self, num_steps: int = 1):
        """
        Update internal step counter.

        Args:
            num_steps: Number of environment steps taken
        """
        self.steps_since_consolidation += num_steps

    def consolidate(
        self,
        agent,
        num_steps: Optional[int] = None,
        strategy: Optional[ConsolidationStrategy] = None
    ) -> ConsolidationResult:
        """
        Perform memory consolidation phase.

        Args:
            agent: Agent with learn_from_batch() and get/set_learning_rate() methods
            num_steps: Number of consolidation steps (overrides default)
            strategy: Consolidation strategy (overrides default)

        Returns:
            ConsolidationResult with statistics
        """
        if len(self.memory) < self.min_memory_size:
            raise ValueError(
                f"Memory too small for consolidation: {len(self.memory)} < {self.min_memory_size}"
            )

        num_steps = num_steps or self.consolidation_steps
        strategy = strategy or self.strategy

        # Store original learning rate
        original_lr = agent.get_learning_rate()
        consolidation_lr = original_lr * self.lr_multiplier
        agent.set_learning_rate(consolidation_lr)

        # Track losses
        losses = []
        start_time = time.time()

        # Consolidation loop
        for step in range(num_steps):
            # Sample batch based on strategy
            batch, indices, weights = self._sample_batch(strategy)

            # Learn from batch
            td_errors, loss = agent.learn_from_batch(batch, weights)

            # Update priorities
            self.memory.update_priorities(indices, td_errors)

            losses.append(loss)

        elapsed = time.time() - start_time

        # Restore original learning rate
        agent.set_learning_rate(original_lr)

        # Create result
        result = ConsolidationResult(
            consolidation_id=self.total_consolidations,
            strategy=strategy.value,
            steps=num_steps,
            batch_size=self.batch_size,
            elapsed_time=elapsed,
            mean_loss=np.mean(losses),
            initial_loss=losses[0],
            final_loss=losses[-1],
            loss_reduction=(losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0.0,
            learning_rate_used=consolidation_lr,
            timestamp=time.time()
        )

        # Update tracking
        self.consolidation_history.append(result)
        self.total_consolidations += 1
        self.total_steps_consolidated += num_steps
        self.steps_since_consolidation = 0
        self.last_consolidation_time = time.time()
        self.total_consolidation_time += elapsed

        return result

    def consolidate_on_performance_drop(
        self,
        agent,
        current_performance: float,
        baseline_performance: float,
        threshold: float = 0.1
    ) -> Optional[ConsolidationResult]:
        """
        Trigger consolidation if performance drops below threshold.

        Args:
            agent: Agent to consolidate
            current_performance: Current performance metric
            baseline_performance: Baseline to compare against
            threshold: Trigger if drop exceeds this fraction (e.g., 0.1 = 10%)

        Returns:
            ConsolidationResult if consolidation performed, None otherwise
        """
        performance_drop = (baseline_performance - current_performance) / baseline_performance

        if performance_drop > threshold:
            # Adaptive consolidation: more steps for bigger drops
            extra_steps = int(self.consolidation_steps * (performance_drop / threshold))
            total_steps = self.consolidation_steps + extra_steps

            return self.consolidate(
                agent,
                num_steps=total_steps,
                strategy=ConsolidationStrategy.PRIORITIZED
            )

        return None

    def consolidate_before_evaluation(
        self,
        agent,
        num_steps: Optional[int] = None
    ) -> ConsolidationResult:
        """
        Perform consolidation before evaluation/testing.

        Useful to ensure agent is at peak performance.

        Args:
            agent: Agent to consolidate
            num_steps: Number of steps (defaults to half of normal)

        Returns:
            ConsolidationResult
        """
        num_steps = num_steps or (self.consolidation_steps // 2)

        return self.consolidate(
            agent,
            num_steps=num_steps,
            strategy=ConsolidationStrategy.PRIORITIZED
        )

    def get_consolidation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive consolidation statistics.

        Returns:
            Dictionary with statistics:
            - total_consolidations: Number of consolidations performed
            - total_steps: Total consolidation steps
            - total_time: Total time spent consolidating
            - mean_loss_reduction: Average loss reduction
            - best_consolidation: Best consolidation result
            - recent_trend: Recent consolidation effectiveness
        """
        if not self.consolidation_history:
            return {
                'total_consolidations': 0,
                'total_steps': 0,
                'total_time': 0.0,
                'mean_loss_reduction': 0.0
            }

        loss_reductions = [c.loss_reduction for c in self.consolidation_history]

        stats = {
            'total_consolidations': self.total_consolidations,
            'total_steps': self.total_steps_consolidated,
            'total_time': self.total_consolidation_time,
            'mean_loss_reduction': np.mean(loss_reductions),
            'median_loss_reduction': np.median(loss_reductions),
            'best_loss_reduction': np.max(loss_reductions),
            'steps_per_second': self.total_steps_consolidated / self.total_consolidation_time if self.total_consolidation_time > 0 else 0,
            'time_since_last': time.time() - self.last_consolidation_time,
        }

        # Best consolidation
        best_idx = np.argmax(loss_reductions)
        stats['best_consolidation'] = {
            'consolidation_id': self.consolidation_history[best_idx].consolidation_id,
            'loss_reduction': self.consolidation_history[best_idx].loss_reduction,
            'strategy': self.consolidation_history[best_idx].strategy
        }

        # Recent trend (last 5 consolidations)
        if len(self.consolidation_history) >= 5:
            recent = loss_reductions[-5:]
            stats['recent_mean_reduction'] = np.mean(recent)
            stats['recent_trend'] = self._compute_trend(recent)

        return stats

    def _sample_batch(self, strategy: ConsolidationStrategy):
        """
        Sample batch based on consolidation strategy.

        Args:
            strategy: Strategy to use

        Returns:
            Tuple of (batch, indices, weights)
        """
        if strategy == ConsolidationStrategy.UNIFORM:
            # Sample with uniform distribution (alpha=0)
            batch, indices, weights = self.memory.buffer.sample(
                batch_size=self.batch_size,
                beta=self.memory.beta
            )
            # Make uniform
            weights = np.ones_like(weights)

        elif strategy == ConsolidationStrategy.PRIORITIZED:
            # Standard prioritized sampling
            batch, indices, weights = self.memory.sample(batch_size=self.batch_size)

        elif strategy == ConsolidationStrategy.RECENT:
            # Sample from recent experiences
            # Get indices of most recent experiences
            memory_size = len(self.memory)
            recent_size = min(self.batch_size * 10, memory_size)
            recent_indices = list(range(memory_size - recent_size, memory_size))

            # Sample from recent subset
            sampled_indices = np.random.choice(recent_indices, size=self.batch_size, replace=False)
            batch = [self.memory.buffer.experiences[i] for i in sampled_indices]
            indices = sampled_indices.tolist()
            weights = np.ones(self.batch_size)

        elif strategy == ConsolidationStrategy.ADAPTIVE:
            # Adapt based on recent performance
            if self.recent_performance and np.mean(self.recent_performance) < 0.5:
                # Poor performance: use prioritized
                batch, indices, weights = self.memory.sample(batch_size=self.batch_size)
            else:
                # Good performance: use uniform
                batch, indices, weights = self.memory.buffer.sample(
                    batch_size=self.batch_size,
                    beta=0.0  # No IS correction
                )

        elif strategy == ConsolidationStrategy.MIXED:
            # Mix of uniform and prioritized
            half = self.batch_size // 2

            # Half prioritized
            batch1, indices1, weights1 = self.memory.sample(batch_size=half)

            # Half uniform
            batch2, indices2, _ = self.memory.buffer.sample(batch_size=self.batch_size - half, beta=0.0)
            weights2 = np.ones(len(batch2))

            batch = batch1 + batch2
            indices = indices1 + indices2
            weights = np.concatenate([weights1, weights2])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return batch, indices, weights

    def _compute_trend(self, values: List[float]) -> str:
        """
        Compute trend in values.

        Args:
            values: List of values

        Returns:
            "improving", "stable", or "degrading"
        """
        if len(values) < 2:
            return "stable"

        # Linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"

    def update_performance(self, performance: float):
        """
        Update performance tracking for adaptive consolidation.

        Args:
            performance: Current performance metric (0 to 1)
        """
        self.recent_performance.append(performance)

        # Keep only recent window
        if len(self.recent_performance) > self.performance_window:
            self.recent_performance.pop(0)

    def reset_consolidation_schedule(self):
        """Reset consolidation schedule (e.g., after major change)."""
        self.steps_since_consolidation = 0

    def __repr__(self) -> str:
        return (
            f"MemoryConsolidator(consolidations={self.total_consolidations}, "
            f"strategy={self.strategy.value}, frequency={self.consolidation_frequency})"
        )
