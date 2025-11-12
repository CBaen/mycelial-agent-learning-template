"""
Transfer Learning Engine for Mycelial Agent Engine (MAE) v3.0

This module implements the core transfer learning capabilities that enable agents
to reuse knowledge from previously learned tasks to accelerate learning on new,
related tasks. The goal is to achieve 10-100x learning speed-up on similar tasks.

Key Components:
- TransferStrategy: Enum defining different transfer approaches
- TransferResult: Results and metrics from transfer operations
- TransferLearningEngine: Main orchestrator for knowledge transfer

Author: MAE Development Team
Date: Week 5-6, Phase 2 Implementation
Big Rock: 8 - Transfer Learning & Meta-Learning
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import threading

from src.core.task_representation import TaskDescriptor, TaskEmbedding, TaskSimilarityMatrix
from src.core.knowledge_base import KnowledgeBase, ExperienceTransition, Episode


logger = logging.getLogger(__name__)


class TransferStrategy(Enum):
    """Enumeration of supported transfer learning strategies"""

    POLICY_TRANSFER = "policy_transfer"
    """Transfer the complete policy from source task to target task"""

    EXPERIENCE_REPLAY = "experience_replay"
    """Transfer experiences for replay-based learning"""

    VALUE_INITIALIZATION = "value_initialization"
    """Initialize value function from source task"""

    FEATURE_EXTRACTION = "feature_extraction"
    """Extract and reuse learned feature representations"""

    CURRICULUM = "curriculum"
    """Progressive transfer through curriculum of related tasks"""

    COMBINED = "combined"
    """Combine multiple transfer strategies"""


@dataclass
class TransferResult:
    """Results and metrics from a transfer learning operation"""

    transfer_id: str
    """Unique identifier for this transfer operation"""

    source_tasks: List[str]
    """Task IDs of source tasks used for transfer"""

    target_task_id: str
    """Task ID of target task"""

    strategy: TransferStrategy
    """Transfer strategy used"""

    agent_id: str
    """Agent that received the transfer"""

    num_experiences_transferred: int = 0
    """Number of experience transitions transferred"""

    num_episodes_transferred: int = 0
    """Number of complete episodes transferred"""

    policy_transferred: bool = False
    """Whether policy was transferred"""

    value_function_transferred: bool = False
    """Whether value function was transferred"""

    transfer_time: float = 0.0
    """Time taken to perform transfer (seconds)"""

    source_task_similarities: Dict[str, float] = field(default_factory=dict)
    """Similarity scores between target and each source task"""

    baseline_performance: Optional[float] = None
    """Performance without transfer (if available)"""

    transfer_performance: Optional[float] = None
    """Performance with transfer"""

    samples_to_threshold: Optional[int] = None
    """Samples needed to reach performance threshold"""

    speedup_factor: Optional[float] = None
    """Learning speed-up: baseline_samples / transfer_samples"""

    timestamp: float = field(default_factory=time.time)
    """Timestamp of transfer operation"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def compute_speedup(self, baseline_samples: int, transfer_samples: int):
        """Compute learning speed-up factor"""
        if transfer_samples > 0:
            self.speedup_factor = baseline_samples / transfer_samples
            self.samples_to_threshold = transfer_samples
        else:
            self.speedup_factor = None
            self.samples_to_threshold = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            'transfer_id': self.transfer_id,
            'source_tasks': self.source_tasks,
            'target_task_id': self.target_task_id,
            'strategy': self.strategy.value,
            'agent_id': self.agent_id,
            'num_experiences_transferred': self.num_experiences_transferred,
            'num_episodes_transferred': self.num_episodes_transferred,
            'policy_transferred': self.policy_transferred,
            'value_function_transferred': self.value_function_transferred,
            'transfer_time': self.transfer_time,
            'source_task_similarities': self.source_task_similarities,
            'speedup_factor': self.speedup_factor,
            'samples_to_threshold': self.samples_to_threshold,
            'timestamp': self.timestamp,
        }


class TransferLearningEngine:
    """
    Main engine for orchestrating transfer learning operations.

    This class coordinates the transfer of knowledge (policies, experiences,
    value functions) from source tasks to target tasks, with the goal of
    achieving 10-100x learning speed-up on related tasks.

    Usage:
        # Initialize engine
        engine = TransferLearningEngine(
            knowledge_base=kb,
            task_embedding=embedding,
            default_strategy=TransferStrategy.COMBINED
        )

        # Initiate transfer for new task
        result = engine.initiate_transfer(
            target_task=new_task_descriptor,
            agent_id="agent_001",
            strategy=TransferStrategy.COMBINED,
            min_similarity=0.5,
            k_source_tasks=3
        )

        # Evaluate transfer effectiveness
        evaluation = engine.evaluate_transfer(
            target_task_id="task_123",
            baseline_performance=0.3,
            transfer_performance=0.8,
            baseline_samples=10000,
            transfer_samples=500
        )
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        task_embedding: TaskEmbedding,
        default_strategy: TransferStrategy = TransferStrategy.COMBINED,
        max_experiences_per_task: int = 10000,
        max_episodes_per_task: int = 100,
        min_source_task_performance: float = 0.5
    ):
        """
        Initialize Transfer Learning Engine.

        Args:
            knowledge_base: KnowledgeBase instance for accessing stored knowledge
            task_embedding: TaskEmbedding instance for computing task similarity
            default_strategy: Default transfer strategy to use
            max_experiences_per_task: Maximum experiences to transfer per source task
            max_episodes_per_task: Maximum episodes to transfer per source task
            min_source_task_performance: Minimum performance required for source task
        """
        self.knowledge_base = knowledge_base
        self.task_embedding = task_embedding
        self.default_strategy = default_strategy
        self.max_experiences_per_task = max_experiences_per_task
        self.max_episodes_per_task = max_episodes_per_task
        self.min_source_task_performance = min_source_task_performance

        # Transfer history
        self.transfer_history: Dict[str, TransferResult] = {}
        self.lock = threading.RLock()

        logger.info(
            f"TransferLearningEngine initialized with strategy={default_strategy.value}, "
            f"max_experiences={max_experiences_per_task}, max_episodes={max_episodes_per_task}"
        )

    def initiate_transfer(
        self,
        target_task: TaskDescriptor,
        agent_id: str,
        strategy: Optional[TransferStrategy] = None,
        min_similarity: float = 0.5,
        k_source_tasks: int = 3,
        same_type_only: bool = False
    ) -> TransferResult:
        """
        Initiate knowledge transfer for a target task.

        This is the main entry point for transfer learning. It identifies similar
        source tasks and transfers knowledge according to the specified strategy.

        Args:
            target_task: Target task descriptor
            agent_id: ID of agent receiving transfer
            strategy: Transfer strategy to use (default: self.default_strategy)
            min_similarity: Minimum similarity threshold for source tasks
            k_source_tasks: Number of source tasks to consider
            same_type_only: Only consider source tasks of same type

        Returns:
            TransferResult with details and metrics
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy

        transfer_id = f"transfer_{target_task.task_id}_{agent_id}_{int(time.time())}"

        logger.info(
            f"Initiating transfer for task={target_task.task_id}, agent={agent_id}, "
            f"strategy={strategy.value}"
        )

        # Find similar source tasks
        similar_tasks = self.knowledge_base.similarity_matrix.find_similar_tasks(
            query_task=target_task,
            k=k_source_tasks,
            min_similarity=min_similarity,
            same_type_only=same_type_only,
            exclude_task_ids=[target_task.task_id]
        )

        if not similar_tasks:
            logger.warning(f"No similar source tasks found for {target_task.task_id}")
            return TransferResult(
                transfer_id=transfer_id,
                source_tasks=[],
                target_task_id=target_task.task_id,
                strategy=strategy,
                agent_id=agent_id,
                transfer_time=time.time() - start_time
            )

        # Filter by performance
        source_tasks = self._filter_by_performance(similar_tasks)

        if not source_tasks:
            logger.warning(f"No high-performing source tasks found for {target_task.task_id}")
            return TransferResult(
                transfer_id=transfer_id,
                source_tasks=[],
                target_task_id=target_task.task_id,
                strategy=strategy,
                agent_id=agent_id,
                transfer_time=time.time() - start_time
            )

        # Create result object
        result = TransferResult(
            transfer_id=transfer_id,
            source_tasks=[task_id for task_id, _ in source_tasks],
            target_task_id=target_task.task_id,
            strategy=strategy,
            agent_id=agent_id,
            source_task_similarities={task_id: sim for task_id, sim in source_tasks}
        )

        # Execute transfer based on strategy
        if strategy == TransferStrategy.POLICY_TRANSFER:
            self._transfer_policy(source_tasks, target_task, agent_id, result)

        elif strategy == TransferStrategy.EXPERIENCE_REPLAY:
            self._transfer_experiences(source_tasks, target_task, agent_id, result)

        elif strategy == TransferStrategy.VALUE_INITIALIZATION:
            self._transfer_value_function(source_tasks, target_task, agent_id, result)

        elif strategy == TransferStrategy.FEATURE_EXTRACTION:
            self._transfer_features(source_tasks, target_task, agent_id, result)

        elif strategy == TransferStrategy.CURRICULUM:
            self._transfer_curriculum(source_tasks, target_task, agent_id, result)

        elif strategy == TransferStrategy.COMBINED:
            # Combine multiple strategies
            self._transfer_policy(source_tasks, target_task, agent_id, result)
            self._transfer_experiences(source_tasks, target_task, agent_id, result)
            self._transfer_value_function(source_tasks, target_task, agent_id, result)

        result.transfer_time = time.time() - start_time

        # Store result
        with self.lock:
            self.transfer_history[transfer_id] = result

        logger.info(
            f"Transfer complete: {result.num_experiences_transferred} experiences, "
            f"{result.num_episodes_transferred} episodes, "
            f"policy={result.policy_transferred}, value={result.value_function_transferred}, "
            f"time={result.transfer_time:.2f}s"
        )

        return result

    def _filter_by_performance(
        self,
        similar_tasks: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Filter source tasks by performance threshold"""
        filtered = []
        for task_id, similarity in similar_tasks:
            stats = self.knowledge_base.get_task_statistics(task_id)
            if stats and stats.get('avg_reward', 0.0) >= self.min_source_task_performance:
                filtered.append((task_id, similarity))
        return filtered

    def _transfer_policy(
        self,
        source_tasks: List[Tuple[str, float]],
        target_task: TaskDescriptor,
        agent_id: str,
        result: TransferResult
    ):
        """Transfer policy from most similar source task"""
        # Use the most similar task
        best_source_id = source_tasks[0][0]

        policy = self.knowledge_base.retrieve_policy(best_source_id)
        if policy is not None:
            # Store policy for target task with agent-specific key
            policy_key = f"{target_task.task_id}_{agent_id}"
            self.knowledge_base.store_policy(policy_key, policy)
            result.policy_transferred = True
            result.metadata['policy_source'] = best_source_id
            logger.info(f"Transferred policy from {best_source_id} to {target_task.task_id}")
        else:
            logger.warning(f"No policy found for source task {best_source_id}")

    def _transfer_experiences(
        self,
        source_tasks: List[Tuple[str, float]],
        target_task: TaskDescriptor,
        agent_id: str,
        result: TransferResult
    ):
        """Transfer experiences for replay-based learning"""
        total_experiences = 0
        total_episodes = 0

        for source_id, similarity in source_tasks:
            # Retrieve experiences
            experiences = self.knowledge_base.retrieve_similar_experiences(
                target_task=target_task,
                n_experiences=self.max_experiences_per_task,
                similarity_threshold=0.0  # Already filtered
            )

            # Filter to only this source task
            source_experiences = [e for e in experiences if e.task_id == source_id]
            total_experiences += len(source_experiences)

            # Retrieve successful episodes
            episodes = self.knowledge_base.retrieve_successful_episodes(
                task_id=source_id,
                k=self.max_episodes_per_task
            )
            total_episodes += len(episodes)

            # Note: Experiences are already in the knowledge base
            # In a full implementation, we might re-weight priorities or
            # create derived experiences for the target task

        result.num_experiences_transferred = total_experiences
        result.num_episodes_transferred = total_episodes

        logger.info(
            f"Transferred {total_experiences} experiences and {total_episodes} episodes "
            f"from {len(source_tasks)} source tasks"
        )

    def _transfer_value_function(
        self,
        source_tasks: List[Tuple[str, float]],
        target_task: TaskDescriptor,
        agent_id: str,
        result: TransferResult
    ):
        """Initialize value function from source task"""
        # Use the most similar task
        best_source_id = source_tasks[0][0]

        value_function = self.knowledge_base.retrieve_value_function(best_source_id)
        if value_function is not None:
            # Store value function for target task with agent-specific key
            vf_key = f"{target_task.task_id}_{agent_id}"
            self.knowledge_base.store_value_function(vf_key, value_function)
            result.value_function_transferred = True
            result.metadata['value_function_source'] = best_source_id
            logger.info(f"Transferred value function from {best_source_id} to {target_task.task_id}")
        else:
            logger.warning(f"No value function found for source task {best_source_id}")

    def _transfer_features(
        self,
        source_tasks: List[Tuple[str, float]],
        target_task: TaskDescriptor,
        agent_id: str,
        result: TransferResult
    ):
        """Extract and reuse learned feature representations"""
        # This is a placeholder for feature extraction transfer
        # In a full implementation, this would extract learned feature
        # representations (e.g., hidden layer weights from neural networks)
        # and use them to initialize the target task's feature extractor

        best_source_id = source_tasks[0][0]
        result.metadata['feature_extraction_source'] = best_source_id
        result.metadata['feature_extraction_note'] = "Feature extraction transfer requires model-specific implementation"

        logger.info(f"Feature extraction transfer from {best_source_id} (placeholder)")

    def _transfer_curriculum(
        self,
        source_tasks: List[Tuple[str, float]],
        target_task: TaskDescriptor,
        agent_id: str,
        result: TransferResult
    ):
        """Progressive transfer through curriculum of related tasks"""
        # Sort source tasks by similarity (already sorted)
        # In curriculum learning, we would train on progressively more similar tasks

        curriculum = [task_id for task_id, _ in source_tasks]
        result.metadata['curriculum'] = curriculum
        result.metadata['curriculum_note'] = "Curriculum transfer requires sequential training implementation"

        logger.info(f"Curriculum transfer planned: {' -> '.join(curriculum)} -> {target_task.task_id}")

    def evaluate_transfer(
        self,
        target_task_id: str,
        baseline_performance: float,
        transfer_performance: float,
        baseline_samples: int,
        transfer_samples: int
    ) -> Dict[str, Any]:
        """
        Evaluate effectiveness of transfer learning.

        Args:
            target_task_id: Target task ID
            baseline_performance: Performance without transfer
            transfer_performance: Performance with transfer
            baseline_samples: Samples needed without transfer
            transfer_samples: Samples needed with transfer

        Returns:
            Dictionary with evaluation metrics including speed-up factor
        """
        speedup = baseline_samples / transfer_samples if transfer_samples > 0 else 0.0
        performance_gain = transfer_performance - baseline_performance

        evaluation = {
            'target_task_id': target_task_id,
            'baseline_performance': baseline_performance,
            'transfer_performance': transfer_performance,
            'performance_gain': performance_gain,
            'baseline_samples': baseline_samples,
            'transfer_samples': transfer_samples,
            'speedup_factor': speedup,
            'meets_10x_target': speedup >= 10.0,
            'meets_100x_target': speedup >= 100.0,
            'timestamp': time.time()
        }

        logger.info(
            f"Transfer evaluation for {target_task_id}: "
            f"speedup={speedup:.2f}x, performance_gain={performance_gain:.3f}"
        )

        # Update transfer result if available
        with self.lock:
            for result in self.transfer_history.values():
                if result.target_task_id == target_task_id:
                    result.baseline_performance = baseline_performance
                    result.transfer_performance = transfer_performance
                    result.compute_speedup(baseline_samples, transfer_samples)

        return evaluation

    def get_transfer_history(
        self,
        target_task_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[TransferResult]:
        """
        Get transfer history, optionally filtered by task or agent.

        Args:
            target_task_id: Filter by target task ID
            agent_id: Filter by agent ID

        Returns:
            List of TransferResult objects
        """
        with self.lock:
            results = list(self.transfer_history.values())

        if target_task_id:
            results = [r for r in results if r.target_task_id == target_task_id]

        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]

        return results

    def get_average_speedup(self, strategy: Optional[TransferStrategy] = None) -> float:
        """
        Get average speed-up factor across all transfers.

        Args:
            strategy: Filter by transfer strategy

        Returns:
            Average speed-up factor
        """
        with self.lock:
            results = list(self.transfer_history.values())

        if strategy:
            results = [r for r in results if r.strategy == strategy]

        speedups = [r.speedup_factor for r in results if r.speedup_factor is not None]

        if not speedups:
            return 0.0

        return np.mean(speedups)

    def clear_history(self):
        """Clear transfer history"""
        with self.lock:
            self.transfer_history.clear()
        logger.info("Transfer history cleared")


def create_transfer_engine(
    knowledge_base: KnowledgeBase,
    strategy: TransferStrategy = TransferStrategy.COMBINED,
    **kwargs
) -> TransferLearningEngine:
    """
    Convenience function to create a TransferLearningEngine.

    Args:
        knowledge_base: KnowledgeBase instance
        strategy: Default transfer strategy
        **kwargs: Additional arguments for TransferLearningEngine

    Returns:
        Configured TransferLearningEngine instance
    """
    return TransferLearningEngine(
        knowledge_base=knowledge_base,
        task_embedding=knowledge_base.task_embedding,
        default_strategy=strategy,
        **kwargs
    )
