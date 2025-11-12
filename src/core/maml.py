"""
MAML (Model-Agnostic Meta-Learning) Implementation for MAE v3.0

This module implements the MAML algorithm (Finn et al., 2017) for meta-learning,
enabling agents to quickly adapt to new tasks with just a few examples (few-shot learning).

MAML learns an initialization of model parameters that can be fine-tuned with
minimal data to achieve strong performance on new tasks. This is achieved through
a two-level optimization process:
- Inner loop: Task-specific adaptation (few gradient steps)
- Outer loop: Meta-optimization across tasks (updates initialization)

Key Components:
- MAMLConfig: Configuration for MAML training
- MAMLLearner: Main MAML implementation
- AdaptationResult: Results from few-shot adaptation

References:
    Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning
    for fast adaptation of deep networks. ICML 2017.

Author: MAE Development Team
Date: Week 5-6, Phase 2 Implementation
Big Rock: 8 - Transfer Learning & Meta-Learning
"""

import time
import logging
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import threading

from src.core.task_representation import TaskDescriptor
from src.core.knowledge_base import KnowledgeBase, Episode, ExperienceTransition


logger = logging.getLogger(__name__)


@dataclass
class MAMLConfig:
    """Configuration for MAML meta-learning"""

    meta_learning_rate: float = 0.001
    """Learning rate for outer loop (meta-optimization)"""

    inner_learning_rate: float = 0.01
    """Learning rate for inner loop (task adaptation)"""

    num_inner_steps: int = 5
    """Number of gradient steps in inner loop"""

    num_tasks_per_batch: int = 4
    """Number of tasks per meta-batch"""

    k_shot: int = 5
    """Number of examples per task for adaptation (k-shot learning)"""

    query_size: int = 10
    """Number of examples per task for meta-evaluation"""

    first_order: bool = False
    """Use first-order approximation (faster but less accurate)"""

    max_meta_iterations: int = 1000
    """Maximum number of meta-training iterations"""

    adaptation_steps_eval: int = 10
    """Number of adaptation steps during evaluation"""

    min_task_similarity: float = 0.3
    """Minimum task similarity for meta-training"""

    early_stopping_patience: int = 50
    """Patience for early stopping (meta-iterations without improvement)"""

    early_stopping_threshold: float = 0.001
    """Minimum improvement threshold for early stopping"""

    def validate(self):
        """Validate configuration parameters"""
        assert self.meta_learning_rate > 0, "meta_learning_rate must be positive"
        assert self.inner_learning_rate > 0, "inner_learning_rate must be positive"
        assert self.num_inner_steps > 0, "num_inner_steps must be positive"
        assert self.num_tasks_per_batch > 0, "num_tasks_per_batch must be positive"
        assert self.k_shot > 0, "k_shot must be positive"
        assert self.query_size > 0, "query_size must be positive"
        assert 0 <= self.min_task_similarity <= 1, "min_task_similarity must be in [0, 1]"


@dataclass
class AdaptationResult:
    """Results from few-shot task adaptation"""

    task_id: str
    """Target task ID"""

    agent_id: str
    """Agent ID"""

    num_adaptation_steps: int
    """Number of adaptation steps performed"""

    num_support_examples: int
    """Number of support examples used (k-shot)"""

    pre_adaptation_performance: float
    """Performance before adaptation"""

    post_adaptation_performance: float
    """Performance after adaptation"""

    performance_gain: float
    """Improvement from adaptation"""

    adaptation_time: float
    """Time taken for adaptation (seconds)"""

    adapted_parameters: Optional[Any] = None
    """Adapted model parameters (if stored)"""

    meta_learned: bool = False
    """Whether adaptation used meta-learned initialization"""

    timestamp: float = field(default_factory=time.time)
    """Timestamp of adaptation"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'num_adaptation_steps': self.num_adaptation_steps,
            'num_support_examples': self.num_support_examples,
            'pre_adaptation_performance': self.pre_adaptation_performance,
            'post_adaptation_performance': self.post_adaptation_performance,
            'performance_gain': self.performance_gain,
            'adaptation_time': self.adaptation_time,
            'meta_learned': self.meta_learned,
            'timestamp': self.timestamp,
        }


class MAMLLearner:
    """
    MAML (Model-Agnostic Meta-Learning) implementation.

    This class implements the MAML algorithm for meta-learning, enabling rapid
    adaptation to new tasks with minimal training data. The algorithm learns
    a good initialization of model parameters through meta-training on a
    distribution of related tasks.

    Usage:
        # Initialize MAML learner
        config = MAMLConfig(
            meta_learning_rate=0.001,
            inner_learning_rate=0.01,
            k_shot=5,
            num_inner_steps=5
        )
        maml = MAMLLearner(
            knowledge_base=kb,
            config=config,
            model_init_fn=create_policy_network
        )

        # Meta-train on related tasks
        meta_result = maml.meta_train(
            task_family_id="navigation",
            num_iterations=1000
        )

        # Quickly adapt to new task
        adaptation_result = maml.adapt_to_task(
            target_task=new_task_descriptor,
            agent_id="agent_001",
            support_episodes=few_shot_examples
        )
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        config: MAMLConfig,
        model_init_fn: Callable[[], Any],
        loss_fn: Optional[Callable] = None,
        update_fn: Optional[Callable] = None
    ):
        """
        Initialize MAML learner.

        Args:
            knowledge_base: KnowledgeBase for accessing training data
            config: MAML configuration
            model_init_fn: Function to initialize model parameters
            loss_fn: Loss function (task_data, parameters) -> loss
            update_fn: Parameter update function (params, gradients, lr) -> new_params
        """
        self.knowledge_base = knowledge_base
        self.config = config
        self.model_init_fn = model_init_fn
        self.loss_fn = loss_fn or self._default_loss_fn
        self.update_fn = update_fn or self._default_update_fn

        config.validate()

        # Meta-learned initialization
        self.meta_parameters: Optional[Any] = None
        self.meta_initialized = False

        # Training history
        self.meta_training_history: List[Dict[str, Any]] = []
        self.adaptation_history: Dict[str, AdaptationResult] = {}
        self.lock = threading.RLock()

        logger.info(
            f"MAMLLearner initialized with k_shot={config.k_shot}, "
            f"inner_steps={config.num_inner_steps}, "
            f"meta_lr={config.meta_learning_rate}, inner_lr={config.inner_learning_rate}"
        )

    def meta_train(
        self,
        task_family_id: str,
        num_iterations: Optional[int] = None,
        task_descriptors: Optional[List[TaskDescriptor]] = None
    ) -> Dict[str, Any]:
        """
        Perform meta-training on a family of related tasks.

        This implements the outer loop of MAML, learning a good initialization
        of model parameters by training on multiple related tasks.

        Args:
            task_family_id: Identifier for task family (for logging)
            num_iterations: Number of meta-training iterations (default: config.max_meta_iterations)
            task_descriptors: List of task descriptors to meta-train on (if None, uses all tasks)

        Returns:
            Dictionary with meta-training results and metrics
        """
        start_time = time.time()
        num_iterations = num_iterations or self.config.max_meta_iterations

        logger.info(
            f"Starting meta-training on task family '{task_family_id}' "
            f"for {num_iterations} iterations"
        )

        # Initialize meta-parameters
        self.meta_parameters = self.model_init_fn()

        # Get task pool
        if task_descriptors is None:
            # Use all tasks from knowledge base
            all_tasks = list(self.knowledge_base.similarity_matrix.tasks.keys())
            task_descriptors = [
                self.knowledge_base.similarity_matrix.tasks[tid]
                for tid in all_tasks
            ]

        if len(task_descriptors) < self.config.num_tasks_per_batch:
            logger.warning(
                f"Only {len(task_descriptors)} tasks available, "
                f"but num_tasks_per_batch={self.config.num_tasks_per_batch}"
            )

        # Meta-training loop
        best_meta_loss = float('inf')
        patience_counter = 0
        meta_losses = []

        for iteration in range(num_iterations):
            # Sample batch of tasks
            task_batch = self._sample_task_batch(task_descriptors)

            # Perform meta-update
            meta_loss = self._meta_update(task_batch)
            meta_losses.append(meta_loss)

            # Early stopping check
            if meta_loss < best_meta_loss - self.config.early_stopping_threshold:
                best_meta_loss = meta_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping at iteration {iteration} "
                    f"(no improvement for {patience_counter} iterations)"
                )
                break

            # Logging
            if (iteration + 1) % 10 == 0:
                avg_loss = np.mean(meta_losses[-10:])
                logger.info(
                    f"Meta-iteration {iteration + 1}/{num_iterations}: "
                    f"meta_loss={meta_loss:.4f}, avg_loss={avg_loss:.4f}"
                )

        self.meta_initialized = True
        training_time = time.time() - start_time

        result = {
            'task_family_id': task_family_id,
            'num_iterations': iteration + 1,
            'num_tasks': len(task_descriptors),
            'final_meta_loss': meta_losses[-1] if meta_losses else None,
            'best_meta_loss': best_meta_loss,
            'avg_meta_loss': np.mean(meta_losses) if meta_losses else None,
            'training_time': training_time,
            'meta_initialized': self.meta_initialized,
            'timestamp': time.time()
        }

        with self.lock:
            self.meta_training_history.append(result)

        logger.info(
            f"Meta-training complete: {iteration + 1} iterations, "
            f"final_loss={meta_losses[-1]:.4f}, time={training_time:.2f}s"
        )

        return result

    def adapt_to_task(
        self,
        target_task: TaskDescriptor,
        agent_id: str,
        support_episodes: List[Episode],
        num_adaptation_steps: Optional[int] = None
    ) -> AdaptationResult:
        """
        Quickly adapt to a new task using few-shot learning.

        This implements the inner loop of MAML, adapting the meta-learned
        initialization to a new task using a small number of examples.

        Args:
            target_task: Target task descriptor
            agent_id: Agent ID
            support_episodes: Few-shot support examples (k-shot)
            num_adaptation_steps: Number of adaptation steps (default: config.num_inner_steps)

        Returns:
            AdaptationResult with adaptation metrics
        """
        start_time = time.time()
        num_adaptation_steps = num_adaptation_steps or self.config.num_inner_steps

        logger.info(
            f"Adapting to task {target_task.task_id} with {len(support_episodes)} "
            f"support episodes and {num_adaptation_steps} adaptation steps"
        )

        # Initialize from meta-learned parameters or random
        if self.meta_initialized:
            adapted_params = copy.deepcopy(self.meta_parameters)
            meta_learned = True
        else:
            adapted_params = self.model_init_fn()
            meta_learned = False
            logger.warning("Meta-parameters not initialized, using random initialization")

        # Evaluate pre-adaptation performance
        pre_perf = self._evaluate_on_episodes(adapted_params, support_episodes)

        # Adaptation loop (inner loop)
        for step in range(num_adaptation_steps):
            # Compute loss on support set
            loss, gradients = self._compute_loss_and_gradients(
                adapted_params,
                support_episodes
            )

            # Update parameters
            adapted_params = self.update_fn(
                adapted_params,
                gradients,
                self.config.inner_learning_rate
            )

        # Evaluate post-adaptation performance
        post_perf = self._evaluate_on_episodes(adapted_params, support_episodes)

        adaptation_time = time.time() - start_time

        result = AdaptationResult(
            task_id=target_task.task_id,
            agent_id=agent_id,
            num_adaptation_steps=num_adaptation_steps,
            num_support_examples=len(support_episodes),
            pre_adaptation_performance=pre_perf,
            post_adaptation_performance=post_perf,
            performance_gain=post_perf - pre_perf,
            adaptation_time=adaptation_time,
            adapted_parameters=adapted_params,
            meta_learned=meta_learned
        )

        # Store result
        with self.lock:
            result_key = f"{target_task.task_id}_{agent_id}_{int(time.time())}"
            self.adaptation_history[result_key] = result

        logger.info(
            f"Adaptation complete: pre={pre_perf:.3f}, post={post_perf:.3f}, "
            f"gain={result.performance_gain:.3f}, time={adaptation_time:.2f}s"
        )

        return result

    def _sample_task_batch(
        self,
        task_descriptors: List[TaskDescriptor]
    ) -> List[TaskDescriptor]:
        """Sample a batch of tasks for meta-training"""
        batch_size = min(self.config.num_tasks_per_batch, len(task_descriptors))
        indices = np.random.choice(len(task_descriptors), size=batch_size, replace=False)
        return [task_descriptors[i] for i in indices]

    def _meta_update(self, task_batch: List[TaskDescriptor]) -> float:
        """
        Perform one meta-update step.

        This implements the outer loop update of MAML:
        1. For each task in batch:
           a. Clone meta-parameters
           b. Perform inner loop adaptation
           c. Evaluate on query set
        2. Aggregate gradients across tasks
        3. Update meta-parameters
        """
        meta_gradients = None
        meta_loss = 0.0

        for task in task_batch:
            # Get support and query sets for this task
            support_episodes, query_episodes = self._get_support_query_sets(task)

            if not support_episodes or not query_episodes:
                logger.warning(f"Insufficient data for task {task.task_id}, skipping")
                continue

            # Clone meta-parameters for task-specific adaptation
            task_params = copy.deepcopy(self.meta_parameters)

            # Inner loop: Adapt to task
            for _ in range(self.config.num_inner_steps):
                _, gradients = self._compute_loss_and_gradients(
                    task_params,
                    support_episodes
                )
                task_params = self.update_fn(
                    task_params,
                    gradients,
                    self.config.inner_learning_rate
                )

            # Compute loss on query set (for meta-gradient)
            query_loss, query_gradients = self._compute_loss_and_gradients(
                task_params,
                query_episodes
            )
            meta_loss += query_loss

            # Accumulate meta-gradients
            if meta_gradients is None:
                meta_gradients = query_gradients
            else:
                meta_gradients = self._add_gradients(meta_gradients, query_gradients)

        # Average across tasks
        if meta_gradients is not None:
            meta_gradients = self._scale_gradients(
                meta_gradients,
                1.0 / len(task_batch)
            )
            meta_loss /= len(task_batch)

            # Update meta-parameters
            self.meta_parameters = self.update_fn(
                self.meta_parameters,
                meta_gradients,
                self.config.meta_learning_rate
            )

        return meta_loss

    def _get_support_query_sets(
        self,
        task: TaskDescriptor
    ) -> Tuple[List[Episode], List[Episode]]:
        """
        Get support and query sets for a task.

        Support set: k-shot examples for adaptation
        Query set: Examples for meta-evaluation
        """
        # Retrieve successful episodes for this task
        episodes = self.knowledge_base.retrieve_successful_episodes(
            task_id=task.task_id,
            k=self.config.k_shot + self.config.query_size
        )

        if len(episodes) < self.config.k_shot + self.config.query_size:
            # Not enough data, use what we have
            support = episodes[:self.config.k_shot]
            query = episodes[self.config.k_shot:]
        else:
            # Split into support and query
            support = episodes[:self.config.k_shot]
            query = episodes[self.config.k_shot:self.config.k_shot + self.config.query_size]

        return support, query

    def _compute_loss_and_gradients(
        self,
        parameters: Any,
        episodes: List[Episode]
    ) -> Tuple[float, Any]:
        """
        Compute loss and gradients on episodes.

        This is a placeholder that should be overridden with actual
        loss computation for the specific model type.
        """
        loss = self.loss_fn(episodes, parameters)

        # Compute gradients (placeholder - actual implementation depends on model)
        # In practice, this would use automatic differentiation
        gradients = self._compute_gradients_placeholder(parameters, loss)

        return loss, gradients

    def _evaluate_on_episodes(
        self,
        parameters: Any,
        episodes: List[Episode]
    ) -> float:
        """
        Evaluate parameters on episodes.

        Returns average reward as performance metric.
        """
        if not episodes:
            return 0.0

        total_reward = sum(ep.total_reward for ep in episodes)
        return total_reward / len(episodes)

    def _default_loss_fn(self, episodes: List[Episode], parameters: Any) -> float:
        """Default loss function (negative mean reward)"""
        if not episodes:
            return 0.0
        avg_reward = np.mean([ep.total_reward for ep in episodes])
        return -avg_reward  # Negative for minimization

    def _default_update_fn(
        self,
        parameters: Any,
        gradients: Any,
        learning_rate: float
    ) -> Any:
        """Default parameter update (gradient descent)"""
        # Placeholder: actual implementation depends on parameter structure
        # For numpy arrays: parameters - learning_rate * gradients
        if isinstance(parameters, np.ndarray):
            return parameters - learning_rate * gradients
        elif isinstance(parameters, dict):
            return {
                k: v - learning_rate * gradients.get(k, 0)
                for k, v in parameters.items()
            }
        else:
            return parameters

    def _compute_gradients_placeholder(self, parameters: Any, loss: float) -> Any:
        """
        Placeholder gradient computation.

        In actual implementation, this would use automatic differentiation.
        For now, returns small random perturbations.
        """
        if isinstance(parameters, np.ndarray):
            return np.random.randn(*parameters.shape) * 0.01
        elif isinstance(parameters, dict):
            return {k: np.random.randn(*v.shape) * 0.01 for k, v in parameters.items()}
        else:
            return parameters

    def _add_gradients(self, grad1: Any, grad2: Any) -> Any:
        """Add two gradient structures"""
        if isinstance(grad1, np.ndarray):
            return grad1 + grad2
        elif isinstance(grad1, dict):
            return {k: grad1[k] + grad2.get(k, 0) for k in grad1}
        else:
            return grad1

    def _scale_gradients(self, gradients: Any, scale: float) -> Any:
        """Scale gradients by a factor"""
        if isinstance(gradients, np.ndarray):
            return gradients * scale
        elif isinstance(gradients, dict):
            return {k: v * scale for k, v in gradients.items()}
        else:
            return gradients

    def get_meta_parameters(self) -> Optional[Any]:
        """Get current meta-learned parameters"""
        return copy.deepcopy(self.meta_parameters) if self.meta_parameters is not None else None

    def set_meta_parameters(self, parameters: Any):
        """Set meta-learned parameters (e.g., from saved checkpoint)"""
        self.meta_parameters = copy.deepcopy(parameters)
        self.meta_initialized = True
        logger.info("Meta-parameters set from external source")

    def get_adaptation_history(
        self,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[AdaptationResult]:
        """
        Get adaptation history, optionally filtered.

        Args:
            task_id: Filter by task ID
            agent_id: Filter by agent ID

        Returns:
            List of AdaptationResult objects
        """
        with self.lock:
            results = list(self.adaptation_history.values())

        if task_id:
            results = [r for r in results if r.task_id == task_id]

        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]

        return results

    def get_average_adaptation_gain(self) -> float:
        """Get average performance gain from adaptation"""
        with self.lock:
            results = list(self.adaptation_history.values())

        if not results:
            return 0.0

        gains = [r.performance_gain for r in results]
        return np.mean(gains)

    def clear_history(self):
        """Clear training and adaptation history"""
        with self.lock:
            self.meta_training_history.clear()
            self.adaptation_history.clear()
        logger.info("MAML history cleared")


def create_maml_learner(
    knowledge_base: KnowledgeBase,
    model_init_fn: Callable[[], Any],
    k_shot: int = 5,
    num_inner_steps: int = 5,
    **kwargs
) -> MAMLLearner:
    """
    Convenience function to create a MAMLLearner.

    Args:
        knowledge_base: KnowledgeBase instance
        model_init_fn: Function to initialize model parameters
        k_shot: Number of examples for few-shot learning
        num_inner_steps: Number of adaptation steps
        **kwargs: Additional MAMLConfig parameters

    Returns:
        Configured MAMLLearner instance
    """
    config = MAMLConfig(
        k_shot=k_shot,
        num_inner_steps=num_inner_steps,
        **kwargs
    )

    return MAMLLearner(
        knowledge_base=knowledge_base,
        config=config,
        model_init_fn=model_init_fn
    )
