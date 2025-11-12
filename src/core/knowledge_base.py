"""
Knowledge Base for Transfer Learning (Big Rock 8)

Centralized storage and retrieval of learned knowledge including:
- Experience transitions
- Complete episodes
- Learned policies
- Value functions
- Task performance data

Enables transfer learning by providing access to past experiences.

Author: MAE Development Team
Date: 2025-11-12
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque, defaultdict
import numpy as np
import threading
import time
import random
import heapq

from src.core.task_representation import TaskDescriptor, TaskEmbedding, TaskSimilarityMatrix


@dataclass
class ExperienceTransition:
    """
    Single experience transition (s, a, r, s', done).

    This is the basic unit of experience stored for replay and transfer.
    """
    task_id: str
    state: np.ndarray
    action: Any  # Can be int, np.ndarray, or other action representation
    reward: float
    next_state: np.ndarray
    done: bool
    agent_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transition to dictionary"""
        return {
            'task_id': self.task_id,
            'state': self.state.tolist(),
            'action': self.action if isinstance(self.action, (int, float, str)) else str(self.action),
            'reward': self.reward,
            'next_state': self.next_state.tolist(),
            'done': self.done,
            'agent_id': self.agent_id,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class Episode:
    """
    Complete episode trajectory.

    Stores full sequence of transitions for a single episode,
    useful for imitation learning and trajectory matching.
    """
    episode_id: str
    task_id: str
    agent_id: str
    transitions: List[ExperienceTransition]
    total_reward: float
    episode_length: int
    success: bool
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_states(self) -> List[np.ndarray]:
        """Extract all states from episode"""
        return [t.state for t in self.transitions]

    def get_actions(self) -> List[Any]:
        """Extract all actions from episode"""
        return [t.action for t in self.transitions]

    def get_rewards(self) -> List[float]:
        """Extract all rewards from episode"""
        return [t.reward for t in self.transitions]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize episode to dictionary"""
        return {
            'episode_id': self.episode_id,
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'transitions': [t.to_dict() for t in self.transitions],
            'total_reward': self.total_reward,
            'episode_length': self.episode_length,
            'success': self.success,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Stores transitions with priority scores, enabling importance sampling
    for more efficient learning.
    """

    def __init__(self, max_size: int = 100000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.

        Args:
            max_size: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        """
        self.max_size = max_size
        self.alpha = alpha

        # Storage
        self.buffer: deque = deque(maxlen=max_size)
        self.priorities: deque = deque(maxlen=max_size)

        # For efficient max priority tracking
        self.max_priority: float = 1.0

    def add(self, transition: ExperienceTransition, priority: Optional[float] = None):
        """
        Add transition with priority.

        Args:
            transition: Experience transition
            priority: Priority score (uses max if None)
        """
        if priority is None:
            priority = self.max_priority

        self.buffer.append(transition)
        self.priorities.append(priority)

        # Update max priority
        if priority > self.max_priority:
            self.max_priority = priority

    def sample(self, batch_size: int) -> Tuple[List[ExperienceTransition], List[int], List[float]]:
        """
        Sample batch of transitions according to priorities.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (transitions, indices, importance_weights)
        """
        if len(self.buffer) == 0:
            return [], [], []

        batch_size = min(batch_size, len(self.buffer))

        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)

        # Sample indices
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)

        # Get transitions
        transitions = [self.buffer[i] for i in indices]

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-1)
        weights = weights / np.max(weights)  # Normalize

        return transitions, list(indices), list(weights)

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority

                if priority > self.max_priority:
                    self.max_priority = priority

    def __len__(self) -> int:
        return len(self.buffer)


class KnowledgeBase:
    """
    Centralized storage for all learned knowledge.

    Stores:
    - Individual transitions (for experience replay)
    - Complete episodes (for imitation/trajectory matching)
    - Learned policies (for policy transfer)
    - Value functions (for value transfer)
    - Task descriptors and embeddings
    - Performance metrics
    """

    def __init__(
        self,
        max_transitions: int = 1000000,
        max_episodes: int = 10000,
        enable_prioritization: bool = True,
        embedding_dim: int = 128
    ):
        """
        Initialize knowledge base.

        Args:
            max_transitions: Maximum number of transitions to store
            max_episodes: Maximum number of episodes to store
            enable_prioritization: Enable prioritized experience replay
            embedding_dim: Dimension for task embeddings
        """
        # Experience storage
        if enable_prioritization:
            self.transitions = PrioritizedReplayBuffer(max_size=max_transitions)
        else:
            self.transitions: deque = deque(maxlen=max_transitions)

        self.enable_prioritization = enable_prioritization

        # Episode storage
        self.episodes: Dict[str, Episode] = {}
        self.episode_index: Dict[str, List[str]] = defaultdict(list)  # task_id -> episode_ids
        self.max_episodes = max_episodes

        # Policy storage: task_id -> policy parameters
        self.policies: Dict[str, Any] = {}

        # Value function storage: task_id -> value function
        self.value_functions: Dict[str, Any] = {}

        # Task knowledge
        self.task_embedding = TaskEmbedding(embedding_dim=embedding_dim)
        self.similarity_matrix = TaskSimilarityMatrix(task_embedding=self.task_embedding)

        # Performance tracking: task_id -> list of rewards
        self.task_performance: Dict[str, List[float]] = defaultdict(list)
        self.task_metadata: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.total_transitions_added: int = 0
        self.total_episodes_added: int = 0

        # Thread safety
        self.lock = threading.RLock()

    def store_transition(
        self,
        transition: ExperienceTransition,
        priority: Optional[float] = None
    ):
        """
        Store single transition with optional priority.

        Args:
            transition: Experience transition
            priority: Priority score (TD-error, novelty, etc.)
        """
        with self.lock:
            if self.enable_prioritization:
                self.transitions.add(transition, priority)
            else:
                self.transitions.append(transition)

            self.total_transitions_added += 1

    def store_episode(self, episode: Episode):
        """
        Store complete episode.

        Args:
            episode: Episode to store
        """
        with self.lock:
            # Check if we need to remove old episodes
            if len(self.episodes) >= self.max_episodes:
                self._evict_oldest_episode()

            # Store episode
            self.episodes[episode.episode_id] = episode
            self.episode_index[episode.task_id].append(episode.episode_id)

            # Update task performance
            self.task_performance[episode.task_id].append(episode.total_reward)

            self.total_episodes_added += 1

    def store_policy(self, task_id: str, policy_params: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Store learned policy for task.

        Args:
            task_id: Task identifier
            policy_params: Policy parameters (dict, np.ndarray, etc.)
            metadata: Optional metadata (training time, performance, etc.)
        """
        with self.lock:
            self.policies[task_id] = policy_params

            if metadata:
                if task_id not in self.task_metadata:
                    self.task_metadata[task_id] = {}
                self.task_metadata[task_id].update(metadata)

    def store_value_function(self, task_id: str, value_function: Any):
        """
        Store value function for task.

        Args:
            task_id: Task identifier
            value_function: Value function parameters
        """
        with self.lock:
            self.value_functions[task_id] = value_function

    def add_task(
        self,
        task: TaskDescriptor,
        performance_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Add task to knowledge base.

        Args:
            task: Task descriptor
            performance_metrics: Optional performance data
        """
        with self.lock:
            self.similarity_matrix.add_task(task, performance_metrics)

    def retrieve_transitions(
        self,
        task_id: Optional[str] = None,
        n_transitions: int = 1000,
        prioritized: bool = True
    ) -> List[ExperienceTransition]:
        """
        Retrieve transitions from knowledge base.

        Args:
            task_id: Optional task filter
            n_transitions: Number of transitions to retrieve
            prioritized: Use prioritized sampling if available

        Returns:
            List of experience transitions
        """
        with self.lock:
            if self.enable_prioritization and prioritized:
                # Prioritized sampling
                transitions, _, _ = self.transitions.sample(n_transitions)
                if task_id:
                    transitions = [t for t in transitions if t.task_id == task_id]
                return transitions
            else:
                # Random sampling
                all_transitions = list(self.transitions) if isinstance(self.transitions, deque) else self.transitions.buffer
                if task_id:
                    all_transitions = [t for t in all_transitions if t.task_id == task_id]

                n = min(n_transitions, len(all_transitions))
                return random.sample(all_transitions, n) if n > 0 else []

    def retrieve_similar_experiences(
        self,
        target_task: TaskDescriptor,
        n_experiences: int = 1000,
        similarity_threshold: float = 0.5
    ) -> List[ExperienceTransition]:
        """
        Retrieve experiences from tasks similar to target task.

        Args:
            target_task: Task to find similar experiences for
            n_experiences: Number of experiences to retrieve
            similarity_threshold: Minimum task similarity

        Returns:
            List of experience transitions from similar tasks
        """
        with self.lock:
            # Find similar tasks
            similar_tasks = self.similarity_matrix.find_similar_tasks(
                target_task,
                k=10,
                min_similarity=similarity_threshold
            )

            if not similar_tasks:
                return []

            # Collect experiences from similar tasks
            similar_task_ids = {task_id for task_id, _ in similar_tasks}

            all_transitions = list(self.transitions) if isinstance(self.transitions, deque) else list(self.transitions.buffer)

            # Filter by similar tasks
            candidate_experiences = [
                t for t in all_transitions
                if t.task_id in similar_task_ids
            ]

            # Weight by similarity and sample
            if len(candidate_experiences) <= n_experiences:
                return candidate_experiences

            # Prioritize by task similarity
            task_sim_dict = {task_id: sim for task_id, sim in similar_tasks}

            # Add similarity weight to metadata
            for exp in candidate_experiences:
                exp.metadata['transfer_similarity'] = task_sim_dict.get(exp.task_id, 0.0)

            # Sort by similarity and take top N
            candidate_experiences.sort(key=lambda x: x.metadata.get('transfer_similarity', 0.0), reverse=True)
            return candidate_experiences[:n_experiences]

    def retrieve_successful_episodes(
        self,
        task_id: str,
        k: int = 10,
        min_reward: Optional[float] = None
    ) -> List[Episode]:
        """
        Retrieve successful episodes for task.

        Args:
            task_id: Task identifier
            k: Number of episodes to retrieve
            min_reward: Optional minimum reward threshold

        Returns:
            List of successful episodes
        """
        with self.lock:
            episode_ids = self.episode_index.get(task_id, [])
            episodes = [self.episodes[eid] for eid in episode_ids if eid in self.episodes]

            # Filter by criteria
            if min_reward is not None:
                episodes = [e for e in episodes if e.total_reward >= min_reward]
            else:
                episodes = [e for e in episodes if e.success]

            # Sort by reward descending
            episodes.sort(key=lambda e: e.total_reward, reverse=True)

            return episodes[:k]

    def retrieve_policy(self, task_id: str) -> Optional[Any]:
        """
        Retrieve stored policy for task.

        Args:
            task_id: Task identifier

        Returns:
            Policy parameters or None if not found
        """
        with self.lock:
            return self.policies.get(task_id)

    def retrieve_value_function(self, task_id: str) -> Optional[Any]:
        """
        Retrieve stored value function for task.

        Args:
            task_id: Task identifier

        Returns:
            Value function or None if not found
        """
        with self.lock:
            return self.value_functions.get(task_id)

    def get_task_statistics(self, task_id: str) -> Dict[str, Any]:
        """
        Get performance statistics for task.

        Args:
            task_id: Task identifier

        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            if task_id not in self.task_performance:
                return {
                    'num_episodes': 0,
                    'avg_reward': 0.0,
                    'max_reward': 0.0,
                    'min_reward': 0.0,
                    'std_reward': 0.0
                }

            rewards = self.task_performance[task_id]

            return {
                'num_episodes': len(rewards),
                'avg_reward': float(np.mean(rewards)),
                'max_reward': float(np.max(rewards)),
                'min_reward': float(np.min(rewards)),
                'std_reward': float(np.std(rewards)),
                'recent_avg': float(np.mean(rewards[-10:])) if len(rewards) >= 10 else float(np.mean(rewards))
            }

    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Get global knowledge base statistics.

        Returns:
            Dictionary with global metrics
        """
        with self.lock:
            num_transitions = len(self.transitions)
            num_episodes = len(self.episodes)
            num_tasks = len(self.similarity_matrix)
            num_policies = len(self.policies)

            return {
                'total_transitions': num_transitions,
                'total_episodes': num_episodes,
                'total_tasks': num_tasks,
                'total_policies': num_policies,
                'total_transitions_added': self.total_transitions_added,
                'total_episodes_added': self.total_episodes_added,
                'similarity_matrix_stats': self.similarity_matrix.get_statistics()
            }

    def find_best_source_task(
        self,
        target_task: TaskDescriptor,
        min_similarity: float = 0.5,
        min_performance: float = 0.0
    ) -> Optional[Tuple[str, float, Dict[str, float]]]:
        """
        Find best source task for transfer learning.

        Selects task with highest (similarity * performance) score.

        Args:
            target_task: Target task
            min_similarity: Minimum similarity threshold
            min_performance: Minimum performance threshold

        Returns:
            Tuple of (task_id, similarity, performance_metrics) or None
        """
        with self.lock:
            # Find similar tasks with performance data
            candidates = self.similarity_matrix.find_similar_tasks_with_performance(
                target_task,
                k=20,
                min_similarity=min_similarity,
                min_performance=min_performance
            )

            if not candidates:
                return None

            # Score by similarity * performance
            best_score = -float('inf')
            best_candidate = None

            for task_id, similarity, perf in candidates:
                avg_reward = perf.get('avg_reward', 0.0)
                score = similarity * avg_reward

                if score > best_score:
                    best_score = score
                    best_candidate = (task_id, similarity, perf)

            return best_candidate

    def _evict_oldest_episode(self):
        """Remove oldest episode to make room for new one"""
        if not self.episodes:
            return

        # Find oldest episode
        oldest_id = min(self.episodes.keys(), key=lambda k: self.episodes[k].timestamp)
        oldest_episode = self.episodes[oldest_id]

        # Remove from storage
        del self.episodes[oldest_id]

        # Remove from index
        task_episodes = self.episode_index.get(oldest_episode.task_id, [])
        if oldest_id in task_episodes:
            task_episodes.remove(oldest_id)

    def clear_task_data(self, task_id: str):
        """
        Clear all data for a specific task.

        Args:
            task_id: Task identifier
        """
        with self.lock:
            # Remove transitions
            if isinstance(self.transitions, deque):
                self.transitions = deque(
                    [t for t in self.transitions if t.task_id != task_id],
                    maxlen=self.transitions.maxlen
                )
            else:
                # PrioritizedReplayBuffer
                new_buffer = deque(maxlen=self.transitions.max_size)
                new_priorities = deque(maxlen=self.transitions.max_size)
                for t, p in zip(self.transitions.buffer, self.transitions.priorities):
                    if t.task_id != task_id:
                        new_buffer.append(t)
                        new_priorities.append(p)
                self.transitions.buffer = new_buffer
                self.transitions.priorities = new_priorities

            # Remove episodes
            episode_ids = self.episode_index.get(task_id, [])
            for eid in episode_ids:
                if eid in self.episodes:
                    del self.episodes[eid]

            if task_id in self.episode_index:
                del self.episode_index[task_id]

            # Remove policy and value function
            if task_id in self.policies:
                del self.policies[task_id]

            if task_id in self.value_functions:
                del self.value_functions[task_id]

            # Remove from similarity matrix
            self.similarity_matrix.remove_task(task_id)

            # Remove performance data
            if task_id in self.task_performance:
                del self.task_performance[task_id]

            if task_id in self.task_metadata:
                del self.task_metadata[task_id]

    def clear_all(self):
        """Clear all stored knowledge"""
        with self.lock:
            if isinstance(self.transitions, deque):
                self.transitions.clear()
            else:
                self.transitions.buffer.clear()
                self.transitions.priorities.clear()

            self.episodes.clear()
            self.episode_index.clear()
            self.policies.clear()
            self.value_functions.clear()
            self.task_performance.clear()
            self.task_metadata.clear()

            # Reset similarity matrix
            self.similarity_matrix = TaskSimilarityMatrix(task_embedding=self.task_embedding)

            self.total_transitions_added = 0
            self.total_episodes_added = 0

    def __len__(self) -> int:
        """Return total number of stored items"""
        return len(self.transitions) + len(self.episodes)

    def __repr__(self) -> str:
        stats = self.get_global_statistics()
        return (
            f"KnowledgeBase(transitions={stats['total_transitions']}, "
            f"episodes={stats['total_episodes']}, "
            f"tasks={stats['total_tasks']}, "
            f"policies={stats['total_policies']})"
        )


# Convenience functions

def create_experience_transition(
    task_id: str,
    state: np.ndarray,
    action: Any,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    agent_id: str,
    **metadata
) -> ExperienceTransition:
    """
    Convenience function to create an experience transition.

    Args:
        task_id: Task identifier
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Next state
        done: Episode termination flag
        agent_id: Agent identifier
        **metadata: Additional metadata

    Returns:
        ExperienceTransition instance
    """
    return ExperienceTransition(
        task_id=task_id,
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        agent_id=agent_id,
        metadata=metadata
    )


def create_episode(
    episode_id: str,
    task_id: str,
    agent_id: str,
    transitions: List[ExperienceTransition],
    success_threshold: Optional[float] = None
) -> Episode:
    """
    Convenience function to create an episode from transitions.

    Args:
        episode_id: Unique episode identifier
        task_id: Task identifier
        agent_id: Agent identifier
        transitions: List of experience transitions
        success_threshold: Optional reward threshold for success

    Returns:
        Episode instance
    """
    total_reward = sum(t.reward for t in transitions)
    episode_length = len(transitions)

    success = True
    if success_threshold is not None:
        success = total_reward >= success_threshold
    else:
        # Use last transition's done flag
        success = transitions[-1].done if transitions else False

    return Episode(
        episode_id=episode_id,
        task_id=task_id,
        agent_id=agent_id,
        transitions=transitions,
        total_reward=total_reward,
        episode_length=episode_length,
        success=success
    )
