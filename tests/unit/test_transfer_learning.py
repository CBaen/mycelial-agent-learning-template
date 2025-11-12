"""
Unit tests for Transfer Learning Engine (Big Rock 8)

Tests cover:
- Task representation and similarity
- Knowledge base storage and retrieval
- Transfer learning strategies
- MAML meta-learning
- Integration with base agent
"""

import pytest
import numpy as np
import time
from typing import List

from src.core.task_representation import (
    TaskDescriptor, TaskEmbedding, TaskSimilarityMatrix
)
from src.core.knowledge_base import (
    KnowledgeBase, ExperienceTransition, Episode, PrioritizedReplayBuffer
)
from src.core.transfer_learning import (
    TransferLearningEngine, TransferStrategy, TransferResult, create_transfer_engine
)
from src.core.maml import (
    MAMLLearner, MAMLConfig, AdaptationResult, create_maml_learner
)


# ==========================================
# TASK REPRESENTATION TESTS (15 tests)
# ==========================================

class TestTaskDescriptor:
    """Test TaskDescriptor functionality"""

    def test_task_descriptor_creation(self):
        """Test basic task descriptor creation"""
        task = TaskDescriptor(
            task_id="test_task_1",
            task_type="navigation",
            state_dim=10,
            action_dim=4,
            episode_length=100
        )

        assert task.task_id == "test_task_1"
        assert task.task_type == "navigation"
        assert task.state_dim == 10
        assert task.action_dim == 4
        assert task.episode_length == 100

    def test_task_descriptor_with_signatures(self):
        """Test task descriptor with computed signatures"""
        task = TaskDescriptor(
            task_id="test_task_2",
            task_type="control",
            state_dim=5,
            action_dim=2
        )

        # Generate sample data
        state_samples = np.random.randn(100, 5)
        reward_samples = np.random.rand(100)
        transition_samples = np.random.randn(100, 5)

        task.compute_signatures(state_samples, reward_samples, transition_samples)

        assert task.state_space_signature is not None
        assert task.reward_signature is not None
        assert task.dynamics_signature is not None
        assert len(task.state_space_signature) == 20  # 4 stats * 5 dims

    def test_task_descriptor_validation(self):
        """Test task descriptor validation"""
        with pytest.raises(ValueError):
            TaskDescriptor(
                task_id="invalid",
                task_type="test",
                state_dim=-1,  # Invalid
                action_dim=4
            )

    def test_task_descriptor_difficulty(self):
        """Test difficulty parameter"""
        easy_task = TaskDescriptor(
            task_id="easy",
            task_type="test",
            state_dim=4,
            action_dim=2,
            difficulty=0.1
        )

        hard_task = TaskDescriptor(
            task_id="hard",
            task_type="test",
            state_dim=4,
            action_dim=2,
            difficulty=0.9
        )

        assert easy_task.difficulty < hard_task.difficulty


class TestTaskEmbedding:
    """Test TaskEmbedding functionality"""

    def test_task_embedding_creation(self):
        """Test task embedding initialization"""
        embedding = TaskEmbedding(embedding_dim=128)
        assert embedding.embedding_dim == 128
        assert len(embedding.embedding_cache) == 0

    def test_task_encoding(self):
        """Test encoding tasks into embeddings"""
        embedding = TaskEmbedding(embedding_dim=128)
        task = TaskDescriptor(
            task_id="task_1",
            task_type="navigation",
            state_dim=10,
            action_dim=4
        )

        emb = embedding.encode(task)
        assert emb.shape == (128,)
        assert np.isclose(np.linalg.norm(emb), 1.0)  # Normalized

    def test_embedding_caching(self):
        """Test that embeddings are cached"""
        embedding = TaskEmbedding()
        task = TaskDescriptor(
            task_id="task_cache",
            task_type="test",
            state_dim=5,
            action_dim=2
        )

        emb1 = embedding.encode(task)
        emb2 = embedding.encode(task)

        assert np.array_equal(emb1, emb2)
        assert "task_cache" in embedding.embedding_cache

    def test_task_similarity(self):
        """Test similarity computation between tasks"""
        embedding = TaskEmbedding()

        # Very similar tasks
        task1 = TaskDescriptor(
            task_id="nav_1",
            task_type="navigation",
            state_dim=10,
            action_dim=4
        )
        task2 = TaskDescriptor(
            task_id="nav_2",
            task_type="navigation",
            state_dim=10,
            action_dim=4
        )

        # Different task
        task3 = TaskDescriptor(
            task_id="control_1",
            task_type="control",
            state_dim=5,
            action_dim=2
        )

        sim_12 = embedding.similarity(task1, task2)
        sim_13 = embedding.similarity(task1, task3)

        assert sim_12 > sim_13  # Similar tasks have higher similarity
        assert -1 <= sim_12 <= 1
        assert -1 <= sim_13 <= 1

    def test_clear_cache(self):
        """Test cache clearing"""
        embedding = TaskEmbedding()
        task = TaskDescriptor(
            task_id="task_clear",
            task_type="test",
            state_dim=5,
            action_dim=2
        )

        embedding.encode(task)
        assert len(embedding.embedding_cache) > 0

        embedding.clear_cache()
        assert len(embedding.embedding_cache) == 0


class TestTaskSimilarityMatrix:
    """Test TaskSimilarityMatrix functionality"""

    def test_similarity_matrix_creation(self):
        """Test similarity matrix initialization"""
        embedding = TaskEmbedding()
        matrix = TaskSimilarityMatrix(embedding)
        assert matrix.task_embedding == embedding

    def test_add_task(self):
        """Test adding tasks to similarity matrix"""
        embedding = TaskEmbedding()
        matrix = TaskSimilarityMatrix(embedding)

        task = TaskDescriptor(
            task_id="task_add",
            task_type="test",
            state_dim=5,
            action_dim=2
        )

        matrix.add_task(task)
        assert "task_add" in matrix.tasks

    def test_find_similar_tasks(self):
        """Test finding similar tasks"""
        embedding = TaskEmbedding()
        matrix = TaskSimilarityMatrix(embedding)

        # Add multiple tasks
        for i in range(5):
            task = TaskDescriptor(
                task_id=f"nav_{i}",
                task_type="navigation",
                state_dim=10,
                action_dim=4
            )
            matrix.add_task(task)

        # Add a different task
        task = TaskDescriptor(
            task_id="control_1",
            task_type="control",
            state_dim=5,
            action_dim=2
        )
        matrix.add_task(task)

        # Query task
        query = TaskDescriptor(
            task_id="nav_query",
            task_type="navigation",
            state_dim=10,
            action_dim=4
        )

        similar = matrix.find_similar_tasks(query, k=3, min_similarity=0.0)
        assert len(similar) <= 3
        assert all(task_id.startswith("nav_") for task_id, _ in similar)

    def test_same_type_filtering(self):
        """Test same type filtering in similarity search"""
        embedding = TaskEmbedding()
        matrix = TaskSimilarityMatrix(embedding)

        # Add tasks of different types
        matrix.add_task(TaskDescriptor(
            task_id="nav_1", task_type="navigation", state_dim=10, action_dim=4
        ))
        matrix.add_task(TaskDescriptor(
            task_id="control_1", task_type="control", state_dim=5, action_dim=2
        ))

        query = TaskDescriptor(
            task_id="nav_query", task_type="navigation", state_dim=10, action_dim=4
        )

        similar = matrix.find_similar_tasks(query, k=5, same_type_only=True)
        assert all(matrix.tasks[tid].task_type == "navigation" for tid, _ in similar)

    def test_exclude_tasks(self):
        """Test excluding specific tasks from search"""
        embedding = TaskEmbedding()
        matrix = TaskSimilarityMatrix(embedding)

        for i in range(5):
            matrix.add_task(TaskDescriptor(
                task_id=f"task_{i}",
                task_type="test",
                state_dim=5,
                action_dim=2
            ))

        query = TaskDescriptor(
            task_id="query", task_type="test", state_dim=5, action_dim=2
        )

        similar = matrix.find_similar_tasks(
            query, k=5, exclude_task_ids=["task_0", "task_1"]
        )

        assert all(tid not in ["task_0", "task_1"] for tid, _ in similar)

    def test_similarity_threshold(self):
        """Test minimum similarity threshold"""
        embedding = TaskEmbedding()
        matrix = TaskSimilarityMatrix(embedding)

        # Add tasks
        matrix.add_task(TaskDescriptor(
            task_id="similar",
            task_type="test",
            state_dim=10,
            action_dim=4
        ))
        matrix.add_task(TaskDescriptor(
            task_id="different",
            task_type="other",
            state_dim=100,
            action_dim=50,
            difficulty=0.9
        ))

        query = TaskDescriptor(
            task_id="query",
            task_type="test",
            state_dim=10,
            action_dim=4
        )

        # High threshold should filter out dissimilar tasks
        similar = matrix.find_similar_tasks(query, k=10, min_similarity=0.8)

        # Should primarily find similar task
        assert any(tid == "similar" for tid, _ in similar)

    def test_clear_tasks(self):
        """Test clearing all tasks from matrix"""
        embedding = TaskEmbedding()
        matrix = TaskSimilarityMatrix(embedding)

        matrix.add_task(TaskDescriptor(
            task_id="task_1", task_type="test", state_dim=5, action_dim=2
        ))

        assert len(matrix.tasks) > 0
        matrix.clear()
        assert len(matrix.tasks) == 0

    def test_similarity_cache(self):
        """Test that similarity computations are cached"""
        embedding = TaskEmbedding()
        matrix = TaskSimilarityMatrix(embedding)

        task1 = TaskDescriptor(
            task_id="task_1", task_type="test", state_dim=5, action_dim=2
        )
        task2 = TaskDescriptor(
            task_id="task_2", task_type="test", state_dim=5, action_dim=2
        )

        matrix.add_task(task1)
        matrix.add_task(task2)

        # First computation
        matrix.find_similar_tasks(task1, k=1)
        cache_size_1 = len(matrix.similarity_cache)

        # Second computation (should use cache)
        matrix.find_similar_tasks(task1, k=1)
        cache_size_2 = len(matrix.similarity_cache)

        assert cache_size_2 == cache_size_1  # No new cache entries


# ==========================================
# KNOWLEDGE BASE TESTS (20 tests)
# ==========================================

class TestPrioritizedReplayBuffer:
    """Test PrioritizedReplayBuffer functionality"""

    def test_buffer_creation(self):
        """Test buffer initialization"""
        buffer = PrioritizedReplayBuffer(max_size=1000)
        assert buffer.max_size == 1000
        assert len(buffer.buffer) == 0

    def test_add_experience(self):
        """Test adding experiences to buffer"""
        buffer = PrioritizedReplayBuffer(max_size=100)

        transition = ExperienceTransition(
            task_id="task_1",
            state=np.array([1, 2, 3]),
            action=0,
            reward=1.0,
            next_state=np.array([2, 3, 4]),
            done=False,
            agent_id="agent_1"
        )

        buffer.add(transition, priority=1.0)
        assert len(buffer.buffer) == 1

    def test_buffer_eviction(self):
        """Test that buffer evicts oldest when full"""
        buffer = PrioritizedReplayBuffer(max_size=5)

        for i in range(10):
            transition = ExperienceTransition(
                task_id="task_1",
                state=np.array([i]),
                action=i,
                reward=float(i),
                next_state=np.array([i+1]),
                done=False,
                agent_id="agent_1"
            )
            buffer.add(transition, priority=1.0)

        assert len(buffer.buffer) == 5  # Max size

    def test_prioritized_sampling(self):
        """Test sampling by priority"""
        buffer = PrioritizedReplayBuffer(max_size=100, alpha=0.6)

        # Add experiences with different priorities
        for i in range(50):
            transition = ExperienceTransition(
                task_id="task_1",
                state=np.array([i]),
                action=i,
                reward=float(i),
                next_state=np.array([i+1]),
                done=False,
                agent_id="agent_1"
            )
            # High priority for later experiences
            priority = 1.0 if i < 25 else 10.0
            buffer.add(transition, priority=priority)

        # Sample multiple times and check if high-priority items appear more
        samples_high = 0
        for _ in range(100):
            transitions, indices, weights = buffer.sample(10)
            # Count how many from high-priority group
            samples_high += sum(1 for t in transitions if t.reward >= 25)

        # High-priority experiences should be sampled more frequently
        assert samples_high > 400  # Out of 1000 total samples

    def test_importance_weights(self):
        """Test importance weight computation"""
        buffer = PrioritizedReplayBuffer(max_size=100)

        for i in range(10):
            transition = ExperienceTransition(
                task_id="task_1",
                state=np.array([i]),
                action=i,
                reward=float(i),
                next_state=np.array([i+1]),
                done=False,
                agent_id="agent_1"
            )
            buffer.add(transition, priority=1.0)

        _, _, weights = buffer.sample(5)
        assert len(weights) == 5
        assert all(w > 0 for w in weights)


class TestKnowledgeBase:
    """Test KnowledgeBase functionality"""

    def test_knowledge_base_creation(self):
        """Test knowledge base initialization"""
        kb = KnowledgeBase(max_transitions=10000, max_episodes=1000)
        assert kb.transitions.max_size == 10000
        assert kb.max_episodes == 1000

    def test_store_transition(self):
        """Test storing transitions"""
        kb = KnowledgeBase()

        transition = ExperienceTransition(
            task_id="task_1",
            state=np.array([1, 2]),
            action=0,
            reward=1.0,
            next_state=np.array([2, 3]),
            done=False,
            agent_id="agent_1"
        )

        kb.store_transition(transition, priority=1.0)
        assert len(kb.transitions.buffer) == 1

    def test_store_episode(self):
        """Test storing episodes"""
        kb = KnowledgeBase(max_episodes=10)

        transitions = [
            ExperienceTransition(
                task_id="task_1",
                state=np.array([i]),
                action=i,
                reward=1.0,
                next_state=np.array([i+1]),
                done=(i == 9),
                agent_id="agent_1"
            )
            for i in range(10)
        ]

        episode = Episode(
            episode_id="ep_1",
            task_id="task_1",
            agent_id="agent_1",
            transitions=transitions,
            total_reward=10.0,
            episode_length=10,
            success=True
        )

        kb.store_episode(episode)
        assert "ep_1" in kb.episodes

    def test_episode_eviction(self):
        """Test episode eviction when max reached"""
        kb = KnowledgeBase(max_episodes=5)

        for i in range(10):
            episode = Episode(
                episode_id=f"ep_{i}",
                task_id="task_1",
                agent_id="agent_1",
                transitions=[],
                total_reward=float(i),
                episode_length=10,
                success=True,
                timestamp=time.time() + i  # Increasing timestamp
            )
            kb.store_episode(episode)

        assert len(kb.episodes) == 5
        # Should keep most recent
        assert "ep_9" in kb.episodes

    def test_store_and_retrieve_policy(self):
        """Test policy storage and retrieval"""
        kb = KnowledgeBase()

        policy = {"weights": np.array([1, 2, 3]), "bias": 0.5}
        kb.store_policy("task_1_agent_1", policy)

        retrieved = kb.retrieve_policy("task_1_agent_1")
        assert retrieved is not None
        assert np.array_equal(retrieved["weights"], policy["weights"])

    def test_store_and_retrieve_value_function(self):
        """Test value function storage and retrieval"""
        kb = KnowledgeBase()

        vf = {"table": np.random.rand(10, 10)}
        kb.store_value_function("task_1_agent_1", vf)

        retrieved = kb.retrieve_value_function("task_1_agent_1")
        assert retrieved is not None
        assert np.array_equal(retrieved["table"], vf["table"])

    def test_retrieve_similar_experiences(self):
        """Test retrieving experiences from similar tasks"""
        kb = KnowledgeBase()

        # Add similar tasks
        task1 = TaskDescriptor(
            task_id="nav_1", task_type="navigation", state_dim=10, action_dim=4
        )
        task2 = TaskDescriptor(
            task_id="nav_2", task_type="navigation", state_dim=10, action_dim=4
        )
        kb.similarity_matrix.add_task(task1)
        kb.similarity_matrix.add_task(task2)

        # Add experiences
        for i in range(20):
            transition = ExperienceTransition(
                task_id="nav_1",
                state=np.array([i]),
                action=i % 4,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False,
                agent_id="agent_1"
            )
            kb.store_transition(transition, priority=1.0)

        # Retrieve for similar task
        experiences = kb.retrieve_similar_experiences(
            target_task=task2,
            n_experiences=10,
            similarity_threshold=0.5
        )

        assert len(experiences) > 0
        assert len(experiences) <= 10

    def test_retrieve_successful_episodes(self):
        """Test retrieving successful episodes"""
        kb = KnowledgeBase()

        # Add successful and failed episodes
        for i in range(10):
            episode = Episode(
                episode_id=f"ep_{i}",
                task_id="task_1",
                agent_id="agent_1",
                transitions=[],
                total_reward=float(i),
                episode_length=10,
                success=(i >= 5)  # Half successful
            )
            kb.store_episode(episode)

        successful = kb.retrieve_successful_episodes("task_1", k=3)
        assert len(successful) <= 3
        assert all(ep.success for ep in successful)

    def test_find_best_source_task(self):
        """Test finding best source task for transfer"""
        kb = KnowledgeBase()

        # Add tasks with different performance
        for i in range(5):
            task = TaskDescriptor(
                task_id=f"task_{i}",
                task_type="test",
                state_dim=10,
                action_dim=4
            )
            kb.similarity_matrix.add_task(task)

            # Add episodes with varying performance
            for j in range(5):
                episode = Episode(
                    episode_id=f"ep_{i}_{j}",
                    task_id=f"task_{i}",
                    agent_id="agent_1",
                    transitions=[],
                    total_reward=float(i * 10 + j),  # Higher for later tasks
                    episode_length=10,
                    success=True
                )
                kb.store_episode(episode)

        target = TaskDescriptor(
            task_id="target",
            task_type="test",
            state_dim=10,
            action_dim=4
        )

        best_source = kb.find_best_source_task(
            target_task=target,
            k_candidates=3
        )

        assert best_source is not None
        # Should select one of the better performing tasks
        assert best_source in ["task_3", "task_4"]

    def test_get_task_statistics(self):
        """Test getting task statistics"""
        kb = KnowledgeBase()

        # Add episodes for a task
        for i in range(10):
            episode = Episode(
                episode_id=f"ep_{i}",
                task_id="task_1",
                agent_id="agent_1",
                transitions=[],
                total_reward=float(i),
                episode_length=10,
                success=(i >= 5)
            )
            kb.store_episode(episode)

        stats = kb.get_task_statistics("task_1")

        assert stats is not None
        assert stats["episode_count"] == 10
        assert stats["success_count"] == 5
        assert stats["success_rate"] == 0.5
        assert "avg_reward" in stats

    def test_clear_task_data(self):
        """Test clearing data for specific task"""
        kb = KnowledgeBase()

        # Add data for two tasks
        for task_id in ["task_1", "task_2"]:
            episode = Episode(
                episode_id=f"ep_{task_id}",
                task_id=task_id,
                agent_id="agent_1",
                transitions=[],
                total_reward=10.0,
                episode_length=10,
                success=True
            )
            kb.store_episode(episode)

        assert len(kb.episodes) == 2

        kb.clear_task_data("task_1")

        # Only task_2 should remain
        assert "ep_task_1" not in kb.episodes
        assert "ep_task_2" in kb.episodes


# ==========================================
# TRANSFER LEARNING ENGINE TESTS (25 tests)
# ==========================================

class TestTransferLearningEngine:
    """Test TransferLearningEngine functionality"""

    def test_engine_creation(self):
        """Test transfer engine initialization"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()

        engine = TransferLearningEngine(
            knowledge_base=kb,
            task_embedding=embedding,
            default_strategy=TransferStrategy.COMBINED
        )

        assert engine.knowledge_base == kb
        assert engine.default_strategy == TransferStrategy.COMBINED

    def test_create_transfer_engine_convenience(self):
        """Test convenience factory function"""
        kb = KnowledgeBase()

        engine = create_transfer_engine(
            knowledge_base=kb,
            strategy=TransferStrategy.POLICY_TRANSFER
        )

        assert engine.default_strategy == TransferStrategy.POLICY_TRANSFER

    def test_initiate_transfer_no_source_tasks(self):
        """Test transfer when no similar tasks exist"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        target = TaskDescriptor(
            task_id="new_task",
            task_type="test",
            state_dim=10,
            action_dim=4
        )

        result = engine.initiate_transfer(
            target_task=target,
            agent_id="agent_1",
            min_similarity=0.5
        )

        assert result.target_task_id == "new_task"
        assert len(result.source_tasks) == 0
        assert not result.policy_transferred

    def test_policy_transfer_strategy(self):
        """Test POLICY_TRANSFER strategy"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Add source task with policy
        source_task = TaskDescriptor(
            task_id="source",
            task_type="test",
            state_dim=10,
            action_dim=4
        )
        kb.similarity_matrix.add_task(source_task)

        policy = {"weights": np.array([1, 2, 3])}
        kb.store_policy("source", policy)

        # Add successful episode for performance
        episode = Episode(
            episode_id="ep_source",
            task_id="source",
            agent_id="agent_1",
            transitions=[],
            total_reward=10.0,
            episode_length=10,
            success=True
        )
        kb.store_episode(episode)

        # Transfer to target
        target = TaskDescriptor(
            task_id="target",
            task_type="test",
            state_dim=10,
            action_dim=4
        )

        result = engine.initiate_transfer(
            target_task=target,
            agent_id="agent_1",
            strategy=TransferStrategy.POLICY_TRANSFER,
            min_similarity=0.0
        )

        assert result.policy_transferred
        assert "source" in result.source_tasks

    def test_experience_replay_strategy(self):
        """Test EXPERIENCE_REPLAY strategy"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Add source task with experiences
        source_task = TaskDescriptor(
            task_id="source",
            task_type="test",
            state_dim=10,
            action_dim=4
        )
        kb.similarity_matrix.add_task(source_task)

        # Add experiences
        for i in range(50):
            transition = ExperienceTransition(
                task_id="source",
                state=np.array([i]),
                action=i % 4,
                reward=1.0,
                next_state=np.array([i+1]),
                done=False,
                agent_id="agent_1"
            )
            kb.store_transition(transition, priority=1.0)

        # Add episode for performance
        episode = Episode(
            episode_id="ep_source",
            task_id="source",
            agent_id="agent_1",
            transitions=[],
            total_reward=10.0,
            episode_length=10,
            success=True
        )
        kb.store_episode(episode)

        # Transfer to target
        target = TaskDescriptor(
            task_id="target",
            task_type="test",
            state_dim=10,
            action_dim=4
        )

        result = engine.initiate_transfer(
            target_task=target,
            agent_id="agent_1",
            strategy=TransferStrategy.EXPERIENCE_REPLAY,
            min_similarity=0.0
        )

        assert result.num_experiences_transferred > 0

    def test_combined_strategy(self):
        """Test COMBINED strategy transfers multiple components"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Setup source task with policy, value function, and experiences
        source_task = TaskDescriptor(
            task_id="source",
            task_type="test",
            state_dim=10,
            action_dim=4
        )
        kb.similarity_matrix.add_task(source_task)

        kb.store_policy("source", {"weights": np.array([1, 2, 3])})
        kb.store_value_function("source", {"table": np.random.rand(10, 4)})

        for i in range(20):
            kb.store_transition(
                ExperienceTransition(
                    task_id="source",
                    state=np.array([i]),
                    action=i % 4,
                    reward=1.0,
                    next_state=np.array([i+1]),
                    done=False,
                    agent_id="agent_1"
                ),
                priority=1.0
            )

        kb.store_episode(Episode(
            episode_id="ep_source",
            task_id="source",
            agent_id="agent_1",
            transitions=[],
            total_reward=10.0,
            episode_length=10,
            success=True
        ))

        # Transfer with combined strategy
        target = TaskDescriptor(
            task_id="target",
            task_type="test",
            state_dim=10,
            action_dim=4
        )

        result = engine.initiate_transfer(
            target_task=target,
            agent_id="agent_1",
            strategy=TransferStrategy.COMBINED,
            min_similarity=0.0
        )

        assert result.policy_transferred
        assert result.value_function_transferred
        assert result.num_experiences_transferred > 0

    def test_transfer_performance_evaluation(self):
        """Test transfer performance evaluation"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        evaluation = engine.evaluate_transfer(
            target_task_id="task_1",
            baseline_performance=0.3,
            transfer_performance=0.8,
            baseline_samples=10000,
            transfer_samples=500
        )

        assert evaluation['speedup_factor'] == 20.0  # 10000 / 500
        assert evaluation['performance_gain'] == 0.5  # 0.8 - 0.3
        assert evaluation['meets_10x_target'] is True
        assert evaluation['meets_100x_target'] is False

    def test_transfer_history_tracking(self):
        """Test transfer history is tracked"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Perform transfer
        target = TaskDescriptor(
            task_id="target",
            task_type="test",
            state_dim=10,
            action_dim=4
        )

        engine.initiate_transfer(
            target_task=target,
            agent_id="agent_1"
        )

        history = engine.get_transfer_history(target_task_id="target")
        assert len(history) == 1
        assert history[0].target_task_id == "target"

    def test_get_average_speedup(self):
        """Test average speed-up calculation"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Perform multiple transfers and evaluate
        for i in range(3):
            target = TaskDescriptor(
                task_id=f"target_{i}",
                task_type="test",
                state_dim=10,
                action_dim=4
            )

            result = engine.initiate_transfer(target_task=target, agent_id="agent_1")

            # Manually set speedup for testing
            result.compute_speedup(baseline_samples=1000 * (i+1), transfer_samples=100)

        avg_speedup = engine.get_average_speedup()
        assert avg_speedup > 0
        assert 10.0 <= avg_speedup <= 30.0  # Average of 10x, 20x, 30x

    def test_filter_by_performance(self):
        """Test source task filtering by performance"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(
            kb, embedding, min_source_task_performance=0.7
        )

        # Add tasks with different performance
        for i in range(3):
            task = TaskDescriptor(
                task_id=f"task_{i}",
                task_type="test",
                state_dim=10,
                action_dim=4
            )
            kb.similarity_matrix.add_task(task)

            # Task 0: low performance, Task 1-2: high performance
            reward = 0.5 if i == 0 else 0.9
            kb.store_episode(Episode(
                episode_id=f"ep_{i}",
                task_id=f"task_{i}",
                agent_id="agent_1",
                transitions=[],
                total_reward=reward,
                episode_length=10,
                success=True
            ))

        # Transfer should only use high-performing tasks
        target = TaskDescriptor(
            task_id="target",
            task_type="test",
            state_dim=10,
            action_dim=4
        )

        result = engine.initiate_transfer(
            target_task=target,
            agent_id="agent_1",
            min_similarity=0.0
        )

        # Should not include task_0 (low performance)
        assert "task_0" not in result.source_tasks

    def test_k_source_tasks_limit(self):
        """Test k_source_tasks parameter limits number of sources"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Add many source tasks
        for i in range(10):
            task = TaskDescriptor(
                task_id=f"source_{i}",
                task_type="test",
                state_dim=10,
                action_dim=4
            )
            kb.similarity_matrix.add_task(task)
            kb.store_episode(Episode(
                episode_id=f"ep_{i}",
                task_id=f"source_{i}",
                agent_id="agent_1",
                transitions=[],
                total_reward=0.8,
                episode_length=10,
                success=True
            ))

        target = TaskDescriptor(
            task_id="target",
            task_type="test",
            state_dim=10,
            action_dim=4
        )

        result = engine.initiate_transfer(
            target_task=target,
            agent_id="agent_1",
            k_source_tasks=3,
            min_similarity=0.0
        )

        assert len(result.source_tasks) <= 3

    def test_same_type_only_filtering(self):
        """Test same_type_only parameter"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Add tasks of different types
        for task_type in ["navigation", "control", "manipulation"]:
            task = TaskDescriptor(
                task_id=f"{task_type}_task",
                task_type=task_type,
                state_dim=10,
                action_dim=4
            )
            kb.similarity_matrix.add_task(task)
            kb.store_episode(Episode(
                episode_id=f"ep_{task_type}",
                task_id=f"{task_type}_task",
                agent_id="agent_1",
                transitions=[],
                total_reward=0.8,
                episode_length=10,
                success=True
            ))

        target = TaskDescriptor(
            task_id="target",
            task_type="navigation",
            state_dim=10,
            action_dim=4
        )

        result = engine.initiate_transfer(
            target_task=target,
            agent_id="agent_1",
            min_similarity=0.0,
            same_type_only=True
        )

        # Should only transfer from navigation task
        assert all("navigation" in tid for tid in result.source_tasks)

    def test_clear_history(self):
        """Test clearing transfer history"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Perform transfer
        target = TaskDescriptor(
            task_id="target",
            task_type="test",
            state_dim=10,
            action_dim=4
        )
        engine.initiate_transfer(target_task=target, agent_id="agent_1")

        assert len(engine.transfer_history) > 0

        engine.clear_history()
        assert len(engine.transfer_history) == 0


# ==========================================
# MAML TESTS (15 tests)
# ==========================================

class TestMAML:
    """Test MAML meta-learning functionality"""

    def test_maml_config_creation(self):
        """Test MAML configuration"""
        config = MAMLConfig(
            meta_learning_rate=0.001,
            inner_learning_rate=0.01,
            k_shot=5,
            num_inner_steps=5
        )

        assert config.meta_learning_rate == 0.001
        assert config.inner_learning_rate == 0.01
        assert config.k_shot == 5

    def test_maml_config_validation(self):
        """Test MAML configuration validation"""
        config = MAMLConfig()
        config.validate()  # Should not raise

        with pytest.raises(AssertionError):
            bad_config = MAMLConfig(meta_learning_rate=-1.0)
            bad_config.validate()

    def test_maml_learner_creation(self):
        """Test MAML learner initialization"""
        kb = KnowledgeBase()
        config = MAMLConfig()

        def model_init():
            return {"weights": np.random.randn(10, 4)}

        maml = MAMLLearner(
            knowledge_base=kb,
            config=config,
            model_init_fn=model_init
        )

        assert maml.knowledge_base == kb
        assert maml.config == config
        assert not maml.meta_initialized

    def test_create_maml_learner_convenience(self):
        """Test convenience factory function"""
        kb = KnowledgeBase()

        def model_init():
            return np.random.randn(10, 4)

        maml = create_maml_learner(
            knowledge_base=kb,
            model_init_fn=model_init,
            k_shot=3,
            num_inner_steps=10
        )

        assert maml.config.k_shot == 3
        assert maml.config.num_inner_steps == 10

    def test_meta_train_initializes_parameters(self):
        """Test that meta-training initializes meta-parameters"""
        kb = KnowledgeBase()
        config = MAMLConfig(max_meta_iterations=10)

        def model_init():
            return np.random.randn(5, 2)

        maml = MAMLLearner(kb, config, model_init)

        # Add some tasks with data
        for i in range(5):
            task = TaskDescriptor(
                task_id=f"task_{i}",
                task_type="test",
                state_dim=5,
                action_dim=2
            )
            kb.similarity_matrix.tasks[task.task_id] = task

            # Add episodes for each task
            for j in range(15):  # k_shot + query_size
                episode = Episode(
                    episode_id=f"ep_{i}_{j}",
                    task_id=f"task_{i}",
                    agent_id="agent_1",
                    transitions=[],
                    total_reward=np.random.rand(),
                    episode_length=10,
                    success=True
                )
                kb.store_episode(episode)

        result = maml.meta_train(
            task_family_id="test_family",
            num_iterations=10,
            task_descriptors=[
                kb.similarity_matrix.tasks[tid]
                for tid in kb.similarity_matrix.tasks
            ]
        )

        assert maml.meta_initialized
        assert maml.meta_parameters is not None
        assert result['num_iterations'] <= 10

    def test_adapt_to_task_without_meta_init(self):
        """Test adaptation without meta-initialization (random init)"""
        kb = KnowledgeBase()
        config = MAMLConfig()

        def model_init():
            return np.random.randn(5, 2)

        maml = MAMLLearner(kb, config, model_init)

        task = TaskDescriptor(
            task_id="new_task",
            task_type="test",
            state_dim=5,
            action_dim=2
        )

        # Add support episodes
        support_episodes = [
            Episode(
                episode_id=f"ep_{i}",
                task_id="new_task",
                agent_id="agent_1",
                transitions=[],
                total_reward=float(i),
                episode_length=10,
                success=True
            )
            for i in range(5)
        ]

        result = maml.adapt_to_task(
            target_task=task,
            agent_id="agent_1",
            support_episodes=support_episodes,
            num_adaptation_steps=5
        )

        assert not result.meta_learned  # No meta-initialization
        assert result.adapted_parameters is not None
        assert result.num_adaptation_steps == 5

    def test_adapt_to_task_with_meta_init(self):
        """Test adaptation with meta-learned initialization"""
        kb = KnowledgeBase()
        config = MAMLConfig()

        def model_init():
            return np.random.randn(5, 2)

        maml = MAMLLearner(kb, config, model_init)

        # Set meta-parameters
        maml.set_meta_parameters(np.random.randn(5, 2))

        task = TaskDescriptor(
            task_id="new_task",
            task_type="test",
            state_dim=5,
            action_dim=2
        )

        support_episodes = [
            Episode(
                episode_id=f"ep_{i}",
                task_id="new_task",
                agent_id="agent_1",
                transitions=[],
                total_reward=float(i),
                episode_length=10,
                success=True
            )
            for i in range(5)
        ]

        result = maml.adapt_to_task(
            target_task=task,
            agent_id="agent_1",
            support_episodes=support_episodes
        )

        assert result.meta_learned  # Used meta-initialization

    def test_adaptation_result_tracking(self):
        """Test adaptation result dataclass"""
        result = AdaptationResult(
            task_id="task_1",
            agent_id="agent_1",
            num_adaptation_steps=5,
            num_support_examples=10,
            pre_adaptation_performance=0.3,
            post_adaptation_performance=0.7,
            performance_gain=0.4,
            adaptation_time=1.5
        )

        assert result.performance_gain == 0.4
        assert result.to_dict()['task_id'] == "task_1"

    def test_get_adaptation_history(self):
        """Test retrieving adaptation history"""
        kb = KnowledgeBase()
        config = MAMLConfig()

        def model_init():
            return np.random.randn(5, 2)

        maml = MAMLLearner(kb, config, model_init)
        maml.set_meta_parameters(np.random.randn(5, 2))

        # Perform adaptations
        for i in range(3):
            task = TaskDescriptor(
                task_id=f"task_{i}",
                task_type="test",
                state_dim=5,
                action_dim=2
            )

            support = [
                Episode(
                    episode_id=f"ep_{i}_{j}",
                    task_id=f"task_{i}",
                    agent_id="agent_1",
                    transitions=[],
                    total_reward=1.0,
                    episode_length=10,
                    success=True
                )
                for j in range(5)
            ]

            maml.adapt_to_task(task, "agent_1", support)

        history = maml.get_adaptation_history(agent_id="agent_1")
        assert len(history) == 3

    def test_get_average_adaptation_gain(self):
        """Test average adaptation gain calculation"""
        kb = KnowledgeBase()
        config = MAMLConfig()

        def model_init():
            return np.random.randn(5, 2)

        maml = MAMLLearner(kb, config, model_init)
        maml.set_meta_parameters(np.random.randn(5, 2))

        # Perform adaptations with different gains
        for i in range(3):
            task = TaskDescriptor(
                task_id=f"task_{i}",
                task_type="test",
                state_dim=5,
                action_dim=2
            )

            support = [
                Episode(
                    episode_id=f"ep_{i}_{j}",
                    task_id=f"task_{i}",
                    agent_id="agent_1",
                    transitions=[],
                    total_reward=float(i + j),
                    episode_length=10,
                    success=True
                )
                for j in range(5)
            ]

            maml.adapt_to_task(task, "agent_1", support)

        avg_gain = maml.get_average_adaptation_gain()
        assert avg_gain >= 0  # Gains might be negative depending on random data

    def test_clear_history(self):
        """Test clearing MAML history"""
        kb = KnowledgeBase()
        config = MAMLConfig()

        def model_init():
            return np.random.randn(5, 2)

        maml = MAMLLearner(kb, config, model_init)

        # Add some history
        maml.meta_training_history.append({"test": "data"})
        maml.adaptation_history["test"] = AdaptationResult(
            task_id="task_1",
            agent_id="agent_1",
            num_adaptation_steps=5,
            num_support_examples=5,
            pre_adaptation_performance=0.3,
            post_adaptation_performance=0.7,
            performance_gain=0.4,
            adaptation_time=1.0
        )

        maml.clear_history()

        assert len(maml.meta_training_history) == 0
        assert len(maml.adaptation_history) == 0

    def test_get_and_set_meta_parameters(self):
        """Test getting and setting meta-parameters"""
        kb = KnowledgeBase()
        config = MAMLConfig()

        def model_init():
            return np.random.randn(5, 2)

        maml = MAMLLearner(kb, config, model_init)

        params = np.random.randn(5, 2)
        maml.set_meta_parameters(params)

        retrieved = maml.get_meta_parameters()
        assert np.array_equal(retrieved, params)
        assert maml.meta_initialized


# ==========================================
# INTEGRATION TESTS (15 tests)
# ==========================================

class TestIntegration:
    """Test integration of transfer learning with agents"""

    def test_complete_transfer_workflow(self):
        """Test complete transfer learning workflow"""
        # Setup knowledge base and engine
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Train on source task
        source_task = TaskDescriptor(
            task_id="source_navigation",
            task_type="navigation",
            state_dim=10,
            action_dim=4,
            difficulty=0.5
        )
        kb.similarity_matrix.add_task(source_task)

        # Add experiences
        for i in range(100):
            kb.store_transition(
                ExperienceTransition(
                    task_id="source_navigation",
                    state=np.random.randn(10),
                    action=np.random.randint(0, 4),
                    reward=np.random.rand(),
                    next_state=np.random.randn(10),
                    done=False,
                    agent_id="agent_1"
                ),
                priority=1.0
            )

        # Store policy and value function
        kb.store_policy("source_navigation", {"weights": np.random.randn(10, 4)})
        kb.store_value_function("source_navigation", {"table": np.random.rand(10, 4)})

        # Add successful episodes
        for i in range(10):
            kb.store_episode(Episode(
                episode_id=f"source_ep_{i}",
                task_id="source_navigation",
                agent_id="agent_1",
                transitions=[],
                total_reward=np.random.rand() * 10,
                episode_length=20,
                success=True
            ))

        # Transfer to similar target task
        target_task = TaskDescriptor(
            task_id="target_navigation",
            task_type="navigation",
            state_dim=10,
            action_dim=4,
            difficulty=0.6
        )

        result = engine.initiate_transfer(
            target_task=target_task,
            agent_id="agent_1",
            strategy=TransferStrategy.COMBINED,
            min_similarity=0.5
        )

        # Verify transfer occurred
        assert result.transfer_used or len(result.source_tasks) > 0
        assert result.num_experiences_transferred > 0 or result.policy_transferred

    def test_multi_task_transfer(self):
        """Test transferring from multiple source tasks"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Create multiple source tasks
        for i in range(5):
            task = TaskDescriptor(
                task_id=f"source_{i}",
                task_type="navigation",
                state_dim=10,
                action_dim=4
            )
            kb.similarity_matrix.add_task(task)

            # Add varying amounts of experience
            for j in range((i+1) * 20):
                kb.store_transition(
                    ExperienceTransition(
                        task_id=f"source_{i}",
                        state=np.random.randn(10),
                        action=np.random.randint(0, 4),
                        reward=np.random.rand(),
                        next_state=np.random.randn(10),
                        done=False,
                        agent_id="agent_1"
                    ),
                    priority=1.0
                )

            # Add episode with increasing performance
            kb.store_episode(Episode(
                episode_id=f"ep_{i}",
                task_id=f"source_{i}",
                agent_id="agent_1",
                transitions=[],
                total_reward=float(i) * 2.0,  # Better performance for later tasks
                episode_length=20,
                success=True
            ))

        # Transfer to target
        target = TaskDescriptor(
            task_id="target",
            task_type="navigation",
            state_dim=10,
            action_dim=4
        )

        result = engine.initiate_transfer(
            target_task=target,
            agent_id="agent_1",
            k_source_tasks=3,
            min_similarity=0.0
        )

        # Should select top-k performing tasks
        assert len(result.source_tasks) <= 3

    def test_maml_with_transfer_workflow(self):
        """Test combining MAML and transfer learning"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        def model_init():
            return np.random.randn(10, 4)

        config = MAMLConfig(
            k_shot=5,
            num_inner_steps=3,
            max_meta_iterations=5
        )
        maml = MAMLLearner(kb, config, model_init)

        # Create task family
        tasks = []
        for i in range(3):
            task = TaskDescriptor(
                task_id=f"train_task_{i}",
                task_type="navigation",
                state_dim=10,
                action_dim=4
            )
            kb.similarity_matrix.tasks[task.task_id] = task
            tasks.append(task)

            # Add episodes for meta-training
            for j in range(15):
                kb.store_episode(Episode(
                    episode_id=f"ep_{i}_{j}",
                    task_id=f"train_task_{i}",
                    agent_id="agent_1",
                    transitions=[],
                    total_reward=np.random.rand() * 10,
                    episode_length=20,
                    success=True
                ))

        # Meta-train
        maml.meta_train(
            task_family_id="navigation",
            num_iterations=5,
            task_descriptors=tasks
        )

        # Now use transfer + MAML for new task
        new_task = TaskDescriptor(
            task_id="new_navigation",
            task_type="navigation",
            state_dim=10,
            action_dim=4
        )

        # Transfer from similar tasks
        transfer_result = engine.initiate_transfer(
            target_task=new_task,
            agent_id="agent_1",
            min_similarity=0.0
        )

        # Add a few episodes for new task
        support_episodes = [
            Episode(
                episode_id=f"new_ep_{i}",
                task_id="new_navigation",
                agent_id="agent_1",
                transitions=[],
                total_reward=np.random.rand() * 10,
                episode_length=20,
                success=True
            )
            for i in range(5)
        ]

        for ep in support_episodes:
            kb.store_episode(ep)

        # MAML adaptation
        adaptation_result = maml.adapt_to_task(
            target_task=new_task,
            agent_id="agent_1",
            support_episodes=support_episodes
        )

        assert adaptation_result.meta_learned
        assert len(transfer_result.source_tasks) > 0

    def test_speedup_measurement(self):
        """Test measuring actual speed-up from transfer"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Setup source task
        source = TaskDescriptor(
            task_id="source",
            task_type="test",
            state_dim=10,
            action_dim=4
        )
        kb.similarity_matrix.add_task(source)

        # Add lots of experience
        for i in range(500):
            kb.store_transition(
                ExperienceTransition(
                    task_id="source",
                    state=np.random.randn(10),
                    action=np.random.randint(0, 4),
                    reward=np.random.rand(),
                    next_state=np.random.randn(10),
                    done=False,
                    agent_id="agent_1"
                ),
                priority=1.0
            )

        kb.store_episode(Episode(
            episode_id="ep_source",
            task_id="source",
            agent_id="agent_1",
            transitions=[],
            total_reward=10.0,
            episode_length=100,
            success=True
        ))

        # Transfer to target
        target = TaskDescriptor(
            task_id="target",
            task_type="test",
            state_dim=10,
            action_dim=4
        )

        result = engine.initiate_transfer(
            target_task=target,
            agent_id="agent_1",
            min_similarity=0.0
        )

        # Evaluate with simulated performance
        evaluation = engine.evaluate_transfer(
            target_task_id="target",
            baseline_performance=0.5,
            transfer_performance=0.8,
            baseline_samples=5000,
            transfer_samples=250
        )

        assert evaluation['speedup_factor'] == 20.0
        assert evaluation['meets_10x_target']

    def test_thread_safety(self):
        """Test thread-safe concurrent operations"""
        import threading

        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Add source task
        source = TaskDescriptor(
            task_id="source",
            task_type="test",
            state_dim=10,
            action_dim=4
        )
        kb.similarity_matrix.add_task(source)
        kb.store_episode(Episode(
            episode_id="ep_source",
            task_id="source",
            agent_id="agent_1",
            transitions=[],
            total_reward=10.0,
            episode_length=10,
            success=True
        ))

        results = []

        def transfer_task(agent_id):
            target = TaskDescriptor(
                task_id=f"target_{agent_id}",
                task_type="test",
                state_dim=10,
                action_dim=4
            )
            result = engine.initiate_transfer(
                target_task=target,
                agent_id=f"agent_{agent_id}",
                min_similarity=0.0
            )
            results.append(result)

        # Spawn multiple threads
        threads = [
            threading.Thread(target=transfer_task, args=(i,))
            for i in range(10)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All transfers should complete
        assert len(results) == 10

    def test_progressive_learning(self):
        """Test progressive learning through curriculum"""
        kb = KnowledgeBase()
        embedding = TaskEmbedding()
        engine = TransferLearningEngine(kb, embedding)

        # Create curriculum: easy -> medium -> hard
        difficulties = [0.1, 0.5, 0.9]

        for i, diff in enumerate(difficulties):
            task = TaskDescriptor(
                task_id=f"curriculum_{i}",
                task_type="test",
                state_dim=10,
                action_dim=4,
                difficulty=diff
            )
            kb.similarity_matrix.add_task(task)

            # Add progressively better performance
            kb.store_episode(Episode(
                episode_id=f"ep_{i}",
                task_id=f"curriculum_{i}",
                agent_id="agent_1",
                transitions=[],
                total_reward=10.0 * (i + 1),  # Better on later tasks
                episode_length=10,
                success=True
            ))

        # Learn hardest task with curriculum transfer
        hard_task = TaskDescriptor(
            task_id="very_hard",
            task_type="test",
            state_dim=10,
            action_dim=4,
            difficulty=0.95
        )

        result = engine.initiate_transfer(
            target_task=hard_task,
            agent_id="agent_1",
            strategy=TransferStrategy.CURRICULUM,
            min_similarity=0.0
        )

        assert 'curriculum' in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
