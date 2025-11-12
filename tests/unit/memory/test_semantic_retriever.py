"""
Tests for SemanticRetriever.

Test Coverage:
- Initialization (with/without Vector DB)
- Experience encoding and storage
- State encoding
- Similarity search
- Reward-based search
- Counterfactual queries
- Numpy fallback mode
- Statistics tracking
- Metadata filtering
"""

import pytest
import numpy as np
import time
from src.memory.semantic_retriever import SemanticRetriever, SemanticQuery


class MockVectorDBCollection:
    """Mock ChromaDB collection for testing."""

    def __init__(self, name):
        self.name = name
        self.embeddings = []
        self.metadatas = []
        self.ids = []

    def add(self, embeddings, metadatas, ids):
        """Add embeddings to collection."""
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def query(self, query_embeddings, n_results, where=None):
        """Query for similar embeddings."""
        if len(self.embeddings) == 0:
            return {'embeddings': [[]], 'metadatas': [[]], 'ids': [[]], 'distances': [[]]}

        # Compute cosine distances
        query = np.array(query_embeddings[0])
        distances = []

        for emb in self.embeddings:
            emb_array = np.array(emb)
            similarity = np.dot(query, emb_array) / (np.linalg.norm(query) * np.linalg.norm(emb_array) + 1e-8)
            distance = 1.0 - similarity
            distances.append(distance)

        # Apply metadata filters
        valid_indices = list(range(len(self.embeddings)))
        if where:
            filtered = []
            for idx in valid_indices:
                if self._matches_filter(self.metadatas[idx], where):
                    filtered.append(idx)
            valid_indices = filtered

        # Sort by distance
        sorted_indices = sorted(valid_indices, key=lambda i: distances[i])
        top_k = sorted_indices[:n_results]

        return {
            'embeddings': [[self.embeddings[i] for i in top_k]],
            'metadatas': [[self.metadatas[i] for i in top_k]],
            'ids': [[self.ids[i] for i in top_k]],
            'distances': [[distances[i] for i in top_k]]
        }

    def get(self, where=None, limit=10):
        """Get experiences by metadata filter."""
        results = []
        result_ids = []

        for idx, metadata in enumerate(self.metadatas):
            if where is None or self._matches_filter(metadata, where):
                results.append(metadata)
                result_ids.append(self.ids[idx])

                if len(results) >= limit:
                    break

        return {
            'metadatas': results,
            'ids': result_ids
        }

    def count(self):
        """Get number of stored embeddings."""
        return len(self.embeddings)

    def _matches_filter(self, metadata, where):
        """Check if metadata matches filter."""
        for key, condition in where.items():
            if key not in metadata:
                return False

            value = metadata[key]

            if isinstance(condition, dict):
                for op, threshold in condition.items():
                    if op == "$gte" and value < threshold:
                        return False
                    elif op == "$lte" and value > threshold:
                        return False
                    elif op == "$gt" and value <= threshold:
                        return False
                    elif op == "$lt" and value >= threshold:
                        return False
            else:
                if value != condition:
                    return False

        return True


class MockVectorDBClient:
    """Mock ChromaDB client for testing."""

    def __init__(self):
        self.collections = {}

    def get_collection(self, name):
        """Get existing collection."""
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist")
        return self.collections[name]

    def create_collection(self, name, metadata=None):
        """Create new collection."""
        collection = MockVectorDBCollection(name)
        self.collections[name] = collection
        return collection

    def delete_collection(self, name):
        """Delete collection."""
        if name in self.collections:
            del self.collections[name]


class TestSemanticRetrieverInitialization:
    """Test SemanticRetriever initialization."""

    def test_init_with_vector_db(self):
        """Test initialization with Vector DB."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db, embedding_dim=64)

        assert retriever.embedding_dim == 64
        assert retriever.collection_name == "episodic_memories"
        assert not retriever.using_fallback
        assert retriever.collection is not None

    def test_init_without_vector_db(self):
        """Test initialization without Vector DB (fallback mode)."""
        retriever = SemanticRetriever(vector_db_client=None, embedding_dim=64)

        assert retriever.embedding_dim == 64
        assert retriever.using_fallback
        assert retriever.collection is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(
            vector_db_client=db,
            embedding_dim=256,
            collection_name="custom_memories",
            distance_metric="l2"
        )

        assert retriever.embedding_dim == 256
        assert retriever.collection_name == "custom_memories"
        assert retriever.distance_metric == "l2"


class TestSemanticRetrieverEncoding:
    """Test experience and state encoding."""

    def test_encode_simple_state(self):
        """Test encoding a simple state vector."""
        retriever = SemanticRetriever(embedding_dim=128)

        state = np.array([1.0, 2.0, 3.0, 4.0])
        embedding = retriever.encode_state(state)

        assert embedding.shape == (128,)
        assert np.isclose(np.linalg.norm(embedding), 1.0)  # Normalized

    def test_encode_experience(self):
        """Test encoding a complete experience."""
        retriever = SemanticRetriever(embedding_dim=128)

        experience = {
            'state': np.array([1.0, 2.0, 3.0]),
            'action': 2,
            'reward': 1.5,
            'next_state': np.array([2.0, 3.0, 4.0]),
            'done': False
        }

        embedding = retriever.encode_experience(experience)

        assert embedding.shape == (128,)
        assert np.isclose(np.linalg.norm(embedding), 1.0)

    def test_encode_experience_continuous_action(self):
        """Test encoding experience with continuous action."""
        retriever = SemanticRetriever(embedding_dim=128)

        experience = {
            'state': np.array([1.0, 2.0]),
            'action': np.array([0.5, -0.5]),
            'reward': 0.8
        }

        embedding = retriever.encode_experience(experience)

        assert embedding.shape == (128,)
        assert np.isclose(np.linalg.norm(embedding), 1.0)

    def test_encode_large_state(self):
        """Test encoding state larger than embedding_dim."""
        retriever = SemanticRetriever(embedding_dim=64)

        state = np.random.randn(200)
        embedding = retriever.encode_state(state)

        assert embedding.shape == (64,)
        assert np.isclose(np.linalg.norm(embedding), 1.0)

    def test_encode_small_state(self):
        """Test encoding state smaller than embedding_dim."""
        retriever = SemanticRetriever(embedding_dim=128)

        state = np.array([1.0, 2.0])
        embedding = retriever.encode_state(state)

        assert embedding.shape == (128,)
        assert np.isclose(np.linalg.norm(embedding), 1.0)


class TestSemanticRetrieverAdd:
    """Test adding experiences to semantic memory."""

    def test_add_single_experience(self):
        """Test adding a single experience."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        experience = {
            'state': np.array([1.0, 2.0, 3.0]),
            'action': 1,
            'reward': 1.0,
            'done': False,
            'timestamp': time.time()
        }

        embedding = retriever.encode_experience(experience)
        exp_id = retriever.add(experience, embedding)

        assert exp_id is not None
        assert retriever.total_added == 1
        assert retriever.get_statistics()['total_experiences'] == 1

    def test_add_multiple_experiences(self):
        """Test adding multiple experiences."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        for i in range(5):
            experience = {
                'state': np.random.randn(10),
                'action': i % 3,
                'reward': float(i),
                'timestamp': time.time() + i
            }
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        assert retriever.total_added == 5
        assert retriever.get_statistics()['total_experiences'] == 5

    def test_add_with_custom_id(self):
        """Test adding experience with custom ID."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        experience = {'state': np.array([1.0, 2.0]), 'reward': 1.0}
        embedding = retriever.encode_experience(experience)

        exp_id = retriever.add(experience, embedding, experience_id="custom_123")

        assert exp_id == "custom_123"

    def test_add_with_wrong_embedding_dim(self):
        """Test that wrong embedding dimension raises error."""
        retriever = SemanticRetriever(embedding_dim=128)

        experience = {'state': np.array([1.0, 2.0]), 'reward': 1.0}
        wrong_embedding = np.random.randn(64)  # Wrong dimension

        with pytest.raises(ValueError, match="Embedding dimension"):
            retriever.add(experience, wrong_embedding)

    def test_add_fallback_mode(self):
        """Test adding experiences in fallback mode."""
        retriever = SemanticRetriever(vector_db_client=None)  # No DB

        experience = {'state': np.array([1.0, 2.0]), 'reward': 1.0, 'timestamp': time.time()}
        embedding = retriever.encode_experience(experience)

        exp_id = retriever.add(experience, embedding)

        assert exp_id is not None
        assert len(retriever.fallback_embeddings) == 1
        assert len(retriever.fallback_experiences) == 1


class TestSemanticRetrieverSearch:
    """Test similarity search."""

    def test_search_basic(self):
        """Test basic similarity search."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db, embedding_dim=64)

        # Add experiences
        for i in range(10):
            experience = {
                'state': np.array([float(i), float(i*2)]),
                'action': i % 3,
                'reward': float(i),
                'timestamp': time.time() + i
            }
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Search for similar experiences
        query_embedding = retriever.encode_state(np.array([5.0, 10.0]))
        result = retriever.search(query_embedding, k=3)

        assert isinstance(result, SemanticQuery)
        assert len(result.experiences) <= 3
        assert len(result.distances) == len(result.experiences)
        assert len(result.ids) == len(result.experiences)
        assert result.query_time >= 0

    def test_search_by_state(self):
        """Test search by state vector."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        # Add experiences with different states
        states = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.9, 0.1, 0.0])  # Similar to first
        ]

        for state in states:
            experience = {'state': state, 'reward': 1.0, 'timestamp': time.time()}
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Search for state similar to [1, 0, 0]
        query_state = np.array([1.0, 0.0, 0.0])
        result = retriever.search_by_state(query_state, k=2)

        assert len(result.experiences) <= 2
        # First result should be most similar (smallest distance)
        assert result.distances[0] <= result.distances[-1]

    def test_search_with_metadata_filter(self):
        """Test search with metadata filtering."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        # Add experiences with varying rewards
        for i in range(5):
            experience = {
                'state': np.random.randn(10),
                'action': i,
                'reward': float(i) * 0.2,  # 0.0, 0.2, 0.4, 0.6, 0.8
                'timestamp': time.time() + i
            }
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Search for high-reward experiences only
        query = retriever.encode_state(np.random.randn(10))
        result = retriever.search(
            query,
            k=3,
            filter_metadata={"reward": {"$gte": 0.5}}
        )

        # Should only return experiences with reward >= 0.5
        for exp in result.experiences:
            assert exp['reward'] >= 0.5

    def test_search_empty_memory(self):
        """Test search on empty memory."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        query = retriever.encode_state(np.array([1.0, 2.0]))
        result = retriever.search(query, k=5)

        assert len(result.experiences) == 0
        assert len(result.distances) == 0

    def test_search_fallback_mode(self):
        """Test search in fallback mode (no Vector DB)."""
        retriever = SemanticRetriever(vector_db_client=None, embedding_dim=64)

        # Add experiences
        for i in range(5):
            experience = {'state': np.array([float(i)]), 'reward': float(i), 'timestamp': time.time()}
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Search
        query = retriever.encode_state(np.array([2.5]))
        result = retriever.search(query, k=3)

        assert len(result.experiences) == 3


class TestSemanticRetrieverRewardSearch:
    """Test reward-based search."""

    def test_search_by_reward(self):
        """Test searching for high-reward experiences."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        # Add experiences with different rewards
        rewards = [0.1, 0.5, 0.9, 0.3, 0.8, 0.2]
        for reward in rewards:
            experience = {
                'state': np.random.randn(5),
                'reward': reward,
                'timestamp': time.time()
            }
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Search for experiences with reward >= 0.6
        result = retriever.search_by_reward(min_reward=0.6, k=3)

        assert len(result.experiences) <= 3
        for exp in result.experiences:
            assert exp['reward'] >= 0.6

    def test_search_by_reward_none_match(self):
        """Test reward search when no experiences match."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        # Add low-reward experiences
        for i in range(3):
            experience = {'state': np.random.randn(5), 'reward': 0.1, 'timestamp': time.time()}
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Search for high reward
        result = retriever.search_by_reward(min_reward=0.9, k=5)

        assert len(result.experiences) == 0

    def test_search_by_reward_fallback(self):
        """Test reward search in fallback mode."""
        retriever = SemanticRetriever(vector_db_client=None)

        rewards = [0.2, 0.7, 0.9, 0.4]
        for reward in rewards:
            experience = {'state': np.random.randn(5), 'reward': reward, 'timestamp': time.time()}
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        result = retriever.search_by_reward(min_reward=0.6, k=3)

        assert len(result.experiences) == 2  # 0.7 and 0.9
        assert all(exp['reward'] >= 0.6 for exp in result.experiences)


class TestSemanticRetrieverCounterfactual:
    """Test counterfactual queries."""

    def test_counterfactual_query(self):
        """Test counterfactual query: 'What happened when action X was taken?'"""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        # Add experiences in similar states with different actions
        base_state = np.array([1.0, 0.0, 0.0])

        for i in range(10):
            # Create states similar to base_state
            state = base_state + np.random.randn(3) * 0.1
            experience = {
                'state': state,
                'action': i % 3,  # Actions 0, 1, 2
                'reward': float(i),
                'timestamp': time.time() + i
            }
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Query: "What happened when action 1 was taken in states similar to base_state?"
        result = retriever.get_counterfactual_experiences(
            state=base_state,
            action=1,
            k=3
        )

        # All returned experiences should have action=1
        for exp in result.experiences:
            assert exp['action'] == 1

        # Should return at most k results
        assert len(result.experiences) <= 3

    def test_counterfactual_no_matches(self):
        """Test counterfactual query when no matching action exists."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        # Add experiences with only action 0
        for i in range(5):
            experience = {
                'state': np.random.randn(5),
                'action': 0,
                'reward': float(i),
                'timestamp': time.time()
            }
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Query for action 2 (doesn't exist)
        result = retriever.get_counterfactual_experiences(
            state=np.array([1.0, 0.0]),
            action=2,
            k=3
        )

        assert len(result.experiences) == 0

    def test_counterfactual_fallback(self):
        """Test counterfactual query in fallback mode."""
        retriever = SemanticRetriever(vector_db_client=None)

        # Add experiences
        for i in range(6):
            experience = {
                'state': np.array([float(i)]),
                'action': i % 2,
                'reward': float(i),
                'timestamp': time.time()
            }
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Query for action 1
        result = retriever.get_counterfactual_experiences(
            state=np.array([3.0]),
            action=1,
            k=2
        )

        assert all(exp['action'] == 1 for exp in result.experiences)


class TestSemanticRetrieverStatistics:
    """Test statistics and management."""

    def test_get_statistics(self):
        """Test getting retriever statistics."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        # Add some experiences
        for i in range(5):
            experience = {'state': np.random.randn(10), 'reward': 1.0, 'timestamp': time.time()}
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Perform some queries
        for _ in range(3):
            query = retriever.encode_state(np.random.randn(10))
            retriever.search(query, k=2)

        stats = retriever.get_statistics()

        assert stats['total_experiences'] == 5
        assert stats['total_added'] == 5
        assert stats['total_queries'] == 3
        assert stats['avg_query_time'] >= 0
        assert stats['embedding_dim'] == 128

    def test_clear_memory(self):
        """Test clearing all memories."""
        db = MockVectorDBClient()
        retriever = SemanticRetriever(vector_db_client=db)

        # Add experiences
        for i in range(3):
            experience = {'state': np.random.randn(5), 'reward': 1.0, 'timestamp': time.time()}
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        assert retriever.get_statistics()['total_experiences'] == 3

        # Clear
        retriever.clear()

        assert retriever.get_statistics()['total_experiences'] == 0
        assert retriever.total_added == 0

    def test_repr(self):
        """Test string representation."""
        retriever = SemanticRetriever(embedding_dim=64)

        repr_str = repr(retriever)

        assert "SemanticRetriever" in repr_str
        assert "dim=64" in repr_str
        assert "using_fallback" in repr_str


class TestSemanticRetrieverFallback:
    """Test numpy fallback functionality."""

    def test_fallback_cosine_distance(self):
        """Test fallback search with cosine distance."""
        retriever = SemanticRetriever(
            vector_db_client=None,
            distance_metric="cosine"
        )

        # Add orthogonal states
        states = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]

        for state in states:
            experience = {'state': state, 'reward': 1.0, 'timestamp': time.time()}
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Query with [1, 0, 0]
        query = retriever.encode_state(np.array([1.0, 0.0, 0.0]))
        result = retriever.search(query, k=2)

        # First result should be most similar
        assert len(result.experiences) == 2

    def test_fallback_l2_distance(self):
        """Test fallback search with L2 distance."""
        retriever = SemanticRetriever(
            vector_db_client=None,
            distance_metric="l2"
        )

        for i in range(5):
            experience = {'state': np.array([float(i)]), 'reward': 1.0, 'timestamp': time.time()}
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        query = retriever.encode_state(np.array([2.5]))
        result = retriever.search(query, k=2)

        assert len(result.experiences) == 2

    def test_fallback_metadata_filter(self):
        """Test metadata filtering in fallback mode."""
        retriever = SemanticRetriever(vector_db_client=None)

        for i in range(5):
            experience = {
                'state': np.array([float(i)]),
                'reward': float(i) * 0.2,
                'timestamp': time.time()
            }
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        query = retriever.encode_state(np.array([2.0]))
        result = retriever.search(
            query,
            k=3,
            filter_metadata={"reward": {"$gte": 0.4}}
        )

        # Should only get experiences with reward >= 0.4
        assert all(exp['reward'] >= 0.4 for exp in result.experiences)


class TestSemanticRetrieverEdgeCases:
    """Test edge cases and error handling."""

    def test_search_k_larger_than_memory(self):
        """Test search when k > number of stored experiences."""
        retriever = SemanticRetriever(vector_db_client=None)

        # Add only 2 experiences
        for i in range(2):
            experience = {'state': np.array([float(i)]), 'reward': 1.0, 'timestamp': time.time()}
            embedding = retriever.encode_experience(experience)
            retriever.add(experience, embedding)

        # Request 10 results
        query = retriever.encode_state(np.array([1.0]))
        result = retriever.search(query, k=10)

        # Should return only 2
        assert len(result.experiences) == 2

    def test_encode_zero_vector(self):
        """Test encoding zero vector."""
        retriever = SemanticRetriever()

        state = np.zeros(10)
        embedding = retriever.encode_state(state)

        # Should still produce valid embedding
        assert embedding.shape == (128,)
        # Norm may be 0 or small
        assert not np.any(np.isnan(embedding))

    def test_semantic_query_dataclass(self):
        """Test SemanticQuery dataclass."""
        query = SemanticQuery(
            experiences=[{'reward': 1.0}],
            distances=[0.5],
            ids=['exp_1'],
            query_time=0.01,
            metadata={'k': 5}
        )

        assert len(query.experiences) == 1
        assert query.distances[0] == 0.5
        assert query.query_time == 0.01
