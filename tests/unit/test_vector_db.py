"""
Unit tests for Vector Database Interface.

Tests the vector database functionality including:
- VectorDBInterface abstract base class
- ChromaDB backend implementation
- Policy embedding storage and retrieval
- Semantic similarity search
- Batch operations
- Metadata management
- Clustering and pattern detection
- Collection management
"""

import pytest
import numpy as np
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any

from src.connectors.vector_db import (
    VectorDBInterface,
    ChromaDBBackend,
    PolicyEmbedding,
    SearchResult,
    create_vector_db
)


class TestPolicyEmbeddingDataclass:
    """Test PolicyEmbedding dataclass."""

    def test_policy_embedding_creation(self):
        """Test creating a PolicyEmbedding instance."""
        embedding = PolicyEmbedding(
            policy_id="policy_001",
            agent_id="agent_123",
            embedding=np.random.rand(128),
            metadata={"version": 1},
            timestamp=time.time(),
            performance=0.85
        )

        assert embedding.policy_id == "policy_001"
        assert embedding.agent_id == "agent_123"
        assert embedding.embedding.shape == (128,)
        assert embedding.metadata["version"] == 1
        assert embedding.performance == 0.85


class TestSearchResultDataclass:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult instance."""
        result = SearchResult(
            policy_id="policy_002",
            agent_id="agent_456",
            similarity_score=0.92,
            metadata={"type": "test"}
        )

        assert result.policy_id == "policy_002"
        assert result.agent_id == "agent_456"
        assert result.similarity_score == 0.92
        assert result.metadata["type"] == "test"
        assert result.embedding is None

    def test_search_result_with_embedding(self):
        """Test SearchResult with embedding included."""
        emb = np.random.rand(128)
        result = SearchResult(
            policy_id="policy_003",
            agent_id="agent_789",
            similarity_score=0.88,
            metadata={},
            embedding=emb
        )

        assert result.embedding is not None
        np.testing.assert_array_equal(result.embedding, emb)


class TestVectorDBInterface:
    """Test VectorDBInterface abstract base class."""

    def test_interface_init(self):
        """Test that interface cannot be instantiated directly."""
        # Abstract classes can't be instantiated
        with pytest.raises(TypeError):
            VectorDBInterface()

    def test_interface_defines_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = [
            'initialize',
            'add_policy_embedding',
            'add_policy_embeddings_batch',
            'search_similar_policies',
            'get_policy_embedding',
            'get_agent_policies',
            'update_policy_metadata',
            'delete_policy',
            'delete_agent_policies',
            'cluster_policies',
            'find_policy_patterns',
            'get_collection_stats',
            'clear_collection',
            'close'
        ]

        for method_name in abstract_methods:
            assert hasattr(VectorDBInterface, method_name)


class TestChromaDBBackendInit:
    """Test ChromaDB backend initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        backend = ChromaDBBackend()

        assert backend.collection_name == "mae_policies"
        assert backend.embedding_dim == 128
        assert backend.persist_directory == "data/chromadb"
        assert backend.client is None
        assert backend.collection is None
        assert backend.is_initialized is False

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = ChromaDBBackend(
                collection_name="custom_collection",
                embedding_dim=256,
                persist_directory=tmpdir
            )

            assert backend.collection_name == "custom_collection"
            assert backend.embedding_dim == 256
            assert backend.persist_directory == tmpdir

    @patch('src.connectors.vector_db.chromadb')
    def test_initialize_creates_client_and_collection(self, mock_chromadb):
        """Test that initialize creates client and collection."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.Client.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        backend = ChromaDBBackend()
        backend.initialize()

        assert backend.is_initialized is True
        assert backend.client is not None
        assert backend.collection is not None
        mock_client.get_or_create_collection.assert_called_once()

    def test_initialize_without_chromadb_installed(self):
        """Test initialization fails gracefully without ChromaDB."""
        with patch('src.connectors.vector_db.chromadb', None):
            backend = ChromaDBBackend()

            # Mock the import to fail
            with patch.dict('sys.modules', {'chromadb': None}):
                with pytest.raises(Exception):
                    backend.initialize()


class TestChromaDBBackendAddOperations:
    """Test adding policy embeddings."""

    @pytest.fixture
    def mock_chromadb_backend(self):
        """Create ChromaDB backend with mocked collection."""
        with patch('src.connectors.vector_db.chromadb'):
            backend = ChromaDBBackend()
            backend.collection = MagicMock()
            backend.is_initialized = True
            yield backend

    def test_add_policy_embedding(self, mock_chromadb_backend):
        """Test adding a single policy embedding."""
        embedding = np.random.rand(128)

        result = mock_chromadb_backend.add_policy_embedding(
            policy_id="policy_001",
            agent_id="agent_123",
            embedding=embedding,
            metadata={"version": 1},
            performance=0.85
        )

        assert result is True
        mock_chromadb_backend.collection.add.assert_called_once()

        # Verify call arguments
        call_args = mock_chromadb_backend.collection.add.call_args
        assert call_args[1]['ids'] == ["policy_001"]
        assert len(call_args[1]['embeddings']) == 1
        assert call_args[1]['metadatas'][0]['agent_id'] == "agent_123"
        assert call_args[1]['metadatas'][0]['performance'] == 0.85

    def test_add_policy_embedding_without_metadata(self, mock_chromadb_backend):
        """Test adding policy without metadata."""
        embedding = np.random.rand(128)

        result = mock_chromadb_backend.add_policy_embedding(
            policy_id="policy_002",
            agent_id="agent_456",
            embedding=embedding
        )

        assert result is True
        call_args = mock_chromadb_backend.collection.add.call_args
        assert 'agent_id' in call_args[1]['metadatas'][0]
        assert 'performance' in call_args[1]['metadatas'][0]

    def test_add_policy_embedding_auto_initializes(self):
        """Test that add_policy_embedding initializes if not initialized."""
        with patch('src.connectors.vector_db.chromadb'):
            backend = ChromaDBBackend()
            backend.collection = MagicMock()
            backend.is_initialized = False

            with patch.object(backend, 'initialize') as mock_init:
                mock_init.side_effect = lambda: setattr(backend, 'is_initialized', True)

                backend.add_policy_embedding(
                    "policy_x",
                    "agent_x",
                    np.random.rand(128)
                )

                mock_init.assert_called_once()

    def test_add_policy_embedding_handles_errors(self, mock_chromadb_backend):
        """Test error handling when adding policy fails."""
        mock_chromadb_backend.collection.add.side_effect = Exception("Database error")

        result = mock_chromadb_backend.add_policy_embedding(
            "policy_error",
            "agent_error",
            np.random.rand(128)
        )

        assert result is False

    def test_add_policy_embeddings_batch(self, mock_chromadb_backend):
        """Test adding multiple policies in batch."""
        embeddings = [
            PolicyEmbedding(
                policy_id=f"policy_{i}",
                agent_id=f"agent_{i}",
                embedding=np.random.rand(128),
                metadata={"index": i},
                timestamp=time.time(),
                performance=0.5 + i * 0.1
            )
            for i in range(5)
        ]

        count = mock_chromadb_backend.add_policy_embeddings_batch(embeddings)

        assert count == 5
        mock_chromadb_backend.collection.add.assert_called_once()

        call_args = mock_chromadb_backend.collection.add.call_args
        assert len(call_args[1]['ids']) == 5
        assert len(call_args[1]['embeddings']) == 5
        assert len(call_args[1]['metadatas']) == 5

    def test_add_policy_embeddings_batch_empty_list(self, mock_chromadb_backend):
        """Test adding empty batch."""
        # Should still work but add nothing
        count = mock_chromadb_backend.add_policy_embeddings_batch([])

        # ChromaDB will be called with empty lists
        assert count == 0

    def test_add_policy_embeddings_batch_error_handling(self, mock_chromadb_backend):
        """Test batch add error handling."""
        mock_chromadb_backend.collection.add.side_effect = Exception("Batch error")

        embeddings = [
            PolicyEmbedding(
                f"p{i}", f"a{i}", np.random.rand(128), {}, time.time(), 0.8
            )
            for i in range(3)
        ]

        count = mock_chromadb_backend.add_policy_embeddings_batch(embeddings)

        assert count == 0


class TestChromaDBBackendSearchOperations:
    """Test search and retrieval operations."""

    @pytest.fixture
    def mock_chromadb_backend(self):
        """Create ChromaDB backend with mocked collection."""
        with patch('src.connectors.vector_db.chromadb'):
            backend = ChromaDBBackend()
            backend.collection = MagicMock()
            backend.is_initialized = True
            yield backend

    def test_search_similar_policies(self, mock_chromadb_backend):
        """Test searching for similar policies."""
        # Mock search results
        mock_chromadb_backend.collection.query.return_value = {
            'ids': [['policy_1', 'policy_2', 'policy_3']],
            'distances': [[0.1, 0.2, 0.3]],
            'metadatas': [[
                {'agent_id': 'agent_1', 'performance': 0.9},
                {'agent_id': 'agent_2', 'performance': 0.8},
                {'agent_id': 'agent_3', 'performance': 0.7}
            ]]
        }

        query_embedding = np.random.rand(128)
        results = mock_chromadb_backend.search_similar_policies(
            query_embedding=query_embedding,
            top_k=3
        )

        assert len(results) == 3
        assert results[0].policy_id == 'policy_1'
        assert results[0].agent_id == 'agent_1'
        assert results[0].similarity_score == 0.9  # 1.0 - 0.1

    def test_search_similar_policies_with_filter(self, mock_chromadb_backend):
        """Test searching with filter criteria."""
        mock_chromadb_backend.collection.query.return_value = {
            'ids': [['policy_x']],
            'distances': [[0.15]],
            'metadatas': [[{'agent_id': 'agent_x'}]]
        }

        query_embedding = np.random.rand(128)
        filter_criteria = {"agent_id": "agent_x"}

        results = mock_chromadb_backend.search_similar_policies(
            query_embedding=query_embedding,
            top_k=10,
            filter_criteria=filter_criteria
        )

        # Verify filter was passed
        call_args = mock_chromadb_backend.collection.query.call_args
        assert call_args[1]['where'] == filter_criteria

    def test_search_similar_policies_no_results(self, mock_chromadb_backend):
        """Test search with no results."""
        mock_chromadb_backend.collection.query.return_value = {
            'ids': [[]],
            'distances': [[]],
            'metadatas': [[]]
        }

        results = mock_chromadb_backend.search_similar_policies(
            np.random.rand(128)
        )

        assert len(results) == 0

    def test_search_similar_policies_error_handling(self, mock_chromadb_backend):
        """Test search error handling."""
        mock_chromadb_backend.collection.query.side_effect = Exception("Search error")

        results = mock_chromadb_backend.search_similar_policies(
            np.random.rand(128)
        )

        assert len(results) == 0

    def test_get_policy_embedding(self, mock_chromadb_backend):
        """Test retrieving a specific policy by ID."""
        mock_chromadb_backend.collection.get.return_value = {
            'ids': ['policy_123'],
            'embeddings': [np.random.rand(128).tolist()],
            'metadatas': [{
                'agent_id': 'agent_456',
                'timestamp': 1234567890.0,
                'performance': 0.92
            }]
        }

        policy = mock_chromadb_backend.get_policy_embedding("policy_123")

        assert policy is not None
        assert policy.policy_id == "policy_123"
        assert policy.agent_id == "agent_456"
        assert policy.performance == 0.92
        assert policy.embedding.shape == (128,)

    def test_get_policy_embedding_not_found(self, mock_chromadb_backend):
        """Test getting non-existent policy."""
        mock_chromadb_backend.collection.get.return_value = {
            'ids': [],
            'embeddings': [],
            'metadatas': []
        }

        policy = mock_chromadb_backend.get_policy_embedding("nonexistent")

        assert policy is None

    def test_get_policy_embedding_error_handling(self, mock_chromadb_backend):
        """Test get policy error handling."""
        mock_chromadb_backend.collection.get.side_effect = Exception("Get error")

        policy = mock_chromadb_backend.get_policy_embedding("policy_error")

        assert policy is None

    def test_get_agent_policies(self, mock_chromadb_backend):
        """Test retrieving all policies for an agent."""
        mock_chromadb_backend.collection.get.return_value = {
            'ids': ['policy_1', 'policy_2', 'policy_3'],
            'embeddings': [
                np.random.rand(128).tolist(),
                np.random.rand(128).tolist(),
                np.random.rand(128).tolist()
            ],
            'metadatas': [
                {'agent_id': 'agent_X', 'timestamp': 1.0, 'performance': 0.8},
                {'agent_id': 'agent_X', 'timestamp': 2.0, 'performance': 0.85},
                {'agent_id': 'agent_X', 'timestamp': 3.0, 'performance': 0.9}
            ]
        }

        policies = mock_chromadb_backend.get_agent_policies("agent_X")

        assert len(policies) == 3
        assert all(p.agent_id == "agent_X" for p in policies)

    def test_get_agent_policies_with_limit(self, mock_chromadb_backend):
        """Test getting agent policies with limit."""
        mock_chromadb_backend.collection.get.return_value = {
            'ids': [],
            'embeddings': [],
            'metadatas': []
        }

        mock_chromadb_backend.get_agent_policies("agent_Y", limit=50)

        call_args = mock_chromadb_backend.collection.get.call_args
        assert call_args[1]['limit'] == 50

    def test_get_agent_policies_error_handling(self, mock_chromadb_backend):
        """Test get agent policies error handling."""
        mock_chromadb_backend.collection.get.side_effect = Exception("Error")

        policies = mock_chromadb_backend.get_agent_policies("agent_error")

        assert len(policies) == 0


class TestChromaDBBackendUpdateOperations:
    """Test update and delete operations."""

    @pytest.fixture
    def mock_chromadb_backend(self):
        """Create ChromaDB backend with mocked collection."""
        with patch('src.connectors.vector_db.chromadb'):
            backend = ChromaDBBackend()
            backend.collection = MagicMock()
            backend.is_initialized = True
            yield backend

    def test_update_policy_metadata(self, mock_chromadb_backend):
        """Test updating policy metadata."""
        new_metadata = {"version": 2, "updated": True}

        result = mock_chromadb_backend.update_policy_metadata(
            "policy_001",
            new_metadata
        )

        assert result is True
        mock_chromadb_backend.collection.update.assert_called_once_with(
            ids=["policy_001"],
            metadatas=[new_metadata]
        )

    def test_update_policy_metadata_error_handling(self, mock_chromadb_backend):
        """Test update metadata error handling."""
        mock_chromadb_backend.collection.update.side_effect = Exception("Update error")

        result = mock_chromadb_backend.update_policy_metadata("policy_x", {})

        assert result is False

    def test_delete_policy(self, mock_chromadb_backend):
        """Test deleting a single policy."""
        result = mock_chromadb_backend.delete_policy("policy_to_delete")

        assert result is True
        mock_chromadb_backend.collection.delete.assert_called_once_with(
            ids=["policy_to_delete"]
        )

    def test_delete_policy_error_handling(self, mock_chromadb_backend):
        """Test delete policy error handling."""
        mock_chromadb_backend.collection.delete.side_effect = Exception("Delete error")

        result = mock_chromadb_backend.delete_policy("policy_error")

        assert result is False

    def test_delete_agent_policies(self, mock_chromadb_backend):
        """Test deleting all policies for an agent."""
        # Mock getting policy IDs
        mock_chromadb_backend.collection.get.return_value = {
            'ids': ['policy_1', 'policy_2', 'policy_3']
        }

        count = mock_chromadb_backend.delete_agent_policies("agent_to_delete")

        assert count == 3
        mock_chromadb_backend.collection.delete.assert_called_once_with(
            ids=['policy_1', 'policy_2', 'policy_3']
        )

    def test_delete_agent_policies_no_policies(self, mock_chromadb_backend):
        """Test deleting agent with no policies."""
        mock_chromadb_backend.collection.get.return_value = {'ids': []}

        count = mock_chromadb_backend.delete_agent_policies("agent_empty")

        assert count == 0

    def test_delete_agent_policies_error_handling(self, mock_chromadb_backend):
        """Test delete agent policies error handling."""
        mock_chromadb_backend.collection.get.side_effect = Exception("Error")

        count = mock_chromadb_backend.delete_agent_policies("agent_error")

        assert count == 0


class TestChromaDBBackendClusteringOperations:
    """Test clustering and pattern detection."""

    @pytest.fixture
    def mock_chromadb_backend(self):
        """Create ChromaDB backend with mocked collection."""
        with patch('src.connectors.vector_db.chromadb'):
            backend = ChromaDBBackend()
            backend.collection = MagicMock()
            backend.is_initialized = True
            yield backend

    @patch('src.connectors.vector_db.KMeans')
    def test_cluster_policies(self, mock_kmeans, mock_chromadb_backend):
        """Test clustering policies."""
        # Mock embeddings
        mock_chromadb_backend.collection.get.return_value = {
            'ids': ['p1', 'p2', 'p3', 'p4', 'p5'],
            'embeddings': [
                np.random.rand(128).tolist() for _ in range(5)
            ]
        }

        # Mock KMeans
        mock_kmeans_instance = MagicMock()
        mock_kmeans.return_value = mock_kmeans_instance
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 0, 1, 1, 2])

        clusters = mock_chromadb_backend.cluster_policies(num_clusters=3)

        assert len(clusters) == 3
        assert len(clusters[0]) == 2  # p1, p2
        assert len(clusters[1]) == 2  # p3, p4
        assert len(clusters[2]) == 1  # p5

    @patch('src.connectors.vector_db.KMeans')
    def test_cluster_policies_with_filter(self, mock_kmeans, mock_chromadb_backend):
        """Test clustering with filter criteria."""
        mock_chromadb_backend.collection.get.return_value = {
            'ids': ['p1', 'p2'],
            'embeddings': [np.random.rand(128).tolist() for _ in range(2)]
        }

        mock_kmeans_instance = MagicMock()
        mock_kmeans.return_value = mock_kmeans_instance
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 0])

        filter_criteria = {"agent_id": "agent_specific"}
        clusters = mock_chromadb_backend.cluster_policies(
            num_clusters=5,
            filter_criteria=filter_criteria
        )

        # Verify filter was passed
        call_args = mock_chromadb_backend.collection.get.call_args
        assert call_args[1]['where'] == filter_criteria

    def test_cluster_policies_no_data(self, mock_chromadb_backend):
        """Test clustering with no data."""
        mock_chromadb_backend.collection.get.return_value = {
            'ids': [],
            'embeddings': []
        }

        clusters = mock_chromadb_backend.cluster_policies()

        assert len(clusters) == 0

    def test_cluster_policies_without_sklearn(self, mock_chromadb_backend):
        """Test clustering without scikit-learn installed."""
        mock_chromadb_backend.collection.get.return_value = {
            'ids': ['p1', 'p2'],
            'embeddings': [np.random.rand(128).tolist() for _ in range(2)]
        }

        with patch('src.connectors.vector_db.KMeans', side_effect=ImportError):
            clusters = mock_chromadb_backend.cluster_policies()

        assert len(clusters) == 0

    def test_find_policy_patterns(self, mock_chromadb_backend):
        """Test finding recurring patterns."""
        with patch.object(mock_chromadb_backend, 'cluster_policies') as mock_cluster:
            mock_cluster.return_value = {
                0: ['p1', 'p2', 'p3', 'p4', 'p5', 'p6'],  # 6 policies
                1: ['p7', 'p8'],  # 2 policies
                2: ['p9', 'p10', 'p11', 'p12', 'p13']  # 5 policies
            }

            patterns = mock_chromadb_backend.find_policy_patterns(
                min_cluster_size=5
            )

            # Should only return clusters with >= 5 policies
            assert len(patterns) == 2  # Cluster 0 (6) and Cluster 2 (5)
            assert patterns[0]['policy_count'] == 6
            assert patterns[1]['policy_count'] == 5

    def test_find_policy_patterns_no_patterns(self, mock_chromadb_backend):
        """Test finding patterns when none meet criteria."""
        with patch.object(mock_chromadb_backend, 'cluster_policies') as mock_cluster:
            mock_cluster.return_value = {
                0: ['p1', 'p2'],
                1: ['p3']
            }

            patterns = mock_chromadb_backend.find_policy_patterns(
                min_cluster_size=5
            )

            assert len(patterns) == 0


class TestChromaDBBackendCollectionManagement:
    """Test collection management operations."""

    @pytest.fixture
    def mock_chromadb_backend(self):
        """Create ChromaDB backend with mocked collection."""
        with patch('src.connectors.vector_db.chromadb'):
            backend = ChromaDBBackend()
            backend.client = MagicMock()
            backend.collection = MagicMock()
            backend.is_initialized = True
            yield backend

    def test_get_collection_stats(self, mock_chromadb_backend):
        """Test getting collection statistics."""
        mock_chromadb_backend.collection.count.return_value = 150

        stats = mock_chromadb_backend.get_collection_stats()

        assert stats['collection_name'] == "mae_policies"
        assert stats['total_policies'] == 150
        assert stats['embedding_dim'] == 128
        assert stats['backend'] == "ChromaDB"

    def test_get_collection_stats_error_handling(self, mock_chromadb_backend):
        """Test stats error handling."""
        mock_chromadb_backend.collection.count.side_effect = Exception("Stats error")

        stats = mock_chromadb_backend.get_collection_stats()

        assert len(stats) == 0

    def test_clear_collection(self, mock_chromadb_backend):
        """Test clearing all embeddings from collection."""
        result = mock_chromadb_backend.clear_collection()

        assert result is True
        mock_chromadb_backend.client.delete_collection.assert_called_once_with(
            name="mae_policies"
        )
        mock_chromadb_backend.client.create_collection.assert_called_once()

    def test_clear_collection_error_handling(self, mock_chromadb_backend):
        """Test clear collection error handling."""
        mock_chromadb_backend.client.delete_collection.side_effect = Exception("Clear error")

        result = mock_chromadb_backend.clear_collection()

        assert result is False

    def test_close(self, mock_chromadb_backend):
        """Test closing the connection."""
        mock_chromadb_backend.close()

        assert mock_chromadb_backend.is_initialized is False


class TestCreateVectorDB:
    """Test factory function for creating vector DB backends."""

    def test_create_chromadb_backend(self):
        """Test creating ChromaDB backend."""
        backend = create_vector_db(backend="chromadb")

        assert isinstance(backend, ChromaDBBackend)

    def test_create_chromadb_with_custom_params(self):
        """Test creating ChromaDB with custom parameters."""
        backend = create_vector_db(
            backend="chromadb",
            collection_name="custom",
            embedding_dim=256
        )

        assert backend.collection_name == "custom"
        assert backend.embedding_dim == 256

    def test_create_unknown_backend(self):
        """Test creating unknown backend raises error."""
        with pytest.raises(ValueError, match="Unknown vector DB backend"):
            create_vector_db(backend="unknown")

    def test_create_unimplemented_backends(self):
        """Test that unimplemented backends raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Milvus backend not yet implemented"):
            create_vector_db(backend="milvus")

        with pytest.raises(NotImplementedError, match="Qdrant backend not yet implemented"):
            create_vector_db(backend="qdrant")


@pytest.mark.integration
@pytest.mark.requires_chromadb
class TestChromaDBIntegration:
    """
    Integration tests requiring actual ChromaDB installation.

    Run with: pytest -m requires_chromadb
    """

    @pytest.fixture
    def real_chromadb_backend(self):
        """Create real ChromaDB backend for integration tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                backend = ChromaDBBackend(
                    collection_name="test_collection",
                    persist_directory=tmpdir
                )
                backend.initialize()
                yield backend
                backend.close()
            except ImportError:
                pytest.skip("ChromaDB not installed")
            except Exception:
                pytest.skip("ChromaDB not available")

    def test_real_add_and_search(self, real_chromadb_backend):
        """Test actual add and search operations."""
        # Add some policies
        for i in range(5):
            embedding = np.random.rand(128)
            real_chromadb_backend.add_policy_embedding(
                policy_id=f"policy_{i}",
                agent_id="agent_test",
                embedding=embedding,
                metadata={"index": i},
                performance=0.5 + i * 0.1
            )

        # Search for similar policies
        query = np.random.rand(128)
        results = real_chromadb_backend.search_similar_policies(query, top_k=3)

        assert len(results) <= 3

    def test_real_batch_operations(self, real_chromadb_backend):
        """Test actual batch operations."""
        embeddings = [
            PolicyEmbedding(
                policy_id=f"batch_policy_{i}",
                agent_id=f"agent_{i % 2}",
                embedding=np.random.rand(128),
                metadata={"batch": True},
                timestamp=time.time(),
                performance=0.8
            )
            for i in range(10)
        ]

        count = real_chromadb_backend.add_policy_embeddings_batch(embeddings)

        assert count == 10

        # Verify they were added
        stats = real_chromadb_backend.get_collection_stats()
        assert stats['total_policies'] >= 10

    def test_real_delete_operations(self, real_chromadb_backend):
        """Test actual delete operations."""
        # Add policies
        for i in range(3):
            real_chromadb_backend.add_policy_embedding(
                f"delete_test_{i}",
                "agent_delete",
                np.random.rand(128)
            )

        # Delete one
        result = real_chromadb_backend.delete_policy("delete_test_0")
        assert result is True

        # Delete all for agent
        count = real_chromadb_backend.delete_agent_policies("agent_delete")
        assert count >= 2
