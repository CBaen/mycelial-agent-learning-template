"""
Vector Database Interface for MAE Collective Memory

This module provides an abstract interface for managing agent policy embeddings
in a vector database. This enables semantic similarity search, pattern recognition,
and collective memory capabilities.

Supports multiple vector DB backends:
- ChromaDB (default, embedded)
- Milvus (scalable, distributed)
- Qdrant (high-performance)
- Weaviate (GraphQL-based)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np

# Optional dependencies - import at module level for easier mocking in tests
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None
    ChromaSettings = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

logger = logging.getLogger(__name__)


@dataclass
class PolicyEmbedding:
    """Container for a policy embedding."""
    policy_id: str
    agent_id: str
    embedding: np.ndarray  # Vector representation
    metadata: Dict[str, Any]
    timestamp: float
    performance: float


@dataclass
class SearchResult:
    """Container for a similarity search result."""
    policy_id: str
    agent_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class VectorDBInterface(ABC):
    """
    Abstract base class for vector database operations.

    This interface enables the MAE system to use different vector database
    backends for storing and retrieving policy embeddings.

    Key capabilities:
    - Store policy embeddings with metadata
    - Semantic similarity search
    - Pattern clustering and retrieval
    - Collective memory management
    """

    def __init__(
        self,
        collection_name: str = "mae_policies",
        embedding_dim: int = 128,
        **kwargs
    ):
        """
        Initialize the vector database interface.

        Args:
            collection_name: Name of the collection/index
            embedding_dim: Dimension of embedding vectors
            **kwargs: Additional backend-specific parameters
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.is_initialized = False

    @abstractmethod
    def initialize(self):
        """
        Initialize the vector database connection and create collections.

        This method should:
        - Establish connection to the vector DB
        - Create collection/index if it doesn't exist
        - Set up any required schemas
        """
        pass

    @abstractmethod
    def add_policy_embedding(
        self,
        policy_id: str,
        agent_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        performance: float = 0.0
    ) -> bool:
        """
        Add a policy embedding to the vector database.

        Args:
            policy_id: Unique identifier for the policy
            agent_id: ID of the agent that owns this policy
            embedding: Vector representation of the policy
            metadata: Additional metadata (policy version, timestamp, etc.)
            performance: Performance metric for this policy

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def add_policy_embeddings_batch(
        self,
        embeddings: List[PolicyEmbedding]
    ) -> int:
        """
        Add multiple policy embeddings in a batch.

        Args:
            embeddings: List of PolicyEmbedding objects

        Returns:
            Number of embeddings successfully added
        """
        pass

    @abstractmethod
    def search_similar_policies(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for policies similar to the query embedding.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_criteria: Optional metadata filters (e.g., agent_id, performance > threshold)

        Returns:
            List of SearchResult objects, sorted by similarity (descending)
        """
        pass

    @abstractmethod
    def get_policy_embedding(
        self,
        policy_id: str
    ) -> Optional[PolicyEmbedding]:
        """
        Retrieve a specific policy embedding by ID.

        Args:
            policy_id: ID of the policy to retrieve

        Returns:
            PolicyEmbedding if found, None otherwise
        """
        pass

    @abstractmethod
    def get_agent_policies(
        self,
        agent_id: str,
        limit: int = 100
    ) -> List[PolicyEmbedding]:
        """
        Retrieve all policies for a specific agent.

        Args:
            agent_id: Agent ID
            limit: Maximum number of policies to return

        Returns:
            List of PolicyEmbedding objects
        """
        pass

    @abstractmethod
    def update_policy_metadata(
        self,
        policy_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a policy.

        Args:
            policy_id: ID of the policy
            metadata: New metadata to merge/update

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def delete_policy(self, policy_id: str) -> bool:
        """
        Delete a policy embedding.

        Args:
            policy_id: ID of the policy to delete

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def delete_agent_policies(self, agent_id: str) -> int:
        """
        Delete all policies for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of policies deleted
        """
        pass

    @abstractmethod
    def cluster_policies(
        self,
        num_clusters: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[int, List[str]]:
        """
        Cluster policies into semantic groups.

        Args:
            num_clusters: Number of clusters to create
            filter_criteria: Optional metadata filters

        Returns:
            Dictionary mapping cluster_id to list of policy_ids
        """
        pass

    @abstractmethod
    def find_policy_patterns(
        self,
        min_cluster_size: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Identify recurring patterns in the policy space.

        Args:
            min_cluster_size: Minimum number of policies to form a pattern
            similarity_threshold: Minimum similarity to be considered same pattern

        Returns:
            List of pattern dictionaries with cluster info and representative policies
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection.

        Returns:
            Dictionary with stats (count, dimensions, etc.)
        """
        pass

    @abstractmethod
    def clear_collection(self) -> bool:
        """
        Clear all embeddings from the collection.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass


class ChromaDBBackend(VectorDBInterface):
    """
    ChromaDB implementation of VectorDBInterface.

    ChromaDB is an embedded vector database that's easy to set up and use.
    Perfect for development and small to medium deployments.
    """

    def __init__(
        self,
        collection_name: str = "mae_policies",
        embedding_dim: int = 128,
        persist_directory: str = "data/chromadb",
        **kwargs
    ):
        """
        Initialize ChromaDB backend.

        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
            persist_directory: Directory to persist database
            **kwargs: Additional ChromaDB parameters
        """
        super().__init__(collection_name, embedding_dim, **kwargs)

        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

        logger.info("ChromaDB backend initialized (persist: %s)", persist_directory)

    def initialize(self):
        """Initialize ChromaDB connection and collection."""
        if chromadb is None:
            logger.error("ChromaDB not installed. Run: pip install chromadb")
            raise ImportError("ChromaDB is not installed")

        try:
            # Create client
            self.client = chromadb.Client(ChromaSettings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            ))

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"embedding_dim": self.embedding_dim}
            )

            self.is_initialized = True
            logger.info("ChromaDB collection '%s' initialized", self.collection_name)

        except Exception as e:
            logger.error("Failed to initialize ChromaDB: %s", e)
            raise

    def add_policy_embedding(
        self,
        policy_id: str,
        agent_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        performance: float = 0.0
    ) -> bool:
        """Add a policy embedding to ChromaDB."""
        if not self.is_initialized:
            self.initialize()

        try:
            meta = metadata or {}
            meta.update({
                "agent_id": agent_id,
                "performance": performance
            })

            self.collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[meta],
                ids=[policy_id]
            )

            logger.debug("Added policy %s to ChromaDB", policy_id)
            return True

        except Exception as e:
            logger.error("Error adding policy to ChromaDB: %s", e)
            return False

    def add_policy_embeddings_batch(
        self,
        embeddings: List[PolicyEmbedding]
    ) -> int:
        """Add multiple policy embeddings in batch."""
        if not self.is_initialized:
            self.initialize()

        try:
            ids = [e.policy_id for e in embeddings]
            vectors = [e.embedding.tolist() for e in embeddings]
            metadatas = []

            for e in embeddings:
                meta = e.metadata.copy()
                meta.update({
                    "agent_id": e.agent_id,
                    "performance": e.performance,
                    "timestamp": e.timestamp
                })
                metadatas.append(meta)

            self.collection.add(
                embeddings=vectors,
                metadatas=metadatas,
                ids=ids
            )

            logger.info("Added %d policies to ChromaDB", len(embeddings))
            return len(embeddings)

        except Exception as e:
            logger.error("Error adding batch to ChromaDB: %s", e)
            return 0

    def search_similar_policies(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar policies in ChromaDB."""
        if not self.is_initialized:
            self.initialize()

        try:
            where = filter_criteria if filter_criteria else None

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where
            )

            search_results = []
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    search_results.append(SearchResult(
                        policy_id=results['ids'][0][i],
                        agent_id=results['metadatas'][0][i].get('agent_id', 'unknown'),
                        similarity_score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                        metadata=results['metadatas'][0][i]
                    ))

            return search_results

        except Exception as e:
            logger.error("Error searching ChromaDB: %s", e)
            return []

    def get_policy_embedding(self, policy_id: str) -> Optional[PolicyEmbedding]:
        """Retrieve a specific policy by ID."""
        if not self.is_initialized:
            self.initialize()

        try:
            results = self.collection.get(ids=[policy_id], include=["embeddings", "metadatas"])

            if results['ids']:
                return PolicyEmbedding(
                    policy_id=results['ids'][0],
                    agent_id=results['metadatas'][0].get('agent_id', 'unknown'),
                    embedding=np.array(results['embeddings'][0]),
                    metadata=results['metadatas'][0],
                    timestamp=results['metadatas'][0].get('timestamp', 0.0),
                    performance=results['metadatas'][0].get('performance', 0.0)
                )

            return None

        except Exception as e:
            logger.error("Error getting policy from ChromaDB: %s", e)
            return None

    def get_agent_policies(self, agent_id: str, limit: int = 100) -> List[PolicyEmbedding]:
        """Retrieve all policies for an agent."""
        if not self.is_initialized:
            self.initialize()

        try:
            results = self.collection.get(
                where={"agent_id": agent_id},
                limit=limit,
                include=["embeddings", "metadatas"]
            )

            policies = []
            for i in range(len(results['ids'])):
                policies.append(PolicyEmbedding(
                    policy_id=results['ids'][i],
                    agent_id=agent_id,
                    embedding=np.array(results['embeddings'][i]),
                    metadata=results['metadatas'][i],
                    timestamp=results['metadatas'][i].get('timestamp', 0.0),
                    performance=results['metadatas'][i].get('performance', 0.0)
                ))

            return policies

        except Exception as e:
            logger.error("Error getting agent policies from ChromaDB: %s", e)
            return []

    def update_policy_metadata(self, policy_id: str, metadata: Dict[str, Any]) -> bool:
        """Update policy metadata."""
        if not self.is_initialized:
            self.initialize()

        try:
            self.collection.update(
                ids=[policy_id],
                metadatas=[metadata]
            )
            return True

        except Exception as e:
            logger.error("Error updating policy metadata: %s", e)
            return False

    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy."""
        if not self.is_initialized:
            self.initialize()

        try:
            self.collection.delete(ids=[policy_id])
            return True

        except Exception as e:
            logger.error("Error deleting policy: %s", e)
            return False

    def delete_agent_policies(self, agent_id: str) -> int:
        """Delete all policies for an agent."""
        if not self.is_initialized:
            self.initialize()

        try:
            # Get all policy IDs for this agent
            results = self.collection.get(where={"agent_id": agent_id})
            policy_ids = results['ids']

            if policy_ids:
                self.collection.delete(ids=policy_ids)

            return len(policy_ids)

        except Exception as e:
            logger.error("Error deleting agent policies: %s", e)
            return 0

    def cluster_policies(
        self,
        num_clusters: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[int, List[str]]:
        """Cluster policies using K-means."""
        if not self.is_initialized:
            self.initialize()

        if KMeans is None:
            logger.error("scikit-learn not installed. Run: pip install scikit-learn")
            return {}

        try:
            # Get all embeddings
            results = self.collection.get(
                where=filter_criteria,
                include=["embeddings"]
            )

            if not results['ids']:
                return {}

            embeddings = np.array(results['embeddings'])
            policy_ids = results['ids']

            # Perform clustering
            kmeans = KMeans(n_clusters=min(num_clusters, len(embeddings)), random_state=42)
            labels = kmeans.fit_predict(embeddings)

            # Group by cluster
            clusters = {}
            for policy_id, label in zip(policy_ids, labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(policy_id)

            return clusters

        except Exception as e:
            logger.error("Error clustering policies: %s", e)
            return {}

    def find_policy_patterns(
        self,
        min_cluster_size: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find recurring patterns in policies."""
        clusters = self.cluster_policies(num_clusters=20)

        patterns = []
        for cluster_id, policy_ids in clusters.items():
            if len(policy_ids) >= min_cluster_size:
                patterns.append({
                    "pattern_id": f"pattern_{cluster_id}",
                    "cluster_id": cluster_id,
                    "policy_count": len(policy_ids),
                    "policy_ids": policy_ids,
                    "representative": policy_ids[0]  # Use first as representative
                })

        return patterns

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.is_initialized:
            self.initialize()

        try:
            count = self.collection.count()

            return {
                "collection_name": self.collection_name,
                "total_policies": count,
                "embedding_dim": self.embedding_dim,
                "backend": "ChromaDB"
            }

        except Exception as e:
            logger.error("Error getting stats: %s", e)
            return {}

    def clear_collection(self) -> bool:
        """Clear all embeddings."""
        if not self.is_initialized:
            self.initialize()

        try:
            # Delete and recreate collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"embedding_dim": self.embedding_dim}
            )
            return True

        except Exception as e:
            logger.error("Error clearing collection: %s", e)
            return False

    def close(self):
        """Close connection."""
        # ChromaDB doesn't require explicit closing
        self.is_initialized = False
        logger.info("ChromaDB connection closed")


def create_vector_db(
    backend: str = "chromadb",
    **kwargs
) -> VectorDBInterface:
    """
    Factory function to create a vector database backend.

    Args:
        backend: Backend type ("chromadb", "milvus", "qdrant", etc.)
        **kwargs: Backend-specific parameters

    Returns:
        VectorDBInterface implementation
    """
    if backend.lower() == "chromadb":
        return ChromaDBBackend(**kwargs)
    elif backend.lower() == "milvus":
        # TODO: Implement MilvusBackend
        raise NotImplementedError("Milvus backend not yet implemented")
    elif backend.lower() == "qdrant":
        # TODO: Implement QdrantBackend
        raise NotImplementedError("Qdrant backend not yet implemented")
    else:
        raise ValueError(f"Unknown vector DB backend: {backend}")
