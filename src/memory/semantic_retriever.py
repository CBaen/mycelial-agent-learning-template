"""
Semantic Retriever for Episodic Memory

This module provides Vector DB integration for semantic memory retrieval,
enabling similarity-based experience lookup and counterfactual queries.

Key Features:
- Store episode embeddings in Vector DB
- Retrieve similar experiences by state/context
- Counterfactual queries ("What happened when I took action X?")
- Integration with TaskEmbedding from Big Rock 8

Biological Inspiration:
- Human semantic memory organization
- Context-based memory retrieval
- Associative memory networks

Based on:
- Lake et al. (2017): Building machines that learn and think like people
- Blundell et al. (2016): Model-Free Episodic Control
- Schaul et al. (2016): Prioritized Experience Replay

Performance Targets:
- >90% semantic retrieval accuracy
- <10ms retrieval time for k=5 neighbors
- Support 100K+ experiences
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class SemanticQuery:
    """Result of a semantic memory query."""
    experiences: List[Dict[str, Any]]
    distances: List[float]
    ids: List[str]
    query_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticRetriever:
    """
    Vector DB integration for semantic memory retrieval.

    Enables similarity-based experience lookup using Vector DB (ChromaDB).
    Supports counterfactual queries and context-aware memory retrieval.

    Example:
        >>> retriever = SemanticRetriever(vector_db_client)
        >>>
        >>> # Store experience
        >>> embedding = retriever.encode_experience(experience)
        >>> retriever.add(experience, embedding)
        >>>
        >>> # Retrieve similar experiences
        >>> similar = retriever.search_by_state(current_state, k=5)
        >>>
        >>> # Counterfactual query
        >>> counterfactual = retriever.get_counterfactual_experiences(
        ...     state=current_state,
        ...     action=2,
        ...     k=3
        ... )
    """

    def __init__(
        self,
        vector_db_client=None,
        embedding_dim: int = 128,
        collection_name: str = "episodic_memories",
        distance_metric: str = "cosine",
        use_fallback: bool = True
    ):
        """
        Initialize semantic retriever.

        Args:
            vector_db_client: ChromaDB client (or None for numpy fallback)
            embedding_dim: Dimension of experience embeddings
            collection_name: Name of vector DB collection
            distance_metric: Distance metric ('cosine', 'l2', 'ip')
            use_fallback: Use numpy fallback if Vector DB unavailable
        """
        self.embedding_dim = embedding_dim
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.use_fallback = use_fallback

        # Vector DB setup
        self.db = vector_db_client
        self.collection = None

        # Fallback storage (if Vector DB unavailable)
        self.fallback_embeddings: List[np.ndarray] = []
        self.fallback_experiences: List[Dict[str, Any]] = []
        self.fallback_ids: List[str] = []

        # Initialize collection
        if self.db is not None:
            try:
                self.collection = self._get_or_create_collection()
                self.using_fallback = False
            except Exception as e:
                if use_fallback:
                    print(f"Warning: Vector DB unavailable, using numpy fallback: {e}")
                    self.using_fallback = True
                else:
                    raise
        else:
            self.using_fallback = True

        # Statistics
        self.total_added = 0
        self.total_queries = 0
        self.total_query_time = 0.0

    def add(
        self,
        experience: Dict[str, Any],
        embedding: np.ndarray,
        experience_id: Optional[str] = None
    ) -> str:
        """
        Add experience to semantic memory.

        Args:
            experience: Experience dictionary with state, action, reward, etc.
            embedding: Pre-computed embedding vector
            experience_id: Optional custom ID (auto-generated if None)

        Returns:
            Experience ID
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} != {self.embedding_dim}"
            )

        # Generate ID if not provided
        if experience_id is None:
            timestamp = experience.get('timestamp', time.time())
            experience_id = f"exp_{int(timestamp * 1e9)}"

        # Extract metadata
        metadata = {
            'timestamp': float(experience.get('timestamp', time.time())),
            'reward': float(experience.get('reward', 0.0)),
            'done': bool(experience.get('done', False)),
        }

        # Store action if present
        if 'action' in experience:
            if isinstance(experience['action'], (int, np.integer)):
                metadata['action'] = int(experience['action'])
            elif isinstance(experience['action'], np.ndarray):
                metadata['action'] = int(experience['action'].item())

        if not self.using_fallback:
            # Store in Vector DB
            try:
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    metadatas=[metadata],
                    ids=[experience_id]
                )
            except Exception as e:
                if self.use_fallback:
                    print(f"Vector DB add failed, using fallback: {e}")
                    self._fallback_add(embedding, metadata, experience_id)
                else:
                    raise
        else:
            # Store in fallback
            self._fallback_add(embedding, metadata, experience_id)

        self.total_added += 1
        return experience_id

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> SemanticQuery:
        """
        Find k most similar experiences.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"reward": {"$gte": 0.5}})

        Returns:
            SemanticQuery with results
        """
        start_time = time.time()

        if not self.using_fallback:
            # Query Vector DB
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=k,
                    where=filter_metadata
                )

                experiences = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                ids = results.get('ids', [[]])[0]

            except Exception as e:
                if self.use_fallback:
                    print(f"Vector DB query failed, using fallback: {e}")
                    experiences, distances, ids = self._fallback_search(query_embedding, k, filter_metadata)
                else:
                    raise
        else:
            # Use fallback
            experiences, distances, ids = self._fallback_search(query_embedding, k, filter_metadata)

        query_time = time.time() - start_time

        # Update statistics
        self.total_queries += 1
        self.total_query_time += query_time

        return SemanticQuery(
            experiences=experiences,
            distances=distances,
            ids=ids,
            query_time=query_time,
            metadata={'k': k, 'filter': filter_metadata}
        )

    def search_by_state(
        self,
        state: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> SemanticQuery:
        """
        Find experiences in similar states.

        Args:
            state: State vector
            k: Number of results
            filter_metadata: Optional filters

        Returns:
            SemanticQuery with similar state experiences
        """
        # Encode state to embedding
        embedding = self.encode_state(state)
        return self.search(embedding, k, filter_metadata)

    def search_by_reward(
        self,
        min_reward: float,
        k: int = 10
    ) -> SemanticQuery:
        """
        Find high-reward experiences.

        Args:
            min_reward: Minimum reward threshold
            k: Number of results

        Returns:
            SemanticQuery with high-reward experiences
        """
        if not self.using_fallback:
            try:
                results = self.collection.get(
                    where={"reward": {"$gte": min_reward}},
                    limit=k
                )

                experiences = results.get('metadatas', [])
                ids = results.get('ids', [])
                distances = [0.0] * len(ids)  # No distance for filter query

                return SemanticQuery(
                    experiences=experiences,
                    distances=distances,
                    ids=ids,
                    query_time=0.0,
                    metadata={'min_reward': min_reward}
                )
            except Exception as e:
                if not self.use_fallback:
                    raise

        # Fallback implementation
        filtered = [
            (exp, exp_id)
            for exp, exp_id in zip(self.fallback_experiences, self.fallback_ids)
            if exp.get('reward', 0.0) >= min_reward
        ]

        # Sort by reward descending
        filtered.sort(key=lambda x: x[0].get('reward', 0.0), reverse=True)
        filtered = filtered[:k]

        experiences = [exp for exp, _ in filtered]
        ids = [exp_id for _, exp_id in filtered]
        distances = [0.0] * len(ids)

        return SemanticQuery(
            experiences=experiences,
            distances=distances,
            ids=ids,
            query_time=0.0,
            metadata={'min_reward': min_reward}
        )

    def get_counterfactual_experiences(
        self,
        state: np.ndarray,
        action: int,
        k: int = 5
    ) -> SemanticQuery:
        """
        Counterfactual query: "What happened when I took action X in similar states?"

        This enables the agent to reason about alternative actions in similar contexts.

        Args:
            state: Current/query state
            action: Action to query about
            k: Number of results

        Returns:
            SemanticQuery with counterfactual experiences
        """
        # Find similar states
        similar_states = self.search_by_state(state, k=k*3)  # Over-sample

        # Filter by action
        counterfactual_experiences = []
        counterfactual_distances = []
        counterfactual_ids = []

        for exp, dist, exp_id in zip(
            similar_states.experiences,
            similar_states.distances,
            similar_states.ids
        ):
            if exp.get('action') == action:
                counterfactual_experiences.append(exp)
                counterfactual_distances.append(dist)
                counterfactual_ids.append(exp_id)

                if len(counterfactual_experiences) >= k:
                    break

        return SemanticQuery(
            experiences=counterfactual_experiences,
            distances=counterfactual_distances,
            ids=counterfactual_ids,
            query_time=similar_states.query_time,
            metadata={'action': action, 'counterfactual': True}
        )

    def encode_experience(self, experience: Dict[str, Any]) -> np.ndarray:
        """
        Encode experience to embedding vector.

        Combines state, action, and context features into a single embedding.

        Args:
            experience: Experience dictionary

        Returns:
            Embedding vector (embedding_dim,)
        """
        features = []

        # State features
        state = experience.get('state', np.array([]))
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        features.append(state.flatten())

        # Action features (one-hot if discrete)
        action = experience.get('action')
        if action is not None:
            if isinstance(action, (int, np.integer)):
                # One-hot encoding (assume max 10 actions)
                action_vec = np.zeros(10)
                action_vec[int(action) % 10] = 1.0
                features.append(action_vec)
            elif isinstance(action, np.ndarray):
                features.append(action.flatten())

        # Reward feature
        reward = experience.get('reward', 0.0)
        features.append(np.array([reward]))

        # Concatenate all features
        embedding = np.concatenate(features)

        # Project to embedding_dim
        embedding = self._project_to_embedding_dim(embedding)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """
        Encode state to embedding vector.

        Args:
            state: State vector

        Returns:
            Embedding vector (embedding_dim,)
        """
        if isinstance(state, (list, tuple)):
            state = np.array(state)

        state = state.flatten()

        # Project to embedding_dim
        embedding = self._project_to_embedding_dim(state)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def clear(self):
        """Clear all memories from semantic storage."""
        if not self.using_fallback:
            try:
                self.db.delete_collection(self.collection_name)
                self.collection = self._get_or_create_collection()
            except Exception as e:
                if not self.use_fallback:
                    raise

        # Clear fallback
        self.fallback_embeddings = []
        self.fallback_experiences = []
        self.fallback_ids = []

        # Reset statistics
        self.total_added = 0
        self.total_queries = 0
        self.total_query_time = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get semantic memory statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.using_fallback and self.collection is not None:
            try:
                count = self.collection.count()
            except:
                count = len(self.fallback_embeddings)
        else:
            count = len(self.fallback_embeddings)

        stats = {
            'total_experiences': count,
            'embedding_dim': self.embedding_dim,
            'collection': self.collection_name,
            'using_fallback': self.using_fallback,
            'total_added': self.total_added,
            'total_queries': self.total_queries,
            'avg_query_time': self.total_query_time / max(self.total_queries, 1),
        }

        return stats

    def _get_or_create_collection(self):
        """Get or create Vector DB collection."""
        try:
            return self.db.get_collection(self.collection_name)
        except Exception:
            return self.db.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Episodic memory storage",
                    "embedding_dim": self.embedding_dim,
                    "distance_metric": self.distance_metric
                }
            )

    def _project_to_embedding_dim(self, vector: np.ndarray) -> np.ndarray:
        """
        Project vector to embedding dimension.

        Uses truncation/padding or random projection as needed.

        Args:
            vector: Input vector

        Returns:
            Vector of size embedding_dim
        """
        if len(vector) == self.embedding_dim:
            return vector
        elif len(vector) > self.embedding_dim:
            # Truncate or use random projection
            # For simplicity, truncate (could use PCA in production)
            return vector[:self.embedding_dim]
        else:
            # Pad with zeros
            return np.pad(vector, (0, self.embedding_dim - len(vector)))

    def _fallback_add(
        self,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        experience_id: str
    ):
        """Add to numpy fallback storage."""
        self.fallback_embeddings.append(embedding)
        self.fallback_experiences.append(metadata)
        self.fallback_ids.append(experience_id)

    def _fallback_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[float], List[str]]:
        """Search using numpy fallback (no Vector DB)."""
        if len(self.fallback_embeddings) == 0:
            return [], [], []

        # Compute distances
        embeddings_array = np.array(self.fallback_embeddings)

        if self.distance_metric == "cosine":
            # Cosine similarity = 1 - cosine distance
            similarities = embeddings_array @ query_embedding
            distances = 1.0 - similarities
        elif self.distance_metric == "l2":
            distances = np.linalg.norm(embeddings_array - query_embedding, axis=1)
        else:  # inner product
            distances = -(embeddings_array @ query_embedding)

        # Apply metadata filters
        valid_indices = list(range(len(self.fallback_embeddings)))

        if filter_metadata:
            filtered_indices = []
            for idx in valid_indices:
                exp = self.fallback_experiences[idx]
                if self._matches_filter(exp, filter_metadata):
                    filtered_indices.append(idx)
            valid_indices = filtered_indices

        if len(valid_indices) == 0:
            return [], [], []

        # Get top k
        valid_distances = [(distances[idx], idx) for idx in valid_indices]
        valid_distances.sort()  # Sort by distance ascending
        top_k = valid_distances[:k]

        # Extract results
        experiences = [self.fallback_experiences[idx] for _, idx in top_k]
        result_distances = [dist for dist, _ in top_k]
        ids = [self.fallback_ids[idx] for _, idx in top_k]

        return experiences, result_distances, ids

    def _matches_filter(
        self,
        experience: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """Check if experience matches metadata filters."""
        for key, condition in filters.items():
            if key not in experience:
                return False

            value = experience[key]

            if isinstance(condition, dict):
                # Comparison operators
                for op, threshold in condition.items():
                    if op == "$gte" and value < threshold:
                        return False
                    elif op == "$lte" and value > threshold:
                        return False
                    elif op == "$gt" and value <= threshold:
                        return False
                    elif op == "$lt" and value >= threshold:
                        return False
                    elif op == "$eq" and value != threshold:
                        return False
                    elif op == "$ne" and value == threshold:
                        return False
            else:
                # Direct equality
                if value != condition:
                    return False

        return True

    def __repr__(self) -> str:
        return (
            f"SemanticRetriever(collection='{self.collection_name}', "
            f"dim={self.embedding_dim}, using_fallback={self.using_fallback}, "
            f"experiences={self.get_statistics()['total_experiences']})"
        )
