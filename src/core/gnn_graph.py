"""
GNN Agent Graph for MAE v3.0 (Big Rock 7)

Provides graph-based representation of agent communication networks
with dynamic edge management and efficient neighbor queries.

Author: MAE Development Team
Date: 2025-11-12
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional, Any
from collections import defaultdict
import threading
import numpy as np
import time


@dataclass
class AgentNode:
    """
    Represents an agent in the communication graph.

    Attributes:
        agent_id: Unique agent identifier
        embedding: d-dimensional feature vector representing agent state/capabilities
        agent_type: Type of agent ("builder", "specialist", "risk_manager")
        capabilities: Set of capability strings
        level: Agent experience level
        position: Spatial position from stigmergy layer
        last_update: Timestamp of last update
        metadata: Additional agent metadata
    """
    agent_id: str
    embedding: np.ndarray
    agent_type: str
    capabilities: Set[str]
    level: int
    position: Tuple[float, ...]
    last_update: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_embedding(self, new_embedding: np.ndarray):
        """Update agent embedding with new state"""
        if len(new_embedding) != len(self.embedding):
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {len(self.embedding)}, got {len(new_embedding)}"
            )
        self.embedding = new_embedding
        self.last_update = time.time()

    def similarity(self, other: 'AgentNode') -> float:
        """Compute cosine similarity with another agent"""
        dot_product = np.dot(self.embedding, other.embedding)
        norm_product = np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding)

        if norm_product == 0:
            return 0.0

        return dot_product / norm_product

    def distance_to(self, other: 'AgentNode') -> float:
        """Compute spatial distance to another agent"""
        if len(self.position) != len(other.position):
            raise ValueError("Position dimension mismatch")

        return float(np.sqrt(sum(
            (a - b) ** 2 for a, b in zip(self.position, other.position)
        )))


@dataclass
class CommunicationEdge:
    """
    Represents a communication link between two agents.

    Attributes:
        source_id: Source agent ID
        target_id: Target agent ID
        weight: Learned importance [0, 1], higher = more valuable connection
        message_count: Number of messages sent on this edge
        last_message_time: Timestamp of last message
        success_rate: Average success/reward from this edge [0, 1]
        edge_type: Type of relationship ("collaboration", "hierarchy", "proximity", "learned")
        metadata: Additional edge metadata
    """
    source_id: str
    target_id: str
    weight: float = 1.0
    message_count: int = 0
    last_message_time: float = 0.0
    success_rate: float = 0.5
    edge_type: str = "learned"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate edge attributes"""
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Edge weight must be in [0, 1], got {self.weight}")
        if not 0 <= self.success_rate <= 1:
            raise ValueError(f"Success rate must be in [0, 1], got {self.success_rate}")

    def update_from_outcome(self, reward: float, learning_rate: float = 0.1):
        """
        Update edge weight based on communication outcome.

        Args:
            reward: Outcome reward [-1, 1] where 1 = very successful
            learning_rate: Learning rate for exponential moving average
        """
        # Normalize reward to [0, 1]
        normalized_reward = (reward + 1) / 2

        # Exponential moving average update
        self.weight = (1 - learning_rate) * self.weight + learning_rate * normalized_reward
        self.success_rate = (1 - learning_rate) * self.success_rate + learning_rate * normalized_reward

        # Clamp to [0, 1]
        self.weight = max(0.0, min(1.0, self.weight))
        self.success_rate = max(0.0, min(1.0, self.success_rate))

    def record_message(self):
        """Record that a message was sent on this edge"""
        self.message_count += 1
        self.last_message_time = time.time()

    def is_active(self, timeout: float = 300.0) -> bool:
        """Check if edge has been used recently"""
        if self.message_count == 0:
            return False

        return (time.time() - self.last_message_time) < timeout

    def is_strong(self, threshold: float = 0.7) -> bool:
        """Check if edge is strong (high weight and success rate)"""
        return self.weight >= threshold and self.success_rate >= threshold


class AgentGraph:
    """
    Dynamic communication graph for multi-agent coordination.

    Maintains agent nodes and communication edges with efficient
    queries for neighbors, subgraphs, and graph statistics.

    Thread-safe for concurrent agent operations.
    """

    def __init__(self, embedding_dim: int = 64):
        """
        Initialize agent graph.

        Args:
            embedding_dim: Dimension of agent embeddings
        """
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.embedding_dim = embedding_dim

        # Core graph structures
        self.nodes: Dict[str, AgentNode] = {}
        self.edges: Dict[Tuple[str, str], CommunicationEdge] = {}

        # Adjacency lists for fast neighbor queries
        self.adjacency_out: Dict[str, Set[str]] = defaultdict(set)  # Outgoing edges
        self.adjacency_in: Dict[str, Set[str]] = defaultdict(set)   # Incoming edges

        # Thread safety
        self.lock = threading.RLock()

    def add_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: Set[str],
        level: int,
        position: Tuple[float, ...],
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add agent to graph.

        Args:
            agent_id: Unique agent identifier
            agent_type: Agent type string
            capabilities: Set of capability strings
            level: Agent level
            position: Spatial position
            embedding: Optional pre-computed embedding
            metadata: Optional metadata dictionary

        Returns:
            True if agent was added, False if already exists
        """
        with self.lock:
            if agent_id in self.nodes:
                return False

            # Initialize embedding if not provided
            if embedding is None:
                embedding = self._initialize_embedding(agent_type, capabilities, level)
            elif len(embedding) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: "
                    f"expected {self.embedding_dim}, got {len(embedding)}"
                )

            self.nodes[agent_id] = AgentNode(
                agent_id=agent_id,
                embedding=embedding,
                agent_type=agent_type,
                capabilities=capabilities,
                level=level,
                position=position,
                metadata=metadata or {}
            )

            return True

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove agent and all its edges from graph.

        Args:
            agent_id: Agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        with self.lock:
            if agent_id not in self.nodes:
                return False

            # Remove all edges involving this agent
            edges_to_remove = [
                (src, tgt) for (src, tgt) in self.edges.keys()
                if src == agent_id or tgt == agent_id
            ]

            for src, tgt in edges_to_remove:
                self.remove_edge(src, tgt)

            # Remove node
            del self.nodes[agent_id]

            return True

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        edge_type: str = "learned",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add communication edge between agents.

        Args:
            source_id: Source agent
            target_id: Target agent
            weight: Initial edge weight [0, 1]
            edge_type: Edge type string
            metadata: Optional metadata

        Returns:
            True if edge was added, False if already exists or invalid
        """
        with self.lock:
            # Validate agents exist
            if source_id not in self.nodes or target_id not in self.nodes:
                raise ValueError(f"Both agents must exist in graph")

            # Check if edge already exists
            edge_key = (source_id, target_id)
            if edge_key in self.edges:
                return False

            # Create edge
            self.edges[edge_key] = CommunicationEdge(
                source_id=source_id,
                target_id=target_id,
                weight=weight,
                edge_type=edge_type,
                metadata=metadata or {}
            )

            # Update adjacency lists
            self.adjacency_out[source_id].add(target_id)
            self.adjacency_in[target_id].add(source_id)

            return True

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        """
        Remove edge from graph.

        Args:
            source_id: Source agent
            target_id: Target agent

        Returns:
            True if edge was removed, False if not found
        """
        with self.lock:
            edge_key = (source_id, target_id)
            if edge_key not in self.edges:
                return False

            # Remove edge
            del self.edges[edge_key]

            # Update adjacency lists
            self.adjacency_out[source_id].discard(target_id)
            self.adjacency_in[target_id].discard(source_id)

            # Clean up empty adjacency entries
            if not self.adjacency_out[source_id]:
                del self.adjacency_out[source_id]
            if not self.adjacency_in[target_id]:
                del self.adjacency_in[target_id]

            return True

    def update_edge_weight(
        self,
        source_id: str,
        target_id: str,
        weight: float
    ) -> bool:
        """
        Update edge weight directly.

        Args:
            source_id: Source agent
            target_id: Target agent
            weight: New weight [0, 1]

        Returns:
            True if updated, False if edge not found
        """
        with self.lock:
            edge = self.edges.get((source_id, target_id))
            if not edge:
                return False

            if not 0 <= weight <= 1:
                raise ValueError(f"Weight must be in [0, 1], got {weight}")

            edge.weight = weight
            return True

    def get_neighbors(
        self,
        agent_id: str,
        k: Optional[int] = None,
        direction: str = "out"
    ) -> List[str]:
        """
        Get neighboring agents.

        Args:
            agent_id: Query agent
            k: Optional limit on number of neighbors (returns top-k by weight)
            direction: "out" (outgoing), "in" (incoming), or "both"

        Returns:
            List of neighbor agent IDs
        """
        with self.lock:
            if agent_id not in self.nodes:
                return []

            # Collect neighbors based on direction
            if direction == "out":
                neighbors = list(self.adjacency_out.get(agent_id, set()))
            elif direction == "in":
                neighbors = list(self.adjacency_in.get(agent_id, set()))
            elif direction == "both":
                out_neighbors = self.adjacency_out.get(agent_id, set())
                in_neighbors = self.adjacency_in.get(agent_id, set())
                neighbors = list(out_neighbors | in_neighbors)
            else:
                raise ValueError(f"Invalid direction: {direction}")

            # Return all if k not specified
            if k is None or k >= len(neighbors):
                return neighbors

            # Sort by edge weight and return top-k
            if direction == "out":
                neighbor_weights = [
                    (neighbor, self.edges[(agent_id, neighbor)].weight)
                    for neighbor in neighbors
                ]
            elif direction == "in":
                neighbor_weights = [
                    (neighbor, self.edges[(neighbor, agent_id)].weight)
                    for neighbor in neighbors
                ]
            else:  # both
                neighbor_weights = []
                for neighbor in neighbors:
                    # Use max weight of incoming or outgoing edge
                    out_weight = self.edges.get((agent_id, neighbor))
                    in_weight = self.edges.get((neighbor, agent_id))
                    weight = max(
                        out_weight.weight if out_weight else 0,
                        in_weight.weight if in_weight else 0
                    )
                    neighbor_weights.append((neighbor, weight))

            # Sort descending by weight and take top-k
            neighbor_weights.sort(key=lambda x: x[1], reverse=True)
            return [neighbor for neighbor, _ in neighbor_weights[:k]]

    def get_subgraph(self, agent_ids: List[str]) -> 'AgentGraph':
        """
        Extract subgraph containing only specified agents.

        Args:
            agent_ids: List of agent IDs to include

        Returns:
            New AgentGraph with only specified nodes and edges between them
        """
        with self.lock:
            subgraph = AgentGraph(self.embedding_dim)

            agent_set = set(agent_ids)

            # Copy nodes
            for agent_id in agent_ids:
                if agent_id in self.nodes:
                    node = self.nodes[agent_id]
                    subgraph.add_agent(
                        agent_id=node.agent_id,
                        agent_type=node.agent_type,
                        capabilities=node.capabilities.copy(),
                        level=node.level,
                        position=node.position,
                        embedding=node.embedding.copy(),
                        metadata=node.metadata.copy()
                    )

            # Copy edges between included nodes
            for (src, tgt), edge in self.edges.items():
                if src in agent_set and tgt in agent_set:
                    subgraph.add_edge(
                        source_id=src,
                        target_id=tgt,
                        weight=edge.weight,
                        edge_type=edge.edge_type,
                        metadata=edge.metadata.copy()
                    )
                    # Copy edge statistics
                    subgraph.edges[(src, tgt)].message_count = edge.message_count
                    subgraph.edges[(src, tgt)].last_message_time = edge.last_message_time
                    subgraph.edges[(src, tgt)].success_rate = edge.success_rate

            return subgraph

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Dictionary with graph metrics
        """
        with self.lock:
            num_nodes = len(self.nodes)
            num_edges = len(self.edges)

            if num_nodes == 0:
                return {
                    'num_nodes': 0,
                    'num_edges': 0,
                    'avg_degree': 0.0,
                    'density': 0.0,
                    'avg_edge_weight': 0.0,
                    'avg_success_rate': 0.0
                }

            # Compute average degree
            total_degree = sum(len(neighbors) for neighbors in self.adjacency_out.values())
            avg_degree = total_degree / num_nodes if num_nodes > 0 else 0

            # Compute density (actual edges / possible edges)
            max_edges = num_nodes * (num_nodes - 1)  # Directed graph
            density = num_edges / max_edges if max_edges > 0 else 0

            # Compute average edge statistics
            if num_edges > 0:
                avg_weight = np.mean([edge.weight for edge in self.edges.values()])
                avg_success = np.mean([edge.success_rate for edge in self.edges.values()])
            else:
                avg_weight = 0.0
                avg_success = 0.0

            # Agent type distribution
            type_counts = {}
            for node in self.nodes.values():
                type_counts[node.agent_type] = type_counts.get(node.agent_type, 0) + 1

            # Edge type distribution
            edge_type_counts = {}
            for edge in self.edges.values():
                edge_type_counts[edge.edge_type] = edge_type_counts.get(edge.edge_type, 0) + 1

            return {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'avg_degree': round(avg_degree, 2),
                'density': round(density, 4),
                'avg_edge_weight': round(float(avg_weight), 3),
                'avg_success_rate': round(float(avg_success), 3),
                'agent_types': type_counts,
                'edge_types': edge_type_counts
            }

    def prune_weak_edges(self, weight_threshold: float = 0.1) -> int:
        """
        Remove edges with weight below threshold.

        Args:
            weight_threshold: Minimum weight to keep

        Returns:
            Number of edges removed
        """
        with self.lock:
            edges_to_remove = [
                (src, tgt) for (src, tgt), edge in self.edges.items()
                if edge.weight < weight_threshold
            ]

            for src, tgt in edges_to_remove:
                self.remove_edge(src, tgt)

            return len(edges_to_remove)

    def prune_inactive_edges(self, timeout: float = 300.0) -> int:
        """
        Remove edges that haven't been used recently.

        Args:
            timeout: Inactivity timeout in seconds

        Returns:
            Number of edges removed
        """
        with self.lock:
            edges_to_remove = [
                (src, tgt) for (src, tgt), edge in self.edges.items()
                if not edge.is_active(timeout)
            ]

            for src, tgt in edges_to_remove:
                self.remove_edge(src, tgt)

            return len(edges_to_remove)

    def strengthen_successful_edges(self, boost_factor: float = 1.1) -> int:
        """
        Boost weights of successful edges.

        Args:
            boost_factor: Multiplicative boost (e.g., 1.1 = 10% increase)

        Returns:
            Number of edges strengthened
        """
        with self.lock:
            count = 0
            for edge in self.edges.values():
                if edge.is_strong():
                    edge.weight = min(1.0, edge.weight * boost_factor)
                    count += 1

            return count

    def _initialize_embedding(
        self,
        agent_type: str,
        capabilities: Set[str],
        level: int
    ) -> np.ndarray:
        """
        Initialize agent embedding from metadata.

        Simple hashing-based initialization. In production, this could
        be replaced with learned embeddings.
        """
        # Use deterministic random seed based on agent properties
        seed = hash((agent_type, frozenset(capabilities), level)) % (2**32)
        rng = np.random.RandomState(seed)

        # Generate embedding
        embedding = rng.randn(self.embedding_dim)

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def __len__(self) -> int:
        """Return number of nodes"""
        return len(self.nodes)

    def __contains__(self, agent_id: str) -> bool:
        """Check if agent is in graph"""
        return agent_id in self.nodes

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"AgentGraph(nodes={stats['num_nodes']}, edges={stats['num_edges']}, "
            f"avg_degree={stats['avg_degree']:.2f}, density={stats['density']:.4f})"
        )
