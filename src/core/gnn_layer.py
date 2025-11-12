"""
GNN Layer with Attention for MAE v3.0 (Big Rock 7)

Implements Graph Neural Network message passing with multi-head attention
for learning optimal communication patterns.

Author: MAE Development Team
Date: 2025-11-12
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import threading
from dataclasses import dataclass

from src.core.gnn_graph import AgentGraph, AgentNode
from src.core.gnn_message import GNNMessage, MessageEncoder


@dataclass
class AttentionWeights:
    """
    Attention weights computed for a node.

    Attributes:
        node_id: Target node ID
        neighbor_weights: Dict mapping neighbor ID to attention weight
        total_attention: Sum of attention weights (should be ~1.0 after softmax)
    """
    node_id: str
    neighbor_weights: Dict[str, float]
    total_attention: float


class GNNLayer:
    """
    Single Graph Neural Network layer with multi-head attention.

    Implements message passing:
    1. For each node, compute attention over neighbors
    2. Aggregate neighbor messages weighted by attention
    3. Update node embeddings with aggregated information
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 4,
        aggregation: str = "mean",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Initialize GNN layer.

        Args:
            embedding_dim: Dimension of node embeddings
            num_heads: Number of attention heads
            aggregation: Aggregation function ("mean", "sum", "max")
            activation: Activation function ("relu", "tanh", "sigmoid", "none")
            dropout: Dropout rate [0, 1] for regularization
        """
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if aggregation not in ["mean", "sum", "max"]:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        if activation not in ["relu", "tanh", "sigmoid", "none"]:
            raise ValueError(f"Unknown activation: {activation}")

        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.aggregation = aggregation
        self.activation = activation
        self.dropout = dropout

        # Learnable parameters (initialized randomly)
        # In production, these would be trained via backpropagation
        self.W_message = self._initialize_weights((embedding_dim, embedding_dim))
        self.W_update = self._initialize_weights((embedding_dim, embedding_dim))
        self.attention_weights = self._initialize_weights((num_heads, embedding_dim))

        # Bias terms
        self.bias_message = np.zeros(embedding_dim)
        self.bias_update = np.zeros(embedding_dim)

        # Thread safety for parameter updates
        self.lock = threading.RLock()

    def forward(
        self,
        graph: AgentGraph,
        node_id: str,
        message_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, AttentionWeights]:
        """
        Forward pass for a single node.

        Args:
            graph: Agent communication graph
            node_id: Target node to update
            message_embeddings: Optional pre-computed message embeddings from neighbors

        Returns:
            Tuple of (updated_embedding, attention_weights)
        """
        with self.lock:
            if node_id not in graph.nodes:
                raise ValueError(f"Node {node_id} not in graph")

            node = graph.nodes[node_id]
            neighbors = graph.get_neighbors(node_id, direction="in")

            if not neighbors:
                # No neighbors, return current embedding
                return node.embedding.copy(), AttentionWeights(
                    node_id=node_id,
                    neighbor_weights={},
                    total_attention=0.0
                )

            # Get neighbor embeddings
            neighbor_nodes = [graph.nodes[n_id] for n_id in neighbors]

            # Compute attention weights
            attention = self._compute_attention(
                node.embedding,
                [n.embedding for n in neighbor_nodes],
                neighbors
            )

            # Aggregate neighbor messages with attention weighting
            if message_embeddings:
                # Use pre-computed message embeddings
                neighbor_embeds = [
                    message_embeddings.get(n_id, graph.nodes[n_id].embedding)
                    for n_id in neighbors
                ]
            else:
                # Use node embeddings directly
                neighbor_embeds = [n.embedding for n in neighbor_nodes]

            aggregated = self._weighted_aggregate(
                neighbor_embeds,
                attention.neighbor_weights,
                neighbors
            )

            # Transform aggregated message
            message = self._transform(aggregated, self.W_message, self.bias_message)

            # Update node embedding
            updated_embedding = self._update_embedding(node.embedding, message)

            return updated_embedding, attention

    def batch_forward(
        self,
        graph: AgentGraph,
        node_ids: Optional[List[str]] = None
    ) -> Dict[str, Tuple[np.ndarray, AttentionWeights]]:
        """
        Forward pass for multiple nodes (batched for efficiency).

        Args:
            graph: Agent communication graph
            node_ids: Optional list of nodes to update (default: all nodes)

        Returns:
            Dictionary mapping node_id to (updated_embedding, attention_weights)
        """
        if node_ids is None:
            node_ids = list(graph.nodes.keys())

        results = {}
        for node_id in node_ids:
            results[node_id] = self.forward(graph, node_id)

        return results

    def _compute_attention(
        self,
        node_embedding: np.ndarray,
        neighbor_embeddings: List[np.ndarray],
        neighbor_ids: List[str]
    ) -> AttentionWeights:
        """
        Compute multi-head attention weights.

        Uses dot-product attention:
        α_ij = softmax(LeakyReLU(a^T [W·h_i || W·h_j]))

        Args:
            node_embedding: Target node embedding
            neighbor_embeddings: List of neighbor embeddings
            neighbor_ids: List of neighbor IDs (for output)

        Returns:
            AttentionWeights object
        """
        if not neighbor_embeddings:
            return AttentionWeights(
                node_id="",
                neighbor_weights={},
                total_attention=0.0
            )

        # Compute attention for each head
        all_head_scores = []

        for head_idx in range(self.num_heads):
            head_weights = self.attention_weights[head_idx]

            # Compute attention scores for this head
            scores = []
            for neighbor_emb in neighbor_embeddings:
                # Concatenate node and neighbor embeddings
                combined = np.concatenate([node_embedding, neighbor_emb])

                # Project to attention dimension
                # Simplified: dot product with head weights
                score = np.dot(combined[:len(head_weights)], head_weights)

                # LeakyReLU activation
                score = self._leaky_relu(score)

                scores.append(score)

            all_head_scores.append(scores)

        # Average scores across heads
        avg_scores = np.mean(all_head_scores, axis=0)

        # Softmax to get attention weights
        attention_probs = self._softmax(avg_scores)

        # Create attention weight dictionary
        neighbor_weights = {
            neighbor_id: float(prob)
            for neighbor_id, prob in zip(neighbor_ids, attention_probs)
        }

        return AttentionWeights(
            node_id="",
            neighbor_weights=neighbor_weights,
            total_attention=float(np.sum(attention_probs))
        )

    def _weighted_aggregate(
        self,
        neighbor_embeddings: List[np.ndarray],
        attention_weights: Dict[str, float],
        neighbor_ids: List[str]
    ) -> np.ndarray:
        """
        Aggregate neighbor embeddings weighted by attention.

        Args:
            neighbor_embeddings: List of neighbor embeddings
            attention_weights: Attention weights from neighbors
            neighbor_ids: List of neighbor IDs

        Returns:
            Aggregated embedding
        """
        if not neighbor_embeddings:
            return np.zeros(self.embedding_dim)

        if self.aggregation == "mean":
            # Weighted mean
            aggregated = np.zeros(self.embedding_dim)
            total_weight = 0.0

            for neighbor_id, emb in zip(neighbor_ids, neighbor_embeddings):
                weight = attention_weights.get(neighbor_id, 0.0)
                aggregated += weight * emb
                total_weight += weight

            if total_weight > 0:
                aggregated = aggregated / total_weight

            return aggregated

        elif self.aggregation == "sum":
            # Weighted sum
            aggregated = np.zeros(self.embedding_dim)

            for neighbor_id, emb in zip(neighbor_ids, neighbor_embeddings):
                weight = attention_weights.get(neighbor_id, 0.0)
                aggregated += weight * emb

            return aggregated

        elif self.aggregation == "max":
            # Max pooling over neighbors (attention used for gating)
            aggregated = np.zeros(self.embedding_dim)

            for neighbor_id, emb in zip(neighbor_ids, neighbor_embeddings):
                weight = attention_weights.get(neighbor_id, 0.0)
                # Gate embeddings by attention
                gated_emb = weight * emb
                aggregated = np.maximum(aggregated, gated_emb)

            return aggregated

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def _transform(
        self,
        embedding: np.ndarray,
        weight_matrix: np.ndarray,
        bias: np.ndarray
    ) -> np.ndarray:
        """
        Apply linear transformation: W·h + b

        Args:
            embedding: Input embedding
            weight_matrix: Weight matrix
            bias: Bias vector

        Returns:
            Transformed embedding
        """
        return np.dot(weight_matrix, embedding) + bias

    def _update_embedding(
        self,
        current_embedding: np.ndarray,
        message: np.ndarray
    ) -> np.ndarray:
        """
        Update node embedding with aggregated message.

        Uses residual connection and activation:
        h' = activation(W_update · message + h)

        Args:
            current_embedding: Current node embedding
            message: Aggregated message from neighbors

        Returns:
            Updated embedding
        """
        # Transform message
        transformed = self._transform(message, self.W_update, self.bias_update)

        # Residual connection
        updated = transformed + current_embedding

        # Apply activation
        updated = self._apply_activation(updated)

        # Dropout (if enabled)
        if self.dropout > 0:
            updated = self._apply_dropout(updated)

        # Normalize to unit length
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm

        return updated

    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "none":
            return x
        else:
            return x

    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout (randomly zero elements)"""
        if self.dropout == 0:
            return x

        mask = np.random.rand(len(x)) > self.dropout
        return x * mask / (1 - self.dropout)  # Scale to maintain expected value

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for stability
        return exp_x / np.sum(exp_x)

    def _leaky_relu(self, x: float, alpha: float = 0.01) -> float:
        """Leaky ReLU activation"""
        return x if x > 0 else alpha * x

    def _initialize_weights(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Initialize weight matrix with Xavier/Glorot initialization.

        Args:
            shape: Weight matrix shape

        Returns:
            Initialized weight matrix
        """
        # Xavier initialization: weights ~ N(0, 2/(n_in + n_out))
        if len(shape) == 2:
            fan_in, fan_out = shape
            std = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            std = 0.1

        return np.random.randn(*shape) * std

    def get_attention_statistics(
        self,
        attention_results: Dict[str, AttentionWeights]
    ) -> Dict[str, any]:
        """
        Compute statistics over attention weights.

        Args:
            attention_results: Attention weights from batch_forward

        Returns:
            Dictionary with attention statistics
        """
        if not attention_results:
            return {
                'avg_neighbors_attended': 0,
                'max_attention_weight': 0.0,
                'min_attention_weight': 0.0,
                'attention_entropy': 0.0
            }

        all_weights = []
        all_neighbor_counts = []

        for node_id, (_, attention) in attention_results.items():
            weights = list(attention.neighbor_weights.values())
            if weights:
                all_weights.extend(weights)
                all_neighbor_counts.append(len(weights))

        if not all_weights:
            return {
                'avg_neighbors_attended': 0,
                'max_attention_weight': 0.0,
                'min_attention_weight': 0.0,
                'attention_entropy': 0.0
            }

        # Compute entropy (measure of attention concentration)
        entropies = []
        for _, (_, attention) in attention_results.items():
            weights = np.array(list(attention.neighbor_weights.values()))
            if len(weights) > 0:
                # Normalize to probabilities
                probs = weights / np.sum(weights)
                # Compute entropy
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)

        return {
            'avg_neighbors_attended': round(np.mean(all_neighbor_counts), 2),
            'max_attention_weight': round(float(np.max(all_weights)), 4),
            'min_attention_weight': round(float(np.min(all_weights)), 4),
            'avg_attention_weight': round(float(np.mean(all_weights)), 4),
            'attention_entropy': round(float(np.mean(entropies)), 4) if entropies else 0.0
        }

    def __repr__(self) -> str:
        return (
            f"GNNLayer(dim={self.embedding_dim}, heads={self.num_heads}, "
            f"agg={self.aggregation}, act={self.activation})"
        )


class MultiLayerGNN:
    """
    Multi-layer GNN for deeper message propagation.

    Stacks multiple GNN layers to enable information flow
    over longer paths in the graph.
    """

    def __init__(
        self,
        num_layers: int = 3,
        embedding_dim: int = 64,
        num_heads: int = 4,
        aggregation: str = "mean"
    ):
        """
        Initialize multi-layer GNN.

        Args:
            num_layers: Number of GNN layers
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads per layer
            aggregation: Aggregation function
        """
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Create layers
        self.layers = [
            GNNLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                aggregation=aggregation
            )
            for _ in range(num_layers)
        ]

    def forward(
        self,
        graph: AgentGraph,
        node_ids: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Multi-layer forward pass.

        Args:
            graph: Agent communication graph
            node_ids: Optional list of nodes to update

        Returns:
            Dictionary mapping node_id to final updated embedding
        """
        if node_ids is None:
            node_ids = list(graph.nodes.keys())

        # Store intermediate embeddings
        current_embeddings = {
            node_id: graph.nodes[node_id].embedding.copy()
            for node_id in node_ids
        }

        # Process each layer
        for layer_idx, layer in enumerate(self.layers):
            # Update embeddings for this layer
            next_embeddings = {}

            for node_id in node_ids:
                # Temporarily update graph with current embeddings
                original_emb = graph.nodes[node_id].embedding.copy()
                graph.nodes[node_id].embedding = current_embeddings[node_id]

                # Forward pass
                updated_emb, _ = layer.forward(graph, node_id)
                next_embeddings[node_id] = updated_emb

                # Restore original
                graph.nodes[node_id].embedding = original_emb

            current_embeddings = next_embeddings

        return current_embeddings

    def __repr__(self) -> str:
        return f"MultiLayerGNN(layers={self.num_layers}, dim={self.embedding_dim})"
