"""
GNN Message Propagator for MAE v3.0 (Big Rock 7)

Intelligent message routing using GNN-learned communication patterns.
Routes messages through optimal paths to reduce overhead.

Author: MAE Development Team
Date: 2025-11-12
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import time
from dataclasses import dataclass, field

from src.core.gnn_graph import AgentGraph, AgentNode
from src.core.gnn_message import GNNMessage, MessageEncoder
from src.core.gnn_layer import GNNLayer, MultiLayerGNN


@dataclass
class RoutingDecision:
    """
    Records a routing decision for analysis/debugging.

    Attributes:
        message_id: Message being routed
        sender_id: Original sender
        recipients: List of selected recipients
        scores: Relevance scores for each recipient
        hop_count: Number of hops from sender
        timestamp: When routing occurred
    """
    message_id: str
    sender_id: str
    recipients: List[str]
    scores: Dict[str, float] = field(default_factory=dict)
    hop_count: int = 0
    timestamp: float = field(default_factory=time.time)


class GNNMessagePropagator:
    """
    Routes messages through agent graph using GNN-based relevance scoring.

    Key features:
    - Multi-hop message propagation with TTL
    - Relevance-based recipient selection (top-k)
    - Cycle prevention (don't revisit nodes)
    - Performance tracking
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 3,
        aggregation: str = "mean",
        default_k: int = 3,
        enable_tracking: bool = True
    ):
        """
        Initialize message propagator.

        Args:
            embedding_dim: Dimension of embeddings
            num_layers: Number of GNN layers
            aggregation: Aggregation function
            default_k: Default number of recipients per hop
            enable_tracking: Enable routing decision tracking
        """
        self.embedding_dim = embedding_dim
        self.default_k = default_k
        self.enable_tracking = enable_tracking

        # GNN for computing relevance
        self.gnn = MultiLayerGNN(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            aggregation=aggregation
        )

        # Message encoder
        self.message_encoder = MessageEncoder(embedding_dim)

        # Routing history (for analysis)
        self.routing_history: List[RoutingDecision] = []

    def propagate(
        self,
        graph: AgentGraph,
        message: GNNMessage,
        max_hops: Optional[int] = None,
        k: Optional[int] = None
    ) -> List[str]:
        """
        Propagate message through graph using GNN routing.

        Args:
            graph: Agent communication graph
            message: Message to propagate
            max_hops: Maximum hops (uses message.ttl if None)
            k: Number of recipients per hop (uses default_k if None)

        Returns:
            List of all recipients (excluding sender)
        """
        if message.sender_id not in graph.nodes:
            return []

        max_hops = max_hops if max_hops is not None else message.ttl
        k = k if k is not None else self.default_k

        all_recipients = []
        current_hop_agents = [message.sender_id]
        visited = {message.sender_id}
        hop_count = 0

        while hop_count < max_hops and current_hop_agents:
            next_hop_agents = []

            for agent_id in current_hop_agents:
                # Get candidates (neighbors not yet visited)
                neighbors = graph.get_neighbors(agent_id, direction="out")
                candidates = [n for n in neighbors if n not in visited]

                if not candidates:
                    continue

                # Compute relevance scores
                scores = self._compute_relevance(
                    agent_id,
                    candidates,
                    message,
                    graph
                )

                # Select top-k most relevant
                selected = self._select_top_k(candidates, scores, k)

                # Record routing decision
                if self.enable_tracking:
                    self.routing_history.append(RoutingDecision(
                        message_id=message.message_id,
                        sender_id=agent_id,
                        recipients=selected,
                        scores={cand: scores[i] for i, cand in enumerate(candidates)},
                        hop_count=hop_count
                    ))

                # Add to recipients and next hop
                for recipient_id in selected:
                    if recipient_id not in visited:
                        all_recipients.append(recipient_id)
                        next_hop_agents.append(recipient_id)
                        visited.add(recipient_id)

            current_hop_agents = next_hop_agents
            hop_count += 1

        return all_recipients

    def propagate_targeted(
        self,
        graph: AgentGraph,
        message: GNNMessage,
        target_ids: List[str],
        max_hops: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Route message to specific targets using shortest weighted paths.

        Args:
            graph: Agent communication graph
            message: Message to route
            target_ids: List of target agent IDs
            max_hops: Maximum path length

        Returns:
            Dictionary mapping target_id to routing path (list of agent IDs)
        """
        if message.sender_id not in graph.nodes:
            return {}

        max_hops = max_hops if max_hops is not None else message.ttl

        paths = {}

        for target_id in target_ids:
            if target_id not in graph.nodes:
                continue

            # Find shortest weighted path
            path = self._find_shortest_path(
                graph,
                message.sender_id,
                target_id,
                max_hops
            )

            if path:
                paths[target_id] = path

        return paths

    def _compute_relevance(
        self,
        sender_id: str,
        candidate_ids: List[str],
        message: GNNMessage,
        graph: AgentGraph
    ) -> np.ndarray:
        """
        Compute relevance scores for candidate recipients.

        Relevance is based on:
        1. Message-agent compatibility (embedding similarity)
        2. Edge weight (learned communication value)
        3. Message priority

        Args:
            sender_id: Sending agent
            candidate_ids: List of candidate recipients
            message: Message being routed
            graph: Agent graph

        Returns:
            Array of relevance scores (one per candidate)
        """
        sender_node = graph.nodes[sender_id]

        # Encode message
        message_emb = self.message_encoder.encode(message, sender_node.embedding)

        scores = []

        for candidate_id in candidate_ids:
            candidate_node = graph.nodes[candidate_id]

            # 1. Message-agent compatibility (cosine similarity)
            compatibility = float(np.dot(message_emb, candidate_node.embedding))

            # 2. Edge weight (communication value)
            edge = graph.edges.get((sender_id, candidate_id))
            edge_weight = edge.weight if edge else 0.5

            # 3. Message priority boost
            priority_boost = 1.0 + message.priority

            # Combined score
            score = compatibility * edge_weight * priority_boost

            scores.append(score)

        return np.array(scores)

    def _select_top_k(
        self,
        candidates: List[str],
        scores: np.ndarray,
        k: int
    ) -> List[str]:
        """
        Select top-k candidates by score.

        Args:
            candidates: List of candidate IDs
            scores: Relevance scores
            k: Number to select

        Returns:
            List of top-k candidate IDs
        """
        if len(candidates) <= k:
            return candidates

        # Get indices of top-k scores
        top_k_indices = np.argsort(scores)[-k:][::-1]

        return [candidates[i] for i in top_k_indices]

    def _find_shortest_path(
        self,
        graph: AgentGraph,
        source_id: str,
        target_id: str,
        max_hops: int
    ) -> Optional[List[str]]:
        """
        Find shortest weighted path from source to target.

        Uses Dijkstra-like algorithm with edge weights.

        Args:
            graph: Agent graph
            source_id: Source agent
            target_id: Target agent
            max_hops: Maximum path length

        Returns:
            Path as list of agent IDs, or None if no path found
        """
        if source_id == target_id:
            return [source_id]

        # Priority queue: (cost, path)
        import heapq
        queue = [(0.0, [source_id])]
        visited = {source_id}
        best_cost = {source_id: 0.0}

        while queue:
            cost, path = heapq.heappop(queue)

            current_id = path[-1]

            # Check if reached target
            if current_id == target_id:
                return path

            # Check max hops
            if len(path) >= max_hops + 1:
                continue

            # Explore neighbors
            neighbors = graph.get_neighbors(current_id, direction="out")

            for neighbor_id in neighbors:
                if neighbor_id in path:
                    # Prevent cycles
                    continue

                edge = graph.edges.get((current_id, neighbor_id))
                edge_cost = 1.0 - edge.weight if edge else 1.0  # Lower weight = higher cost

                new_cost = cost + edge_cost
                new_path = path + [neighbor_id]

                # Only explore if better than previous best
                if neighbor_id not in best_cost or new_cost < best_cost[neighbor_id]:
                    best_cost[neighbor_id] = new_cost
                    heapq.heappush(queue, (new_cost, new_path))
                    visited.add(neighbor_id)

        return None  # No path found

    def get_routing_statistics(self) -> Dict[str, any]:
        """
        Get statistics about routing decisions.

        Returns:
            Dictionary with routing metrics
        """
        if not self.routing_history:
            return {
                'total_routing_decisions': 0,
                'avg_recipients_per_hop': 0.0,
                'avg_score': 0.0,
                'max_hop_count': 0
            }

        total_decisions = len(self.routing_history)
        total_recipients = sum(len(d.recipients) for d in self.routing_history)

        all_scores = []
        for decision in self.routing_history:
            all_scores.extend(decision.scores.values())

        max_hop = max(d.hop_count for d in self.routing_history)

        return {
            'total_routing_decisions': total_decisions,
            'avg_recipients_per_hop': round(total_recipients / total_decisions, 2),
            'avg_score': round(float(np.mean(all_scores)), 4) if all_scores else 0.0,
            'max_hop_count': max_hop,
            'total_recipients': total_recipients
        }

    def clear_history(self):
        """Clear routing history"""
        self.routing_history.clear()

    def __repr__(self) -> str:
        return (
            f"GNNMessagePropagator(dim={self.embedding_dim}, "
            f"layers={self.gnn.num_layers}, k={self.default_k})"
        )


class RoutingOptimizer:
    """
    Optimizes routing patterns based on historical outcomes.

    Analyzes which routes led to successful coordination and
    adjusts edge weights accordingly.
    """

    def __init__(self, learning_rate: float = 0.1):
        """
        Initialize routing optimizer.

        Args:
            learning_rate: Learning rate for edge weight updates
        """
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")

        self.learning_rate = learning_rate

        # Track outcomes: message_id -> (path, reward)
        self.outcome_buffer: Dict[str, Tuple[List[str], float]] = {}

    def record_outcome(
        self,
        message_id: str,
        path: List[str],
        reward: float
    ):
        """
        Record outcome of a message routing.

        Args:
            message_id: Message identifier
            path: Routing path taken
            reward: Outcome reward [-1, 1]
        """
        self.outcome_buffer[message_id] = (path, reward)

    def optimize_graph(
        self,
        graph: AgentGraph,
        min_samples: int = 10
    ) -> int:
        """
        Update graph edge weights based on recorded outcomes.

        Args:
            graph: Agent graph to optimize
            min_samples: Minimum outcomes needed before optimizing

        Returns:
            Number of edges updated
        """
        if len(self.outcome_buffer) < min_samples:
            return 0

        # Group outcomes by edge
        edge_outcomes: Dict[Tuple[str, str], List[float]] = {}

        for message_id, (path, reward) in self.outcome_buffer.items():
            # Extract edges from path
            for i in range(len(path) - 1):
                edge_key = (path[i], path[i + 1])
                if edge_key not in edge_outcomes:
                    edge_outcomes[edge_key] = []
                edge_outcomes[edge_key].append(reward)

        # Update edge weights
        updated_count = 0

        for edge_key, rewards in edge_outcomes.items():
            if edge_key in graph.edges:
                edge = graph.edges[edge_key]

                # Compute average reward for this edge
                avg_reward = np.mean(rewards)

                # Update edge weight
                edge.update_from_outcome(avg_reward, self.learning_rate)
                updated_count += 1

        # Clear buffer after optimization
        self.outcome_buffer.clear()

        return updated_count

    def get_statistics(self) -> Dict[str, any]:
        """Get optimizer statistics"""
        if not self.outcome_buffer:
            return {
                'buffered_outcomes': 0,
                'avg_reward': 0.0
            }

        rewards = [reward for _, reward in self.outcome_buffer.values()]

        return {
            'buffered_outcomes': len(self.outcome_buffer),
            'avg_reward': round(float(np.mean(rewards)), 4),
            'min_reward': round(float(np.min(rewards)), 4),
            'max_reward': round(float(np.max(rewards)), 4)
        }

    def __repr__(self) -> str:
        return f"RoutingOptimizer(lr={self.learning_rate}, buffered={len(self.outcome_buffer)})"
