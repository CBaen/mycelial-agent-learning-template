"""
SimpleFRL - Reference Implementation of Federated Reinforcement Learning

This module provides a simple, production-ready implementation of the
FederatedLearningEngine interface. It uses basic federated averaging
and trust-based peer selection.

Use this as:
1. A starting point for your own FRL implementation
2. A reference for understanding the FRL interface
3. A functional engine for prototyping and testing

For production systems with advanced requirements (differential privacy,
Byzantine-robust aggregation, etc.), extend this implementation or
create your own.
"""

import sys
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.frl_base import (
    FederatedLearningEngine,
    PolicyUpdateStrategy,
    AggregationMethod
)
from src.connectors.redis_client import RedisClient

logger = logging.getLogger(__name__)


class SimpleFRL(FederatedLearningEngine):
    """
    Simple Federated Reinforcement Learning Engine.

    Features:
    - Federated averaging for policy aggregation
    - Trust-based peer selection
    - Byzantine resistance through validation
    - Redis-backed peer discovery and communication

    This implementation uses:
    - Weighted average aggregation (based on trust scores)
    - Performance-based peer selection
    - Simple anomaly detection for validation
    """

    def __init__(
        self,
        agent_id: str,
        redis_client: RedisClient,
        policy_update_strategy: PolicyUpdateStrategy = PolicyUpdateStrategy.PERFORMANCE_BASED,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        max_peers: int = 10,
        trust_threshold: float = 0.5,
        share_frequency: int = 10,
        validation_enabled: bool = True
    ):
        """
        Initialize SimpleFRL engine.

        Args:
            agent_id: Unique identifier for this agent
            redis_client: Redis client for communication
            policy_update_strategy: How to select peers for sharing
            aggregation_method: How to combine peer policies
            max_peers: Maximum number of peers to maintain
            trust_threshold: Minimum trust score to accept updates
            share_frequency: Share policy every N steps
            validation_enabled: Enable policy validation
        """
        self.agent_id = agent_id
        self.redis_client = redis_client
        self.policy_update_strategy = policy_update_strategy
        self.aggregation_method = aggregation_method
        self.max_peers = max_peers
        self.trust_threshold = trust_threshold
        self.share_frequency = share_frequency
        self.validation_enabled = validation_enabled

        # Peer management
        self.connected_peers: Set[str] = set()
        self.trust_scores: Dict[str, float] = {}  # peer_id -> trust score
        self.peer_metadata: Dict[str, Dict[str, Any]] = {}  # peer_id -> metadata
        self.peer_history: Dict[str, List[float]] = defaultdict(list)  # peer_id -> performance history

        # Policy management
        self.local_policy_version: int = 0
        self.update_count: int = 0
        self.share_count: int = 0

        # Statistics
        self.total_shares: int = 0
        self.total_receives: int = 0
        self.total_validations_passed: int = 0
        self.total_validations_failed: int = 0

        # Redis channels
        self.policy_channel = f"frl:policies:{agent_id}"
        self.discovery_channel = "frl:discovery"

        logger.info("SimpleFRL initialized for agent %s", agent_id)

    # =========================================================================
    # Policy Sharing (Outbound)
    # =========================================================================

    def share_policy(
        self,
        policy_state: Dict[str, Any],
        performance_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Alias for share_policy_update with performance_score support."""
        metadata = metadata or {}
        if performance_score is not None:
            metadata["performance"] = performance_score
        return self.share_policy_update(policy_state, metadata)

    def share_policy_update(
        self,
        policy_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Share local policy update with selected peers.

        Args:
            policy_state: Current policy parameters
            metadata: Additional info (performance, version, etc.)

        Returns:
            Number of peers that received the update
        """
        # Select peers based on strategy
        target_peers = self._select_peers_for_sharing()

        if not target_peers:
            logger.debug("No peers available for sharing")
            return 0

        # Increment version
        self.local_policy_version += 1

        # Prepare update package
        update_package = {
            "agent_id": self.agent_id,
            "policy_state": policy_state,
            "version": self.local_policy_version,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        # Share with each peer
        shared_count = 0
        for peer_id in target_peers:
            try:
                # Publish to peer's channel
                channel = f"frl:policies:{peer_id}"
                self.redis_client.publish(channel, update_package)
                shared_count += 1

            except Exception as e:
                logger.error("Failed to share with peer %s: %s", peer_id, e)

        self.share_count += 1
        self.total_shares += shared_count

        logger.debug("Shared policy v%d with %d peers",
                    self.local_policy_version, shared_count)

        return shared_count

    def _select_peers_for_sharing(self) -> List[str]:
        """Select peers to share policy with based on strategy."""
        if not self.connected_peers:
            return []

        if self.policy_update_strategy == PolicyUpdateStrategy.BROADCAST:
            # Share with all connected peers
            return list(self.connected_peers)

        elif self.policy_update_strategy == PolicyUpdateStrategy.SELECTIVE:
            # Share with trusted peers only
            return [
                peer_id for peer_id in self.connected_peers
                if self.trust_scores.get(peer_id, 0.5) >= self.trust_threshold
            ]

        elif self.policy_update_strategy == PolicyUpdateStrategy.PERFORMANCE_BASED:
            # Share with top-performing peers
            peer_performance = []
            for peer_id in self.connected_peers:
                history = self.peer_history.get(peer_id, [])
                avg_performance = np.mean(history) if history else 0.5
                peer_performance.append((peer_id, avg_performance))

            # Sort by performance and take top N
            peer_performance.sort(key=lambda x: x[1], reverse=True)
            top_peers = [p[0] for p in peer_performance[:self.max_peers // 2]]
            return top_peers

        elif self.policy_update_strategy == PolicyUpdateStrategy.NEIGHBORHOOD:
            # Share with nearby peers (geographically or in state space)
            # For simplicity, share with peers with similar IDs
            return list(self.connected_peers)[:self.max_peers // 2]

        elif self.policy_update_strategy == PolicyUpdateStrategy.RANDOM:
            # Share with random subset
            peers_list = list(self.connected_peers)
            np.random.shuffle(peers_list)
            return peers_list[:self.max_peers // 2]

        return []

    def select_peers_for_sharing(
        self,
        num_peers: Optional[int] = None
    ) -> List[str]:
        """
        Select which peers to share policy updates with (abstract method implementation).

        Args:
            num_peers: Optional limit on number of peers to select

        Returns:
            List of peer IDs to share with
        """
        selected = self._select_peers_for_sharing()

        if num_peers is not None and len(selected) > num_peers:
            selected = selected[:num_peers]

        return selected

    # =========================================================================
    # Policy Aggregation (Inbound)
    # =========================================================================

    def aggregate_policy_updates(
        self,
        local_policy: Dict[str, Any],
        peer_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate local policy with updates from peers.

        Args:
            local_policy: Current local policy
            peer_updates: List of policy updates from peers

        Returns:
            Aggregated policy
        """
        if not peer_updates:
            return local_policy

        # Filter out invalid updates
        valid_updates = []
        for update in peer_updates:
            peer_id = update.get("agent_id")
            if self.validation_enabled and peer_id:
                if self.validate_policy_update(peer_id, update):
                    valid_updates.append(update)
                    self.total_validations_passed += 1
                else:
                    self.total_validations_failed += 1
            else:
                valid_updates.append(update)

        if not valid_updates:
            logger.debug("No valid peer updates to aggregate")
            return local_policy

        self.total_receives += len(valid_updates)

        # Aggregate based on method
        if self.aggregation_method == AggregationMethod.FEDERATED_AVERAGING:
            return self._simple_average_aggregation(local_policy, valid_updates)

        elif self.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_aggregation(local_policy, valid_updates)

        elif self.aggregation_method == AggregationMethod.MEDIAN:
            return self._median_aggregation(local_policy, valid_updates)

        elif self.aggregation_method == AggregationMethod.TRUST_WEIGHTED:
            return self._trust_weighted_aggregation(local_policy, valid_updates)

        else:
            # Default to simple average
            return self._simple_average_aggregation(local_policy, valid_updates)

    def _simple_average_aggregation(
        self,
        local_policy: Dict[str, Any],
        peer_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simple federated averaging."""
        aggregated = local_policy.copy()

        # Extract policy states
        all_policies = [local_policy] + [u["policy_state"] for u in peer_updates]
        num_policies = len(all_policies)

        # Average each parameter
        for key in local_policy.keys():
            values = []
            for policy in all_policies:
                if key in policy:
                    value = policy[key]
                    if isinstance(value, (int, float, np.ndarray)):
                        values.append(value)

            if values:
                if isinstance(values[0], np.ndarray):
                    aggregated[key] = np.mean(values, axis=0)
                else:
                    aggregated[key] = np.mean(values)

        return aggregated

    def _weighted_average_aggregation(
        self,
        local_policy: Dict[str, Any],
        peer_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Weighted average based on peer performance."""
        aggregated = local_policy.copy()

        # Get weights (performance-based)
        weights = [1.0]  # Local policy weight
        for update in peer_updates:
            peer_id = update.get("agent_id")
            performance = update.get("metadata", {}).get("performance", 0.5)
            self.peer_history[peer_id].append(performance)
            weights.append(performance)

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        # Extract policies
        all_policies = [local_policy] + [u["policy_state"] for u in peer_updates]

        # Weighted average for each parameter
        for key in local_policy.keys():
            values = []
            valid_weights = []

            for i, policy in enumerate(all_policies):
                if key in policy:
                    value = policy[key]
                    if isinstance(value, (int, float, np.ndarray)):
                        values.append(value)
                        valid_weights.append(weights[i])

            if values:
                valid_weights = np.array(valid_weights)
                valid_weights = valid_weights / valid_weights.sum()

                if isinstance(values[0], np.ndarray):
                    aggregated[key] = sum(w * v for w, v in zip(valid_weights, values))
                else:
                    aggregated[key] = sum(w * v for w, v in zip(valid_weights, values))

        return aggregated

    def _median_aggregation(
        self,
        local_policy: Dict[str, Any],
        peer_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Median aggregation (Byzantine-resistant)."""
        aggregated = local_policy.copy()

        # Extract policies
        all_policies = [local_policy] + [u["policy_state"] for u in peer_updates]

        # Median for each parameter
        for key in local_policy.keys():
            values = []
            for policy in all_policies:
                if key in policy:
                    value = policy[key]
                    if isinstance(value, (int, float, np.ndarray)):
                        values.append(value)

            if values:
                if isinstance(values[0], np.ndarray):
                    aggregated[key] = np.median(values, axis=0)
                else:
                    aggregated[key] = np.median(values)

        return aggregated

    def _trust_weighted_aggregation(
        self,
        local_policy: Dict[str, Any],
        peer_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Trust-weighted aggregation."""
        aggregated = local_policy.copy()

        # Get trust weights
        weights = [1.0]  # Local policy
        for update in peer_updates:
            peer_id = update.get("agent_id")
            trust = self.trust_scores.get(peer_id, 0.5)
            weights.append(trust)

        weights = np.array(weights)
        weights = weights / weights.sum()

        # Extract policies
        all_policies = [local_policy] + [u["policy_state"] for u in peer_updates]

        # Weighted average
        for key in local_policy.keys():
            values = []
            valid_weights = []

            for i, policy in enumerate(all_policies):
                if key in policy:
                    value = policy[key]
                    if isinstance(value, (int, float, np.ndarray)):
                        values.append(value)
                        valid_weights.append(weights[i])

            if values:
                valid_weights = np.array(valid_weights)
                valid_weights = valid_weights / valid_weights.sum()

                if isinstance(values[0], np.ndarray):
                    aggregated[key] = sum(w * v for w, v in zip(valid_weights, values))
                else:
                    aggregated[key] = sum(w * v for w, v in zip(valid_weights, values))

        return aggregated

    # =========================================================================
    # Policy Validation
    # =========================================================================

    def validate_policy_update(
        self,
        peer_id: str,
        policy_update: Dict[str, Any]
    ) -> bool:
        """
        Validate policy update from peer.

        Simple validation checks:
        1. Structure validation
        2. Value range checks
        3. Trust score check
        4. Anomaly detection
        """
        # Check trust score
        trust = self.trust_scores.get(peer_id, 0.5)
        if trust < self.trust_threshold:
            logger.debug("Rejected update from %s (low trust: %.2f)", peer_id, trust)
            return False

        # Structure validation
        if "policy_state" not in policy_update:
            logger.warning("Invalid policy update structure from %s", peer_id)
            return False

        policy_state = policy_update["policy_state"]

        # Simple anomaly detection: check for extreme values
        for key, value in policy_state.items():
            if isinstance(value, (int, float)):
                if abs(value) > 1e6:  # Suspiciously large value
                    logger.warning("Extreme value detected in update from %s", peer_id)
                    return False
            elif isinstance(value, np.ndarray):
                if np.any(np.abs(value) > 1e6):
                    logger.warning("Extreme array values from %s", peer_id)
                    return False

        return True

    def update_trust_score(
        self,
        peer_id: str,
        feedback: float,
        weight: float = 0.1
    ):
        """
        Update trust score for a peer based on feedback.

        Args:
            peer_id: Peer to update
            feedback: Feedback value (0.0 to 1.0)
            weight: Learning rate for trust update
        """
        current_trust = self.trust_scores.get(peer_id, 0.5)
        new_trust = (1 - weight) * current_trust + weight * feedback
        self.trust_scores[peer_id] = np.clip(new_trust, 0.0, 1.0)

        logger.debug("Updated trust for %s: %.3f -> %.3f",
                    peer_id, current_trust, new_trust)

    def update_peer_trust(
        self,
        peer_id: str,
        update_quality: float,
        performance_delta: float
    ):
        """
        Update trust score for a peer based on update quality (abstract method implementation).

        Args:
            peer_id: ID of the peer to update trust for
            update_quality: Quality metric for the received update (0.0 to 1.0)
            performance_delta: Change in performance after applying update
        """
        # Combine update quality and performance delta into feedback
        # Positive performance delta increases trust, negative decreases it
        feedback = update_quality * 0.5 + (0.5 if performance_delta > 0 else 0.0)
        feedback = np.clip(feedback, 0.0, 1.0)

        self.update_trust_score(peer_id, feedback)

    # =========================================================================
    # Peer Management
    # =========================================================================

    def discover_peers(
        self,
        search_criteria: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Discover peers in the network.

        Args:
            search_criteria: Optional filters (team_id, specialization, etc.)

        Returns:
            List of discovered peer IDs
        """
        try:
            # Announce presence
            announcement = {
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "metadata": search_criteria or {}
            }
            self.redis_client.publish(self.discovery_channel, announcement)

            # Query Redis for active agents
            # In a real implementation, would query a peer registry
            # For now, simulate with Redis keys
            pattern = "agent:state:*"
            keys = self.redis_client.client.keys(pattern)

            discovered = []
            for key in keys:
                # Extract agent ID from key
                agent_id = key.split(":")[-1]
                if agent_id != self.agent_id:
                    discovered.append(agent_id)

            logger.info("Discovered %d peers", len(discovered))
            return discovered

        except Exception as e:
            logger.error("Peer discovery failed: %s", e)
            return []

    def connect_to_peer(
        self,
        peer_id: str,
        connection_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Connect to a specific peer."""
        if peer_id in self.connected_peers:
            logger.debug("Already connected to %s", peer_id)
            return True

        if len(self.connected_peers) >= self.max_peers:
            logger.warning("Max peers reached, cannot connect to %s", peer_id)
            return False

        try:
            # Add to connected peers
            self.connected_peers.add(peer_id)
            self.trust_scores[peer_id] = 0.5  # Neutral trust initially
            self.peer_metadata[peer_id] = connection_metadata or {}

            logger.info("Connected to peer %s", peer_id)
            return True

        except Exception as e:
            logger.error("Failed to connect to %s: %s", peer_id, e)
            return False

    def disconnect_from_peer(
        self,
        peer_id: str,
        reason: Optional[str] = None
    ):
        """Disconnect from a peer."""
        if peer_id not in self.connected_peers:
            return

        self.connected_peers.remove(peer_id)
        logger.info("Disconnected from %s (reason: %s)", peer_id, reason or "manual")

    def get_peer_reputation(self, peer_id: str) -> float:
        """Get reputation score for a peer."""
        return self.trust_scores.get(peer_id, 0.5)

    def get_connected_peers(self) -> List[str]:
        """Get list of currently connected peers."""
        return list(self.connected_peers)

    @property
    def peers(self) -> Set[str]:
        """Alias for connected_peers (for test compatibility)."""
        return self.connected_peers

    def aggregate_policies(
        self,
        policies: Dict[str, Dict[str, Any]],
        trust_scores: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate policies from multiple peers (test-compatible API).

        Args:
            policies: Dict of agent_id -> policy_state
            trust_scores: Dict of agent_id -> trust_score

        Returns:
            Aggregated policy or None if no valid policies
        """
        if not policies:
            return None

        # Filter by trust threshold
        valid_policies = []
        for agent_id, policy in policies.items():
            trust = trust_scores.get(agent_id, 0.5)
            if trust >= self.trust_threshold:
                # Convert to update format
                update = {
                    "agent_id": agent_id,
                    "policy_state": policy,
                    "metadata": {"performance": trust}
                }
                valid_policies.append(update)

        if not valid_policies:
            return None

        # Use a dummy local policy (all zeros or first policy)
        if valid_policies:
            local_policy = valid_policies[0]["policy_state"].copy()
        else:
            return None

        # Aggregate
        return self.aggregate_policy_updates(local_policy, valid_policies)

    # =========================================================================
    # Additional Abstract Method Implementations
    # =========================================================================

    def receive_policy_updates(
        self,
        timeout_ms: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Receive policy updates from peers (abstract method implementation).

        Args:
            timeout_ms: Optional timeout in milliseconds for blocking receive

        Returns:
            List of policy updates from peers
        """
        # In SimpleFRL, we use push-based updates via Redis pub/sub
        # This method returns buffered updates
        updates = self.update_buffer.copy()
        self.update_buffer.clear()
        return updates

    def compute_update_weight(self, peer_id: str) -> float:
        """
        Compute the weight/importance for a peer's update during aggregation.

        Args:
            peer_id: ID of the peer

        Returns:
            Weight value (typically 0.0 to 1.0)
        """
        # Weight based on trust and performance
        trust = self.trust_scores.get(peer_id, 0.5)

        # Get average performance from history
        history = self.peer_history.get(peer_id, [])
        performance = np.mean(history) if history else 0.5

        # Combine trust and performance
        weight = 0.6 * trust + 0.4 * performance

        return float(np.clip(weight, 0.0, 1.0))

    def get_policy_divergence(
        self,
        policy_a: Dict[str, Any],
        policy_b: Dict[str, Any]
    ) -> float:
        """
        Measure divergence between two policies.

        Args:
            policy_a: First policy to compare
            policy_b: Second policy to compare

        Returns:
            Divergence metric (higher = more different)
        """
        # Compute L2 norm of differences
        divergence = 0.0
        compared_keys = 0

        for key in policy_a.keys():
            if key in policy_b:
                val_a = policy_a[key]
                val_b = policy_b[key]

                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    divergence += (val_a - val_b) ** 2
                    compared_keys += 1
                elif isinstance(val_a, np.ndarray) and isinstance(val_b, np.ndarray):
                    if val_a.shape == val_b.shape:
                        divergence += np.sum((val_a - val_b) ** 2)
                        compared_keys += 1

        if compared_keys == 0:
            return 0.0

        return float(np.sqrt(divergence / compared_keys))

    def prune_policy_updates(
        self,
        updates: List[Dict[str, Any]],
        max_updates: int
    ) -> List[Dict[str, Any]]:
        """
        Prune/filter policy updates to keep only the most valuable ones.

        Args:
            updates: List of policy updates
            max_updates: Maximum number of updates to keep

        Returns:
            Filtered list of most valuable updates
        """
        if len(updates) <= max_updates:
            return updates

        # Score each update based on trust and performance
        scored_updates = []
        for update in updates:
            peer_id = update.get("agent_id")
            if peer_id:
                weight = self.compute_update_weight(peer_id)
                scored_updates.append((weight, update))
            else:
                scored_updates.append((0.5, update))

        # Sort by score (descending) and take top max_updates
        scored_updates.sort(key=lambda x: x[0], reverse=True)

        return [update for _, update in scored_updates[:max_updates]]

    # =========================================================================
    # Statistics and Management
    # =========================================================================

    def get_sharing_statistics(self) -> Dict[str, Any]:
        """Get statistics about policy sharing."""
        return {
            "total_shares": self.total_shares,
            "total_receives": self.total_receives,
            "share_count": self.share_count,
            "connected_peers": len(self.connected_peers),
            "validations_passed": self.total_validations_passed,
            "validations_failed": self.total_validations_failed,
            "local_policy_version": self.local_policy_version,
            "avg_trust_score": np.mean(list(self.trust_scores.values())) if self.trust_scores else 0.5
        }

    def reset_sharing_history(self):
        """Reset sharing statistics."""
        self.total_shares = 0
        self.total_receives = 0
        self.share_count = 0
        self.total_validations_passed = 0
        self.total_validations_failed = 0

    def set_policy_update_strategy(self, strategy: PolicyUpdateStrategy):
        """Change the policy update strategy."""
        self.policy_update_strategy = strategy
        logger.info("Policy update strategy changed to %s", strategy)

    def set_aggregation_method(self, method: AggregationMethod):
        """Change the aggregation method."""
        self.aggregation_method = method
        logger.info("Aggregation method changed to %s", method)

    def shutdown(self):
        """Shutdown the FRL engine."""
        logger.info("SimpleFRL shutting down for agent %s", self.agent_id)
        self.connected_peers.clear()


# =========================================================================
# Factory Function
# =========================================================================

def create_frl_engine(
    agent_id: str,
    redis_client: RedisClient,
    config: Optional[Dict[str, Any]] = None
) -> SimpleFRL:
    """
    Factory function to create SimpleFRL engine.

    Args:
        agent_id: Agent identifier
        redis_client: Redis client instance
        config: Optional configuration dictionary

    Returns:
        SimpleFRL engine instance
    """
    config = config or {}

    return SimpleFRL(
        agent_id=agent_id,
        redis_client=redis_client,
        policy_update_strategy=config.get("policy_update_strategy", PolicyUpdateStrategy.PERFORMANCE_BASED),
        aggregation_method=config.get("aggregation_method", AggregationMethod.WEIGHTED_AVERAGE),
        max_peers=config.get("max_peers", 10),
        trust_threshold=config.get("trust_threshold", 0.5),
        share_frequency=config.get("share_frequency", 10),
        validation_enabled=config.get("validation_enabled", True)
    )
