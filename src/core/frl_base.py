"""
Federated Reinforcement Learning (FRL) Abstract Base Class

This module defines the abstract interface for federated learning in a
peer-to-peer "mycelial" network architecture, where agents share policy
updates in a decentralized manner without a central aggregator.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PolicyUpdateStrategy(Enum):
    """Strategies for selecting which peers to share updates with."""
    BROADCAST = "broadcast"  # Share with all connected peers
    SELECTIVE = "selective"  # Share with subset based on similarity/performance
    NEIGHBORHOOD = "neighborhood"  # Share with spatial/logical neighbors only
    PERFORMANCE_BASED = "performance_based"  # Share with top-performing peers
    RANDOM = "random"  # Share with random subset


class AggregationMethod(Enum):
    """Methods for aggregating received policy updates."""
    WEIGHTED_AVERAGE = "weighted_average"  # Weight by peer performance/trust
    MEDIAN = "median"  # Use median values (robust to outliers)
    FEDERATED_AVERAGING = "federated_averaging"  # Standard FedAvg
    TRUST_WEIGHTED = "trust_weighted"  # Weight by peer trust score
    GRADIENT_BASED = "gradient_based"  # Aggregate gradients instead of weights


class FederatedLearningEngine(ABC):
    """
    Abstract base class for Federated Reinforcement Learning.

    This engine enables agents to learn collaboratively in a decentralized
    P2P fashion, sharing policy updates through a mycelial network structure
    where knowledge flows organically between connected peers.

    Key Principles:
    - No central aggregator (pure P2P)
    - Asynchronous policy sharing
    - Privacy-preserving (only share policy updates, not raw experiences)
    - Resilient to Byzantine/adversarial peers
    - Adaptive peer selection based on trust and performance
    """

    def __init__(
        self,
        agent_id: str,
        redis_client: Any,
        update_strategy: PolicyUpdateStrategy = PolicyUpdateStrategy.SELECTIVE,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    ):
        """
        Initialize the Federated Learning Engine.

        Args:
            agent_id: Unique identifier for this agent
            redis_client: Redis client for P2P communication
            update_strategy: Strategy for selecting peers to share with
            aggregation_method: Method for aggregating received updates
        """
        self.agent_id = agent_id
        self.redis_client = redis_client
        self.update_strategy = update_strategy
        self.aggregation_method = aggregation_method

        # Track connected peers in the mycelial network
        self.connected_peers: Set[str] = set()

        # Trust scores for each peer (0.0 to 1.0)
        self.peer_trust_scores: Dict[str, float] = {}

        # Performance metrics for each peer
        self.peer_performance: Dict[str, float] = {}

        # Local policy version tracking
        self.policy_version: int = 0

        # Buffer for received policy updates
        self.update_buffer: List[Dict[str, Any]] = []

    @abstractmethod
    def share_policy_update(
        self,
        policy_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Share a local policy update with selected peers.

        Args:
            policy_state: The policy parameters/weights to share
            metadata: Optional metadata (performance metrics, version, etc.)

        Returns:
            Number of peers the update was shared with
        """
        pass

    @abstractmethod
    def receive_policy_updates(
        self,
        timeout_ms: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Receive policy updates from peers.

        Args:
            timeout_ms: Optional timeout in milliseconds for blocking receive

        Returns:
            List of policy updates from peers, each containing:
            - peer_id: ID of the sending peer
            - policy_state: The policy parameters/weights
            - metadata: Additional information (version, performance, etc.)
        """
        pass

    @abstractmethod
    def aggregate_policy_updates(
        self,
        local_policy: Dict[str, Any],
        peer_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate received policy updates with local policy.

        This implements the core federated learning logic, combining
        multiple policy updates into a single updated policy.

        Args:
            local_policy: The agent's current local policy
            peer_updates: List of policy updates from peers

        Returns:
            Aggregated policy incorporating peer knowledge
        """
        pass

    @abstractmethod
    def select_peers_for_sharing(
        self,
        num_peers: Optional[int] = None
    ) -> List[str]:
        """
        Select which peers to share policy updates with.

        Uses the configured update_strategy to determine peer selection.

        Args:
            num_peers: Optional limit on number of peers to select

        Returns:
            List of peer IDs to share with
        """
        pass

    @abstractmethod
    def update_peer_trust(
        self,
        peer_id: str,
        update_quality: float,
        performance_delta: float
    ):
        """
        Update trust score for a peer based on update quality.

        This implements a reputation system to handle Byzantine peers
        and prioritize high-quality policy updates.

        Args:
            peer_id: ID of the peer to update trust for
            update_quality: Quality metric for the received update (0.0 to 1.0)
            performance_delta: Change in performance after applying update
        """
        pass

    @abstractmethod
    def validate_policy_update(
        self,
        peer_id: str,
        policy_update: Dict[str, Any]
    ) -> bool:
        """
        Validate a received policy update before applying it.

        Checks for:
        - Malformed/corrupted updates
        - Byzantine/adversarial behavior
        - Version compatibility
        - Structural consistency

        Args:
            peer_id: ID of the peer who sent the update
            policy_update: The policy update to validate

        Returns:
            True if update is valid and should be applied
        """
        pass

    @abstractmethod
    def discover_peers(self) -> Set[str]:
        """
        Discover available peers in the mycelial network.

        Uses Redis pub/sub or service discovery to find other agents
        willing to participate in federated learning.

        Returns:
            Set of discovered peer IDs
        """
        pass

    @abstractmethod
    def connect_to_peer(self, peer_id: str) -> bool:
        """
        Establish a connection with a peer in the mycelial network.

        Args:
            peer_id: ID of the peer to connect to

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect_from_peer(self, peer_id: str):
        """
        Disconnect from a peer (e.g., due to low trust or poor performance).

        Args:
            peer_id: ID of the peer to disconnect from
        """
        pass

    @abstractmethod
    def compute_update_weight(self, peer_id: str) -> float:
        """
        Compute the weight/importance for a peer's update during aggregation.

        Considers factors like:
        - Trust score
        - Performance metrics
        - Recency of update
        - Network proximity

        Args:
            peer_id: ID of the peer

        Returns:
            Weight value (typically 0.0 to 1.0)
        """
        pass

    @abstractmethod
    def get_policy_divergence(
        self,
        policy_a: Dict[str, Any],
        policy_b: Dict[str, Any]
    ) -> float:
        """
        Measure divergence between two policies.

        Used to assess how different a peer's policy is from local policy.
        High divergence might indicate different exploration or adversarial behavior.

        Args:
            policy_a: First policy to compare
            policy_b: Second policy to compare

        Returns:
            Divergence metric (higher = more different)
        """
        pass

    @abstractmethod
    def prune_policy_updates(
        self,
        updates: List[Dict[str, Any]],
        max_updates: int
    ) -> List[Dict[str, Any]]:
        """
        Prune/filter policy updates to keep only the most valuable ones.

        Useful when receiving many updates to avoid information overload
        and maintain computational efficiency.

        Args:
            updates: List of policy updates
            max_updates: Maximum number of updates to keep

        Returns:
            Filtered list of most valuable updates
        """
        pass

    @abstractmethod
    def get_sharing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about federated learning activity.

        Returns:
            Dictionary containing metrics like:
            - Number of updates shared
            - Number of updates received
            - Average trust score
            - Aggregation performance
            - Active peer count
        """
        pass

    def increment_policy_version(self):
        """Increment the local policy version counter."""
        self.policy_version += 1
        logger.debug("Agent %s policy version: %d", self.agent_id, self.policy_version)

    def get_connected_peer_count(self) -> int:
        """Get the number of currently connected peers."""
        return len(self.connected_peers)

    def is_peer_connected(self, peer_id: str) -> bool:
        """Check if a specific peer is currently connected."""
        return peer_id in self.connected_peers

    def get_peer_trust_score(self, peer_id: str) -> Optional[float]:
        """Get the trust score for a specific peer."""
        return self.peer_trust_scores.get(peer_id)

    def clear_update_buffer(self):
        """Clear the buffer of received policy updates."""
        self.update_buffer.clear()
        logger.debug("Cleared update buffer for agent %s", self.agent_id)
