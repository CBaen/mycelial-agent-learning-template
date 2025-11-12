"""
GNN Message Encoding for MAE v3.0 (Big Rock 7)

Defines message structures and encoding mechanisms for GNN-based
communication routing.

Author: MAE Development Team
Date: 2025-11-12
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import numpy as np
import hashlib
import json


@dataclass
class GNNMessage:
    """
    Message structure for GNN-routed communication.

    Attributes:
        message_id: Unique message identifier
        sender_id: Sending agent ID
        content: Message payload (arbitrary dictionary)
        message_type: Message type string
        priority: Message priority [0, 1]
        ttl: Time-to-live (hops remaining)
        path: List of agents visited (routing path)
        timestamp: Creation timestamp
        metadata: Additional message metadata
    """
    message_id: str
    sender_id: str
    content: Dict[str, Any]
    message_type: str
    priority: float = 0.5
    ttl: int = 3
    path: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate message attributes"""
        if not 0 <= self.priority <= 1:
            raise ValueError(f"Priority must be in [0, 1], got {self.priority}")

        if self.ttl < 0:
            raise ValueError(f"TTL must be non-negative, got {self.ttl}")

        # Ensure sender is in path
        if self.sender_id not in self.path:
            self.path.insert(0, self.sender_id)

    def decrement_ttl(self) -> bool:
        """
        Decrement TTL by 1.

        Returns:
            True if message is still alive (TTL > 0), False if expired
        """
        self.ttl -= 1
        return self.ttl > 0

    def add_to_path(self, agent_id: str):
        """Add agent to routing path"""
        self.path.append(agent_id)

    def has_visited(self, agent_id: str) -> bool:
        """Check if agent is in routing path (prevents cycles)"""
        return agent_id in self.path

    def age(self) -> float:
        """Get message age in seconds"""
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary (for serialization)"""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'content': self.content,
            'message_type': self.message_type,
            'priority': self.priority,
            'ttl': self.ttl,
            'path': self.path,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GNNMessage':
        """Create message from dictionary"""
        return cls(
            message_id=data['message_id'],
            sender_id=data['sender_id'],
            content=data['content'],
            message_type=data['message_type'],
            priority=data.get('priority', 0.5),
            ttl=data.get('ttl', 3),
            path=data.get('path', []),
            timestamp=data.get('timestamp', time.time()),
            metadata=data.get('metadata', {})
        )


class MessageType:
    """
    Standard message types for GNN communication.

    These complement electrical signals (Big Rock 5) by providing
    structured, routed communication for non-urgent coordination.
    """

    # Coordination messages
    COLLABORATION_REQUEST = "COLLABORATION_REQUEST"
    """Request for team formation or collaboration"""

    COLLABORATION_RESPONSE = "COLLABORATION_RESPONSE"
    """Response to collaboration request (accept/reject)"""

    TASK_ASSIGNMENT = "TASK_ASSIGNMENT"
    """Assign task to specific agent(s)"""

    TASK_COMPLETION = "TASK_COMPLETION"
    """Report task completion"""

    # Information sharing
    KNOWLEDGE_SHARE = "KNOWLEDGE_SHARE"
    """Share learned knowledge or policy updates"""

    STATE_UPDATE = "STATE_UPDATE"
    """Share agent state information"""

    CAPABILITY_BROADCAST = "CAPABILITY_BROADCAST"
    """Advertise agent capabilities"""

    # Queries
    QUERY = "QUERY"
    """Request information from network"""

    QUERY_RESPONSE = "QUERY_RESPONSE"
    """Response to query"""

    SPECIALIST_REQUEST = "SPECIALIST_REQUEST"
    """Request for specialist with specific capability"""

    # Coordination
    SYNC_REQUEST = "SYNC_REQUEST"
    """Request for policy/state synchronization"""

    VOTE = "VOTE"
    """Cast vote in distributed decision"""

    CONSENSUS = "CONSENSUS"
    """Report consensus reached"""

    # Broadcast
    BROADCAST = "BROADCAST"
    """General broadcast message"""


class MessageEncoder:
    """
    Encodes GNN messages into fixed-size embeddings for neural processing.

    Combines message content, type, priority, and sender information
    into a dense vector representation.
    """

    def __init__(self, embedding_dim: int = 64):
        """
        Initialize message encoder.

        Args:
            embedding_dim: Dimension of message embeddings
        """
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.embedding_dim = embedding_dim

        # Reserve dimensions for different features
        self.type_dim = 16
        self.meta_dim = 8
        self.content_dim = embedding_dim - self.type_dim - self.meta_dim

        # Message type to embedding mapping
        self.type_embeddings = self._initialize_type_embeddings()

    def encode(
        self,
        message: GNNMessage,
        sender_embedding: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Encode message into fixed-size embedding.

        Args:
            message: Message to encode
            sender_embedding: Optional sender's node embedding

        Returns:
            Message embedding vector
        """
        # Encode message type
        type_features = self._encode_type(message.message_type)

        # Encode content
        content_features = self._encode_content(message.content, sender_embedding)

        # Encode metadata (priority, ttl, age)
        meta_features = self._encode_metadata(message)

        # Concatenate all features
        embedding = np.concatenate([
            type_features,
            content_features,
            meta_features
        ])

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _encode_type(self, message_type: str) -> np.ndarray:
        """
        Encode message type into fixed-size vector.

        Uses learned type embeddings for known types,
        hash-based encoding for unknown types.
        """
        if message_type in self.type_embeddings:
            return self.type_embeddings[message_type]

        # Hash-based encoding for unknown types
        type_hash = int(hashlib.md5(message_type.encode()).hexdigest(), 16)
        rng = np.random.RandomState(type_hash % (2**32))
        embedding = rng.randn(self.type_dim)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _encode_content(
        self,
        content: Dict[str, Any],
        sender_embedding: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Encode message content into fixed-size vector.

        Strategy:
        1. If sender embedding provided, use first half for sender
        2. Hash content keys and values for second half
        """
        # Initialize content embedding
        content_emb = np.zeros(self.content_dim)

        # Use sender embedding if provided
        if sender_embedding is not None:
            sender_dim = min(len(sender_embedding), self.content_dim // 2)
            content_emb[:sender_dim] = sender_embedding[:sender_dim]

        # Hash content dictionary
        try:
            content_str = json.dumps(content, sort_keys=True, default=str)
            content_hash = int(hashlib.md5(content_str.encode()).hexdigest(), 16)

            rng = np.random.RandomState(content_hash % (2**32))
            content_hash_emb = rng.randn(self.content_dim // 2)

            # Normalize
            norm = np.linalg.norm(content_hash_emb)
            if norm > 0:
                content_hash_emb = content_hash_emb / norm

            # Place in second half
            start_idx = self.content_dim // 2
            end_idx = start_idx + len(content_hash_emb)
            if end_idx <= self.content_dim:
                content_emb[start_idx:end_idx] = content_hash_emb
        except (TypeError, ValueError):
            # If content is not JSON-serializable, use random encoding
            pass

        return content_emb

    def _encode_metadata(self, message: GNNMessage) -> np.ndarray:
        """
        Encode message metadata (priority, ttl, age, path length).

        Returns:
            Metadata feature vector
        """
        # Normalize features to [0, 1] range
        priority = message.priority  # Already in [0, 1]
        ttl_normalized = min(1.0, message.ttl / 10.0)  # Assume max TTL ~10
        age_normalized = min(1.0, message.age() / 60.0)  # Normalize by 60 seconds
        path_length_normalized = min(1.0, len(message.path) / 10.0)  # Assume max path ~10

        # Create feature vector
        meta = np.array([
            priority,
            ttl_normalized,
            age_normalized,
            path_length_normalized,
            # Pad to meta_dim
            *([0.0] * (self.meta_dim - 4))
        ])

        return meta[:self.meta_dim]

    def _initialize_type_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Initialize embeddings for known message types.

        Uses deterministic random initialization based on type name.
        """
        embeddings = {}

        # Get all message types from MessageType class
        message_types = [
            attr for attr in dir(MessageType)
            if not attr.startswith('_') and isinstance(getattr(MessageType, attr), str)
        ]

        for msg_type_attr in message_types:
            msg_type = getattr(MessageType, msg_type_attr)

            # Deterministic random embedding
            type_hash = int(hashlib.md5(msg_type.encode()).hexdigest(), 16)
            rng = np.random.RandomState(type_hash % (2**32))
            embedding = rng.randn(self.type_dim)

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            embeddings[msg_type] = embedding

        return embeddings

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between message embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity [-1, 1]
        """
        if len(emb1) != len(emb2):
            raise ValueError("Embeddings must have same dimension")

        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)

        if norm_product == 0:
            return 0.0

        return dot_product / norm_product


def create_message(
    sender_id: str,
    message_type: str,
    content: Dict[str, Any],
    priority: float = 0.5,
    ttl: int = 3,
    metadata: Optional[Dict[str, Any]] = None
) -> GNNMessage:
    """
    Convenience function to create a GNN message.

    Args:
        sender_id: Sending agent ID
        message_type: Type of message (use MessageType constants)
        content: Message payload
        priority: Message priority [0, 1]
        ttl: Time-to-live (hops)
        metadata: Optional metadata

    Returns:
        GNNMessage instance
    """
    message_id = f"msg_{time.time_ns()}_{sender_id}"

    return GNNMessage(
        message_id=message_id,
        sender_id=sender_id,
        content=content,
        message_type=message_type,
        priority=priority,
        ttl=ttl,
        path=[sender_id],
        metadata=metadata or {}
    )


def get_message_types() -> List[str]:
    """
    Get list of all standard message types.

    Returns:
        List of message type strings
    """
    return [
        getattr(MessageType, attr)
        for attr in dir(MessageType)
        if not attr.startswith('_') and isinstance(getattr(MessageType, attr), str)
    ]


def is_query_type(message_type: str) -> bool:
    """Check if message type is a query"""
    return message_type in [
        MessageType.QUERY,
        MessageType.QUERY_RESPONSE,
        MessageType.SPECIALIST_REQUEST
    ]


def is_coordination_type(message_type: str) -> bool:
    """Check if message type is coordination-related"""
    return message_type in [
        MessageType.COLLABORATION_REQUEST,
        MessageType.COLLABORATION_RESPONSE,
        MessageType.TASK_ASSIGNMENT,
        MessageType.TASK_COMPLETION,
        MessageType.SYNC_REQUEST,
        MessageType.VOTE,
        MessageType.CONSENSUS
    ]


def is_information_type(message_type: str) -> bool:
    """Check if message type is information sharing"""
    return message_type in [
        MessageType.KNOWLEDGE_SHARE,
        MessageType.STATE_UPDATE,
        MessageType.CAPABILITY_BROADCAST
    ]
