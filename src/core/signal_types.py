"""
Standard Signal Types for MAE v3.0 Electrical Signaling

This module defines the standard signal types used throughout the
mycelial agent network for ultra-fast coordination and communication.

Signal Type Categories:
- Critical Alerts: DANGER, SYSTEM_FAILURE, RESOURCE_EXHAUSTION
- Opportunities: OPPORTUNITY, RESOURCE_AVAILABLE
- Coordination: COLLABORATION_REQUEST, CONVERGENCE, POLICY_UPDATE
- Status: HEARTBEAT, STATUS_UPDATE, PERFORMANCE_REPORT

Author: MAE Development Team
Date: 2025-11-12
"""

from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass


class SignalType:
    """Standard signal types for the mycelial network"""

    # === CRITICAL ALERTS (Priority: CRITICAL) ===
    DANGER = "DANGER"
    """
    Critical risk detected that requires immediate attention.
    Payload: {
        'risk_level': float (0-1),
        'risk_type': str (e.g., 'policy_divergence', 'performance_collapse'),
        'description': str,
        'recommended_action': str
    }
    """

    SYSTEM_FAILURE = "SYSTEM_FAILURE"
    """
    System-level failure detected.
    Payload: {
        'component': str (e.g., 'redis', 'vector_db', 'frl'),
        'error_message': str,
        'severity': str ('critical', 'high', 'medium')
    }
    """

    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    """
    Resource exhaustion warning.
    Payload: {
        'resource_type': str (e.g., 'memory', 'cpu', 'connections'),
        'current_usage': float,
        'threshold': float,
        'remaining_capacity': float
    }
    """

    # === OPPORTUNITIES (Priority: HIGH) ===
    OPPORTUNITY = "OPPORTUNITY"
    """
    High-reward opportunity discovered.
    Payload: {
        'opportunity_type': str (e.g., 'high_reward_state', 'optimal_strategy'),
        'expected_reward': float,
        'confidence': float (0-1),
        'state_description': Dict[str, Any],
        'recommended_action': str
    }
    """

    RESOURCE_AVAILABLE = "RESOURCE_AVAILABLE"
    """
    Resource has become available for use.
    Payload: {
        'resource_type': str (e.g., 'compute', 'data', 'specialist'),
        'resource_id': str,
        'capacity': float,
        'availability_duration': float (seconds, 0=infinite)
    }
    """

    # === COORDINATION (Priority: HIGH/NORMAL) ===
    COLLABORATION_REQUEST = "COLLABORATION_REQUEST"
    """
    Agent requesting collaboration from peers.
    Payload: {
        'task_type': str,
        'required_capabilities': List[str],
        'urgency': str ('high', 'medium', 'low'),
        'expected_duration': float (seconds),
        'reward_share': float (0-1)
    }
    """

    CONVERGENCE = "CONVERGENCE"
    """
    Agent has reached policy convergence.
    Payload: {
        'agent_level': int,
        'satisfaction_score': float,
        'policy_summary': Dict[str, Any],
        'performance_metrics': Dict[str, float]
    }
    """

    POLICY_UPDATE = "POLICY_UPDATE"
    """
    Agent has updated its policy significantly.
    Payload: {
        'update_magnitude': float,
        'improvement': float,
        'policy_version': int,
        'key_changes': List[str]
    }
    """

    DISCOVERY = "DISCOVERY"
    """
    Novel state or pattern discovered.
    Payload: {
        'discovery_type': str (e.g., 'new_state', 'new_pattern', 'anomaly'),
        'novelty_score': float (0-1),
        'description': str,
        'state_data': Dict[str, Any]
    }
    """

    # === STATUS (Priority: NORMAL/LOW) ===
    HEARTBEAT = "HEARTBEAT"
    """
    Regular heartbeat signal for liveness detection.
    Payload: {
        'timestamp': float,
        'status': str ('healthy', 'degraded', 'overloaded'),
        'metrics': Dict[str, float]
    }
    """

    STATUS_UPDATE = "STATUS_UPDATE"
    """
    General status update from agent.
    Payload: {
        'status': str,
        'message': str,
        'metadata': Dict[str, Any]
    }
    """

    PERFORMANCE_REPORT = "PERFORMANCE_REPORT"
    """
    Performance metrics report.
    Payload: {
        'average_reward': float,
        'success_rate': float,
        'learning_iterations': int,
        'agent_level': int,
        'satisfaction_score': float
    }
    """

    ACHIEVEMENT_UNLOCKED = "ACHIEVEMENT_UNLOCKED"
    """
    Agent unlocked an achievement.
    Payload: {
        'achievement_name': str,
        'agent_level': int,
        'experience_points': int,
        'description': str
    }
    """

    # === LEARNING (Priority: NORMAL) ===
    KNOWLEDGE_SHARE = "KNOWLEDGE_SHARE"
    """
    Agent sharing learned knowledge.
    Payload: {
        'knowledge_type': str (e.g., 'policy', 'strategy', 'state_value'),
        'confidence': float (0-1),
        'performance_context': Dict[str, float],
        'data': Dict[str, Any]
    }
    """

    LEARNING_MILESTONE = "LEARNING_MILESTONE"
    """
    Significant learning milestone reached.
    Payload: {
        'milestone_type': str (e.g., 'level_up', 'convergence', 'mastery'),
        'iterations': int,
        'performance_improvement': float,
        'description': str
    }
    """


@dataclass
class SignalTypeInfo:
    """Metadata about a signal type"""
    name: str
    category: str
    default_priority: str
    description: str
    payload_schema: Dict[str, str]


# Signal type registry for introspection
SIGNAL_TYPE_REGISTRY: Dict[str, SignalTypeInfo] = {
    SignalType.DANGER: SignalTypeInfo(
        name="DANGER",
        category="Critical Alert",
        default_priority="CRITICAL",
        description="Critical risk detected requiring immediate attention",
        payload_schema={
            'risk_level': 'float',
            'risk_type': 'str',
            'description': 'str',
            'recommended_action': 'str'
        }
    ),
    SignalType.SYSTEM_FAILURE: SignalTypeInfo(
        name="SYSTEM_FAILURE",
        category="Critical Alert",
        default_priority="CRITICAL",
        description="System-level failure detected",
        payload_schema={
            'component': 'str',
            'error_message': 'str',
            'severity': 'str'
        }
    ),
    SignalType.RESOURCE_EXHAUSTION: SignalTypeInfo(
        name="RESOURCE_EXHAUSTION",
        category="Critical Alert",
        default_priority="CRITICAL",
        description="Resource exhaustion warning",
        payload_schema={
            'resource_type': 'str',
            'current_usage': 'float',
            'threshold': 'float',
            'remaining_capacity': 'float'
        }
    ),
    SignalType.OPPORTUNITY: SignalTypeInfo(
        name="OPPORTUNITY",
        category="Opportunity",
        default_priority="HIGH",
        description="High-reward opportunity discovered",
        payload_schema={
            'opportunity_type': 'str',
            'expected_reward': 'float',
            'confidence': 'float',
            'state_description': 'dict',
            'recommended_action': 'str'
        }
    ),
    SignalType.RESOURCE_AVAILABLE: SignalTypeInfo(
        name="RESOURCE_AVAILABLE",
        category="Opportunity",
        default_priority="HIGH",
        description="Resource has become available",
        payload_schema={
            'resource_type': 'str',
            'resource_id': 'str',
            'capacity': 'float',
            'availability_duration': 'float'
        }
    ),
    SignalType.COLLABORATION_REQUEST: SignalTypeInfo(
        name="COLLABORATION_REQUEST",
        category="Coordination",
        default_priority="HIGH",
        description="Agent requesting collaboration",
        payload_schema={
            'task_type': 'str',
            'required_capabilities': 'list',
            'urgency': 'str',
            'expected_duration': 'float',
            'reward_share': 'float'
        }
    ),
    SignalType.CONVERGENCE: SignalTypeInfo(
        name="CONVERGENCE",
        category="Coordination",
        default_priority="HIGH",
        description="Agent reached policy convergence",
        payload_schema={
            'agent_level': 'int',
            'satisfaction_score': 'float',
            'policy_summary': 'dict',
            'performance_metrics': 'dict'
        }
    ),
    SignalType.POLICY_UPDATE: SignalTypeInfo(
        name="POLICY_UPDATE",
        category="Coordination",
        default_priority="NORMAL",
        description="Agent updated its policy",
        payload_schema={
            'update_magnitude': 'float',
            'improvement': 'float',
            'policy_version': 'int',
            'key_changes': 'list'
        }
    ),
    SignalType.DISCOVERY: SignalTypeInfo(
        name="DISCOVERY",
        category="Coordination",
        default_priority="HIGH",
        description="Novel state or pattern discovered",
        payload_schema={
            'discovery_type': 'str',
            'novelty_score': 'float',
            'description': 'str',
            'state_data': 'dict'
        }
    ),
    SignalType.HEARTBEAT: SignalTypeInfo(
        name="HEARTBEAT",
        category="Status",
        default_priority="LOW",
        description="Regular heartbeat for liveness",
        payload_schema={
            'timestamp': 'float',
            'status': 'str',
            'metrics': 'dict'
        }
    ),
    SignalType.STATUS_UPDATE: SignalTypeInfo(
        name="STATUS_UPDATE",
        category="Status",
        default_priority="NORMAL",
        description="General status update",
        payload_schema={
            'status': 'str',
            'message': 'str',
            'metadata': 'dict'
        }
    ),
    SignalType.PERFORMANCE_REPORT: SignalTypeInfo(
        name="PERFORMANCE_REPORT",
        category="Status",
        default_priority="LOW",
        description="Performance metrics report",
        payload_schema={
            'average_reward': 'float',
            'success_rate': 'float',
            'learning_iterations': 'int',
            'agent_level': 'int',
            'satisfaction_score': 'float'
        }
    ),
    SignalType.ACHIEVEMENT_UNLOCKED: SignalTypeInfo(
        name="ACHIEVEMENT_UNLOCKED",
        category="Status",
        default_priority="LOW",
        description="Agent unlocked achievement",
        payload_schema={
            'achievement_name': 'str',
            'agent_level': 'int',
            'experience_points': 'int',
            'description': 'str'
        }
    ),
    SignalType.KNOWLEDGE_SHARE: SignalTypeInfo(
        name="KNOWLEDGE_SHARE",
        category="Learning",
        default_priority="NORMAL",
        description="Agent sharing learned knowledge",
        payload_schema={
            'knowledge_type': 'str',
            'confidence': 'float',
            'performance_context': 'dict',
            'data': 'dict'
        }
    ),
    SignalType.LEARNING_MILESTONE: SignalTypeInfo(
        name="LEARNING_MILESTONE",
        category="Learning",
        default_priority="NORMAL",
        description="Significant learning milestone",
        payload_schema={
            'milestone_type': 'str',
            'iterations': 'int',
            'performance_improvement': 'float',
            'description': 'str'
        }
    )
}


def get_signal_info(signal_type: str) -> SignalTypeInfo:
    """
    Get metadata about a signal type.

    Args:
        signal_type: Signal type string

    Returns:
        SignalTypeInfo object or None if not found
    """
    return SIGNAL_TYPE_REGISTRY.get(signal_type)


def list_signal_types(category: str = None) -> list[str]:
    """
    List all signal types, optionally filtered by category.

    Args:
        category: Filter by category (e.g., 'Critical Alert', 'Opportunity')

    Returns:
        List of signal type names
    """
    if category:
        return [
            sig_type for sig_type, info in SIGNAL_TYPE_REGISTRY.items()
            if info.category == category
        ]
    return list(SIGNAL_TYPE_REGISTRY.keys())


def validate_signal_payload(signal_type: str, payload: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate that a payload matches the expected schema for a signal type.

    Args:
        signal_type: Signal type string
        payload: Payload dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    info = get_signal_info(signal_type)
    if not info:
        return False, f"Unknown signal type: {signal_type}"

    schema = info.payload_schema
    missing_keys = set(schema.keys()) - set(payload.keys())

    if missing_keys:
        return False, f"Missing required keys: {missing_keys}"

    return True, ""
