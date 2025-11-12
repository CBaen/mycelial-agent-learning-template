"""
Stigmergic Marker Types for MAE v3.0 (Big Rock 6)

Defines standard marker types for pheromone-like environmental communication.
Inspired by ant colony stigmergy and swarm intelligence patterns.

Author: MAE Development Team
Date: 2025-11-12
"""

from dataclasses import dataclass
from typing import Dict


class MarkerType:
    """
    Standard stigmergic marker types for indirect agent coordination.

    These markers are deposited in the environment and sensed by agents,
    enabling emergent swarm behaviors without direct communication.
    """

    SUCCESS = "SUCCESS"
    """
    Marks successful actions/outcomes (analogous to food trail pheromones).

    Usage: Agents deposit SUCCESS markers after positive rewards.
    Effect: Attracts other agents to replicate successful behaviors.
    Typical decay: Moderate (0.1-0.2) - persists while still valuable.

    Example payload:
    {
        'reward': float,
        'action': str,
        'confidence': float
    }
    """

    DANGER = "DANGER"
    """
    Marks dangerous areas or failed actions (analogous to alarm pheromones).

    Usage: Agents deposit DANGER markers after negative outcomes.
    Effect: Repels agents, warns them to avoid this region.
    Typical decay: Fast (0.3-0.5) - danger may be temporary.

    Example payload:
    {
        'risk_level': float,
        'risk_type': str,
        'failure_reason': str
    }
    """

    EXPLORATION = "EXPLORATION"
    """
    Marks explored areas to reduce redundant exploration.

    Usage: Agents deposit EXPLORATION markers when visiting locations.
    Effect: Guides agents toward unexplored regions (novelty seeking).
    Typical decay: Slow (0.05-0.1) - exploration history persists.

    Example payload:
    {
        'visit_count': int,
        'last_visited': float,
        'findings': str
    }
    """

    RESOURCE = "RESOURCE"
    """
    Marks resource locations (data, compute, specialists).

    Usage: Agents deposit RESOURCE markers when finding valuable resources.
    Effect: Attracts agents needing resources, enables efficient allocation.
    Typical decay: Moderate (0.1-0.15) - resources may deplete.

    Example payload:
    {
        'resource_type': str,
        'quantity': float,
        'quality': float
    }
    """

    CONVERGENCE = "CONVERGENCE"
    """
    Marks areas where agents have converged on policies.

    Usage: Agents deposit CONVERGENCE markers when reaching stability.
    Effect: Helps coordinate team convergence, signals maturity.
    Typical decay: Very slow (0.01-0.05) - convergence is stable.

    Example payload:
    {
        'satisfaction_score': float,
        'policy_version': int,
        'agent_level': int
    }
    """

    COLLABORATION = "COLLABORATION"
    """
    Marks areas where agents successfully collaborated.

    Usage: Agents deposit COLLABORATION markers after teamwork.
    Effect: Encourages team formation in beneficial regions.
    Typical decay: Moderate (0.1-0.2) - collaboration opportunities vary.

    Example payload:
    {
        'team_size': int,
        'collaboration_type': str,
        'success_rate': float
    }
    """

    NOVELTY = "NOVELTY"
    """
    Marks novel states or discoveries.

    Usage: Agents deposit NOVELTY markers when discovering new patterns.
    Effect: Attracts curious agents, promotes knowledge sharing.
    Typical decay: Fast (0.2-0.3) - novelty fades quickly.

    Example payload:
    {
        'novelty_score': float,
        'discovery_type': str,
        'significance': float
    }
    """


@dataclass
class MarkerTypeInfo:
    """Metadata about a marker type"""
    name: str
    category: str
    typical_decay: tuple  # (min, max) range
    effect: str  # "attractive" or "repulsive"
    description: str
    example_metadata: Dict[str, str]


# Marker type registry for introspection
MARKER_TYPE_REGISTRY: Dict[str, MarkerTypeInfo] = {
    MarkerType.SUCCESS: MarkerTypeInfo(
        name="SUCCESS",
        category="Positive Reinforcement",
        typical_decay=(0.1, 0.2),
        effect="attractive",
        description="Marks successful actions/outcomes, attracts replication",
        example_metadata={
            'reward': 'float',
            'action': 'str',
            'confidence': 'float'
        }
    ),
    MarkerType.DANGER: MarkerTypeInfo(
        name="DANGER",
        category="Negative Reinforcement",
        typical_decay=(0.3, 0.5),
        effect="repulsive",
        description="Marks dangerous areas, warns agents to avoid",
        example_metadata={
            'risk_level': 'float',
            'risk_type': 'str',
            'failure_reason': 'str'
        }
    ),
    MarkerType.EXPLORATION: MarkerTypeInfo(
        name="EXPLORATION",
        category="Information",
        typical_decay=(0.05, 0.1),
        effect="repulsive",  # Repels from explored areas
        description="Marks explored areas, guides toward novelty",
        example_metadata={
            'visit_count': 'int',
            'last_visited': 'float',
            'findings': 'str'
        }
    ),
    MarkerType.RESOURCE: MarkerTypeInfo(
        name="RESOURCE",
        category="Opportunity",
        typical_decay=(0.1, 0.15),
        effect="attractive",
        description="Marks resource locations, enables allocation",
        example_metadata={
            'resource_type': 'str',
            'quantity': 'float',
            'quality': 'float'
        }
    ),
    MarkerType.CONVERGENCE: MarkerTypeInfo(
        name="CONVERGENCE",
        category="Coordination",
        typical_decay=(0.01, 0.05),
        effect="attractive",
        description="Marks policy convergence, coordinates team",
        example_metadata={
            'satisfaction_score': 'float',
            'policy_version': 'int',
            'agent_level': 'int'
        }
    ),
    MarkerType.COLLABORATION: MarkerTypeInfo(
        name="COLLABORATION",
        category="Social",
        typical_decay=(0.1, 0.2),
        effect="attractive",
        description="Marks successful teamwork, encourages formation",
        example_metadata={
            'team_size': 'int',
            'collaboration_type': 'str',
            'success_rate': 'float'
        }
    ),
    MarkerType.NOVELTY: MarkerTypeInfo(
        name="NOVELTY",
        category="Information",
        typical_decay=(0.2, 0.3),
        effect="attractive",
        description="Marks discoveries, attracts curious agents",
        example_metadata={
            'novelty_score': 'float',
            'discovery_type': 'str',
            'significance': 'float'
        }
    )
}


def get_marker_info(marker_type: str) -> MarkerTypeInfo:
    """
    Get metadata about a marker type.

    Args:
        marker_type: Marker type string

    Returns:
        MarkerTypeInfo object or None if not found
    """
    return MARKER_TYPE_REGISTRY.get(marker_type)


def list_marker_types(category: str = None) -> list:
    """
    List all marker types, optionally filtered by category.

    Args:
        category: Filter by category (e.g., 'Positive Reinforcement')

    Returns:
        List of marker type names
    """
    if category:
        return [
            mtype for mtype, info in MARKER_TYPE_REGISTRY.items()
            if info.category == category
        ]
    return list(MARKER_TYPE_REGISTRY.keys())


def get_recommended_decay_rate(marker_type: str) -> float:
    """
    Get recommended decay rate for a marker type.

    Args:
        marker_type: Marker type string

    Returns:
        Recommended decay rate (midpoint of typical range)
    """
    info = get_marker_info(marker_type)
    if not info:
        return 0.1  # Default

    min_decay, max_decay = info.typical_decay
    return (min_decay + max_decay) / 2


def is_attractive(marker_type: str) -> bool:
    """
    Check if marker type is attractive or repulsive.

    Args:
        marker_type: Marker type string

    Returns:
        True if attractive, False if repulsive
    """
    info = get_marker_info(marker_type)
    if not info:
        return True  # Default to attractive

    return info.effect == "attractive"
