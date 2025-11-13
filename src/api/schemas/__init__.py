"""
API Schemas for MAE REST API

Pydantic models for request/response validation and OpenAPI documentation.
"""

from .agents import (
    AgentCreate,
    AgentResponse,
    AgentList,
    AgentStats
)

from .training import (
    TrainingConfig,
    TrainingStart,
    TrainingStatus,
    TrainingResponse
)

from .metrics import (
    MetricsQuery,
    MetricsResponse,
    SystemMetrics
)

from .policies import (
    PolicyExport,
    PolicyImport,
    PolicyCompare
)

from .system import (
    HealthResponse,
    VersionResponse,
    SystemStats
)

__all__ = [
    # Agents
    "AgentCreate",
    "AgentResponse",
    "AgentList",
    "AgentStats",
    # Training
    "TrainingConfig",
    "TrainingStart",
    "TrainingStatus",
    "TrainingResponse",
    # Metrics
    "MetricsQuery",
    "MetricsResponse",
    "SystemMetrics",
    # Policies
    "PolicyExport",
    "PolicyImport",
    "PolicyCompare",
    # System
    "HealthResponse",
    "VersionResponse",
    "SystemStats",
]
