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
    SystemMetrics,
    MetricType,
    TimeRange
)

from .policies import (
    PolicyExport,
    PolicyImport,
    PolicyCompare,
    PolicyMetadata
)

from .system import (
    HealthResponse,
    VersionResponse,
    SystemStats,
    HealthStatus,
    ComponentHealth,
    ResourceUsage
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
    "MetricType",
    "TimeRange",
    # Policies
    "PolicyExport",
    "PolicyImport",
    "PolicyCompare",
    "PolicyMetadata",
    # System
    "HealthResponse",
    "VersionResponse",
    "SystemStats",
    "HealthStatus",
    "ComponentHealth",
    "ResourceUsage",
]
