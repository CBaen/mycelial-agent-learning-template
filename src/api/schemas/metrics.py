"""
Metrics API Schemas

Pydantic models for metrics query endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MetricType(str, Enum):
    """Metric type categories."""
    AGENT = "agent"
    SYSTEM = "system"
    COMMUNICATION = "communication"
    MEMORY = "memory"


class TimeRange(BaseModel):
    """Time range for metric queries."""

    start: datetime = Field(..., description="Start timestamp")
    end: datetime = Field(..., description="End timestamp")

    class Config:
        schema_extra = {
            "example": {
                "start": "2025-11-12T09:00:00Z",
                "end": "2025-11-12T10:00:00Z"
            }
        }


class MetricsQuery(BaseModel):
    """Query parameters for metrics."""

    metric_types: Optional[List[MetricType]] = Field(
        None,
        description="Types of metrics to query"
    )
    agent_ids: Optional[List[str]] = Field(
        None,
        description="Specific agent IDs to query"
    )
    time_range: Optional[TimeRange] = Field(
        None,
        description="Time range for query"
    )
    aggregation: Optional[str] = Field(
        "avg",
        description="Aggregation function (avg, sum, min, max, count)"
    )

    class Config:
        schema_extra = {
            "example": {
                "metric_types": ["agent", "system"],
                "agent_ids": ["agent_1", "agent_2"],
                "time_range": {
                    "start": "2025-11-12T09:00:00Z",
                    "end": "2025-11-12T10:00:00Z"
                },
                "aggregation": "avg"
            }
        }


class MetricDataPoint(BaseModel):
    """Single metric data point."""

    timestamp: datetime = Field(..., description="Timestamp")
    value: float = Field(..., description="Metric value")
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Metric labels"
    )


class MetricsResponse(BaseModel):
    """Response for metrics queries."""

    metric_name: str = Field(..., description="Metric name")
    metric_type: MetricType = Field(..., description="Metric type")
    data: List[MetricDataPoint] = Field(..., description="Metric data points")
    aggregated_value: Optional[float] = Field(
        None,
        description="Aggregated value"
    )

    class Config:
        schema_extra = {
            "example": {
                "metric_name": "agent_reward",
                "metric_type": "agent",
                "data": [
                    {
                        "timestamp": "2025-11-12T10:00:00Z",
                        "value": 15.5,
                        "labels": {"agent_id": "agent_1"}
                    }
                ],
                "aggregated_value": 15.5
            }
        }


class SystemMetrics(BaseModel):
    """Current system-wide metrics."""

    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_bytes: int = Field(..., description="Memory usage in bytes")
    active_agents: int = Field(..., description="Number of active agents")
    training_sessions: int = Field(..., description="Active training sessions")
    total_learning_steps: int = Field(..., description="Total learning steps")
    average_reward: float = Field(..., description="Average reward across agents")
    uptime_seconds: float = Field(..., description="System uptime")

    class Config:
        schema_extra = {
            "example": {
                "cpu_percent": 45.2,
                "memory_bytes": 2147483648,
                "active_agents": 10,
                "training_sessions": 1,
                "total_learning_steps": 50000,
                "average_reward": 15.5,
                "uptime_seconds": 3600.0
            }
        }
