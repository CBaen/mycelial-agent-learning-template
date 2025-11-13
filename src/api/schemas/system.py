"""
System API Schemas

Pydantic models for system health and info endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class HealthStatus(str, Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a system component."""

    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Status message")
    last_check: datetime = Field(..., description="Last health check")


class HealthResponse(BaseModel):
    """System health check response."""

    status: HealthStatus = Field(..., description="Overall system status")
    components: List[ComponentHealth] = Field(..., description="Component statuses")
    timestamp: datetime = Field(..., description="Check timestamp")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "components": [
                    {
                        "name": "redis",
                        "status": "healthy",
                        "message": "Connected",
                        "last_check": "2025-11-12T10:00:00Z"
                    },
                    {
                        "name": "chromadb",
                        "status": "healthy",
                        "message": "Connected",
                        "last_check": "2025-11-12T10:00:00Z"
                    }
                ],
                "timestamp": "2025-11-12T10:00:00Z"
            }
        }


class VersionResponse(BaseModel):
    """System version information."""

    version: str = Field(..., description="MAE version")
    build_date: str = Field(..., description="Build date")
    git_commit: Optional[str] = Field(None, description="Git commit hash")
    python_version: str = Field(..., description="Python version")
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Key dependency versions"
    )

    class Config:
        schema_extra = {
            "example": {
                "version": "3.0.0",
                "build_date": "2025-11-12",
                "git_commit": "abc123",
                "python_version": "3.11.9",
                "dependencies": {
                    "fastapi": "0.104.0",
                    "pydantic": "2.5.0",
                    "prometheus-client": "0.19.0"
                }
            }
        }


class ResourceUsage(BaseModel):
    """System resource usage."""

    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_used_mb: float = Field(..., description="Memory used (MB)")
    memory_total_mb: float = Field(..., description="Total memory (MB)")
    disk_used_gb: float = Field(..., description="Disk used (GB)")
    disk_total_gb: float = Field(..., description="Total disk (GB)")


class SystemStats(BaseModel):
    """Comprehensive system statistics."""

    uptime_seconds: float = Field(..., description="System uptime")
    resource_usage: ResourceUsage = Field(..., description="Resource usage")
    agent_count: int = Field(..., description="Total agents")
    active_agents: int = Field(..., description="Active agents")
    training_sessions: int = Field(..., description="Active training sessions")
    total_learning_steps: int = Field(..., description="Total learning steps")
    total_requests: int = Field(..., description="Total API requests")
    average_response_time_ms: float = Field(..., description="Avg response time (ms)")

    class Config:
        schema_extra = {
            "example": {
                "uptime_seconds": 3600.0,
                "resource_usage": {
                    "cpu_percent": 45.2,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 16384.0,
                    "disk_used_gb": 50.0,
                    "disk_total_gb": 500.0
                },
                "agent_count": 10,
                "active_agents": 8,
                "training_sessions": 1,
                "total_learning_steps": 50000,
                "total_requests": 1234,
                "average_response_time_ms": 25.5
            }
        }
