"""
Agent API Schemas

Pydantic models for agent management endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class AgentCreate(BaseModel):
    """Schema for creating a new agent."""

    agent_type: str = Field(
        ...,
        description="Type of agent (specialist, builder, risk_manager)",
        example="specialist"
    )
    team_id: str = Field(
        ...,
        description="Team identifier for the agent",
        example="team_alpha"
    )
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Agent-specific configuration"
    )

    @validator('agent_type')
    def validate_agent_type(cls, v):
        """Validate agent type."""
        valid_types = ['specialist', 'builder', 'risk_manager']
        if v not in valid_types:
            raise ValueError(f"agent_type must be one of {valid_types}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "agent_type": "specialist",
                "team_id": "team_alpha",
                "config": {
                    "learning_rate": 0.001,
                    "gamma": 0.99
                }
            }
        }


class AgentStats(BaseModel):
    """Agent performance statistics."""

    learning_steps: int = Field(..., description="Total learning steps")
    average_reward: float = Field(..., description="Average reward")
    convergence_score: float = Field(..., ge=0, le=1, description="Convergence score")
    satisfaction_score: float = Field(..., ge=0, le=1, description="Satisfaction score")
    xp: int = Field(..., description="Experience points")


class AgentResponse(BaseModel):
    """Response schema for agent operations."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Type of agent")
    team_id: str = Field(..., description="Team identifier")
    status: str = Field(..., description="Agent status (active, idle, stopped)")
    created_at: datetime = Field(..., description="Creation timestamp")
    stats: Optional[AgentStats] = Field(None, description="Agent statistics")

    class Config:
        schema_extra = {
            "example": {
                "agent_id": "agent_123",
                "agent_type": "specialist",
                "team_id": "team_alpha",
                "status": "active",
                "created_at": "2025-11-12T10:00:00Z",
                "stats": {
                    "learning_steps": 1000,
                    "average_reward": 15.5,
                    "convergence_score": 0.85,
                    "satisfaction_score": 0.92,
                    "xp": 5000
                }
            }
        }


class AgentList(BaseModel):
    """List of agents with pagination."""

    agents: List[AgentResponse] = Field(..., description="List of agents")
    total: int = Field(..., description="Total number of agents")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Page size")

    class Config:
        schema_extra = {
            "example": {
                "agents": [],
                "total": 10,
                "page": 1,
                "page_size": 10
            }
        }
