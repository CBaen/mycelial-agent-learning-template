"""
Training API Schemas

Pydantic models for training control endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TrainingStatus(str, Enum):
    """Training session status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration."""

    learning_rate: float = Field(
        0.001,
        gt=0,
        le=1,
        description="Learning rate"
    )
    gamma: float = Field(
        0.99,
        ge=0,
        le=1,
        description="Discount factor"
    )
    batch_size: int = Field(
        32,
        ge=1,
        le=1024,
        description="Batch size for training"
    )
    epsilon: float = Field(
        0.1,
        ge=0,
        le=1,
        description="Exploration rate"
    )
    max_episodes: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum number of episodes"
    )
    max_steps_per_episode: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum steps per episode"
    )

    class Config:
        schema_extra = {
            "example": {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "batch_size": 32,
                "epsilon": 0.1,
                "max_episodes": 1000,
                "max_steps_per_episode": 500
            }
        }


class TrainingStart(BaseModel):
    """Request to start training session."""

    config: Optional[TrainingConfig] = Field(
        None,
        description="Training configuration (uses defaults if not provided)"
    )
    agents: Optional[list[str]] = Field(
        None,
        description="Specific agent IDs to train (all if not provided)"
    )
    resume: bool = Field(
        False,
        description="Resume from previous checkpoint"
    )

    class Config:
        schema_extra = {
            "example": {
                "config": {
                    "learning_rate": 0.001,
                    "batch_size": 32
                },
                "agents": ["agent_1", "agent_2"],
                "resume": False
            }
        }


class TrainingMetrics(BaseModel):
    """Training session metrics."""

    episodes_completed: int = Field(..., description="Episodes completed")
    total_steps: int = Field(..., description="Total training steps")
    average_reward: float = Field(..., description="Average reward")
    average_loss: float = Field(..., description="Average loss")
    convergence_rate: float = Field(..., ge=0, le=1, description="Convergence rate")


class TrainingResponse(BaseModel):
    """Response for training operations."""

    session_id: str = Field(..., description="Training session ID")
    status: TrainingStatus = Field(..., description="Current status")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    stopped_at: Optional[datetime] = Field(None, description="Stop timestamp")
    config: TrainingConfig = Field(..., description="Training configuration")
    metrics: Optional[TrainingMetrics] = Field(None, description="Training metrics")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "status": "running",
                "started_at": "2025-11-12T10:00:00Z",
                "stopped_at": None,
                "config": {
                    "learning_rate": 0.001,
                    "gamma": 0.99,
                    "batch_size": 32,
                    "epsilon": 0.1
                },
                "metrics": {
                    "episodes_completed": 50,
                    "total_steps": 25000,
                    "average_reward": 15.5,
                    "average_loss": 0.05,
                    "convergence_rate": 0.75
                }
            }
        }
