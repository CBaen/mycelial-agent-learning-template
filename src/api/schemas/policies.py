"""
Policy API Schemas

Pydantic models for policy management endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PolicyExport(BaseModel):
    """Schema for exporting agent policies."""

    agent_ids: Optional[List[str]] = Field(
        None,
        description="Specific agent IDs to export (all if not provided)"
    )
    format: str = Field(
        "json",
        description="Export format (json, pickle, onnx)"
    )
    include_metadata: bool = Field(
        True,
        description="Include training metadata"
    )

    class Config:
        schema_extra = {
            "example": {
                "agent_ids": ["agent_1", "agent_2"],
                "format": "json",
                "include_metadata": True
            }
        }


class PolicyMetadata(BaseModel):
    """Metadata for exported policy."""

    agent_id: str = Field(..., description="Agent identifier")
    agent_type: str = Field(..., description="Type of agent")
    exported_at: datetime = Field(..., description="Export timestamp")
    training_steps: int = Field(..., description="Training steps completed")
    average_reward: float = Field(..., description="Average reward")
    convergence_score: float = Field(..., description="Convergence score")


class PolicyImport(BaseModel):
    """Schema for importing agent policies."""

    policy_data: str = Field(..., description="Base64-encoded policy data")
    agent_id: Optional[str] = Field(
        None,
        description="Target agent ID (creates new if not provided)"
    )
    overwrite: bool = Field(
        False,
        description="Overwrite existing policy"
    )

    class Config:
        schema_extra = {
            "example": {
                "policy_data": "eyJ...",  # Base64 encoded
                "agent_id": "agent_1",
                "overwrite": False
            }
        }


class PolicyComparison(BaseModel):
    """Comparison between two policies."""

    metric: str = Field(..., description="Comparison metric name")
    agent1_value: float = Field(..., description="Agent 1 value")
    agent2_value: float = Field(..., description="Agent 2 value")
    difference: float = Field(..., description="Absolute difference")
    relative_difference: float = Field(..., description="Relative difference")


class PolicyCompare(BaseModel):
    """Result of policy comparison."""

    agent1_id: str = Field(..., description="First agent ID")
    agent2_id: str = Field(..., description="Second agent ID")
    comparisons: List[PolicyComparison] = Field(
        ...,
        description="List of metric comparisons"
    )
    overall_similarity: float = Field(
        ...,
        ge=0,
        le=1,
        description="Overall similarity score"
    )

    class Config:
        schema_extra = {
            "example": {
                "agent1_id": "agent_1",
                "agent2_id": "agent_2",
                "comparisons": [
                    {
                        "metric": "average_reward",
                        "agent1_value": 15.5,
                        "agent2_value": 14.2,
                        "difference": 1.3,
                        "relative_difference": 0.084
                    }
                ],
                "overall_similarity": 0.92
            }
        }
