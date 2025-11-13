"""
Authentication Models

Pydantic models for users, tokens, and authentication.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class Token(BaseModel):
    """JWT token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class TokenData(BaseModel):
    """Data stored in JWT token."""
    username: Optional[str] = None
    role: Optional[UserRole] = None
    scopes: List[str] = Field(default_factory=list)


class User(BaseModel):
    """User model for API responses."""
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[str] = Field(None, pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    full_name: Optional[str] = None
    role: UserRole = Field(default=UserRole.VIEWER)
    disabled: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "role": "operator",
                "disabled": False,
                "created_at": "2025-11-12T10:00:00Z"
            }
        }


class UserInDB(User):
    """User model with hashed password for database storage."""
    hashed_password: str


class UserCreate(BaseModel):
    """Schema for creating new users."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    email: Optional[str] = Field(None, pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    full_name: Optional[str] = None
    role: UserRole = Field(default=UserRole.VIEWER)

    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

    class Config:
        schema_extra = {
            "example": {
                "username": "john_doe",
                "password": "SecurePass123",
                "email": "john@example.com",
                "full_name": "John Doe",
                "role": "operator"
            }
        }


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str = Field(...)
    password: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "username": "john_doe",
                "password": "SecurePass123"
            }
        }
