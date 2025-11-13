"""
Role-Based Access Control (RBAC)

Implements permission checking and role-based authorization.
"""

from typing import List, Callable
from functools import wraps
from fastapi import Depends, HTTPException, status

from .models import User, UserRole
from .jwt_handler import get_current_active_user


# Permission definitions
PERMISSIONS = {
    UserRole.ADMIN: [
        "agents:create", "agents:read", "agents:update", "agents:delete",
        "training:start", "training:stop", "training:read", "training:update",
        "metrics:read", "metrics:export",
        "policies:read", "policies:write", "policies:delete",
        "system:read", "system:write",
        "users:create", "users:read", "users:update", "users:delete"
    ],
    UserRole.OPERATOR: [
        "agents:create", "agents:read", "agents:update",
        "training:start", "training:stop", "training:read", "training:update",
        "metrics:read", "metrics:export",
        "policies:read", "policies:write",
        "system:read"
    ],
    UserRole.VIEWER: [
        "agents:read",
        "training:read",
        "metrics:read",
        "policies:read",
        "system:read"
    ]
}


def check_permission(user: User, permission: str) -> bool:
    """
    Check if user has specific permission.

    Args:
        user: User to check
        permission: Permission string (e.g., "agents:create")

    Returns:
        True if user has permission
    """
    user_permissions = PERMISSIONS.get(user.role, [])
    return permission in user_permissions


def require_permission(permission: str):
    """
    Dependency that requires specific permission.

    Args:
        permission: Required permission

    Returns:
        Dependency function
    """
    async def permission_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if not check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission}"
            )
        return current_user

    return permission_checker


def require_role(allowed_roles: List[UserRole]):
    """
    Dependency that requires specific role(s).

    Args:
        allowed_roles: List of allowed roles

    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}"
            )
        return current_user

    return role_checker


def require_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """
    Dependency that requires admin role.

    Args:
        current_user: Current authenticated user

    Returns:
        Admin user

    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# Permission decorators for route protection
def admin_only(func: Callable) -> Callable:
    """Decorator to restrict endpoint to admins only."""
    @wraps(func)
    async def wrapper(*args, current_user: User = Depends(require_admin), **kwargs):
        return await func(*args, current_user=current_user, **kwargs)
    return wrapper


def operator_or_admin(func: Callable) -> Callable:
    """Decorator to restrict endpoint to operators and admins."""
    checker = require_role([UserRole.ADMIN, UserRole.OPERATOR])

    @wraps(func)
    async def wrapper(*args, current_user: User = Depends(checker), **kwargs):
        return await func(*args, current_user=current_user, **kwargs)
    return wrapper
