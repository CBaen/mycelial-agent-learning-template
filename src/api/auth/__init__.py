"""
Authentication and Authorization Module

JWT-based authentication with RBAC support.
"""

from .jwt_handler import (
    create_access_token,
    verify_token,
    get_current_user,
    get_current_active_user
)

from .rbac import (
    UserRole,
    check_permission,
    require_role,
    require_admin
)

from .models import (
    User,
    Token,
    TokenData,
    UserInDB
)

__all__ = [
    # JWT handlers
    "create_access_token",
    "verify_token",
    "get_current_user",
    "get_current_active_user",
    # RBAC
    "UserRole",
    "check_permission",
    "require_role",
    "require_admin",
    # Models
    "User",
    "Token",
    "TokenData",
    "UserInDB",
]
