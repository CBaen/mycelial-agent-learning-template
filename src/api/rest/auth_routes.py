"""
Authentication Routes

Endpoints for user authentication, token management, and user operations.
"""

from datetime import timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import (
    create_access_token,
    authenticate_user,
    get_current_active_user,
    require_admin,
    get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from src.api.auth.models import (
    User,
    Token,
    LoginRequest,
    UserCreate,
    UserInDB
)
from src.api.auth.jwt_handler import fake_users_db, get_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=Token, status_code=200)
async def login(login_data: LoginRequest) -> Token:
    """
    Authenticate user and return JWT token.

    **Credentials:**
    - `admin` / `secret` - Full admin access
    - `operator` / `secret` - Operator access
    - `viewer` / `secret` - Read-only access

    Returns:
        JWT access token
    """
    user = authenticate_user(login_data.username, login_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is disabled"
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get current authenticated user information.

    Requires valid JWT token in Authorization header.

    Returns:
        Current user details
    """
    return current_user


@router.post("/users", response_model=User, status_code=201)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(require_admin)
) -> User:
    """
    Create new user (admin only).

    **Requires:** Admin role

    Args:
        user_data: New user details

    Returns:
        Created user
    """
    # Check if user already exists
    if user_data.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists"
        )

    # Create user
    hashed_password = get_password_hash(user_data.password)
    new_user = UserInDB(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        role=user_data.role,
        hashed_password=hashed_password,
        disabled=False
    )

    # Store in database (mock)
    fake_users_db[user_data.username] = new_user.dict()

    # Return user without hashed password
    return User(**new_user.dict(exclude={"hashed_password"}))


@router.get("/users", response_model=List[User])
async def list_users(
    current_user: User = Depends(require_admin)
) -> List[User]:
    """
    List all users (admin only).

    **Requires:** Admin role

    Returns:
        List of all users
    """
    users = []
    for username in fake_users_db:
        user_data = fake_users_db[username]
        users.append(User(**{k: v for k, v in user_data.items() if k != "hashed_password"}))

    return users


@router.get("/users/{username}", response_model=User)
async def get_user_by_username(
    username: str,
    current_user: User = Depends(require_admin)
) -> User:
    """
    Get user by username (admin only).

    **Requires:** Admin role

    Args:
        username: Username to look up

    Returns:
        User details
    """
    user = get_user(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return User(**user.dict(exclude={"hashed_password"}))


@router.delete("/users/{username}", status_code=204)
async def delete_user(
    username: str,
    current_user: User = Depends(require_admin)
) -> None:
    """
    Delete user (admin only).

    **Requires:** Admin role

    Args:
        username: Username to delete
    """
    # Prevent deleting yourself
    if username == current_user.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    if username not in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    del fake_users_db[username]


@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_active_user)
) -> Token:
    """
    Refresh JWT token.

    Requires valid JWT token in Authorization header.

    Returns:
        New JWT access token
    """
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": current_user.username, "role": current_user.role},
        expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
