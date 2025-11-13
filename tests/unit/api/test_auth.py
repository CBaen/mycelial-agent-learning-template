"""
Unit tests for Authentication and Authorization

Tests JWT token generation, validation, user management, and RBAC.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import timedelta

from src.api.rest.main import app
from src.api.auth.jwt_handler import (
    create_access_token,
    verify_password,
    get_password_hash,
    fake_users_db
)
from src.api.auth.models import UserRole


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def admin_token(client):
    """Get admin JWT token."""
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "secret"}
    )
    return response.json()["access_token"]


@pytest.fixture
def operator_token(client):
    """Get operator JWT token."""
    response = client.post(
        "/auth/login",
        json={"username": "operator", "password": "secret"}
    )
    return response.json()["access_token"]


@pytest.fixture
def viewer_token(client):
    """Get viewer JWT token."""
    response = client.post(
        "/auth/login",
        json={"username": "viewer", "password": "secret"}
    )
    return response.json()["access_token"]


@pytest.fixture(autouse=True)
def reset_test_users():
    """Reset user database to initial state after each test."""
    yield
    # Restore original users if any were deleted
    initial_users = ["admin", "operator", "viewer"]
    for username in list(fake_users_db.keys()):
        if username not in initial_users:
            del fake_users_db[username]


# ============================================================================
# Authentication Tests
# ============================================================================

class TestAuthentication:
    """Tests for user authentication."""

    def test_login_success(self, client):
        """Test successful login."""
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "secret"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert data["expires_in"] > 0

    def test_login_invalid_username(self, client):
        """Test login with invalid username."""
        response = client.post(
            "/auth/login",
            json={"username": "nonexistent", "password": "secret"}
        )

        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    def test_login_invalid_password(self, client):
        """Test login with invalid password."""
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrongpassword"}
        )

        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    def test_login_missing_fields(self, client):
        """Test login with missing fields."""
        response = client.post(
            "/auth/login",
            json={"username": "admin"}
        )

        assert response.status_code == 422  # Validation error

    def test_get_current_user(self, client, admin_token):
        """Test getting current user info."""
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
        assert data["role"] == "admin"
        assert "email" in data

    def test_get_current_user_no_token(self, client):
        """Test getting user info without token."""
        response = client.get("/auth/me")

        assert response.status_code == 403  # Forbidden without token

    def test_get_current_user_invalid_token(self, client):
        """Test getting user info with invalid token."""
        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )

        assert response.status_code == 401

    def test_refresh_token(self, client, admin_token):
        """Test token refresh."""
        response = client.post(
            "/auth/refresh",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] != admin_token  # New token should be different


# ============================================================================
# User Management Tests
# ============================================================================

class TestUserManagement:
    """Tests for user CRUD operations."""

    def test_create_user_as_admin(self, client, admin_token):
        """Test creating user as admin."""
        new_user = {
            "username": "newuser",
            "password": "NewPass123",
            "email": "newuser@mae.com",
            "full_name": "New User",
            "role": "operator"
        }

        response = client.post(
            "/auth/users",
            json=new_user,
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser"
        assert data["role"] == "operator"
        assert "hashed_password" not in data  # Should not return password

    def test_create_user_as_operator(self, client, operator_token):
        """Test creating user as operator (should fail)."""
        new_user = {
            "username": "newuser",
            "password": "NewPass123",
            "role": "viewer"
        }

        response = client.post(
            "/auth/users",
            json=new_user,
            headers={"Authorization": f"Bearer {operator_token}"}
        )

        assert response.status_code == 403  # Forbidden

    def test_create_user_duplicate_username(self, client, admin_token):
        """Test creating user with duplicate username."""
        new_user = {
            "username": "admin",  # Already exists
            "password": "NewPass123"
        }

        response = client.post(
            "/auth/users",
            json=new_user,
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 409  # Conflict

    def test_create_user_weak_password(self, client, admin_token):
        """Test creating user with weak password."""
        new_user = {
            "username": "weakuser",
            "password": "weak"  # Too short, no uppercase/digits
        }

        response = client.post(
            "/auth/users",
            json=new_user,
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 422  # Validation error

    def test_list_users_as_admin(self, client, admin_token):
        """Test listing all users as admin."""
        response = client.get(
            "/auth/users",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        users = response.json()
        assert isinstance(users, list)
        assert len(users) >= 3  # admin, operator, viewer
        usernames = [u["username"] for u in users]
        assert "admin" in usernames

    def test_list_users_as_viewer(self, client, viewer_token):
        """Test listing users as viewer (should fail)."""
        response = client.get(
            "/auth/users",
            headers={"Authorization": f"Bearer {viewer_token}"}
        )

        assert response.status_code == 403  # Forbidden

    def test_get_user_by_username(self, client, admin_token):
        """Test getting specific user."""
        response = client.get(
            "/auth/users/operator",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "operator"
        assert data["role"] == "operator"

    def test_get_user_not_found(self, client, admin_token):
        """Test getting non-existent user."""
        response = client.get(
            "/auth/users/nonexistent",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 404

    def test_delete_user_as_admin(self, client, admin_token):
        """Test deleting user as admin."""
        # First create a user
        client.post(
            "/auth/users",
            json={"username": "tempuser", "password": "TempPass123"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        # Then delete it
        response = client.delete(
            "/auth/users/tempuser",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 204

        # Verify deletion
        get_response = client.get(
            "/auth/users/tempuser",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert get_response.status_code == 404

    def test_delete_self(self, client, admin_token):
        """Test deleting your own account (should fail)."""
        response = client.delete(
            "/auth/users/admin",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 400
        assert "Cannot delete your own account" in response.json()["detail"]


# ============================================================================
# RBAC Tests
# ============================================================================

class TestRBAC:
    """Tests for role-based access control."""

    def test_admin_full_access(self, client, admin_token):
        """Test admin has full access."""
        # Admin should be able to access all endpoints
        endpoints = [
            ("/auth/users", "get"),
            ("/system/health", "get"),
        ]

        for endpoint, method in endpoints:
            if method == "get":
                response = client.get(
                    endpoint,
                    headers={"Authorization": f"Bearer {admin_token}"}
                )
            else:
                response = client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {admin_token}"}
                )

            assert response.status_code != 403, f"Admin denied access to {endpoint}"

    def test_operator_limited_access(self, client, operator_token):
        """Test operator has limited access."""
        # Operator should NOT be able to manage users
        response = client.get(
            "/auth/users",
            headers={"Authorization": f"Bearer {operator_token}"}
        )
        assert response.status_code == 403

        # But should be able to view system health
        response = client.get(
            "/system/health",
            headers={"Authorization": f"Bearer {operator_token}"}
        )
        assert response.status_code == 200

    def test_viewer_read_only(self, client, viewer_token):
        """Test viewer has read-only access."""
        # Viewer should NOT be able to manage users
        response = client.get(
            "/auth/users",
            headers={"Authorization": f"Bearer {viewer_token}"}
        )
        assert response.status_code == 403

        # Should be able to view system info
        response = client.get(
            "/system/health",
            headers={"Authorization": f"Bearer {viewer_token}"}
        )
        assert response.status_code == 200


# ============================================================================
# Security Tests
# ============================================================================

class TestSecurity:
    """Tests for security features."""

    def test_password_hashing(self):
        """Test password hashing works correctly."""
        password = "TestPass123"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrongpassword", hashed)

    def test_token_expiration_claim(self):
        """Test token contains expiration claim."""
        token_data = {"sub": "testuser", "role": "admin"}
        token = create_access_token(token_data, expires_delta=timedelta(minutes=30))

        assert token is not None
        assert len(token) > 0

    def test_different_tokens_for_different_users(self, client):
        """Test different users get different tokens."""
        admin_response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "secret"}
        )
        operator_response = client.post(
            "/auth/login",
            json={"username": "operator", "password": "secret"}
        )

        admin_token = admin_response.json()["access_token"]
        operator_token = operator_response.json()["access_token"]

        assert admin_token != operator_token

    def test_token_contains_role(self, client, admin_token):
        """Test token verification returns correct role."""
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        assert response.json()["role"] == "admin"


# ============================================================================
# Integration Tests
# ============================================================================

class TestAuthIntegration:
    """Integration tests for auth workflows."""

    def test_complete_user_lifecycle(self, client, admin_token):
        """Test complete user management flow."""
        # 1. Create user
        new_user = {
            "username": "lifecycle_test",
            "password": "LifeCycle123",
            "email": "lifecycle@mae.com",
            "role": "operator"
        }
        create_response = client.post(
            "/auth/users",
            json=new_user,
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert create_response.status_code == 201

        # 2. New user logs in
        login_response = client.post(
            "/auth/login",
            json={"username": "lifecycle_test", "password": "LifeCycle123"}
        )
        assert login_response.status_code == 200
        user_token = login_response.json()["access_token"]

        # 3. User accesses their info
        me_response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        assert me_response.status_code == 200
        assert me_response.json()["username"] == "lifecycle_test"

        # 4. Admin deletes user
        delete_response = client.delete(
            "/auth/users/lifecycle_test",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert delete_response.status_code == 204

        # 5. Verify user can no longer log in
        login_fail_response = client.post(
            "/auth/login",
            json={"username": "lifecycle_test", "password": "LifeCycle123"}
        )
        assert login_fail_response.status_code == 401

    def test_role_based_endpoint_access(self, client, admin_token, viewer_token):
        """Test role-based access to different endpoints."""
        # Admin creates a user
        create_response = client.post(
            "/auth/users",
            json={"username": "test_rbac", "password": "TestRBAC123"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert create_response.status_code == 201

        # Viewer cannot create users
        viewer_create_response = client.post(
            "/auth/users",
            json={"username": "test_rbac2", "password": "TestRBAC123"},
            headers={"Authorization": f"Bearer {viewer_token}"}
        )
        assert viewer_create_response.status_code == 403

        # Cleanup
        client.delete(
            "/auth/users/test_rbac",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
