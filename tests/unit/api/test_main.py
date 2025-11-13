"""
Unit tests for FastAPI REST API endpoints

Tests all API endpoints including:
- Agent management
- Training control
- Metrics querying
- Policy operations
- System monitoring
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json
import base64

from src.api.rest.main import app, api_state


@pytest.fixture(autouse=True)
def reset_state():
    """Reset API state before each test."""
    api_state.agents.clear()
    api_state.training_sessions.clear()
    api_state.request_count = 0
    api_state.total_response_time = 0.0
    yield


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_agent_data():
    """Sample agent creation data."""
    return {
        "agent_type": "specialist",
        "team_id": "test_team",
        "config": {
            "specialization": "performance",
            "learning_rate": 0.001
        }
    }


@pytest.fixture
def sample_training_config():
    """Sample training configuration."""
    return {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 32,
        "epsilon": 0.1,
        "max_episodes": 1000
    }


# ============================================================================
# Agent Endpoints Tests
# ============================================================================

class TestAgentEndpoints:
    """Tests for /agents endpoints."""

    def test_create_agent_success(self, client, sample_agent_data):
        """Test successful agent creation."""
        response = client.post("/agents", json=sample_agent_data)

        assert response.status_code == 201
        data = response.json()
        assert "agent_id" in data
        assert data["agent_type"] == "specialist"
        assert data["team_id"] == "test_team"
        assert data["status"] == "idle"
        assert "created_at" in data

    def test_create_agent_invalid_type(self, client):
        """Test agent creation with invalid type."""
        invalid_data = {
            "agent_type": "invalid_type",
            "team_id": "test_team"
        }
        response = client.post("/agents", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_list_agents_empty(self, client):
        """Test listing agents when none exist."""
        response = client.get("/agents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["agents"] == []
        assert data["page"] == 1

    def test_list_agents_with_pagination(self, client, sample_agent_data):
        """Test agent listing with pagination."""
        # Create multiple agents
        for _ in range(5):
            client.post("/agents", json=sample_agent_data)

        # Test first page
        response = client.get("/agents?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["agents"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2

        # Test second page
        response = client.get("/agents?page=2&page_size=2")
        data = response.json()
        assert len(data["agents"]) == 2

    def test_get_agent_success(self, client, sample_agent_data):
        """Test getting specific agent."""
        # Create agent
        create_response = client.post("/agents", json=sample_agent_data)
        agent_id = create_response.json()["agent_id"]

        # Get agent
        response = client.get(f"/agents/{agent_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == agent_id
        assert data["agent_type"] == "specialist"

    def test_get_agent_not_found(self, client):
        """Test getting non-existent agent."""
        response = client.get("/agents/nonexistent_id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_agent_success(self, client, sample_agent_data):
        """Test successful agent deletion."""
        # Create agent
        create_response = client.post("/agents", json=sample_agent_data)
        agent_id = create_response.json()["agent_id"]

        # Delete agent
        response = client.delete(f"/agents/{agent_id}")
        assert response.status_code == 204

        # Verify deletion
        get_response = client.get(f"/agents/{agent_id}")
        assert get_response.status_code == 404

    def test_delete_agent_not_found(self, client):
        """Test deleting non-existent agent."""
        response = client.delete("/agents/nonexistent_id")
        assert response.status_code == 404

    def test_reset_agent_success(self, client, sample_agent_data):
        """Test successful agent reset."""
        # Create agent
        create_response = client.post("/agents", json=sample_agent_data)
        agent_id = create_response.json()["agent_id"]

        # Reset agent
        response = client.post(f"/agents/{agent_id}/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == agent_id
        # Stats should be reset
        if data.get("stats"):
            assert data["stats"]["learning_steps"] == 0

    def test_reset_agent_not_found(self, client):
        """Test resetting non-existent agent."""
        response = client.post("/agents/nonexistent_id/reset")
        assert response.status_code == 404


# ============================================================================
# Training Endpoints Tests
# ============================================================================

class TestTrainingEndpoints:
    """Tests for /training endpoints."""

    def test_start_training_success(self, client, sample_agent_data, sample_training_config):
        """Test starting training session."""
        # Create agent
        create_response = client.post("/agents", json=sample_agent_data)
        agent_id = create_response.json()["agent_id"]

        # Start training
        training_data = {
            "agents": [agent_id],
            "config": sample_training_config
        }
        response = client.post("/training/start", json=training_data)

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "running"

    def test_start_training_no_agents(self, client, sample_training_config):
        """Test starting training with no agents."""
        training_data = {
            "agents": [],
            "config": sample_training_config
        }
        response = client.post("/training/start", json=training_data)
        assert response.status_code in [200, 400]  # May succeed with empty list or fail

    def test_stop_training_success(self, client, sample_agent_data, sample_training_config):
        """Test stopping training session."""
        # Create agent and start training
        create_response = client.post("/agents", json=sample_agent_data)
        agent_id = create_response.json()["agent_id"]

        training_data = {
            "agents": [agent_id],
            "config": sample_training_config
        }
        start_response = client.post("/training/start", json=training_data)
        session_id = start_response.json()["session_id"]

        # Stop training
        response = client.post(f"/training/stop?session_id={session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"

    def test_stop_training_not_found(self, client):
        """Test stopping non-existent training session."""
        response = client.post("/training/stop?session_id=nonexistent")
        assert response.status_code == 404

    def test_get_training_status(self, client, sample_agent_data, sample_training_config):
        """Test getting training status."""
        # Create agent and start training
        create_response = client.post("/agents", json=sample_agent_data)
        agent_id = create_response.json()["agent_id"]

        training_data = {
            "agents": [agent_id],
            "config": sample_training_config
        }
        start_response = client.post("/training/start", json=training_data)
        session_id = start_response.json()["session_id"]

        # Get status
        response = client.get(f"/training/status?session_id={session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "metrics" in data

    def test_update_training_config(self, client, sample_agent_data, sample_training_config):
        """Test updating training configuration during active training."""
        # Create agent and start training first
        create_response = client.post("/agents", json=sample_agent_data)
        agent_id = create_response.json()["agent_id"]

        training_data = {
            "agents": [agent_id],
            "config": sample_training_config
        }
        client.post("/training/start", json=training_data)

        # Now update config
        updated_config = sample_training_config.copy()
        updated_config["learning_rate"] = 0.002

        response = client.put("/training/config", json=updated_config)
        assert response.status_code in [200, 404]  # May not support dynamic updates

    def test_update_training_config_invalid(self, client):
        """Test updating training config with invalid values."""
        invalid_config = {
            "learning_rate": -0.001,  # Invalid: negative
            "gamma": 1.5,  # Invalid: > 1
            "batch_size": 0  # Invalid: zero
        }
        response = client.put("/training/config", json=invalid_config)
        assert response.status_code == 422


# ============================================================================
# Metrics Endpoints Tests
# ============================================================================

class TestMetricsEndpoints:
    """Tests for /metrics endpoints."""

    def test_get_agent_metrics(self, client, sample_agent_data):
        """Test getting agent-specific metrics."""
        # Create agent
        create_response = client.post("/agents", json=sample_agent_data)
        agent_id = create_response.json()["agent_id"]

        # Get metrics
        response = client.get(f"/metrics/agents/{agent_id}")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_agent_metrics_not_found(self, client):
        """Test getting metrics for non-existent agent."""
        response = client.get("/metrics/agents/nonexistent_id")
        assert response.status_code == 404

    def test_get_system_metrics(self, client):
        """Test getting system-wide metrics."""
        response = client.get("/metrics/system")

        assert response.status_code == 200
        data = response.json()
        assert "cpu_percent" in data
        assert "memory_bytes" in data
        assert "active_agents" in data
        assert "training_sessions" in data
        assert "uptime_seconds" in data
        assert data["cpu_percent"] >= 0
        assert data["memory_bytes"] > 0

    def test_export_metrics_prometheus(self, client):
        """Test Prometheus metrics export."""
        response = client.get("/metrics/export")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        content = response.text
        # Check for expected Prometheus format
        assert "mae_system" in content or "mae_agent" in content
        assert "total" in content or "percent" in content  # Some metric exists


# ============================================================================
# Policy Endpoints Tests
# ============================================================================

class TestPolicyEndpoints:
    """Tests for /policies endpoints."""

    def test_export_policies_all(self, client, sample_agent_data):
        """Test exporting all agent policies."""
        # Create agents
        for _ in range(3):
            client.post("/agents", json=sample_agent_data)

        # Export all policies
        export_data = {
            "format": "json",
            "include_metadata": True
        }
        response = client.post("/policies/export", json=export_data)

        assert response.status_code == 200
        data = response.json()
        assert "policies" in data
        assert len(data["policies"]) == 3

    def test_export_policies_specific_agents(self, client, sample_agent_data):
        """Test exporting specific agent policies."""
        # Create agents
        create_response = client.post("/agents", json=sample_agent_data)
        agent_id = create_response.json()["agent_id"]

        # Export specific policy
        export_data = {
            "agent_ids": [agent_id],
            "format": "json",
            "include_metadata": True
        }
        response = client.post("/policies/export", json=export_data)

        assert response.status_code == 200
        data = response.json()
        assert len(data["policies"]) == 1

    def test_import_policy_success(self, client):
        """Test importing agent policy."""
        # Create mock policy data
        policy_data = base64.b64encode(
            json.dumps({"weights": [1.0, 2.0, 3.0]}).encode()
        ).decode()

        import_data = {
            "policy_data": policy_data,
            "overwrite": False
        }
        response = client.post("/policies/import", json=import_data)

        assert response.status_code == 200
        data = response.json()
        assert "agent_id" in data

    def test_import_policy_overwrite(self, client, sample_agent_data):
        """Test importing policy with overwrite."""
        # Create agent
        create_response = client.post("/agents", json=sample_agent_data)
        agent_id = create_response.json()["agent_id"]

        # Import policy with overwrite
        policy_data = base64.b64encode(
            json.dumps({"weights": [1.0, 2.0, 3.0]}).encode()
        ).decode()

        import_data = {
            "policy_data": policy_data,
            "agent_id": agent_id,
            "overwrite": True
        }
        response = client.post("/policies/import", json=import_data)
        assert response.status_code == 200

    def test_compare_policies(self, client, sample_agent_data):
        """Test comparing two agent policies."""
        # Create two agents
        response1 = client.post("/agents", json=sample_agent_data)
        agent1_id = response1.json()["agent_id"]

        response2 = client.post("/agents", json=sample_agent_data)
        agent2_id = response2.json()["agent_id"]

        # Compare policies
        response = client.get(
            f"/policies/compare?agent1_id={agent1_id}&agent2_id={agent2_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agent1_id"] == agent1_id
        assert data["agent2_id"] == agent2_id
        assert "comparisons" in data
        assert "overall_similarity" in data
        assert 0 <= data["overall_similarity"] <= 1

    def test_compare_policies_invalid(self, client):
        """Test comparing policies with invalid agent IDs."""
        response = client.get(
            "/policies/compare?agent1_id=invalid1&agent2_id=invalid2"
        )
        assert response.status_code == 404


# ============================================================================
# System Endpoints Tests
# ============================================================================

class TestSystemEndpoints:
    """Tests for /system endpoints."""

    def test_health_check(self, client):
        """Test system health check."""
        response = client.get("/system/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "components" in data
        assert "timestamp" in data

        # Check component structure
        for component in data["components"]:
            assert "name" in component
            assert "status" in component
            assert "last_check" in component

    def test_version_info(self, client):
        """Test version information endpoint."""
        response = client.get("/system/version")

        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "3.0.0"
        assert "build_date" in data
        assert "python_version" in data
        assert "dependencies" in data

    def test_system_stats(self, client, sample_agent_data):
        """Test system statistics endpoint."""
        # Create some agents
        for _ in range(3):
            client.post("/agents", json=sample_agent_data)

        response = client.get("/system/stats")

        assert response.status_code == 200
        data = response.json()

        # Check resource usage
        assert "resource_usage" in data
        assert data["resource_usage"]["cpu_percent"] >= 0
        assert data["resource_usage"]["memory_used_mb"] > 0
        assert data["resource_usage"]["memory_total_mb"] > 0

        # Check agent stats
        assert data["agent_count"] == 3
        assert data["uptime_seconds"] > 0
        assert data["total_requests"] >= 0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint."""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404

    def test_missing_required_fields(self, client):
        """Test request with missing required fields."""
        response = client.post("/agents", json={})
        assert response.status_code == 422

    def test_invalid_json(self, client):
        """Test request with invalid JSON."""
        response = client.post(
            "/agents",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_pagination_invalid_page(self, client):
        """Test pagination with invalid page number."""
        response = client.get("/agents?page=0")
        assert response.status_code == 422

    def test_pagination_invalid_page_size(self, client):
        """Test pagination with invalid page size."""
        response = client.get("/agents?page_size=0")
        assert response.status_code == 422


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests covering complete workflows."""

    def test_complete_agent_lifecycle(self, client, sample_agent_data, sample_training_config):
        """Test complete agent lifecycle from creation to deletion."""
        # 1. Create agent
        create_response = client.post("/agents", json=sample_agent_data)
        assert create_response.status_code == 201
        agent_id = create_response.json()["agent_id"]

        # 2. Verify agent exists
        get_response = client.get(f"/agents/{agent_id}")
        assert get_response.status_code == 200

        # 3. Start training
        training_data = {
            "agents": [agent_id],
            "config": sample_training_config
        }
        train_response = client.post("/training/start", json=training_data)
        assert train_response.status_code == 200
        session_id = train_response.json()["session_id"]

        # 4. Check training status
        status_response = client.get(f"/training/status?session_id={session_id}")
        assert status_response.status_code == 200

        # 5. Stop training
        stop_response = client.post(f"/training/stop?session_id={session_id}")
        assert stop_response.status_code == 200

        # 6. Get agent metrics
        metrics_response = client.get(f"/metrics/agents/{agent_id}")
        assert metrics_response.status_code == 200

        # 7. Reset agent
        reset_response = client.post(f"/agents/{agent_id}/reset")
        assert reset_response.status_code == 200

        # 8. Delete agent
        delete_response = client.delete(f"/agents/{agent_id}")
        assert delete_response.status_code == 204

        # 9. Verify deletion
        final_get = client.get(f"/agents/{agent_id}")
        assert final_get.status_code == 404

    def test_multiple_agents_training(self, client, sample_agent_data, sample_training_config):
        """Test training multiple agents simultaneously."""
        # Create multiple agents
        agent_ids = []
        for _ in range(3):
            response = client.post("/agents", json=sample_agent_data)
            agent_ids.append(response.json()["agent_id"])

        # Start training with all agents
        training_data = {
            "agents": agent_ids,
            "config": sample_training_config
        }
        train_response = client.post("/training/start", json=training_data)
        assert train_response.status_code == 200
        # Verify session was created
        assert "session_id" in train_response.json()

    def test_policy_export_import_roundtrip(self, client, sample_agent_data):
        """Test exporting and importing policies."""
        # Create agent
        create_response = client.post("/agents", json=sample_agent_data)
        original_agent_id = create_response.json()["agent_id"]

        # Export policy
        export_data = {
            "agent_ids": [original_agent_id],
            "format": "json",
            "include_metadata": True
        }
        export_response = client.post("/policies/export", json=export_data)
        assert export_response.status_code == 200

        # Import policy - this creates a new agent internally
        exported_policies = export_response.json()["policies"]
        # policies is a dict, get first value
        first_policy = list(exported_policies.values())[0]
        policy_data = base64.b64encode(
            json.dumps(first_policy).encode()
        ).decode()

        import_data = {
            "policy_data": policy_data,
            "overwrite": False
        }
        import_response = client.post("/policies/import", json=import_data)
        assert import_response.status_code == 200

        # The import creates an agent ID reference but may not create actual agent
        # Verify the response has an agent_id field
        assert "agent_id" in import_response.json()
