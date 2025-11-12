"""
Pytest configuration and shared fixtures for MAE testing.

This module provides common fixtures and configuration for all tests,
including mock objects, test data, and setup/teardown utilities.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Redis Fixtures
# =============================================================================

@pytest.fixture
def mock_redis_client():
    """
    Mock Redis client for testing without actual Redis connection.
    """
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.client = MagicMock()
    mock_client.pubsub = MagicMock()

    # Mock storage
    mock_client._storage = {}

    # Mock methods with side effects for storage
    def mock_set(key, value):
        mock_client._storage[key] = value
        return True

    def mock_get(key):
        return mock_client._storage.get(key)

    def mock_exists(key):
        return key in mock_client._storage

    def mock_delete(key):
        return mock_client._storage.pop(key, None) is not None

    # Use MagicMock with side_effect to allow both functionality and assertions
    mock_client.set_key_value = MagicMock(side_effect=mock_set)
    mock_client.get_key_value = MagicMock(side_effect=mock_get)
    mock_client.client.exists = MagicMock(side_effect=mock_exists)
    mock_client.client.delete = MagicMock(side_effect=mock_delete)

    # Stream methods
    mock_client.write_to_stream = MagicMock(return_value="1-0")
    mock_client.read_from_stream = MagicMock(return_value=[])

    # Pub/Sub methods
    mock_client.publish = MagicMock(return_value=1)
    mock_client.subscribe = MagicMock()

    # Connection methods
    mock_client.ping = MagicMock(return_value=True)
    mock_client.close = MagicMock()
    mock_client.flushdb = MagicMock()

    return mock_client


@pytest.fixture
def redis_config():
    """Redis configuration for tests."""
    return {
        "host": "localhost",
        "port": 6379,
        "db": 15,  # Use high DB number for tests
        "password": None,
        "decode_responses": True
    }


# =============================================================================
# Vector DB Fixtures
# =============================================================================

@pytest.fixture
def temp_vector_db_dir():
    """Temporary directory for Vector DB persistence."""
    temp_dir = tempfile.mkdtemp(prefix="test_vector_db_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_vector_db():
    """Mock Vector DB for testing."""
    from unittest.mock import MagicMock

    mock_db = MagicMock()
    mock_db._storage = {}
    mock_db.is_initialized = False

    def mock_initialize():
        mock_db.is_initialized = True

    def mock_add_policy(policy_id, agent_id, embedding, metadata):
        mock_db._storage[policy_id] = {
            "agent_id": agent_id,
            "embedding": embedding,
            "metadata": metadata
        }

    def mock_search(query_embedding, top_k=5, filter_dict=None):
        results = []
        for policy_id, data in list(mock_db._storage.items())[:top_k]:
            results.append({
                "policy_id": policy_id,
                "agent_id": data["agent_id"],
                "similarity": 0.85,
                "metadata": data["metadata"]
            })
        return results

    mock_db.initialize = mock_initialize
    mock_db.add_policy_embedding = mock_add_policy
    mock_db.search_similar_policies = mock_search
    mock_db.close = MagicMock()

    return mock_db


# =============================================================================
# SQLite Fixtures
# =============================================================================

@pytest.fixture
def temp_sqlite_db():
    """Temporary SQLite database file."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sql_logger(temp_sqlite_db):
    """SQLite logger instance with temporary database."""
    from connectors.sql_logger import SQLiteLogger

    logger = SQLiteLogger(
        db_path=str(temp_sqlite_db),
        queue_size=100,
        batch_size=10,
        flush_interval=1.0
    )

    yield logger

    logger.stop(timeout=2.0)


@pytest.fixture
def mock_sql_logger():
    """Mock SQLite logger for testing."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.log_agent_event = MagicMock()
    mock.log_pattern_detected = MagicMock()
    mock.log_performance_metric = MagicMock()
    mock.log_system_event = MagicMock()
    mock.log_risk_event = MagicMock()
    mock.flush = MagicMock()
    mock.stop = MagicMock()

    return mock


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def mock_mesa_model():
    """Mock Mesa model for agent testing (Mesa 3.x compatible)."""
    from unittest.mock import MagicMock

    model = MagicMock()
    model.current_step = 0
    model.schedule = MagicMock()
    model.schedule.agents = []
    model.active_agents = {}
    model.hibernated_agents = {}
    model.total_agents_created = 0

    # Mesa 3.x auto-assigns unique_id via next_id()
    model._next_id = 0
    def get_next_id():
        model._next_id += 1
        return model._next_id
    model.next_id = MagicMock(side_effect=get_next_id)

    # Mock methods
    model.add_agent = MagicMock()
    model.remove_agent = MagicMock()
    model.get_agent_by_id = MagicMock(return_value=None)

    # Mock learning components (FRL, VDN, Vector DB)
    # These are accessed by agents via model.frl_engine, model.vdn_engine, etc.
    model.frl_engine = MagicMock()
    model.frl_engine.share_policy_update = MagicMock(return_value=0)
    model.vdn_engine = MagicMock()
    model.vdn_engine.assign_credit = MagicMock(return_value=0.0)
    model.vector_db = MagicMock()

    return model


# =============================================================================
# Agent Fixtures
# =============================================================================

@pytest.fixture
def agent_config():
    """Standard agent configuration for tests."""
    return {
        "learning_rate": 0.01,
        "exploration_rate": 0.1,
        "buffer_size": 100,
        "batch_size": 10
    }


@pytest.fixture
def sample_policy_parameters():
    """Sample policy parameters for testing."""
    return {
        "weights": {"feature_1": 0.5, "feature_2": 0.3},
        "bias": 0.1,
        "initialized": True
    }


@pytest.fixture
def sample_policy_embedding():
    """Sample policy embedding vector."""
    embedding = np.random.rand(128)
    return (embedding / np.linalg.norm(embedding)).tolist()


# =============================================================================
# Simulation Fixtures
# =============================================================================

@pytest.fixture
def adversarial_config():
    """Configuration for adversarial simulations."""
    return {
        "num_healthy_agents": 5,
        "num_toxic_agents": 1,
        "test_scenario": "policy_contagion",
        "num_steps": 50,
        "random_seed": 42
    }


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_agent_state():
    """Sample agent state dictionary."""
    return {
        "agent_id": "TestAgent_1",
        "agent_type": "SpecialistAgent",
        "team_id": "test_team",
        "step_count": 10,
        "cumulative_reward": 5.5,
        "policy_version": 2,
        "risk_score": 0.3,
        "is_isolated": False,
        "performance_history": [0.5, 0.6, 0.55, 0.58, 0.52]
    }


@pytest.fixture
def sample_stream_data():
    """Sample Redis stream data."""
    return [
        ("1-0", {"data": "value1", "timestamp": "2024-01-01"}),
        ("1-1", {"data": "value2", "timestamp": "2024-01-02"}),
        ("1-2", {"data": "value3", "timestamp": "2024-01-03"})
    ]


# =============================================================================
# HAVEN and Builder Fixtures
# =============================================================================

@pytest.fixture
def mock_haven_coordinator():
    """Mock HAVEN risk coordinator."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.register_agent = MagicMock()
    mock.unregister_agent = MagicMock()
    mock.assess_agent_risk = MagicMock()
    mock.assess_system_risk = MagicMock(return_value=0.0)
    mock.recommend_intervention = MagicMock()
    mock.execute_intervention = MagicMock(return_value=True)
    mock.detect_policy_contagion = MagicMock()
    mock.identify_contagion_source = MagicMock(return_value=[])
    mock.get_system_health_report = MagicMock(return_value={})
    mock.export_risk_analytics = MagicMock(return_value={})
    return mock


# =============================================================================
# Cleanup Utilities
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_numpy_random():
    """Reset numpy random state after each test."""
    yield
    np.random.seed(None)


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_redis: marks tests that require Redis connection"
    )
    config.addinivalue_line(
        "markers", "requires_chromadb: marks tests that require ChromaDB"
    )
