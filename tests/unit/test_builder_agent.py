"""
Unit tests for BuilderAgent.

Tests the builder agent functionality including:
- Initialization and configuration
- Agent spawning from blueprints
- Agent hibernation and restoration
- Agent termination
- Agent cloning
- Auto-scaling decisions
- Resource constraint checking
- Spawn queue management
- Lifecycle hooks
- State management
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any

from src.agents.builder_agent import BuilderAgent
from src.core.builder_base import (
    AgentBlueprint,
    AgentSnapshot,
    SpawnTrigger,
    HibernationTrigger,
    AgentState
)


class TestBuilderAgentInit:
    """Test BuilderAgent initialization."""

    def test_init_with_defaults(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Test initialization with default parameters."""
        agent = BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger
        )

        assert agent.unique_id == 1
        assert agent.team_id == "builders"
        assert agent.agent_type == "BuilderAgent"

    def test_init_with_custom_team(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Test initialization with custom team."""
        agent = BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger,
            team_id="custom_builders"
        )

        assert agent.team_id == "custom_builders"

    def test_init_with_custom_config(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Test initialization with custom configuration."""
        config = {
            "scaling_check_interval": 20,
            "spawn_cooldown": 10,
            "max_agents": 50,
            "min_agents": 5
        }

        agent = BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger,
            agent_config=config
        )

        assert agent.scaling_check_interval == 20
        assert agent.spawn_cooldown == 10
        assert agent.max_agents == 50
        assert agent.min_agents == 5

    def test_init_metrics(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Test initialization of builder metrics."""
        agent = BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger
        )

        assert agent.spawn_requests == 0
        assert agent.successful_spawns == 0
        assert agent.failed_spawns == 0
        assert agent.last_spawn_time == 0


class TestBuilderAgentSpawning:
    """Test agent spawning functionality."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Create test builder agent."""
        return BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger
        )

    @pytest.fixture
    def mock_agent_class(self):
        """Create mock agent class for spawning."""
        mock_class = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_instance.agent_id = "TestAgent_123"
        mock_class.return_value = mock_agent_instance
        return mock_class

    def test_spawn_agent_success(self, agent, mock_agent_class, mock_mesa_model):
        """Test successful agent spawning."""
        mock_mesa_model.active_agents = {}
        mock_mesa_model.total_agents_created = 10
        mock_mesa_model.add_agent = MagicMock()

        blueprint = AgentBlueprint(
            agent_type="TestAgent",
            agent_class=mock_agent_class,
            config={"test_param": "value"},
            spawn_trigger=SpawnTrigger.MANUAL
        )

        spawned_agent = agent.spawn_agent(blueprint)

        assert spawned_agent is not None
        assert agent.successful_spawns == 1
        assert agent.spawn_requests == 1
        mock_mesa_model.add_agent.assert_called_once()

    def test_spawn_agent_increments_metrics(self, agent, mock_agent_class, mock_mesa_model):
        """Test that spawning increments metrics."""
        mock_mesa_model.active_agents = {}
        mock_mesa_model.total_agents_created = 0

        blueprint = AgentBlueprint(
            agent_type="TestAgent",
            agent_class=mock_agent_class,
            config={},
            spawn_trigger=SpawnTrigger.WORKLOAD_SPIKE
        )

        initial_requests = agent.spawn_requests
        initial_successful = agent.successful_spawns

        agent.spawn_agent(blueprint)

        assert agent.spawn_requests == initial_requests + 1
        assert agent.successful_spawns == initial_successful + 1

    def test_spawn_agent_exceeds_max_limit(self, agent, mock_agent_class, mock_mesa_model):
        """Test spawning fails when agent limit reached."""
        # Set up model with max agents
        mock_mesa_model.active_agents = {f"agent_{i}": MagicMock() for i in range(100)}
        agent.max_agents = 100

        blueprint = AgentBlueprint(
            agent_type="TestAgent",
            agent_class=mock_agent_class,
            config={},
            spawn_trigger=SpawnTrigger.MANUAL
        )

        with pytest.raises(RuntimeError, match="Resource constraints violated"):
            agent.spawn_agent(blueprint)

        assert agent.failed_spawns == 1

    def test_spawn_agent_calls_lifecycle_hook(self, agent, mock_agent_class, mock_mesa_model):
        """Test that spawning calls lifecycle hook."""
        mock_mesa_model.active_agents = {}
        mock_mesa_model.total_agents_created = 0

        with patch.object(agent, 'on_agent_spawned') as mock_hook:
            blueprint = AgentBlueprint(
                agent_type="TestAgent",
                agent_class=mock_agent_class,
                config={},
                spawn_trigger=SpawnTrigger.MANUAL
            )

            spawned_agent = agent.spawn_agent(blueprint)

            mock_hook.assert_called_once_with(spawned_agent)

    def test_can_spawn_respects_cooldown(self, agent):
        """Test that _can_spawn respects cooldown period."""
        agent.spawn_cooldown = 5
        agent.last_spawn_time = agent.step_count

        assert agent._can_spawn() is False

        agent.step_count += 10  # Exceed cooldown
        assert agent._can_spawn() is True


class TestBuilderAgentHibernation:
    """Test agent hibernation functionality."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Create test builder agent."""
        return BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger
        )

    def test_hibernate_agent_success(self, agent, mock_mesa_model):
        """Test successful agent hibernation."""
        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.agent_id = "agent_to_hibernate"
        mock_agent.agent_type = "SpecialistAgent"

        mock_mesa_model.get_agent_by_id = MagicMock(return_value=mock_agent)
        mock_mesa_model.remove_agent = MagicMock()

        with patch.object(agent, 'save_agent_state', return_value={"state": "data"}):
            snapshot = agent.hibernate_agent(
                agent_id="agent_to_hibernate",
                trigger=HibernationTrigger.POOR_PERFORMANCE
            )

            assert snapshot is not None
            assert snapshot.agent_id == "agent_to_hibernate"
            assert snapshot.agent_type == "SpecialistAgent"
            assert snapshot.can_restore is True
            assert snapshot.hibernation_trigger == HibernationTrigger.POOR_PERFORMANCE
            mock_mesa_model.remove_agent.assert_called_once()

    def test_hibernate_agent_not_found(self, agent, mock_mesa_model):
        """Test hibernation fails for non-existent agent."""
        mock_mesa_model.get_agent_by_id = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="Agent not found"):
            agent.hibernate_agent("nonexistent_agent", HibernationTrigger.LOW_WORKLOAD)

    def test_hibernate_agent_already_hibernated(self, agent, mock_mesa_model):
        """Test hibernation fails for already hibernated agent."""
        mock_agent = MagicMock()
        mock_agent.agent_id = "agent_123"

        mock_mesa_model.get_agent_by_id = MagicMock(return_value=mock_agent)

        # Add to hibernation snapshots
        agent.hibernation_snapshots["agent_123"] = MagicMock()

        with pytest.raises(ValueError, match="already hibernated"):
            agent.hibernate_agent("agent_123", HibernationTrigger.RESOURCE_PRESSURE)

    def test_hibernate_agent_calls_lifecycle_hook(self, agent, mock_mesa_model):
        """Test that hibernation calls lifecycle hook."""
        mock_agent = MagicMock()
        mock_agent.agent_id = "agent_hook_test"
        mock_agent.agent_type = "TestAgent"

        mock_mesa_model.get_agent_by_id = MagicMock(return_value=mock_agent)

        with patch.object(agent, 'save_agent_state', return_value={}):
            with patch.object(agent, 'on_agent_hibernated') as mock_hook:
                snapshot = agent.hibernate_agent(
                    "agent_hook_test",
                    HibernationTrigger.SCHEDULED
                )

                mock_hook.assert_called_once()


class TestBuilderAgentRestoration:
    """Test agent restoration from hibernation."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Create test builder agent."""
        return BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger
        )

    def test_restore_agent_success(self, agent, mock_mesa_model):
        """Test successful agent restoration."""
        # Create mock snapshot
        snapshot = AgentSnapshot(
            agent_id="hibernated_agent",
            agent_type="SpecialistAgent",
            state={"config": {}},
            hibernation_time=time.time(),
            hibernation_trigger=HibernationTrigger.LOW_WORKLOAD,
            memory_size_bytes=1024,
            can_restore=True
        )

        agent.hibernation_snapshots["hibernated_agent"] = snapshot

        mock_mesa_model.active_agents = {}
        mock_mesa_model.total_agents_created = 0

        with patch.object(agent, 'spawn_agent') as mock_spawn:
            with patch.object(agent, 'load_agent_state'):
                mock_new_agent = MagicMock()
                mock_new_agent.agent_id = "restored_agent"
                mock_spawn.return_value = mock_new_agent

                restored_agent = agent.restore_agent("hibernated_agent")

                assert restored_agent is not None
                mock_spawn.assert_called_once()

    def test_restore_agent_not_found(self, agent):
        """Test restoration fails for non-existent snapshot."""
        with pytest.raises(ValueError, match="Snapshot not found"):
            agent.restore_agent("nonexistent_snapshot")

    def test_restore_agent_cannot_restore(self, agent):
        """Test restoration fails when snapshot marked as non-restorable."""
        snapshot = AgentSnapshot(
            agent_id="broken_agent",
            agent_type="TestAgent",
            state={},
            hibernation_time=time.time(),
            hibernation_trigger=HibernationTrigger.MANUAL,
            memory_size_bytes=0,
            can_restore=False
        )

        agent.hibernation_snapshots["broken_agent"] = snapshot

        with pytest.raises(ValueError, match="cannot be restored"):
            agent.restore_agent("broken_agent")


class TestBuilderAgentTermination:
    """Test agent termination functionality."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Create test builder agent."""
        return BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger
        )

    def test_terminate_agent_with_state_preservation(self, agent, mock_redis_client, mock_mesa_model):
        """Test termination with state preservation."""
        mock_agent = MagicMock()
        mock_agent.agent_id = "agent_to_terminate"

        mock_mesa_model.get_agent_by_id = MagicMock(return_value=mock_agent)
        mock_mesa_model.remove_agent = MagicMock()

        with patch.object(agent, 'save_agent_state', return_value={"state": "preserved"}):
            agent.terminate_agent("agent_to_terminate", preserve_state=True)

            mock_redis_client.set_key_value.assert_called_once()
            mock_mesa_model.remove_agent.assert_called_once()

    def test_terminate_agent_without_state_preservation(self, agent, mock_redis_client, mock_mesa_model):
        """Test termination without state preservation."""
        mock_agent = MagicMock()
        mock_agent.agent_id = "agent_discard"

        mock_mesa_model.get_agent_by_id = MagicMock(return_value=mock_agent)
        mock_mesa_model.remove_agent = MagicMock()

        agent.terminate_agent("agent_discard", preserve_state=False)

        # Should not save state
        mock_redis_client.set_key_value.assert_not_called()
        mock_mesa_model.remove_agent.assert_called_once()

    def test_terminate_nonexistent_agent(self, agent, mock_mesa_model):
        """Test terminating non-existent agent (should not error)."""
        mock_mesa_model.get_agent_by_id = MagicMock(return_value=None)

        # Should not raise error
        agent.terminate_agent("nonexistent", preserve_state=True)


class TestBuilderAgentCloning:
    """Test agent cloning functionality."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Create test builder agent."""
        return BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger
        )

    def test_clone_agent_success(self, agent, mock_mesa_model):
        """Test successful agent cloning."""
        source_agent = MagicMock()
        source_agent.agent_id = "source_agent"
        source_agent.agent_type = "SpecialistAgent"

        mock_mesa_model.get_agent_by_id = MagicMock(return_value=source_agent)
        mock_mesa_model.active_agents = {}
        mock_mesa_model.total_agents_created = 0

        with patch.object(agent, 'save_agent_state', return_value={"config": {}}):
            with patch.object(agent, 'spawn_agent') as mock_spawn:
                with patch.object(agent, 'load_agent_state'):
                    mock_clone = MagicMock()
                    mock_clone.agent_id = "cloned_agent"
                    mock_spawn.return_value = mock_clone

                    clone = agent.clone_agent("source_agent")

                    assert clone is not None
                    mock_spawn.assert_called_once()

    def test_clone_agent_with_modifications(self, agent, mock_mesa_model):
        """Test cloning with parameter modifications."""
        source_agent = MagicMock()
        source_agent.agent_id = "source"
        source_agent.agent_type = "TestAgent"

        mock_mesa_model.get_agent_by_id = MagicMock(return_value=source_agent)

        modifications = {"learning_rate": 0.001}

        with patch.object(agent, 'save_agent_state', return_value={"config": {}}):
            with patch.object(agent, 'spawn_agent', return_value=MagicMock()):
                with patch.object(agent, 'load_agent_state') as mock_load:
                    agent.clone_agent("source", modifications=modifications)

                    # Verify modifications were applied
                    loaded_state = mock_load.call_args[0][1]
                    assert "learning_rate" in loaded_state

    def test_clone_nonexistent_agent(self, agent, mock_mesa_model):
        """Test cloning non-existent agent fails."""
        mock_mesa_model.get_agent_by_id = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="Source agent not found"):
            agent.clone_agent("nonexistent")


class TestBuilderAgentAutoScaling:
    """Test auto-scaling decision logic."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Create test builder agent."""
        config = {"auto_scaling_enabled": True}
        return BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger,
            agent_config=config
        )

    def test_should_spawn_workload_spike(self, agent, mock_mesa_model):
        """Test spawning decision on workload spike."""
        mock_mesa_model.active_agents = {}

        metrics = {"workload": 0.9}

        should_spawn, trigger = agent.should_spawn_agent(metrics)

        assert should_spawn is True
        assert trigger.value == SpawnTrigger.WORKLOAD_SPIKE.value

    def test_should_spawn_performance_drop(self, agent, mock_mesa_model):
        """Test spawning decision on performance drop."""
        mock_mesa_model.active_agents = {}

        metrics = {"workload": 0.5, "average_performance": 0.3}

        should_spawn, trigger = agent.should_spawn_agent(metrics)

        assert should_spawn is True
        assert trigger.value == SpawnTrigger.PERFORMANCE_DROP.value

    def test_should_spawn_below_minimum(self, agent, mock_mesa_model):
        """Test spawning when below minimum agent count."""
        agent.min_agents = 5
        mock_mesa_model.active_agents = {}  # 0 agents

        metrics = {"workload": 0.5}

        should_spawn, trigger = agent.should_spawn_agent(metrics)

        assert should_spawn is True
        assert trigger.value == SpawnTrigger.SCHEDULED.value

    def test_should_not_spawn_normal_conditions(self, agent, mock_mesa_model):
        """Test no spawning under normal conditions."""
        agent.min_agents = 5
        agent.max_agents = 10
        mock_mesa_model.active_agents = {f"agent_{i}": MagicMock() for i in range(7)}

        metrics = {"workload": 0.5, "average_performance": 0.8}

        should_spawn, trigger = agent.should_spawn_agent(metrics)

        assert should_spawn is False
        assert trigger is None

    def test_should_hibernate_poor_performance(self, agent, mock_mesa_model):
        """Test hibernation decision for poor performance."""
        agent.min_agents = 5
        mock_mesa_model.active_agents = {f"agent_{i}": MagicMock() for i in range(10)}

        agent_metrics = {"recent_performance": 0.05}
        system_metrics = {"workload": 0.5}

        should_hibernate, trigger = agent.should_hibernate_agent(
            "test_agent",
            agent_metrics,
            system_metrics
        )

        assert should_hibernate is True
        assert trigger.value == HibernationTrigger.POOR_PERFORMANCE.value

    def test_should_hibernate_low_workload(self, agent, mock_mesa_model):
        """Test hibernation decision for low workload."""
        agent.min_agents = 5
        mock_mesa_model.active_agents = {f"agent_{i}": MagicMock() for i in range(10)}

        agent_metrics = {"recent_performance": 0.8}
        system_metrics = {"workload": 0.1}

        should_hibernate, trigger = agent.should_hibernate_agent(
            "test_agent",
            agent_metrics,
            system_metrics
        )

        assert should_hibernate is True
        assert trigger.value == HibernationTrigger.LOW_WORKLOAD.value

    def test_should_not_hibernate_normal_conditions(self, agent, mock_mesa_model):
        """Test no hibernation under normal conditions."""
        agent.min_agents = 5
        mock_mesa_model.active_agents = {f"agent_{i}": MagicMock() for i in range(7)}

        agent_metrics = {"recent_performance": 0.8}
        system_metrics = {"workload": 0.5}

        should_hibernate, trigger = agent.should_hibernate_agent(
            "test_agent",
            agent_metrics,
            system_metrics
        )

        assert should_hibernate is False
        assert trigger is None


class TestBuilderAgentStep:
    """Test step execution."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_vector_db, mock_sql_logger):
        """Create test builder agent."""
        return BuilderAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            vector_db=mock_vector_db,
            sql_logger=mock_sql_logger
        )

    def test_step_increments_count(self, agent):
        """Test that step increments step count."""
        initial_count = agent.step_count

        agent.step()

        assert agent.step_count == initial_count + 1

    def test_step_updates_risk_score(self, agent):
        """Test that step updates risk score based on resources."""
        with patch.object(agent, 'check_resource_constraints', return_value={"memory_ok": True}):
            agent.step()

            assert agent.risk_score >= 0.0
            assert agent.risk_score <= 1.0

    def test_step_calculates_reward(self, agent, mock_mesa_model):
        """Test that step calculates reward based on agent count."""
        agent.min_agents = 5
        agent.max_agents = 15
        mock_mesa_model.active_agents = {f"agent_{i}": MagicMock() for i in range(10)}

        initial_reward = agent.cumulative_reward
        agent.step()

        # Reward should be calculated
        assert agent.last_reward >= 0.0
        assert agent.cumulative_reward >= initial_reward
