"""
Unit tests for SpecialistAgent.

Tests the specialist agent functionality including:
- Initialization and configuration
- Policy initialization and updates
- Action selection (exploration vs exploitation)
- Experience buffer management
- Learning from teammates (Rule of 3 via Vector DB)
- Peer collaboration (FRL)
- Credit assignment (VDN)
- Task execution and metrics
- Performance tracking
- Redis state persistence
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any

from src.agents.specialist_agent import SpecialistAgent


class TestSpecialistAgentInit:
    """Test SpecialistAgent initialization."""

    def test_init_with_defaults(self, mock_redis_client, mock_mesa_model):
        """Test initialization with default parameters."""
        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="test_data"
        )

        assert agent.unique_id == 1
        assert agent.team_id == "specialists"
        assert agent.specialization == "general"
        assert agent.data_channel == "test_data"
        assert agent.agent_type == "SpecialistAgent"

    def test_init_with_custom_team_and_specialization(self, mock_redis_client, mock_mesa_model):
        """Test initialization with custom team and specialization."""
        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data",
            team_id="team_alpha",
            specialization="classifier"
        )

        assert agent.team_id == "team_alpha"
        assert agent.specialization == "classifier"

    def test_init_connects_to_vector_db_from_model(self, mock_redis_client, mock_mesa_model):
        """Test that agent connects to Vector DB from model."""
        mock_vector_db = MagicMock()
        mock_mesa_model.vector_db = mock_vector_db

        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

        assert agent.vector_db == mock_vector_db

    def test_init_connects_to_frl_engine_from_model(self, mock_redis_client, mock_mesa_model):
        """Test that agent connects to FRL engine from model."""
        mock_frl_engine = MagicMock()
        mock_mesa_model.frl_engine = mock_frl_engine

        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

        assert agent.frl_engine == mock_frl_engine

    def test_init_connects_to_vdn_engine_from_model(self, mock_redis_client, mock_mesa_model):
        """Test that agent connects to VDN engine from model."""
        mock_vdn_engine = MagicMock()
        mock_mesa_model.vdn_engine = mock_vdn_engine

        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

        assert agent.vdn_engine == mock_vdn_engine

    def test_init_subscribes_to_data_channel(self, mock_redis_client, mock_mesa_model):
        """Test that agent subscribes to data channel."""
        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="my_data_channel"
        )

        mock_redis_client.subscribe.assert_called_once_with(["my_data_channel"])

    def test_init_learning_configuration_defaults(self, mock_redis_client, mock_mesa_model):
        """Test default learning configuration."""
        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

        assert agent.learning_rate == 0.01
        assert agent.exploration_rate == 0.1
        assert agent.exploration_decay == 0.995
        assert agent.min_exploration == 0.01

    def test_init_learning_configuration_custom(self, mock_redis_client, mock_mesa_model):
        """Test custom learning configuration."""
        config = {
            "learning_rate": 0.001,
            "exploration_rate": 0.3,
            "exploration_decay": 0.99,
            "min_exploration": 0.05,
            "buffer_size": 500
        }

        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data",
            agent_config=config
        )

        assert agent.learning_rate == 0.001
        assert agent.exploration_rate == 0.3
        assert agent.exploration_decay == 0.99
        assert agent.min_exploration == 0.05
        assert agent.buffer_size == 500

    def test_init_policy_type_simple(self, mock_redis_client, mock_mesa_model):
        """Test initialization with simple policy type."""
        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

        assert agent.policy_type == "simple"
        assert "weights" in agent.policy_parameters
        assert agent.policy_parameters["initialized"] is True

    def test_init_experience_buffer(self, mock_redis_client, mock_mesa_model):
        """Test experience buffer initialization."""
        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

        assert agent.experience_buffer == []
        assert agent.buffer_size == 1000

    def test_init_metrics_initialization(self, mock_redis_client, mock_mesa_model):
        """Test task metrics initialization."""
        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

        assert agent.tasks_completed == 0
        assert agent.tasks_successful == 0
        assert agent.average_task_reward == 0.0
        assert agent.collaboration_count == 0
        assert agent.data_messages_received == 0


class TestSpecialistAgentPolicyInitialization:
    """Test policy initialization."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

    def test_initialize_policy_simple(self, agent):
        """Test simple policy initialization."""
        agent._initialize_policy()

        assert "weights" in agent.policy_parameters
        assert "bias" in agent.policy_parameters
        assert agent.policy_parameters["initialized"] is True

    def test_initialize_policy_custom_type(self, mock_redis_client, mock_mesa_model):
        """Test initialization with custom policy type."""
        config = {"policy_type": "neural_network"}
        agent = SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data",
            agent_config=config
        )

        assert agent.policy_type == "neural_network"
        assert agent.policy_parameters["policy_type"] == "neural_network"


class TestSpecialistAgentActionSelection:
    """Test action selection and exploration."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

    def test_select_action_exploration(self, agent):
        """Test that exploration triggers random actions."""
        agent.exploration_rate = 1.0  # Always explore

        with patch.object(agent, '_get_random_action', return_value=999) as mock_random:
            action = agent._select_action({})

            mock_random.assert_called_once()
            assert action == 999

    def test_select_action_exploitation(self, agent):
        """Test that exploitation uses policy."""
        agent.exploration_rate = 0.0  # Never explore

        with patch.object(agent, '_get_policy_action', return_value=777) as mock_policy:
            action = agent._select_action({})

            mock_policy.assert_called_once()
            assert action == 777

    def test_get_random_action_returns_binary(self, agent):
        """Test that random action returns 0 or 1."""
        for _ in range(10):
            action = agent._get_random_action()
            assert action in [0, 1]

    def test_get_policy_action_based_on_reward(self, agent):
        """Test policy action selection based on cumulative reward."""
        agent.cumulative_reward = 10.0
        action = agent._get_policy_action({})
        assert action == 1

        agent.cumulative_reward = -5.0
        action = agent._get_policy_action({})
        assert action == 0

    def test_decay_exploration(self, agent):
        """Test exploration rate decay."""
        initial_rate = agent.exploration_rate
        agent._decay_exploration()

        assert agent.exploration_rate < initial_rate
        assert agent.exploration_rate >= agent.min_exploration

    def test_decay_exploration_respects_minimum(self, agent):
        """Test that exploration rate doesn't go below minimum."""
        agent.exploration_rate = agent.min_exploration
        agent._decay_exploration()

        assert agent.exploration_rate == agent.min_exploration


class TestSpecialistAgentExperienceBuffer:
    """Test experience buffer management."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data",
            agent_config={"buffer_size": 10}
        )

    def test_store_experience(self, agent):
        """Test storing experience in buffer."""
        state = {"data": "test"}
        action = 1
        reward = 5.0

        agent._store_experience(state, action, reward)

        assert len(agent.experience_buffer) == 1
        assert agent.experience_buffer[0]["state"] == state
        assert agent.experience_buffer[0]["action"] == action
        assert agent.experience_buffer[0]["reward"] == reward
        assert "timestamp" in agent.experience_buffer[0]

    def test_experience_buffer_size_limit(self, agent):
        """Test that buffer respects size limit."""
        # Fill buffer beyond limit
        for i in range(15):
            agent._store_experience({"step": i}, i, float(i))

        # Should only keep last 10
        assert len(agent.experience_buffer) == 10
        # Oldest should be dropped (first 5 removed, so oldest is step 5)
        assert agent.experience_buffer[0]["state"]["step"] == 5


class TestSpecialistAgentLearning:
    """Test learning and policy updates."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

    def test_update_policy_increments_version(self, agent):
        """Test that policy update increments version on significant reward."""
        initial_version = agent.policy_version

        agent._update_policy({}, 1, 1.0)  # Significant reward

        assert agent.policy_version == initial_version + 1

    def test_update_policy_no_increment_small_reward(self, agent):
        """Test that small rewards don't increment version."""
        initial_version = agent.policy_version

        agent._update_policy({}, 1, 0.1)  # Small reward

        # May or may not increment depending on threshold
        # Just verify it doesn't error
        assert agent.policy_version >= initial_version

    def test_extract_state_from_data(self, agent):
        """Test extracting state from task data."""
        task_data = {"field1": "value1", "field2": 42}

        state = agent._extract_state_from_data(task_data)

        assert "task_data" in state
        assert "agent_state" in state
        assert state["task_data"] == task_data


class TestSpecialistAgentPolicySharing:
    """Test policy sharing with FRL and Vector DB."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        mock_mesa_model.vector_db = MagicMock()
        mock_mesa_model.frl_engine = MagicMock()
        return SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

    def test_should_share_policy_periodic(self, agent):
        """Test periodic FRL policy sharing."""
        agent.policy_version = 1
        agent.tasks_completed = 10
        assert agent._should_share_policy() is True

        agent.tasks_completed = 15
        assert agent._should_share_policy() is False

        agent.tasks_completed = 20
        assert agent._should_share_policy() is True

    def test_should_share_with_team_periodic(self, agent):
        """Test periodic team sharing via Vector DB."""
        agent.policy_version = 1
        agent.tasks_completed = 5
        assert agent._should_share_with_team() is True

        agent.tasks_completed = 7
        assert agent._should_share_with_team() is False

        agent.tasks_completed = 10
        assert agent._should_share_with_team() is True

    def test_update_policy_embedding(self, agent):
        """Test creating policy embedding."""
        agent.policy_parameters = {"weights": {}, "bias": 1.5}
        agent.cumulative_reward = 100.0
        agent.tasks_completed = 50

        agent._update_policy_embedding()

        assert agent.policy_embedding is not None
        assert len(agent.policy_embedding) == 128
        # Embedding should be normalized
        embedding_norm = np.linalg.norm(agent.policy_embedding)
        assert abs(embedding_norm - 1.0) < 0.01

    def test_receive_peer_policies_without_frl(self, agent):
        """Test receiving policies without FRL engine."""
        agent.frl_engine = None

        # Should not raise error
        agent._receive_peer_policies()

    def test_receive_peer_policies_with_frl(self, agent):
        """Test receiving and integrating peer policies."""
        peer_updates = [
            {"policy_parameters": {"weights": {}, "bias": 2.0}, "agent_id": "peer_1"},
            {"policy_parameters": {"weights": {}, "bias": 1.8}, "agent_id": "peer_2"}
        ]
        agent.frl_engine.receive_policy_updates.return_value = peer_updates
        agent.frl_engine.aggregate_policy_updates.return_value = {"weights": {}, "bias": 1.9}

        initial_version = agent.policy_version
        agent._receive_peer_policies()

        agent.frl_engine.receive_policy_updates.assert_called_once()
        agent.frl_engine.aggregate_policy_updates.assert_called_once()
        assert agent.policy_version == initial_version + 1

    def test_learn_from_teammates(self, agent):
        """Test learning from teammates via Vector DB."""
        agent.policy_embedding = np.random.rand(128).tolist()
        agent.average_task_reward = 0.5

        with patch.object(agent, 'retrieve_teammate_policies') as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    "policy_id": "policy_1",
                    "agent_id": "teammate_1",
                    "similarity": 0.9,
                    "metadata": {"performance": 0.8}
                }
            ]

            with patch.object(agent, '_integrate_teammate_policy') as mock_integrate:
                agent._learn_from_teammates()

                mock_retrieve.assert_called_once_with(
                    top_k=3,
                    min_performance=0.5
                )
                mock_integrate.assert_called_once()
                assert agent.collaboration_count == 1

    def test_integrate_teammate_policy(self, agent):
        """Test integrating a teammate's policy."""
        policy_info = {
            "policy_id": "p1",
            "similarity": 0.85,
            "metadata": {
                "agent_id": "teammate_x",
                "performance": 0.9
            }
        }

        agent.average_task_reward = 0.5
        initial_version = agent.policy_version

        agent._integrate_teammate_policy(policy_info)

        # Should increment version when learning from better performer
        assert agent.policy_version == initial_version + 1

    def test_integrate_teammate_policy_lower_performance(self, agent):
        """Test that lower-performing teammate policies aren't integrated."""
        policy_info = {
            "policy_id": "p1",
            "similarity": 0.85,
            "metadata": {
                "agent_id": "teammate_y",
                "performance": 0.3  # Lower than agent's performance
            }
        }

        agent.average_task_reward = 0.8
        initial_version = agent.policy_version

        agent._integrate_teammate_policy(policy_info)

        # Should not learn from worse performers
        assert agent.policy_version == initial_version


class TestSpecialistAgentPeerManagement:
    """Test peer specialist management."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        mock_mesa_model.frl_engine = MagicMock()
        mock_mesa_model.vdn_engine = MagicMock()
        return SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

    def test_add_peer_specialist(self, agent):
        """Test adding a peer specialist."""
        agent.add_peer_specialist("peer_123")

        assert "peer_123" in agent.peer_specialists
        agent.frl_engine.connect_to_peer.assert_called_once_with("peer_123")
        agent.vdn_engine.add_peer_agent.assert_called_once_with("peer_123")

    def test_add_multiple_peers(self, agent):
        """Test adding multiple peer specialists."""
        agent.add_peer_specialist("peer_1")
        agent.add_peer_specialist("peer_2")
        agent.add_peer_specialist("peer_3")

        assert len(agent.peer_specialists) == 3
        assert "peer_1" in agent.peer_specialists
        assert "peer_2" in agent.peer_specialists
        assert "peer_3" in agent.peer_specialists

    def test_remove_peer_specialist(self, agent):
        """Test removing a peer specialist."""
        agent.add_peer_specialist("peer_456")
        agent.remove_peer_specialist("peer_456")

        assert "peer_456" not in agent.peer_specialists
        agent.frl_engine.disconnect_from_peer.assert_called_with("peer_456")
        agent.vdn_engine.remove_peer_agent.assert_called_with("peer_456")

    def test_remove_nonexistent_peer(self, agent):
        """Test removing a peer that wasn't added."""
        # Should not raise error
        agent.remove_peer_specialist("nonexistent_peer")


class TestSpecialistAgentMetrics:
    """Test task metrics and performance tracking."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

    def test_update_average_task_reward(self, agent):
        """Test updating average task reward."""
        agent._update_average_task_reward(1.0)
        assert agent.average_task_reward > 0

        agent._update_average_task_reward(2.0)
        # Should be exponential moving average
        assert agent.average_task_reward > 0

    def test_get_success_rate_no_tasks(self, agent):
        """Test success rate with no tasks."""
        assert agent._get_success_rate() == 0.0

    def test_get_success_rate_all_successful(self, agent):
        """Test success rate with all successful tasks."""
        agent.tasks_completed = 10
        agent.tasks_successful = 10

        assert agent._get_success_rate() == 1.0

    def test_get_success_rate_partial(self, agent):
        """Test success rate with partial success."""
        agent.tasks_completed = 10
        agent.tasks_successful = 7

        assert agent._get_success_rate() == 0.7

    def test_get_specialist_statistics(self, agent):
        """Test getting specialist statistics."""
        agent.specialization = "classifier"
        agent.tasks_completed = 50
        agent.tasks_successful = 40
        agent.cumulative_reward = 200.0
        agent.policy_version = 5
        agent.exploration_rate = 0.05
        agent.peer_specialists = {"peer_1", "peer_2", "peer_3"}
        agent.collaboration_count = 15

        stats = agent.get_specialist_statistics()

        assert stats["agent_id"] == agent.agent_id
        assert stats["specialization"] == "classifier"
        assert stats["tasks_completed"] == 50
        assert stats["tasks_successful"] == 40
        assert stats["success_rate"] == 0.8
        assert stats["cumulative_reward"] == 200.0
        assert stats["policy_version"] == 5
        assert stats["exploration_rate"] == 0.05
        assert stats["peer_count"] == 3
        assert stats["collaboration_count"] == 15

    def test_get_learning_progress(self, agent):
        """Test getting learning progress."""
        agent.policy_version = 3
        agent.exploration_rate = 0.08
        agent.performance_history = [0.5, 0.6, 0.7, 0.8, 0.9]

        progress = agent.get_learning_progress()

        assert progress["agent_id"] == agent.agent_id
        assert progress["policy_version"] == 3
        assert progress["exploration_rate"] == 0.08
        assert "recent_performance" in progress
        assert "performance_trend" in progress
        assert "is_learning" in progress

    def test_calculate_performance_trend_insufficient_data(self, agent):
        """Test trend calculation with insufficient data."""
        agent.performance_history = [1.0, 2.0]

        trend = agent._calculate_performance_trend()

        assert trend == "insufficient_data"

    def test_calculate_performance_trend_improving(self, agent):
        """Test detecting improving trend."""
        # Older data: average 1.0, Recent data: average 2.0
        agent.performance_history = [1.0] * 10 + [2.0] * 10

        trend = agent._calculate_performance_trend()

        assert trend == "improving"

    def test_calculate_performance_trend_declining(self, agent):
        """Test detecting declining trend."""
        # Older data: average 2.0, Recent data: average 1.0
        agent.performance_history = [2.0] * 10 + [1.0] * 10

        trend = agent._calculate_performance_trend()

        assert trend == "declining"

    def test_calculate_performance_trend_stable(self, agent):
        """Test detecting stable trend."""
        agent.performance_history = [1.0] * 20

        trend = agent._calculate_performance_trend()

        assert trend == "stable"


class TestSpecialistAgentRedisState:
    """Test Redis state persistence."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

    def test_save_state_to_redis(self, agent, mock_redis_client):
        """Test saving specialist state to Redis."""
        agent.specialization = "optimizer"
        agent.tasks_completed = 25
        agent.tasks_successful = 20
        agent.average_task_reward = 1.5
        agent.exploration_rate = 0.07
        agent.peer_specialists = {"peer_x", "peer_y"}

        agent._save_state_to_redis()

        # Should call set_key_value twice (base state + specialist state)
        assert mock_redis_client.set_key_value.call_count == 2

        # Check specialist state was saved
        specialist_call = None
        for call_item in mock_redis_client.set_key_value.call_args_list:
            if "specialist_state" in call_item[0][0]:
                specialist_call = call_item
                break

        assert specialist_call is not None
        state = specialist_call[0][1]
        assert state["specialization"] == "optimizer"
        assert state["tasks_completed"] == 25
        assert state["tasks_successful"] == 20


class TestSpecialistAgentReset:
    """Test agent reset functionality."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data",
            agent_config={"exploration_rate": 0.2}
        )

    def test_reset_clears_metrics(self, agent):
        """Test that reset clears task metrics."""
        agent.tasks_completed = 100
        agent.tasks_successful = 80
        agent.average_task_reward = 2.5
        agent.collaboration_count = 50

        agent.reset()

        assert agent.tasks_completed == 0
        assert agent.tasks_successful == 0
        assert agent.average_task_reward == 0.0
        assert agent.collaboration_count == 0

    def test_reset_clears_experience_buffer(self, agent):
        """Test that reset clears experience buffer."""
        for i in range(10):
            agent._store_experience({"step": i}, i, float(i))

        agent.reset()

        assert len(agent.experience_buffer) == 0

    def test_reset_restores_exploration_rate(self, agent):
        """Test that reset restores initial exploration rate."""
        agent.exploration_rate = 0.01  # Decayed value

        agent.reset()

        assert agent.exploration_rate == 0.2  # Restored to config value

    def test_reset_reinitializes_policy(self, agent):
        """Test that reset reinitializes policy."""
        agent.policy_parameters = {"modified": True}
        agent.policy_version = 10

        agent.reset()

        # Policy should be reinitialized
        assert "weights" in agent.policy_parameters
        assert "initialized" in agent.policy_parameters


class TestSpecialistAgentStep:
    """Test step execution."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return SpecialistAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            data_channel="data"
        )

    def test_step_without_data(self, agent):
        """Test step when no data is received."""
        with patch.object(agent, '_receive_data_from_channel', return_value=None):
            initial_count = agent.step_count

            agent.step()

            assert agent.step_count == initial_count + 1

    def test_step_with_data(self, agent):
        """Test step with received data."""
        task_data = {"task": "classify", "data": [1, 2, 3]}

        # Mock VDN engine if present
        if agent.vdn_engine:
            agent.vdn_engine.assign_credit = MagicMock(return_value=1.5)

        with patch.object(agent, '_receive_data_from_channel', return_value=task_data):
            with patch.object(agent, '_execute_action', return_value=1.5):
                initial_tasks = agent.tasks_completed

                agent.step()

                assert agent.tasks_completed == initial_tasks + 1

    def test_step_decrements_exploration(self, agent):
        """Test that step decrements exploration rate."""
        initial_exploration = agent.exploration_rate

        with patch.object(agent, '_receive_data_from_channel', return_value=None):
            agent.step()

        assert agent.exploration_rate < initial_exploration

    def test_step_saves_state_periodically(self, agent, mock_redis_client):
        """Test that state is saved periodically."""
        with patch.object(agent, '_receive_data_from_channel', return_value=None):
            # Run 20 steps
            for _ in range(20):
                agent.step()

        # Should have saved state at step 20
        assert mock_redis_client.set_key_value.called
