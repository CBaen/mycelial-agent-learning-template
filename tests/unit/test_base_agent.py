"""
Unit tests for MycelialAgent base agent class.

Tests the base agent functionality including:
- Initialization and configuration
- Step execution and lifecycle
- Policy sharing (FRL)
- Credit assignment (VDN)
- Team collaboration (Rule of 3)
- Redis state persistence
- Vector DB integration
- Risk management (HAVEN compatibility)
- Performance tracking
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List

from src.agents.base_agent import MycelialAgent


class TestMycelialAgentInit:
    """Test MycelialAgent initialization."""

    def test_init_with_defaults(self, mock_redis_client, mock_mesa_model):
        """Test initialization with default parameters."""
        agent = MycelialAgent(
            model=mock_mesa_model,
            redis_client=mock_redis_client
        )

        assert agent.unique_id == 1
        assert agent.model == mock_mesa_model
        assert agent.redis_client == mock_redis_client
        assert agent.team_id == "default"
        assert agent.agent_type == "MycelialAgent"
        assert agent.agent_id == "MycelialAgent_1"
        assert agent.step_count == 0
        assert agent.cumulative_reward == 0.0

    def test_init_with_custom_team_id(self, mock_redis_client, mock_mesa_model):
        """Test initialization with custom team_id."""
        agent = MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            team_id="team_alpha"
        )

        assert agent.team_id == "team_alpha"

    def test_init_with_agent_config(self, mock_redis_client, mock_mesa_model):
        """Test initialization with agent configuration."""
        config = {"learning_rate": 0.01, "discount": 0.99}
        agent = MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            agent_config=config
        )

        assert agent.agent_config == config

    def test_init_sets_learning_components_to_none(self, mock_redis_client, mock_mesa_model):
        """Test that learning components are None by default."""
        agent = MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client
        )

        assert agent.frl_engine is None
        assert agent.vdn_engine is None
        assert agent.vector_db is None

    def test_init_empty_teammates_list(self, mock_redis_client, mock_mesa_model):
        """Test that teammates list starts empty."""
        agent = MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client
        )

        assert agent.teammates == []

    def test_init_performance_tracking(self, mock_redis_client, mock_mesa_model):
        """Test initialization of performance tracking."""
        agent = MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client
        )

        assert agent.performance_history == []
        assert agent.policies_shared_with_team == 0
        assert agent.policies_received_from_team == 0

    def test_init_risk_metrics(self, mock_redis_client, mock_mesa_model):
        """Test initialization of risk metrics."""
        agent = MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client
        )

        assert agent.risk_score == 0.0
        assert agent.is_isolated is False


class TestMycelialAgentStep:
    """Test agent step execution."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client
        )

    def test_step_increments_count(self, agent):
        """Test that step increments step_count."""
        initial_count = agent.step_count

        agent.step()

        assert agent.step_count == initial_count + 1

    def test_step_calls_observe_state(self, agent):
        """Test that step calls _observe_state."""
        with patch.object(agent, '_observe_state', return_value={}) as mock_observe:
            agent.step()

            mock_observe.assert_called_once()

    def test_step_calls_select_action(self, agent):
        """Test that step calls _select_action."""
        with patch.object(agent, '_select_action', return_value=None) as mock_select:
            agent.step()

            mock_select.assert_called_once()

    def test_step_calls_execute_action(self, agent):
        """Test that step calls _execute_action."""
        with patch.object(agent, '_execute_action', return_value=0.0) as mock_execute:
            agent.step()

            mock_execute.assert_called_once()

    def test_step_updates_cumulative_reward(self, agent):
        """Test that step updates cumulative reward."""
        with patch.object(agent, '_execute_action', return_value=5.0):
            initial_reward = agent.cumulative_reward

            agent.step()

            assert agent.cumulative_reward == initial_reward + 5.0

    def test_step_updates_performance_history(self, agent):
        """Test that step updates performance history."""
        with patch.object(agent, '_execute_action', return_value=3.5):
            initial_len = len(agent.performance_history)

            agent.step()

            assert len(agent.performance_history) == initial_len + 1
            assert agent.performance_history[-1] == 3.5

    def test_step_calls_save_state_to_redis(self, agent):
        """Test that step saves state to Redis."""
        with patch.object(agent, '_save_state_to_redis') as mock_save:
            agent.step()

            mock_save.assert_called_once()

    def test_step_with_vdn_engine(self, agent):
        """Test step execution with VDN engine."""
        agent.vdn_engine = MagicMock()
        agent.vdn_engine.assign_credit.return_value = 2.5

        with patch.object(agent, '_execute_action', return_value=5.0):
            with patch.object(agent, '_update_policy') as mock_update:
                with patch.object(agent, 'get_local_reward', return_value=2.5) as mock_credit:
                    agent.step()

                    mock_update.assert_called_once()
                    # Verify it was called with the local credit
                    assert mock_update.call_args[0][2] == 2.5

    def test_step_with_frl_engine_shares_policy(self, agent):
        """Test step shares policy when FRL enabled."""
        agent.frl_engine = MagicMock()
        agent.frl_engine.share_policy_update.return_value = 3

        # Set up so sharing is triggered
        agent.step_count = 9  # Next step will be 10, triggering sharing

        with patch.object(agent, 'share_policy') as mock_share:
            agent.step()

            mock_share.assert_called_once()


class TestMycelialAgentPolicySharing:
    """Test policy sharing functionality."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client
        )

    def test_share_policy_without_frl_engine(self, agent):
        """Test that sharing without FRL engine returns 0."""
        result = agent.share_policy()

        assert result == 0

    def test_share_policy_with_frl_engine(self, agent):
        """Test policy sharing with FRL engine."""
        agent.frl_engine = MagicMock()
        agent.frl_engine.share_policy_update.return_value = 5
        agent.policy_parameters = {"weights": [1, 2, 3]}
        agent.policy_version = 2
        agent.cumulative_reward = 100.0

        result = agent.share_policy()

        assert result == 5
        agent.frl_engine.share_policy_update.assert_called_once()

        # Verify policy update structure
        call_args = agent.frl_engine.share_policy_update.call_args
        policy_update = call_args[0][0]
        assert "policy_parameters" in policy_update
        assert "policy_version" in policy_update
        assert "performance" in policy_update

    def test_should_share_policy_default(self, agent):
        """Test default sharing logic (every 10 steps)."""
        agent.step_count = 5
        assert agent._should_share_policy() is False

        agent.step_count = 10
        assert agent._should_share_policy() is True

        agent.step_count = 20
        assert agent._should_share_policy() is True

        agent.step_count = 15
        assert agent._should_share_policy() is False


class TestMycelialAgentCreditAssignment:
    """Test credit assignment with VDN."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client
        )

    def test_get_local_reward_without_vdn(self, agent):
        """Test that without VDN, agent receives full global reward."""
        global_reward = 10.0

        local_credit = agent.get_local_reward(global_reward)

        assert local_credit == global_reward

    def test_get_local_reward_with_vdn(self, agent):
        """Test credit assignment with VDN engine."""
        agent.vdn_engine = MagicMock()
        agent.vdn_engine.assign_credit.return_value = 3.5

        global_reward = 10.0
        local_credit = agent.get_local_reward(global_reward)

        assert local_credit == 3.5
        agent.vdn_engine.assign_credit.assert_called_once()

        # Verify call arguments
        call_args = agent.vdn_engine.assign_credit.call_args
        assert call_args[1]['global_reward'] == global_reward


class TestMycelialAgentProtectedMethods:
    """Test protected methods (meant to be overridden)."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client
        )

    def test_observe_state_default(self, agent, mock_mesa_model):
        """Test default _observe_state returns basic state."""
        mock_mesa_model.current_step = 42

        state = agent._observe_state()

        assert "step" in state
        assert "agent_id" in state
        assert state["step"] == 42
        assert state["agent_id"] == agent.agent_id

    def test_select_action_default(self, agent):
        """Test default _select_action returns None."""
        action = agent._select_action({})

        assert action is None

    def test_execute_action_default(self, agent):
        """Test default _execute_action returns 0.0."""
        reward = agent._execute_action(None)

        assert reward == 0.0

    def test_update_policy_default(self, agent):
        """Test default _update_policy does nothing."""
        # Should not raise error
        agent._update_policy({}, None, 0.0)

    def test_get_joint_action_default(self, agent):
        """Test default _get_joint_action returns only own action."""
        agent.last_action = "test_action"

        joint_action = agent._get_joint_action()

        assert agent.agent_id in joint_action
        assert joint_action[agent.agent_id] == "test_action"

    def test_get_recent_performance_empty_history(self, agent):
        """Test getting recent performance with no history."""
        performance = agent._get_recent_performance()

        assert performance == 0.0

    def test_get_recent_performance_with_history(self, agent):
        """Test getting recent performance average."""
        agent.performance_history = [1.0, 2.0, 3.0, 4.0, 5.0]

        performance = agent._get_recent_performance()

        assert performance == 3.0  # Average of all 5

    def test_get_recent_performance_limits_to_10(self, agent):
        """Test that recent performance uses last 10 steps."""
        agent.performance_history = list(range(1, 21))  # 1 to 20

        performance = agent._get_recent_performance()

        # Should average last 10: 11, 12, 13, ..., 20
        expected = sum(range(11, 21)) / 10
        assert performance == expected

    def test_update_performance_metrics(self, agent):
        """Test updating performance metrics."""
        agent._update_performance_metrics(5.0)

        assert len(agent.performance_history) == 1
        assert agent.performance_history[0] == 5.0

    def test_update_performance_metrics_limits_history(self, agent):
        """Test that performance history is limited to 1000."""
        # Fill history with 1001 items
        agent.performance_history = list(range(1001))

        agent._update_performance_metrics(999.0)

        assert len(agent.performance_history) == 1000


class TestMycelialAgentRedisOperations:
    """Test Redis state persistence."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            team_id="team_test"
        )

    def test_save_state_to_redis(self, agent, mock_redis_client):
        """Test saving agent state to Redis."""
        agent.step_count = 100
        agent.cumulative_reward = 50.5
        agent.policy_version = 3
        agent.risk_score = 0.3

        agent._save_state_to_redis()

        # Verify Redis set was called
        mock_redis_client.set_key_value.assert_called_once()

        # Verify call arguments
        call_args = mock_redis_client.set_key_value.call_args
        key = call_args[0][0]
        state_data = call_args[0][1]

        assert key == f"agent:state:{agent.agent_id}"
        assert state_data["agent_id"] == agent.agent_id
        assert state_data["team_id"] == "team_test"
        assert state_data["step_count"] == 100
        assert state_data["cumulative_reward"] == 50.5

    def test_load_state_from_redis(self, agent, mock_redis_client):
        """Test loading agent state from Redis."""
        saved_state = {
            "agent_id": agent.agent_id,
            "step_count": 200,
            "cumulative_reward": 75.0
        }
        # Store in mock's internal storage (used by side_effect)
        key = f"agent:state:{agent.agent_id}"
        mock_redis_client._storage[key] = saved_state

        loaded_state = agent._load_state_from_redis()

        assert loaded_state == saved_state
        mock_redis_client.get_key_value.assert_called_once_with(key)

    def test_load_state_from_redis_not_found(self, agent, mock_redis_client):
        """Test loading state when none exists."""
        mock_redis_client.get_key_value.return_value = None

        loaded_state = agent._load_state_from_redis()

        assert loaded_state is None

    def test_publish_message(self, agent, mock_redis_client):
        """Test publishing message to channel."""
        mock_redis_client.publish.return_value = 3

        subscribers = agent.publish_message("test_channel", {"msg": "hello"})

        assert subscribers == 3
        mock_redis_client.publish.assert_called_once_with(
            "test_channel",
            {"msg": "hello"}
        )

    def test_subscribe_to_channel(self, agent, mock_redis_client):
        """Test subscribing to channel."""
        agent.subscribe_to_channel("updates")

        mock_redis_client.subscribe.assert_called_once_with(["updates"])

    def test_write_to_stream(self, agent, mock_redis_client):
        """Test writing to Redis stream."""
        mock_redis_client.write_to_stream.return_value = "1234567890-0"

        entry_id = agent.write_to_stream("data_stream", {"value": 42})

        assert entry_id == "1234567890-0"
        mock_redis_client.write_to_stream.assert_called_once_with(
            "data_stream",
            {"value": 42}
        )

    def test_read_from_stream(self, agent, mock_redis_client):
        """Test reading from Redis stream."""
        mock_redis_client.read_from_stream.return_value = [
            ("1-0", {"data": "test1"}),
            ("1-1", {"data": "test2"})
        ]

        entries = agent.read_from_stream("data_stream", last_id="0-0", count=10)

        assert len(entries) == 2
        mock_redis_client.read_from_stream.assert_called_once()


class TestMycelialAgentRiskManagement:
    """Test risk management (HAVEN compatibility)."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client
        )

    def test_set_risk_score(self, agent):
        """Test setting risk score."""
        agent.set_risk_score(0.75)

        assert agent.risk_score == 0.75

    def test_set_risk_score_clamps_to_range(self, agent):
        """Test that risk score is clamped to [0, 1]."""
        agent.set_risk_score(1.5)
        assert agent.risk_score == 1.0

        agent.set_risk_score(-0.5)
        assert agent.risk_score == 0.0

    def test_isolate(self, agent):
        """Test isolating agent."""
        agent.isolate()

        assert agent.is_isolated is True

    def test_restore(self, agent):
        """Test restoring agent from isolation."""
        agent.is_isolated = True

        agent.restore()

        assert agent.is_isolated is False

    def test_get_state_summary(self, agent):
        """Test getting state summary."""
        agent.step_count = 50
        agent.cumulative_reward = 25.0
        agent.policy_version = 2
        agent.risk_score = 0.4

        summary = agent.get_state_summary()

        assert summary["agent_id"] == agent.agent_id
        assert summary["step_count"] == 50
        assert summary["cumulative_reward"] == 25.0
        assert summary["policy_version"] == 2
        assert summary["risk_score"] == 0.4
        assert summary["is_isolated"] is False

    def test_get_state_summary_with_frl(self, agent):
        """Test state summary includes FRL peer count."""
        agent.frl_engine = MagicMock()
        agent.frl_engine.get_connected_peer_count.return_value = 7

        summary = agent.get_state_summary()

        assert summary["connected_peers"] == 7


class TestMycelialAgentTeamCollaboration:
    """Test team collaboration (Rule of 3)."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            team_id="team_alpha"
        )

    def test_get_teammates(self, agent, mock_redis_client):
        """Test querying teammates."""
        # Mock Redis keys response
        mock_redis_client.client.keys.return_value = [
            "agent:state:Agent_1",
            "agent:state:Agent_2",
            "agent:state:Agent_3"
        ]

        # Mock get_key_value responses
        def get_state(key):
            if key == "agent:state:Agent_1":
                return {"agent_id": "Agent_1", "team_id": "team_alpha"}
            elif key == "agent:state:Agent_2":
                return {"agent_id": "Agent_2", "team_id": "team_alpha"}
            elif key == "agent:state:Agent_3":
                return {"agent_id": "Agent_3", "team_id": "team_beta"}
            return None

        mock_redis_client.get_key_value.side_effect = get_state

        # Agent's own ID should be filtered out
        agent.agent_id = "Agent_1"

        teammates = agent.get_teammates()

        # Should only include Agent_2 (same team, not self)
        assert "Agent_2" in teammates
        assert "Agent_1" not in teammates  # Self excluded
        assert "Agent_3" not in teammates  # Different team

    def test_get_teammates_empty_team(self, agent, mock_redis_client):
        """Test getting teammates when alone in team."""
        mock_redis_client.client.keys.return_value = [
            "agent:state:Agent_1"
        ]
        mock_redis_client.get_key_value.return_value = {
            "agent_id": "Agent_1",
            "team_id": "team_alpha"
        }

        agent.agent_id = "Agent_1"

        teammates = agent.get_teammates()

        assert len(teammates) == 0

    def test_share_policy_with_team_without_vector_db(self, agent):
        """Test sharing policy without Vector DB returns None."""
        result = agent.share_policy_with_team()

        assert result is None

    def test_share_policy_with_team_without_embedding(self, agent):
        """Test sharing policy without embedding returns None."""
        agent.vector_db = MagicMock()
        agent.policy_embedding = None

        result = agent.share_policy_with_team()

        assert result is None

    def test_share_policy_with_team(self, agent):
        """Test sharing policy with team via Vector DB."""
        agent.vector_db = MagicMock()
        agent.policy_embedding = np.random.rand(128)
        agent.policy_version = 3
        agent.cumulative_reward = 100.0

        policy_id = agent.share_policy_with_team()

        assert policy_id is not None
        assert agent.policies_shared_with_team == 1
        agent.vector_db.add_policy_embedding.assert_called_once()

        # Verify call includes team_id metadata
        call_args = agent.vector_db.add_policy_embedding.call_args
        metadata = call_args[1]['metadata']
        assert metadata['team_id'] == "team_alpha"

    def test_share_policy_with_team_error_handling(self, agent):
        """Test error handling when sharing fails."""
        agent.vector_db = MagicMock()
        agent.policy_embedding = np.random.rand(128)
        agent.vector_db.add_policy_embedding.side_effect = Exception("DB error")

        with pytest.raises(RuntimeError, match="Failed to share policy"):
            agent.share_policy_with_team()

    def test_retrieve_teammate_policies_without_vector_db(self, agent):
        """Test retrieving policies without Vector DB returns empty list."""
        result = agent.retrieve_teammate_policies()

        assert result == []

    def test_retrieve_teammate_policies_without_embedding(self, agent):
        """Test retrieving policies without embedding returns empty list."""
        agent.vector_db = MagicMock()
        agent.policy_embedding = None

        result = agent.retrieve_teammate_policies()

        assert result == []

    def test_retrieve_teammate_policies(self, agent):
        """Test retrieving teammate policies from Vector DB."""
        agent.vector_db = MagicMock()
        agent.policy_embedding = np.random.rand(128)

        # Mock search results
        agent.vector_db.search_similar_policies.return_value = [
            {
                "policy_id": "policy_1",
                "agent_id": "Agent_2",
                "similarity": 0.9,
                "metadata": {"team_id": "team_alpha", "performance": 0.8}
            },
            {
                "policy_id": "policy_2",
                "agent_id": agent.agent_id,  # Self, should be filtered
                "similarity": 1.0,
                "metadata": {"team_id": "team_alpha", "performance": 0.9}
            },
            {
                "policy_id": "policy_3",
                "agent_id": "Agent_3",
                "similarity": 0.85,
                "metadata": {"team_id": "team_alpha", "performance": 0.75}
            }
        ]

        policies = agent.retrieve_teammate_policies(top_k=5)

        # Should filter out self
        assert len(policies) == 2
        assert agent.policies_received_from_team == 2
        assert all(p["agent_id"] != agent.agent_id for p in policies)

    def test_retrieve_teammate_policies_with_performance_filter(self, agent):
        """Test retrieving policies with performance threshold."""
        agent.vector_db = MagicMock()
        agent.policy_embedding = np.random.rand(128)

        agent.vector_db.search_similar_policies.return_value = [
            {
                "policy_id": "p1",
                "agent_id": "Agent_2",
                "metadata": {"performance": 0.9}
            },
            {
                "policy_id": "p2",
                "agent_id": "Agent_3",
                "metadata": {"performance": 0.5}  # Below threshold
            },
            {
                "policy_id": "p3",
                "agent_id": "Agent_4",
                "metadata": {"performance": 0.85}
            }
        ]

        policies = agent.retrieve_teammate_policies(
            top_k=10,
            min_performance=0.7
        )

        # Should only include policies with performance >= 0.7
        assert len(policies) == 2
        assert all(p["metadata"]["performance"] >= 0.7 for p in policies)

    def test_retrieve_teammate_policies_error_handling(self, agent):
        """Test error handling when retrieval fails."""
        agent.vector_db = MagicMock()
        agent.policy_embedding = np.random.rand(128)
        agent.vector_db.search_similar_policies.side_effect = Exception("Search error")

        with pytest.raises(RuntimeError, match="Failed to retrieve policies"):
            agent.retrieve_teammate_policies()

    def test_get_team_statistics(self, agent, mock_redis_client):
        """Test getting team statistics."""
        # Mock teammates
        with patch.object(agent, 'get_teammates', return_value=["Agent_2", "Agent_3"]):
            # Mock teammate states
            def get_state(key):
                if key == "agent:state:Agent_2":
                    return {"cumulative_reward": 80.0, "risk_score": 0.3}
                elif key == "agent:state:Agent_3":
                    return {"cumulative_reward": 60.0, "risk_score": 0.2}
                return None

            mock_redis_client.get_key_value.side_effect = get_state

            agent.cumulative_reward = 100.0
            agent.policies_shared_with_team = 5
            agent.policies_received_from_team = 3

            stats = agent.get_team_statistics()

            assert stats["team_id"] == "team_alpha"
            assert stats["team_size"] == 2
            assert stats["teammates"] == ["Agent_2", "Agent_3"]
            assert stats["policies_shared"] == 5
            assert stats["policies_received"] == 3
            assert stats["team_avg_reward"] == 70.0  # (80 + 60) / 2
            assert stats["team_avg_risk"] == 0.25  # (0.3 + 0.2) / 2
            assert stats["my_contribution_rank"] == 1  # Best performance

    def test_calculate_team_rank(self, agent):
        """Test calculating team rank."""
        agent.cumulative_reward = 75.0

        team_rewards = [100.0, 80.0, 60.0, 40.0]

        rank = agent._calculate_team_rank(team_rewards)

        # Should be 3rd: 100, 80, 75, 60, 40
        assert rank == 3

    def test_calculate_team_rank_best_performer(self, agent):
        """Test rank when agent is best performer."""
        agent.cumulative_reward = 150.0

        team_rewards = [100.0, 80.0, 60.0]

        rank = agent._calculate_team_rank(team_rewards)

        assert rank == 1

    def test_calculate_team_rank_empty_team(self, agent):
        """Test rank with no teammates."""
        agent.cumulative_reward = 50.0

        rank = agent._calculate_team_rank([])

        assert rank == 1


class TestMycelialAgentReset:
    """Test agent reset functionality."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model):
        """Create test agent."""
        return MycelialAgent(model=mock_mesa_model,
            redis_client=mock_redis_client
        )

    def test_reset_clears_state(self, agent):
        """Test that reset clears current state."""
        agent.current_state = {"data": "test"}
        agent.last_action = "action"
        agent.last_reward = 5.0

        agent.reset()

        assert agent.current_state == {}
        assert agent.last_action is None
        assert agent.last_reward == 0.0

    def test_reset_preserves_cumulative_metrics(self, agent):
        """Test that reset preserves cumulative reward and step count."""
        agent.cumulative_reward = 100.0
        agent.step_count = 50

        agent.reset()

        # These should NOT be reset by default
        assert agent.cumulative_reward == 100.0
        assert agent.step_count == 50
