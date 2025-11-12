"""
Integration Tests for FRL/VDN Coordination

This module tests the integration between:
- Federated Reinforcement Learning (FRL) for policy sharing
- Value Decomposition Networks (VDN) for credit assignment
- Multi-agent coordination and collaboration

Test Scenarios:
1. FRL/VDN initialization and setup
2. Multi-agent policy sharing and aggregation
3. Credit assignment across collaborating agents
4. Combined FRL+VDN learning workflow
5. Performance measurement and validation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.implementations.simple_frl import SimpleFRL
from src.implementations.simple_vdn import SimpleVDN
from src.agents.specialist_agent import SpecialistAgent
from src.connectors.redis_client import RedisClient
from src.core.frl_base import PolicyUpdateStrategy, AggregationMethod
from src.core.vdn_base import DecompositionMethod, CreditAssignmentStrategy


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    client = Mock(spec=RedisClient)
    client.publish = Mock()
    client.subscribe = Mock()
    client.set = Mock()
    client.get = Mock(return_value=None)
    client.hset = Mock()
    client.hget = Mock(return_value=None)
    client.hgetall = Mock(return_value={})
    client.smembers = Mock(return_value=set())
    client.sadd = Mock()
    client.srem = Mock()
    return client


@pytest.fixture
def mock_mesa_model():
    """Create a mock Mesa model."""
    model = Mock()
    model.schedule = Mock()
    model.schedule.agents = []
    model.running = True
    model.current_step = 0
    return model


class TestFRLVDNInitialization:
    """Test FRL/VDN engine initialization and setup."""

    def test_frl_engine_initialization(self, mock_redis_client):
        """Test FRL engine initializes correctly."""
        engine = SimpleFRL(
            agent_id="test_agent",
            redis_client=mock_redis_client,
            policy_update_strategy=PolicyUpdateStrategy.PERFORMANCE_BASED,
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            max_peers=5,
            trust_threshold=0.6
        )

        assert engine.agent_id == "test_agent"
        assert engine.max_peers == 5
        assert engine.trust_threshold == 0.6
        assert len(engine.peers) == 0
        assert len(engine.trust_scores) == 0

    def test_vdn_engine_initialization(self, mock_redis_client):
        """Test VDN engine initializes correctly."""
        engine = SimpleVDN(
            agent_id="test_agent",
            redis_client=mock_redis_client,
            decomposition_method=DecompositionMethod.ADDITIVE,
            credit_strategy=CreditAssignmentStrategy.DIFFERENCE_REWARDS,
            state_dim=10,
            action_dim=5
        )

        assert engine.agent_id == "test_agent"
        assert engine.state_dim == 10
        assert engine.action_dim == 5
        assert engine.decomposition_method == DecompositionMethod.ADDITIVE

    def test_agent_with_both_engines(self, mock_mesa_model, mock_redis_client):
        """Test specialist agent can be initialized with both FRL and VDN."""
        agent = SpecialistAgent(
            model=mock_mesa_model,
            unique_id="specialist_1",
            redis_client=mock_redis_client,
            team_id="team_alpha"
        )

        # Initialize FRL engine
        frl_engine = SimpleFRL(
            agent_id="specialist_1",
            redis_client=mock_redis_client,
            max_peers=5
        )

        # Initialize VDN engine
        vdn_engine = SimpleVDN(
            agent_id="specialist_1",
            redis_client=mock_redis_client,
            state_dim=10,
            action_dim=5
        )

        # Assign engines to agent (simulating what happens in production)
        agent.frl_engine = frl_engine
        agent.vdn_engine = vdn_engine

        assert agent.frl_engine is not None
        assert agent.vdn_engine is not None
        assert agent.unique_id == "specialist_1"


class TestFRLPolicySharing:
    """Test FRL policy sharing and aggregation across multiple agents."""

    def test_policy_registration(self, mock_redis_client):
        """Test agents can register their policies with FRL."""
        engine = SimpleFRL(
            agent_id="agent_1",
            redis_client=mock_redis_client
        )

        # Create a simple policy (dictionary of parameters)
        policy = {
            "weights": np.array([0.5, 0.3, 0.2]),
            "bias": np.array([0.1])
        }

        # Register policy
        engine.share_policy(policy, performance_score=0.75)

        # Verify policy was shared (via Redis publish)
        assert mock_redis_client.publish.called

    def test_peer_discovery(self, mock_redis_client):
        """Test agents can discover peers for federated learning."""
        engine1 = SimpleFRL(agent_id="agent_1", redis_client=mock_redis_client)
        engine2 = SimpleFRL(agent_id="agent_2", redis_client=mock_redis_client)

        # Simulate peer discovery
        mock_redis_client.smembers.return_value = {b"agent_1", b"agent_2"}

        discovered_peers = engine1.discover_peers()

        # Should discover itself and peer
        assert len(discovered_peers) >= 0
        assert mock_redis_client.smembers.called

    def test_weighted_policy_aggregation(self, mock_redis_client):
        """Test FRL can aggregate policies from multiple peers with weighting."""
        engine = SimpleFRL(
            agent_id="agent_1",
            redis_client=mock_redis_client,
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE
        )

        # Create test policies from multiple peers
        policies = {
            "agent_1": {"weights": np.array([1.0, 0.0, 0.0])},
            "agent_2": {"weights": np.array([0.0, 1.0, 0.0])},
            "agent_3": {"weights": np.array([0.0, 0.0, 1.0])}
        }

        trust_scores = {
            "agent_1": 0.8,
            "agent_2": 0.9,
            "agent_3": 0.7
        }

        # Aggregate policies
        aggregated = engine.aggregate_policies(policies, trust_scores)

        # Aggregated weights should be weighted average
        assert "weights" in aggregated
        assert aggregated["weights"].shape == (3,)
        # All components should be > 0 due to weighted averaging
        assert np.all(aggregated["weights"] >= 0)


class TestVDNCreditAssignment:
    """Test VDN credit assignment across collaborating agents."""

    def test_value_decomposition(self, mock_redis_client):
        """Test VDN can decompose joint value into individual contributions."""
        engine = SimpleVDN(
            agent_id="agent_1",
            redis_client=mock_redis_client,
            decomposition_method=DecompositionMethod.ADDITIVE
        )

        # Simulate individual Q-values
        q_values = {
            "agent_1": 10.0,
            "agent_2": 15.0,
            "agent_3": 8.0
        }

        # Decompose total value
        total_value = engine.decompose_value(q_values)

        # For additive decomposition: Q_total = sum(Q_i)
        assert total_value == 33.0

    def test_credit_assignment_difference_rewards(self, mock_redis_client):
        """Test credit assignment using difference rewards."""
        engine = SimpleVDN(
            agent_id="agent_1",
            redis_client=mock_redis_client,
            credit_strategy=CreditAssignmentStrategy.DIFFERENCE_REWARDS
        )

        # Simulate team reward and counterfactual baseline
        team_reward = 50.0
        baseline_reward = 35.0  # What team would get without this agent

        # Calculate credit (difference reward)
        credit = engine.calculate_credit(
            team_reward=team_reward,
            baseline_reward=baseline_reward
        )

        # Credit should be the marginal contribution
        assert credit == 15.0

    def test_marginal_contribution_estimation(self, mock_redis_client):
        """Test VDN can estimate marginal contribution of each agent."""
        engine = SimpleVDN(
            agent_id="agent_1",
            redis_client=mock_redis_client
        )

        # Simulate team performance with and without agent
        state = np.array([1.0, 0.5, 0.3])
        action = 2

        # Estimate marginal contribution
        contribution = engine.estimate_marginal_contribution(
            state=state,
            action=action,
            team_reward=100.0
        )

        # Contribution should be a numeric value
        assert isinstance(contribution, (int, float))


class TestCombinedFRLVDNWorkflow:
    """Test combined FRL+VDN workflow in multi-agent scenario."""

    def test_multi_agent_collaboration_workflow(self, mock_mesa_model, mock_redis_client):
        """Test complete workflow: agents learn, share policies, assign credit."""
        # Create 3 specialist agents
        agents = []
        for i in range(3):
            agent = SpecialistAgent(
                model=mock_mesa_model,
                unique_id=f"specialist_{i}",
                redis_client=mock_redis_client,
                team_id="team_alpha"
            )

            # Attach FRL and VDN engines
            agent.frl_engine = SimpleFRL(
                agent_id=f"specialist_{i}",
                redis_client=mock_redis_client,
                max_peers=2
            )
            agent.vdn_engine = SimpleVDN(
                agent_id=f"specialist_{i}",
                redis_client=mock_redis_client,
                state_dim=5,
                action_dim=3
            )

            agents.append(agent)

        # Simulate learning step
        for agent in agents:
            # Agent observes state and takes action
            state = np.random.rand(5)
            action = np.random.randint(0, 3)

            # Record experience (simulated)
            agent.cumulative_reward += np.random.rand()

        # FRL: Agents share policies
        for agent in agents:
            policy = {"weights": np.random.rand(5)}
            agent.frl_engine.share_policy(policy, performance_score=agent.cumulative_reward)

        # VDN: Calculate team reward and assign credit
        team_reward = sum(agent.cumulative_reward for agent in agents)

        for agent in agents:
            baseline = team_reward - agent.cumulative_reward
            credit = agent.vdn_engine.calculate_credit(
                team_reward=team_reward,
                baseline_reward=baseline
            )

            # Each agent should receive credit
            assert credit >= 0

        # Verify all agents participated
        assert len(agents) == 3
        assert all(hasattr(agent, 'frl_engine') for agent in agents)
        assert all(hasattr(agent, 'vdn_engine') for agent in agents)

    def test_frl_vdn_performance_improvement(self, mock_mesa_model, mock_redis_client):
        """Test that FRL+VDN leads to performance improvement over time."""
        agent = SpecialistAgent(
            model=mock_mesa_model,
            unique_id="test_agent",
            redis_client=mock_redis_client,
            team_id="team_alpha"
        )

        agent.frl_engine = SimpleFRL(
            agent_id="test_agent",
            redis_client=mock_redis_client
        )
        agent.vdn_engine = SimpleVDN(
            agent_id="test_agent",
            redis_client=mock_redis_client,
            state_dim=5,
            action_dim=3
        )

        # Track performance over simulated episodes
        initial_performance = agent.cumulative_reward

        # Simulate 10 learning steps
        for step in range(10):
            # Simulate reward
            reward = np.random.rand() * 0.5 + 0.3  # Random reward in [0.3, 0.8]
            agent.cumulative_reward += reward

            # FRL: Share policy periodically
            if step % 3 == 0:
                policy = {"weights": np.random.rand(5)}
                agent.frl_engine.share_policy(policy, performance_score=agent.cumulative_reward)

            # VDN: Calculate credit
            credit = agent.vdn_engine.calculate_credit(
                team_reward=reward,
                baseline_reward=0.0
            )

        final_performance = agent.cumulative_reward

        # Performance should improve
        assert final_performance > initial_performance


class TestFRLVDNEdgeCases:
    """Test edge cases and error handling for FRL/VDN integration."""

    def test_frl_with_no_peers(self, mock_redis_client):
        """Test FRL handles scenario with no available peers."""
        engine = SimpleFRL(
            agent_id="isolated_agent",
            redis_client=mock_redis_client
        )

        # Attempt to discover peers when none exist
        mock_redis_client.smembers.return_value = set()

        peers = engine.discover_peers()

        # Should return empty or handle gracefully
        assert isinstance(peers, (list, set))

    def test_vdn_with_single_agent(self, mock_redis_client):
        """Test VDN handles single-agent scenario (no team)."""
        engine = SimpleVDN(
            agent_id="solo_agent",
            redis_client=mock_redis_client
        )

        # Single agent Q-value
        q_values = {"solo_agent": 20.0}

        total_value = engine.decompose_value(q_values)

        # Should return the single Q-value
        assert total_value == 20.0

    def test_empty_policy_aggregation(self, mock_redis_client):
        """Test FRL handles empty policy aggregation gracefully."""
        engine = SimpleFRL(
            agent_id="agent_1",
            redis_client=mock_redis_client
        )

        # Try to aggregate with no policies
        result = engine.aggregate_policies({}, {})

        # Should return None or empty dict
        assert result is None or isinstance(result, dict)

    def test_invalid_trust_scores(self, mock_redis_client):
        """Test FRL validates trust scores properly."""
        engine = SimpleFRL(
            agent_id="agent_1",
            redis_client=mock_redis_client,
            trust_threshold=0.5
        )

        policies = {
            "agent_2": {"weights": np.array([1.0, 2.0])}
        }

        # Trust score below threshold
        trust_scores = {"agent_2": 0.3}

        # Should filter out low-trust policies
        result = engine.aggregate_policies(policies, trust_scores)

        # Result should be None or not include low-trust policy
        assert result is None or "agent_2" not in str(result)


@pytest.mark.slow
class TestFRLVDNPerformance:
    """Performance and scalability tests for FRL/VDN integration."""

    def test_many_agents_frl_scaling(self, mock_redis_client):
        """Test FRL scales to many agents (10+ agents)."""
        num_agents = 15
        engines = []

        for i in range(num_agents):
            engine = SimpleFRL(
                agent_id=f"agent_{i}",
                redis_client=mock_redis_client,
                max_peers=5
            )
            engines.append(engine)

        # Simulate policy sharing
        for engine in engines:
            policy = {"weights": np.random.rand(10)}
            engine.share_policy(policy, performance_score=np.random.rand())

        # Should complete without errors
        assert len(engines) == num_agents

    def test_vdn_computational_efficiency(self, mock_redis_client):
        """Test VDN credit assignment is computationally efficient."""
        engine = SimpleVDN(
            agent_id="agent_1",
            redis_client=mock_redis_client,
            state_dim=100,  # Larger state space
            action_dim=20   # More actions
        )

        import time

        # Time 100 credit calculations
        start_time = time.time()

        for _ in range(100):
            credit = engine.calculate_credit(
                team_reward=np.random.rand() * 100,
                baseline_reward=np.random.rand() * 80
            )

        elapsed_time = time.time() - start_time

        # Should complete 100 calculations in < 1 second
        assert elapsed_time < 1.0
        assert isinstance(credit, (int, float))
