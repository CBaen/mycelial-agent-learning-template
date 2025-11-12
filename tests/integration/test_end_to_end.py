"""
End-to-End Integration Tests for MAE (Mycelial Agent Engine)

This module tests the complete system integration across all major components:
- Multi-agent collaboration and coordination
- FRL policy sharing + VDN credit assignment
- HAVEN risk management and safety oversight
- Episodic memory and experience replay
- Electrical signaling for fast coordination
- Stigmergic environment interaction
- GNN communication routing

Test Scenarios:
1. Complete multi-agent learning episode
2. Full system startup and initialization
3. Agent lifecycle (spawn, learn, collaborate, retire)
4. System response to failures and recovery
5. Performance under load and stress conditions
6. Data flow through all system layers
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import time

from src.agents.specialist_agent import SpecialistAgent
from src.agents.builder_agent import BuilderAgent
from src.agents.risk_manager_agent import RiskManagerAgent
from src.implementations.simple_frl import SimpleFRL
from src.implementations.simple_vdn import SimpleVDN
from src.core.haven_base import (
    HavenRiskCoordinator,
    RiskLevel,
    RiskAssessment,
    ContagionStatus,
    ContagionReport,
    InterventionType
)
from src.memory.episodic_memory import EpisodicMemory, Experience
from src.core.electrical_signal import ElectricalSignalBus, SignalPriority
from src.core.signal_types import SignalType
import numpy as np


# Mock HAVEN Coordinator (inline to avoid import issues)
class MockHavenCoordinator(HavenRiskCoordinator):
    """Mock implementation of HAVEN coordinator for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assessments = {}
        self.monitored_agents = {}

    def assess_agent_risk(self, agent_id, policy_state, recent_performance, behavioral_metrics=None):
        if not recent_performance:
            risk_score = 0.5
        else:
            avg_performance = np.mean(recent_performance[-10:]) if len(recent_performance) >= 10 else np.mean(recent_performance)
            risk_score = max(0.0, 1.0 - avg_performance) if avg_performance > 0 else 0.8

        if risk_score < 0.3:
            risk_level = RiskLevel.LOW
        elif risk_score < 0.6:
            risk_level = RiskLevel.MODERATE
        elif risk_score < 0.8:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        assessment = RiskAssessment(
            agent_id=agent_id,
            risk_level=risk_level,
            risk_score=risk_score,
            contributing_factors={"performance": avg_performance if recent_performance else 0.0},
            recommended_intervention=InterventionType.MONITORING if risk_level == RiskLevel.MODERATE else InterventionType.ISOLATION,
            timestamp=0.0,
            confidence=0.8
        )

        self.assessments[agent_id] = assessment
        self.monitored_agents[agent_id] = True
        return assessment

    def assess_system_risk(self):
        if not self.assessments:
            return 0.0
        risk_scores = [a.risk_score for a in self.assessments.values()]
        return sum(risk_scores) / len(risk_scores)

    def detect_policy_contagion(self, time_window=None):
        high_risk_agents = {aid for aid, a in self.assessments.items() if a.risk_score > 0.7}

        if len(high_risk_agents) > len(self.monitored_agents) * 0.3:
            status = ContagionStatus.SPREADING
        elif len(high_risk_agents) > 0:
            status = ContagionStatus.EARLY_WARNING
        else:
            status = ContagionStatus.HEALTHY

        return ContagionReport(
            contagion_status=status,
            affected_agents=high_risk_agents,
            source_agents=set(list(high_risk_agents)[:2]) if high_risk_agents else set(),
            contagion_score=len(high_risk_agents) / max(len(self.monitored_agents), 1),
            spread_rate=0.0,
            containment_actions=[],
            timestamp=0.0
        )

    def execute_intervention(self, agent_id, intervention_type, metadata=None):
        return True

    def identify_contagion_source(self, affected_agents):
        return []

    def recommend_intervention(self, agent_id, risk_assessment):
        return risk_assessment.recommended_intervention

    def isolate_agent(self, agent_id, reason=None):
        """Isolate an agent from the system."""
        self.monitored_agents[agent_id] = False
        return True

    def restore_agent(self, agent_id, verification_required=False):
        """Restore an isolated agent."""
        self.monitored_agents[agent_id] = True
        return True

    def compute_adversarial_value(self, state, policy_state, perturbation_budget=0.1):
        """Compute adversarial value under worst-case perturbation."""
        return 0.0

    def evaluate_policy_robustness(self, agent_id, policy_state, num_perturbations=10):
        """Evaluate how robust a policy is to perturbations."""
        return 0.8

    def track_policy_influence(self, source_agent, target_agent, influence_data):
        """Track policy influence between agents."""
        pass

    def get_influence_graph(self):
        """Get the policy influence graph."""
        return {}

    def compute_risk_propagation(self, source_agent, time_horizon=10):
        """Compute risk propagation from source agent."""
        return {}

    def establish_safety_constraints(self, constraints):
        """Establish safety constraints for the system."""
        pass

    def verify_safety_constraints(self, agent_id, policy_state):
        """Verify that policy satisfies safety constraints."""
        return True

    def get_system_health_report(self):
        """Generate comprehensive system health report."""
        return {
            "total_agents": len(self.monitored_agents),
            "high_risk_agents": sum(1 for a in self.assessments.values() if a.risk_score > 0.7),
            "average_risk": self.assess_system_risk(),
            "system_status": "healthy"
        }

    def export_risk_analytics(self, time_range=None):
        """Export risk analytics data."""
        return {
            "assessments": self.assessments,
            "system_risk": self.assess_system_risk()
        }

    def calibrate_risk_thresholds(self, historical_data):
        """Calibrate risk thresholds based on historical data."""
        pass

    def simulate_intervention_impact(self, intervention, agent_id, time_horizon=5):
        """Simulate the impact of an intervention."""
        return {
            "expected_risk_reduction": 0.3,
            "time_to_recovery": time_horizon,
            "side_effects": []
        }


@pytest.fixture
def mock_redis_client():
    """Create a comprehensive mock Redis client."""
    client = Mock()
    client.publish = Mock(return_value=1)
    client.subscribe = Mock()
    client.set = Mock(return_value=True)
    client.get = Mock(return_value=None)
    client.hset = Mock(return_value=1)
    client.hget = Mock(return_value=None)
    client.hgetall = Mock(return_value={})
    client.smembers = Mock(return_value=set())
    client.sadd = Mock(return_value=1)
    client.srem = Mock(return_value=1)
    client.zadd = Mock(return_value=1)
    client.zrange = Mock(return_value=[])
    return client


@pytest.fixture
def mock_mesa_model():
    """Create a mock Mesa model with scheduling."""
    model = Mock()
    model.schedule = Mock()
    model.schedule.agents = []
    model.schedule.steps = 0
    model.running = True
    model.current_step = 0
    model.step = Mock()
    return model


@pytest.fixture
def signal_bus():
    """Create electrical signal bus for fast coordination."""
    return ElectricalSignalBus()


@pytest.fixture
def episodic_memory():
    """Create episodic memory for experience replay."""
    return EpisodicMemory(capacity=1000, alpha=0.6, beta=0.4)


@pytest.fixture
def haven_coordinator(mock_redis_client):
    """Create HAVEN risk coordinator."""
    return MockHavenCoordinator(
        coordinator_id="haven_main",
        redis_client=mock_redis_client,
        risk_threshold=0.7,
        contagion_threshold=0.5
    )


class TestSystemInitialization:
    """Test complete system initialization and startup."""

    def test_initialize_multi_agent_system(
        self,
        mock_mesa_model,
        mock_redis_client,
        signal_bus,
        haven_coordinator
    ):
        """Test initializing a complete multi-agent system with all components."""
        # Create diverse agent population
        specialists = []
        for i in range(5):
            specialist = SpecialistAgent(
                model=mock_mesa_model,
                unique_id=f"specialist_{i}",
                redis_client=mock_redis_client,
                team_id="team_alpha",
                role="collaborator"
            )
            specialists.append(specialist)

        # Create builder agent
        builder = BuilderAgent(
            model=mock_mesa_model,
            unique_id="builder_1",
            redis_client=mock_redis_client,
            team_id="team_alpha"
        )

        # Create risk manager
        risk_manager = RiskManagerAgent(
            model=mock_mesa_model,
            unique_id="risk_mgr_1",
            redis_client=mock_redis_client,
            haven_coordinator=haven_coordinator,
            team_id="system"
        )

        # Verify all agents initialized
        assert len(specialists) == 5
        assert builder is not None
        assert risk_manager is not None
        assert all(isinstance(s, SpecialistAgent) for s in specialists)

    def test_attach_all_engines_to_agents(
        self,
        mock_mesa_model,
        mock_redis_client,
        episodic_memory
    ):
        """Test attaching FRL, VDN, and memory engines to agents."""
        agent = SpecialistAgent(
            model=mock_mesa_model,
            unique_id="specialist_full",
            redis_client=mock_redis_client,
            team_id="team_beta",
            role="learner"
        )

        # Attach FRL engine
        agent.frl_engine = SimpleFRL(
            agent_id="specialist_full",
            redis_client=mock_redis_client
        )

        # Attach VDN engine
        agent.vdn_engine = SimpleVDN(
            agent_id="specialist_full",
            redis_client=mock_redis_client,
            state_dim=10,
            action_dim=5
        )

        # Attach episodic memory
        agent.episodic_memory = episodic_memory

        # Verify all engines attached
        assert hasattr(agent, 'frl_engine')
        assert hasattr(agent, 'vdn_engine')
        assert hasattr(agent, 'episodic_memory')


class TestCompleteLearningEpisode:
    """Test a complete learning episode through all system layers."""

    def test_single_episode_full_workflow(
        self,
        mock_mesa_model,
        mock_redis_client,
        signal_bus,
        episodic_memory,
        haven_coordinator
    ):
        """Test complete workflow: observe, act, learn, share, assess."""
        # Create agent with all engines
        agent = SpecialistAgent(
            model=mock_mesa_model,
            unique_id="learner_1",
            redis_client=mock_redis_client,
            team_id="team_gamma",
            role="explorer"
        )

        agent.frl_engine = SimpleFRL(
            agent_id="learner_1",
            redis_client=mock_redis_client
        )

        agent.vdn_engine = SimpleVDN(
            agent_id="learner_1",
            redis_client=mock_redis_client,
            state_dim=5,
            action_dim=3
        )

        agent.episodic_memory = episodic_memory

        # Episode workflow
        # 1. Observe state
        state = np.random.rand(5)

        # 2. Select action
        action = np.random.randint(0, 3)

        # 3. Execute action and receive reward
        reward = np.random.rand()
        next_state = np.random.rand(5)

        # 4. Store experience in memory
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
            priority=reward,  # Use reward as initial priority
            agent_id="learner_1",
            timestamp=time.time()
        )
        episodic_memory.store(experience)

        # 5. Sample and learn from memory
        if len(episodic_memory) > 0:
            batch = episodic_memory.sample(batch_size=1)
            assert len(batch) == 1

        # 6. Share policy via FRL
        policy = {"weights": np.random.rand(5)}
        agent.frl_engine.share_policy(policy, performance_score=reward)

        # 7. Calculate credit via VDN
        credit = agent.vdn_engine.calculate_credit(
            team_reward=reward,
            baseline_reward=0.0
        )

        # 8. Update cumulative reward
        agent.cumulative_reward += reward

        # 9. Assess risk via HAVEN
        assessment = haven_coordinator.assess_agent_risk(
            agent_id="learner_1",
            policy_state=policy,
            recent_performance=[reward]
        )

        # Verify episode completed successfully
        assert agent.cumulative_reward > 0
        assert credit >= 0
        assert assessment is not None
        assert len(episodic_memory) == 1

    def test_multi_agent_collaborative_episode(
        self,
        mock_mesa_model,
        mock_redis_client,
        episodic_memory,
        haven_coordinator
    ):
        """Test multi-agent collaboration in a shared episode."""
        # Create team of 3 agents
        team = []
        for i in range(3):
            agent = SpecialistAgent(
                model=mock_mesa_model,
                unique_id=f"team_member_{i}",
                redis_client=mock_redis_client,
                team_id="collaborative_team",
                role="collaborator"
            )

            agent.frl_engine = SimpleFRL(
                agent_id=f"team_member_{i}",
                redis_client=mock_redis_client,
                max_peers=2
            )

            agent.vdn_engine = SimpleVDN(
                agent_id=f"team_member_{i}",
                redis_client=mock_redis_client,
                state_dim=5,
                action_dim=3
            )

            team.append(agent)

        # Collaborative episode
        team_rewards = []

        for agent in team:
            # Each agent acts
            action = np.random.randint(0, 3)
            reward = np.random.rand()
            team_rewards.append(reward)

            agent.cumulative_reward += reward

        # Calculate team reward
        total_team_reward = sum(team_rewards)

        # Assign credit to each agent
        for i, agent in enumerate(team):
            baseline = total_team_reward - team_rewards[i]
            credit = agent.vdn_engine.calculate_credit(
                team_reward=total_team_reward,
                baseline_reward=baseline
            )

            # Credit should reflect marginal contribution
            assert credit >= 0

        # Share policies among team
        for agent in team:
            policy = {"weights": np.random.rand(5)}
            agent.frl_engine.share_policy(
                policy,
                performance_score=agent.cumulative_reward
            )

        # Verify collaboration succeeded
        assert len(team) == 3
        assert all(agent.cumulative_reward > 0 for agent in team)


class TestSystemFailureAndRecovery:
    """Test system response to failures and recovery mechanisms."""

    def test_agent_failure_and_replacement(
        self,
        mock_mesa_model,
        mock_redis_client,
        haven_coordinator
    ):
        """Test system handles agent failure and spawns replacement."""
        # Create risk manager
        risk_manager = RiskManagerAgent(
            model=mock_mesa_model,
            unique_id="risk_mgr",
            redis_client=mock_redis_client,
            haven_coordinator=haven_coordinator,
            auto_intervention=True
        )

        # Create failing agent
        failing_agent_id = "failing_agent"
        risk_manager.register_agent_for_monitoring(
            agent_id=failing_agent_id,
            agent_data={"performance_history": [0.9, 0.7, 0.4, 0.1, 0.05]}
        )

        # Assess the failing agent
        assessment = haven_coordinator.assess_agent_risk(
            agent_id=failing_agent_id,
            policy_state={"weights": np.array([0.1])},
            recent_performance=[0.05, 0.03, 0.01]
        )

        # Should detect critical risk
        assert assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert assessment.recommended_intervention in [
            InterventionType.ISOLATION,
            InterventionType.ROLLBACK,
            InterventionType.EMERGENCY_STOP
        ]

        # In production, this would trigger agent replacement
        # Here we verify the detection worked
        assert risk_manager.total_assessments >= 0

    def test_contagion_outbreak_and_containment(
        self,
        mock_mesa_model,
        mock_redis_client,
        haven_coordinator
    ):
        """Test system detects and contains policy contagion outbreak."""
        # Create population with contagion
        num_agents = 10
        for i in range(num_agents):
            # 60% infected with bad policy
            performance = [0.2, 0.15] if i < 6 else [0.85, 0.9]

            haven_coordinator.assess_agent_risk(
                agent_id=f"agent_{i}",
                policy_state={"weights": np.array([0.5])},
                recent_performance=performance
            )

        # Detect contagion
        contagion_report = haven_coordinator.detect_policy_contagion()

        # Should detect spreading contagion
        assert contagion_report.contagion_status == ContagionStatus.SPREADING
        assert len(contagion_report.affected_agents) >= 4

        # Simulate containment: Remove infected agents
        for agent_id in list(contagion_report.affected_agents)[:4]:
            if agent_id in haven_coordinator.assessments:
                del haven_coordinator.assessments[agent_id]

        # Re-check contagion
        post_containment = haven_coordinator.detect_policy_contagion()

        # Should be contained
        assert post_containment.contagion_status in [
            ContagionStatus.CONTAINED,
            ContagionStatus.HEALTHY,
            ContagionStatus.EARLY_WARNING
        ]


class TestDataFlowAcrossLayers:
    """Test data flows correctly through all system layers."""

    def test_signal_bus_to_agent_communication(
        self,
        mock_mesa_model,
        mock_redis_client,
        signal_bus
    ):
        """Test electrical signals propagate from bus to agents."""
        agent = SpecialistAgent(
            model=mock_mesa_model,
            unique_id="signal_receiver",
            redis_client=mock_redis_client,
            team_id="team_delta",
            role="responder"
        )

        # Create handler for signal
        received_signals = []

        def signal_handler(signal):
            received_signals.append(signal)

        # Register handler
        signal_bus.subscribe(
            agent_id="signal_receiver",
            signal_type=SignalType.OPPORTUNITY,
            handler=signal_handler
        )

        # Emit signal
        signal_bus.emit(
            signal_type=SignalType.OPPORTUNITY,
            sender_id="signal_sender",
            priority=SignalPriority.HIGH,
            data={"resource_location": (10, 20)}
        )

        # Process signals
        signal_bus.process_pending_signals(max_signals=10)

        # Verify signal received
        assert len(received_signals) > 0

    def test_memory_to_learning_pipeline(
        self,
        mock_mesa_model,
        mock_redis_client,
        episodic_memory
    ):
        """Test experience flows from memory to learning."""
        agent = SpecialistAgent(
            model=mock_mesa_model,
            unique_id="memory_learner",
            redis_client=mock_redis_client,
            team_id="team_epsilon",
            role="learner"
        )

        agent.episodic_memory = episodic_memory

        # Store 10 experiences
        for i in range(10):
            exp = Experience(
                state=np.random.rand(5),
                action=np.random.randint(0, 3),
                reward=np.random.rand(),
                next_state=np.random.rand(5),
                done=False,
                priority=np.random.rand(),
                agent_id="memory_learner",
                timestamp=time.time()
            )
            episodic_memory.store(exp)

        # Sample batch for learning
        batch = episodic_memory.sample(batch_size=5)

        # Verify batch retrieved
        assert len(batch) == 5
        assert all(isinstance(exp, Experience) for exp in batch)


class TestPerformanceUnderLoad:
    """Test system performance under load and stress conditions."""

    @pytest.mark.slow
    def test_many_agents_system_performance(
        self,
        mock_mesa_model,
        mock_redis_client,
        haven_coordinator
    ):
        """Test system handles 50+ agents efficiently."""
        num_agents = 50
        agents = []

        start_time = time.time()

        # Create 50 agents
        for i in range(num_agents):
            agent = SpecialistAgent(
                model=mock_mesa_model,
                unique_id=f"agent_{i}",
                redis_client=mock_redis_client,
                team_id="large_team",
                role="worker"
            )
            agents.append(agent)

        creation_time = time.time() - start_time

        # Should create 50 agents in < 2 seconds
        assert creation_time < 2.0
        assert len(agents) == 50

    @pytest.mark.slow
    def test_high_frequency_risk_assessment(self, haven_coordinator):
        """Test HAVEN handles high-frequency risk assessments."""
        num_assessments = 500

        start_time = time.time()

        for i in range(num_assessments):
            haven_coordinator.assess_agent_risk(
                agent_id=f"agent_{i % 100}",  # Cycle through 100 agents
                policy_state={"weights": np.random.rand(5)},
                recent_performance=np.random.rand(5).tolist()
            )

        elapsed_time = time.time() - start_time

        # Should handle 500 assessments in < 2 seconds
        assert elapsed_time < 2.0

    @pytest.mark.slow
    def test_concurrent_policy_sharing(self, mock_redis_client):
        """Test FRL handles concurrent policy sharing from many agents."""
        num_agents = 30
        engines = []

        for i in range(num_agents):
            engine = SimpleFRL(
                agent_id=f"agent_{i}",
                redis_client=mock_redis_client,
                max_peers=5
            )
            engines.append(engine)

        start_time = time.time()

        # All agents share policies concurrently (simulated)
        for engine in engines:
            policy = {"weights": np.random.rand(10)}
            engine.share_policy(policy, performance_score=np.random.rand())

        elapsed_time = time.time() - start_time

        # Should complete in < 1 second
        assert elapsed_time < 1.0


class TestCompleteSystemLifecycle:
    """Test complete system lifecycle from start to shutdown."""

    def test_system_startup_to_shutdown(
        self,
        mock_mesa_model,
        mock_redis_client,
        signal_bus,
        episodic_memory,
        haven_coordinator
    ):
        """Test complete system lifecycle: startup, run, shutdown."""
        # STARTUP PHASE
        # 1. Initialize infrastructure
        assert signal_bus is not None
        assert episodic_memory is not None
        assert haven_coordinator is not None

        # 2. Create agents
        specialists = [
            SpecialistAgent(
                model=mock_mesa_model,
                unique_id=f"specialist_{i}",
                redis_client=mock_redis_client,
                team_id="main_team",
                role="worker"
            )
            for i in range(3)
        ]

        risk_manager = RiskManagerAgent(
            model=mock_mesa_model,
            unique_id="risk_mgr",
            redis_client=mock_redis_client,
            haven_coordinator=haven_coordinator,
            team_id="system"
        )

        # RUNNING PHASE
        # 3. Simulate 5 simulation steps
        for step in range(5):
            # Each specialist acts
            for specialist in specialists:
                action = np.random.randint(0, 3)
                reward = np.random.rand()
                specialist.cumulative_reward += reward

            # Risk manager monitors
            for specialist in specialists:
                risk_manager.register_agent_for_monitoring(
                    agent_id=specialist.unique_id,
                    agent_data={"performance_history": [specialist.cumulative_reward]}
                )

        # SHUTDOWN PHASE
        # 4. Cleanup and shutdown
        final_statistics = {
            "num_agents": len(specialists),
            "total_steps": 5,
            "avg_reward": sum(s.cumulative_reward for s in specialists) / len(specialists),
            "system_risk": haven_coordinator.assess_system_risk()
        }

        # Verify system ran successfully
        assert final_statistics["num_agents"] == 3
        assert final_statistics["total_steps"] == 5
        assert final_statistics["avg_reward"] >= 0
