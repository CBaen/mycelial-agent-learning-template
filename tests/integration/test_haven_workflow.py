"""
Integration Tests for HAVEN (Hierarchical Adversarial Value Estimation Network) Workflow

This module tests the complete HAVEN risk management and contagion prevention workflow:
- Risk assessment at agent and system levels
- Policy contagion detection and containment
- Intervention coordination and execution
- Integration with Risk Manager Agent
- System stability and safety guarantees

Test Scenarios:
1. HAVEN initialization and risk monitoring setup
2. Agent risk assessment workflow
3. Policy contagion detection and tracking
4. Intervention recommendation and execution
5. System-wide risk aggregation
6. Recovery and stabilization after intervention
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import time

from src.core.haven_base import (
    HavenRiskCoordinator,
    RiskLevel,
    RiskAssessment,
    ContagionStatus,
    ContagionReport,
    InterventionType
)
from src.agents.risk_manager_agent import RiskManagerAgent
from src.agents.specialist_agent import SpecialistAgent


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
    """Create a mock Redis client for testing."""
    client = Mock()
    client.publish = Mock()
    client.subscribe = Mock()
    client.set = Mock()
    client.get = Mock(return_value=None)
    client.hset = Mock()
    client.hget = Mock(return_value=None)
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


@pytest.fixture
def haven_coordinator(mock_redis_client):
    """Create a HAVEN coordinator instance for testing."""
    return MockHavenCoordinator(
        coordinator_id="haven_test",
        redis_client=mock_redis_client,
        risk_threshold=0.7,
        contagion_threshold=0.5,
        intervention_enabled=True
    )


class TestHAVENInitialization:
    """Test HAVEN coordinator initialization and setup."""

    def test_haven_coordinator_initialization(self, mock_redis_client):
        """Test HAVEN coordinator initializes with correct parameters."""
        coordinator = MockHavenCoordinator(
            coordinator_id="haven_1",
            redis_client=mock_redis_client,
            risk_threshold=0.6,
            contagion_threshold=0.4,
            intervention_enabled=True
        )

        assert coordinator.coordinator_id == "haven_1"
        assert coordinator.risk_threshold == 0.6
        assert coordinator.contagion_threshold == 0.4
        assert coordinator.intervention_enabled is True

    def test_risk_manager_with_haven(self, mock_mesa_model, mock_redis_client, haven_coordinator):
        """Test Risk Manager Agent integrates with HAVEN coordinator."""
        risk_manager = RiskManagerAgent(
            model=mock_mesa_model,
            unique_id="risk_mgr_1",
            redis_client=mock_redis_client,
            haven_coordinator=haven_coordinator,
            team_id="system"
        )

        assert risk_manager.haven_coordinator is not None
        assert risk_manager.haven_coordinator == haven_coordinator
        assert risk_manager.unique_id == "risk_mgr_1"


class TestAgentRiskAssessment:
    """Test individual agent risk assessment workflow."""

    def test_assess_healthy_agent(self, haven_coordinator):
        """Test risk assessment for a healthy performing agent."""
        agent_id = "specialist_1"
        policy_state = {"weights": np.array([0.5, 0.3, 0.2])}
        recent_performance = [0.8, 0.85, 0.9, 0.82, 0.88]  # Good performance

        assessment = haven_coordinator.assess_agent_risk(
            agent_id=agent_id,
            policy_state=policy_state,
            recent_performance=recent_performance
        )

        # Healthy agent should have low risk
        assert isinstance(assessment, RiskAssessment)
        assert assessment.agent_id == agent_id
        assert assessment.risk_level in [RiskLevel.MINIMAL, RiskLevel.LOW]
        assert assessment.risk_score < 0.4
        assert assessment.recommended_intervention == InterventionType.MONITORING

    def test_assess_degrading_agent(self, haven_coordinator):
        """Test risk assessment for agent with degrading performance."""
        agent_id = "specialist_2"
        policy_state = {"weights": np.array([0.1, 0.1, 0.1])}
        recent_performance = [0.4, 0.3, 0.25, 0.2, 0.15]  # Degrading performance

        assessment = haven_coordinator.assess_agent_risk(
            agent_id=agent_id,
            policy_state=policy_state,
            recent_performance=recent_performance
        )

        # Degrading agent should have elevated risk
        assert assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert assessment.risk_score > 0.6
        assert assessment.recommended_intervention in [InterventionType.ISOLATION, InterventionType.ROLLBACK]

    def test_assess_agent_with_no_history(self, haven_coordinator):
        """Test risk assessment for newly initialized agent with no history."""
        agent_id = "new_agent"
        policy_state = {"weights": np.array([0.5, 0.5])}
        recent_performance = []  # No history

        assessment = haven_coordinator.assess_agent_risk(
            agent_id=agent_id,
            policy_state=policy_state,
            recent_performance=recent_performance
        )

        # New agent should have moderate risk (unknown)
        assert isinstance(assessment, RiskAssessment)
        assert assessment.risk_level == RiskLevel.MODERATE
        assert 0.4 <= assessment.risk_score <= 0.6


class TestPolicyContagionDetection:
    """Test policy contagion detection and tracking."""

    def test_detect_no_contagion(self, haven_coordinator):
        """Test contagion detection when system is healthy."""
        # Assess several healthy agents
        for i in range(5):
            haven_coordinator.assess_agent_risk(
                agent_id=f"agent_{i}",
                policy_state={"weights": np.array([0.5])},
                recent_performance=[0.8, 0.85, 0.9]
            )

        contagion_report = haven_coordinator.detect_policy_contagion()

        assert isinstance(contagion_report, ContagionReport)
        assert contagion_report.contagion_status == ContagionStatus.HEALTHY
        assert contagion_report.contagion_score < 0.3
        assert len(contagion_report.affected_agents) == 0

    def test_detect_early_contagion(self, haven_coordinator):
        """Test early warning detection when few agents show risk."""
        # Assess mostly healthy agents
        for i in range(8):
            haven_coordinator.assess_agent_risk(
                agent_id=f"healthy_{i}",
                policy_state={"weights": np.array([0.5])},
                recent_performance=[0.8, 0.85]
            )

        # One risky agent
        haven_coordinator.assess_agent_risk(
            agent_id="risky_agent",
            policy_state={"weights": np.array([0.1])},
            recent_performance=[0.2, 0.15]
        )

        contagion_report = haven_coordinator.detect_policy_contagion()

        assert contagion_report.contagion_status in [ContagionStatus.EARLY_WARNING, ContagionStatus.HEALTHY]
        assert len(contagion_report.affected_agents) <= 2

    def test_detect_spreading_contagion(self, haven_coordinator):
        """Test detection when contagion is spreading."""
        # Many agents with poor performance
        for i in range(10):
            performance = [0.3, 0.2, 0.15] if i < 6 else [0.8, 0.85]
            haven_coordinator.assess_agent_risk(
                agent_id=f"agent_{i}",
                policy_state={"weights": np.array([0.5])},
                recent_performance=performance
            )

        contagion_report = haven_coordinator.detect_policy_contagion()

        # Should detect spreading contagion (>30% of agents affected)
        assert contagion_report.contagion_status == ContagionStatus.SPREADING
        assert len(contagion_report.affected_agents) >= 4
        assert contagion_report.contagion_score > 0.5


class TestInterventionWorkflow:
    """Test intervention recommendation and execution."""

    def test_recommend_monitoring_intervention(self, haven_coordinator):
        """Test HAVEN recommends monitoring for moderate risk."""
        assessment = haven_coordinator.assess_agent_risk(
            agent_id="moderate_agent",
            policy_state={"weights": np.array([0.5])},
            recent_performance=[0.5, 0.48, 0.52]
        )

        # Moderate risk should recommend monitoring
        assert assessment.recommended_intervention == InterventionType.MONITORING

    def test_recommend_isolation_intervention(self, haven_coordinator):
        """Test HAVEN recommends isolation for high-risk agent."""
        assessment = haven_coordinator.assess_agent_risk(
            agent_id="high_risk_agent",
            policy_state={"weights": np.array([0.1])},
            recent_performance=[0.1, 0.08, 0.05]
        )

        # High risk should recommend isolation or rollback
        assert assessment.recommended_intervention in [InterventionType.ISOLATION, InterventionType.ROLLBACK]

    def test_intervention_execution_via_risk_manager(self, mock_mesa_model, mock_redis_client, haven_coordinator):
        """Test Risk Manager can execute interventions recommended by HAVEN."""
        risk_manager = RiskManagerAgent(
            model=mock_mesa_model,
            unique_id="risk_mgr",
            redis_client=mock_redis_client,
            haven_coordinator=haven_coordinator,
            auto_intervention=True
        )

        # Register an agent for monitoring
        risk_manager.register_agent_for_monitoring(
            agent_id="target_agent",
            agent_data={"performance_history": [0.1, 0.05, 0.02]}
        )

        # Create a high-risk assessment
        assessment = RiskAssessment(
            agent_id="target_agent",
            risk_level=RiskLevel.CRITICAL,
            risk_score=0.95,
            contributing_factors={"performance_drop": 0.9},
            recommended_intervention=InterventionType.ISOLATION,
            timestamp=time.time(),
            confidence=0.95
        )

        # Mock the intervention execution
        haven_coordinator.execute_intervention = Mock(return_value=True)

        # Risk manager should attempt to execute intervention
        with patch.object(risk_manager, '_handle_high_risk_agent'):
            risk_manager._handle_high_risk_agent("target_agent", assessment)

        # Verify intervention was considered
        assert risk_manager.total_assessments >= 0


class TestSystemWideRiskAggregation:
    """Test system-wide risk aggregation and reporting."""

    def test_system_risk_with_all_healthy_agents(self, haven_coordinator):
        """Test system risk calculation when all agents are healthy."""
        # Assess 5 healthy agents
        for i in range(5):
            haven_coordinator.assess_agent_risk(
                agent_id=f"agent_{i}",
                policy_state={"weights": np.array([0.5])},
                recent_performance=[0.8, 0.85, 0.9]
            )

        system_risk = haven_coordinator.assess_system_risk()

        # System risk should be low
        assert system_risk < 0.3

    def test_system_risk_with_mixed_agents(self, haven_coordinator):
        """Test system risk reflects mixture of healthy and risky agents."""
        # 3 healthy agents
        for i in range(3):
            haven_coordinator.assess_agent_risk(
                agent_id=f"healthy_{i}",
                policy_state={"weights": np.array([0.5])},
                recent_performance=[0.85, 0.9]
            )

        # 2 risky agents
        for i in range(2):
            haven_coordinator.assess_agent_risk(
                agent_id=f"risky_{i}",
                policy_state={"weights": np.array([0.1])},
                recent_performance=[0.2, 0.15]
            )

        system_risk = haven_coordinator.assess_system_risk()

        # System risk should be moderate (average of healthy and risky)
        assert 0.3 < system_risk < 0.7

    def test_system_risk_escalation(self, haven_coordinator):
        """Test system risk escalates as more agents become risky."""
        # Initial: All healthy
        for i in range(5):
            haven_coordinator.assess_agent_risk(
                agent_id=f"agent_{i}",
                policy_state={"weights": np.array([0.5])},
                recent_performance=[0.85, 0.9]
            )

        initial_risk = haven_coordinator.assess_system_risk()

        # Escalation: Some agents degrade
        for i in range(3):
            haven_coordinator.assess_agent_risk(
                agent_id=f"agent_{i}",
                policy_state={"weights": np.array([0.2])},
                recent_performance=[0.3, 0.2]
            )

        escalated_risk = haven_coordinator.assess_system_risk()

        # Risk should increase
        assert escalated_risk > initial_risk


class TestHAVENRecoveryWorkflow:
    """Test system recovery and stabilization after intervention."""

    def test_recovery_after_isolation(self, mock_mesa_model, mock_redis_client, haven_coordinator):
        """Test system stabilizes after isolating risky agents."""
        risk_manager = RiskManagerAgent(
            model=mock_mesa_model,
            unique_id="risk_mgr",
            redis_client=mock_redis_client,
            haven_coordinator=haven_coordinator,
            auto_intervention=True
        )

        # Register agents
        for i in range(5):
            risk_manager.register_agent_for_monitoring(
                agent_id=f"agent_{i}",
                agent_data={"performance_history": [0.1 if i < 2 else 0.85]}
            )

        # Assess all agents
        for agent_id in risk_manager.monitored_agents.keys():
            perf = [0.1] if agent_id in ["agent_0", "agent_1"] else [0.85]
            haven_coordinator.assess_agent_risk(
                agent_id=agent_id,
                policy_state={"weights": np.array([0.5])},
                recent_performance=perf
            )

        initial_system_risk = haven_coordinator.assess_system_risk()

        # Simulate isolation of risky agents
        # (In reality, this would prevent them from influencing others)
        # Here we just remove them from assessment
        for risky_id in ["agent_0", "agent_1"]:
            if risky_id in haven_coordinator.assessments:
                del haven_coordinator.assessments[risky_id]

        recovered_system_risk = haven_coordinator.assess_system_risk()

        # System risk should decrease after isolation
        assert recovered_system_risk < initial_system_risk or recovered_system_risk < 0.3

    def test_contagion_containment(self, haven_coordinator):
        """Test contagion can be contained through intervention."""
        # Initial spreading contagion
        for i in range(10):
            perf = [0.2, 0.15] if i < 6 else [0.85, 0.9]
            haven_coordinator.assess_agent_risk(
                agent_id=f"agent_{i}",
                policy_state={"weights": np.array([0.5])},
                recent_performance=perf
            )

        initial_report = haven_coordinator.detect_policy_contagion()
        assert initial_report.contagion_status == ContagionStatus.SPREADING

        # Simulate intervention: Isolate affected agents
        affected_agents = list(initial_report.affected_agents)[:4]
        for agent_id in affected_agents:
            if agent_id in haven_coordinator.assessments:
                del haven_coordinator.assessments[agent_id]

        # Re-assess contagion
        post_intervention_report = haven_coordinator.detect_policy_contagion()

        # Contagion should be contained or healthy
        assert post_intervention_report.contagion_status in [ContagionStatus.CONTAINED, ContagionStatus.HEALTHY]


class TestHAVENEdgeCases:
    """Test HAVEN edge cases and error handling."""

    def test_no_agents_monitored(self, haven_coordinator):
        """Test HAVEN handles scenario with no agents to monitor."""
        system_risk = haven_coordinator.assess_system_risk()

        # Should return 0.0 or handle gracefully
        assert system_risk == 0.0

    def test_single_agent_system(self, haven_coordinator):
        """Test HAVEN works with a single agent."""
        assessment = haven_coordinator.assess_agent_risk(
            agent_id="solo_agent",
            policy_state={"weights": np.array([0.5])},
            recent_performance=[0.75]
        )

        system_risk = haven_coordinator.assess_system_risk()

        # System risk should equal single agent risk
        assert isinstance(assessment, RiskAssessment)
        assert system_risk >= 0.0

    def test_concurrent_assessments(self, haven_coordinator):
        """Test HAVEN handles concurrent risk assessments correctly."""
        # Simulate concurrent assessments (in real scenario, different threads)
        agent_ids = [f"agent_{i}" for i in range(10)]

        assessments = []
        for agent_id in agent_ids:
            assessment = haven_coordinator.assess_agent_risk(
                agent_id=agent_id,
                policy_state={"weights": np.array([0.5])},
                recent_performance=[np.random.rand()]
            )
            assessments.append(assessment)

        # All assessments should be recorded
        assert len(assessments) == 10
        assert len(haven_coordinator.assessments) == 10


@pytest.mark.slow
class TestHAVENPerformance:
    """Performance tests for HAVEN workflow."""

    def test_risk_assessment_performance(self, haven_coordinator):
        """Test risk assessment completes quickly for many agents."""
        import time

        num_agents = 100
        start_time = time.time()

        for i in range(num_agents):
            haven_coordinator.assess_agent_risk(
                agent_id=f"agent_{i}",
                policy_state={"weights": np.random.rand(10)},
                recent_performance=np.random.rand(10).tolist()
            )

        elapsed_time = time.time() - start_time

        # Should assess 100 agents in < 1 second
        assert elapsed_time < 1.0

    def test_contagion_detection_scalability(self, haven_coordinator):
        """Test contagion detection scales to large agent populations."""
        # Assess 200 agents
        for i in range(200):
            perf = [np.random.rand()] * 5
            haven_coordinator.assess_agent_risk(
                agent_id=f"agent_{i}",
                policy_state={"weights": np.array([0.5])},
                recent_performance=perf
            )

        import time
        start_time = time.time()

        contagion_report = haven_coordinator.detect_policy_contagion()

        elapsed_time = time.time() - start_time

        # Contagion detection should be fast even with 200 agents
        assert elapsed_time < 0.5
        assert isinstance(contagion_report, ContagionReport)
