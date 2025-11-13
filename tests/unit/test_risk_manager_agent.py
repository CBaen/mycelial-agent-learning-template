"""
Unit tests for RiskManagerAgent.

Tests the risk manager agent functionality including:
- Initialization and configuration
- Agent registration and monitoring
- Risk assessment (HAVEN integration)
- Contagion detection
- Intervention execution
- Alert management
- System health reporting
- Metrics and analytics
- Event logging and history
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any

from src.agents.risk_manager_agent import RiskManagerAgent
from src.core.haven_base import (
    RiskLevel,
    ContagionStatus,
    InterventionType,
    RiskAssessment,
    ContagionReport
)


class TestRiskManagerAgentInit:
    """Test RiskManagerAgent initialization."""

    def test_init_with_defaults(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Test initialization with default parameters."""
        agent = RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

        assert agent.unique_id == 1
        assert agent.team_id == "risk_managers"
        assert agent.agent_type == "RiskManagerAgent"
        assert agent.haven_coordinator == mock_haven_coordinator

    def test_init_with_custom_team(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Test initialization with custom team."""
        agent = RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator,
            team_id="custom_risk_managers"
        )

        assert agent.team_id == "custom_risk_managers"

    def test_init_with_custom_config(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Test initialization with custom configuration."""
        config = {
            "monitoring_interval": 10,
            "risk_threshold": 0.8,
            "auto_intervention": False
        }

        agent = RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator,
            agent_config=config
        )

        assert agent.monitoring_interval == 10
        assert agent.risk_threshold == 0.8
        assert agent.auto_intervention is False

    def test_init_subscribes_to_agent_state_channel(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Test that agent subscribes to agent state channel."""
        agent = RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

        mock_redis_client.subscribe.assert_called_once_with(["abm:agent_states"])

    def test_init_metrics(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Test initialization of risk metrics."""
        agent = RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

        assert agent.total_assessments == 0
        assert agent.high_risk_detections == 0
        assert agent.interventions_executed == 0
        assert agent.contagions_detected == 0
        assert agent.current_alert_level == RiskLevel.MINIMAL
        assert agent.active_alerts == []


class TestRiskManagerAgentRegistration:
    """Test agent registration and tracking."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Create test risk manager agent."""
        return RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

    def test_register_agent_for_monitoring(self, agent, mock_haven_coordinator):
        """Test registering an agent for monitoring."""
        agent_data = {
            "agent_id": "specialist_123",
            "policy_parameters": {},
            "cumulative_reward": 50.0
        }

        agent.register_agent_for_monitoring("specialist_123", agent_data)

        assert "specialist_123" in agent.monitored_agents
        assert agent.monitored_agents["specialist_123"] == agent_data
        mock_haven_coordinator.register_agent.assert_called_once_with("specialist_123")

    def test_register_multiple_agents(self, agent):
        """Test registering multiple agents."""
        for i in range(5):
            agent.register_agent_for_monitoring(
                f"agent_{i}",
                {"agent_id": f"agent_{i}"}
            )

        assert len(agent.monitored_agents) == 5

    def test_unregister_agent(self, agent, mock_haven_coordinator):
        """Test unregistering an agent."""
        agent.register_agent_for_monitoring("agent_x", {"data": "test"})

        agent.unregister_agent("agent_x")

        assert "agent_x" not in agent.monitored_agents
        mock_haven_coordinator.unregister_agent.assert_called_once_with("agent_x")

    def test_unregister_nonexistent_agent(self, agent):
        """Test unregistering non-existent agent (should not error)."""
        agent.unregister_agent("nonexistent")

    def test_update_agent_state(self, agent):
        """Test updating agent state."""
        agent.register_agent_for_monitoring(
            "agent_update",
            {"cumulative_reward": 10.0}
        )

        agent.update_agent_state("agent_update", {"cumulative_reward": 20.0})

        assert agent.monitored_agents["agent_update"]["cumulative_reward"] == 20.0

    def test_update_nonexistent_agent_state(self, agent):
        """Test updating state of non-existent agent (should not error)."""
        agent.update_agent_state("nonexistent", {"data": "test"})


class TestRiskManagerAgentRiskAssessment:
    """Test risk assessment functionality."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Create test risk manager agent."""
        return RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

    def test_assess_agent_risk(self, agent, mock_haven_coordinator):
        """Test assessing risk for a specific agent."""
        agent_data = {
            "policy_parameters": {"weights": {}},
            "performance_history": [0.5, 0.6, 0.7],
            "cumulative_reward": 100.0,
            "policy_version": 5,
            "exploration_rate": 0.1
        }

        mock_risk_assessment = RiskAssessment(
            agent_id="test_agent",
            risk_level=RiskLevel.MODERATE,
            risk_score=0.55,
            contributing_factors={"high_variance": 0.3},
            recommended_intervention=InterventionType.MONITORING,
            timestamp=1234567890.0,
            confidence=0.85
        )
        mock_haven_coordinator.assess_agent_risk.return_value = mock_risk_assessment

        result = agent._assess_agent_risk("test_agent", agent_data)

        assert result.risk_level == RiskLevel.MODERATE
        assert result.risk_score == 0.55
        mock_haven_coordinator.assess_agent_risk.assert_called_once()

    def test_assess_system_risk(self, agent, mock_haven_coordinator):
        """Test assessing overall system risk."""
        # Register some agents so the check passes
        agent.register_agent_for_monitoring("agent_1", {"performance_history": [0.8]})
        agent.register_agent_for_monitoring("agent_2", {"performance_history": [0.6]})

        mock_haven_coordinator.assess_system_risk.return_value = 0.35

        risk = agent._assess_system_risk()

        assert risk == 0.35
        mock_haven_coordinator.assess_system_risk.assert_called_once()

    def test_assess_system_risk_no_agents(self, agent):
        """Test system risk assessment with no monitored agents."""
        risk = agent._assess_system_risk()

        assert risk == 0.0


class TestRiskManagerAgentInterventions:
    """Test intervention execution."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Create test risk manager agent."""
        return RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

    def test_execute_intervention_success(self, agent, mock_haven_coordinator):
        """Test successful intervention execution."""
        mock_haven_coordinator.execute_intervention.return_value = True

        risk_assessment = RiskAssessment(
            agent_id="risky_agent",
            risk_level=RiskLevel.HIGH,
            risk_score=0.85,
            contributing_factors={"erratic_behavior": 0.5},
            recommended_intervention=InterventionType.ISOLATION,
            timestamp=1234567890.0,
            confidence=0.9
        )

        success = agent._execute_intervention(
            "risky_agent",
            InterventionType.ISOLATION,
            risk_assessment
        )

        assert success is True
        assert agent.interventions_executed == 1
        mock_haven_coordinator.execute_intervention.assert_called_once()

    def test_execute_intervention_failure(self, agent, mock_haven_coordinator):
        """Test failed intervention execution."""
        mock_haven_coordinator.execute_intervention.return_value = False

        success = agent._execute_intervention(
            "agent_x",
            InterventionType.ROLLBACK,
            {}
        )

        assert success is False
        assert agent.interventions_executed == 0

    def test_handle_high_risk_agent_with_auto_intervention(self, agent, mock_haven_coordinator):
        """Test handling high-risk agent with auto-intervention enabled."""
        agent.auto_intervention = True

        risk_assessment = RiskAssessment(
            agent_id="high_risk",
            risk_level=RiskLevel.CRITICAL,
            risk_score=0.95,
            contributing_factors={"policy_corruption": 0.7},
            recommended_intervention=InterventionType.ISOLATION,
            timestamp=1234567890.0,
            confidence=0.95
        )

        mock_haven_coordinator.recommend_intervention.return_value = InterventionType.ISOLATION
        mock_haven_coordinator.execute_intervention.return_value = True

        with patch.object(agent, '_log_risk_event'):
            with patch.object(agent, '_publish_risk_alert'):
                agent._handle_high_risk_agent("high_risk", risk_assessment)

                mock_haven_coordinator.execute_intervention.assert_called_once()

    def test_handle_high_risk_agent_without_auto_intervention(self, agent, mock_haven_coordinator):
        """Test handling high-risk agent without auto-intervention."""
        agent.auto_intervention = False

        risk_assessment = RiskAssessment(
            agent_id="high_risk",
            risk_level=RiskLevel.HIGH,
            risk_score=0.82,
            contributing_factors={},
            recommended_intervention=InterventionType.MONITORING,
            timestamp=1234567890.0,
            confidence=0.85
        )

        mock_haven_coordinator.recommend_intervention.return_value = InterventionType.MONITORING

        with patch.object(agent, '_log_risk_event'):
            agent._handle_high_risk_agent("high_risk", risk_assessment)

            # Should not execute intervention
            mock_haven_coordinator.execute_intervention.assert_not_called()


class TestRiskManagerAgentContagionDetection:
    """Test policy contagion detection."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Create test risk manager agent."""
        return RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

    def test_check_for_contagion_healthy(self, agent, mock_haven_coordinator):
        """Test contagion check with healthy system."""
        contagion_report = ContagionReport(
            contagion_status=ContagionStatus.HEALTHY,
            contagion_score=0.05,
            affected_agents=set(),
            source_agents=set(),
            spread_rate=0.0,
            containment_actions=[],
            timestamp=1234567890.0
        )
        mock_haven_coordinator.detect_policy_contagion.return_value = contagion_report

        agent._check_for_contagion()

        # Should not handle contagion for healthy status
        assert agent.contagions_detected == 0

    def test_check_for_contagion_detected(self, agent, mock_haven_coordinator):
        """Test contagion detection and handling."""
        contagion_report = ContagionReport(
            contagion_status=ContagionStatus.SPREADING,
            contagion_score=0.75,
            affected_agents={"agent_1", "agent_2", "agent_3"},
            source_agents={"agent_1"},
            spread_rate=0.6,
            containment_actions=[InterventionType.ISOLATION, InterventionType.ROLLBACK],
            timestamp=1234567890.0
        )
        mock_haven_coordinator.detect_policy_contagion.return_value = contagion_report
        mock_haven_coordinator.identify_contagion_source.return_value = [("agent_1", 0.9)]

        with patch.object(agent, '_handle_contagion') as mock_handle:
            agent._check_for_contagion()

            assert agent.contagions_detected == 1
            mock_handle.assert_called_once_with(contagion_report)

    def test_handle_contagion(self, agent, mock_haven_coordinator):
        """Test handling detected contagion."""
        contagion_report = ContagionReport(
            contagion_status=ContagionStatus.SPREADING,
            contagion_score=0.82,
            affected_agents={"agent_a", "agent_b"},
            source_agents={"source_agent"},
            spread_rate=0.7,
            containment_actions=[InterventionType.ISOLATION],
            timestamp=1234567890.0
        )

        mock_haven_coordinator.identify_contagion_source.return_value = [
            ("source_agent", 0.95)
        ]
        mock_haven_coordinator.execute_intervention.return_value = True

        with patch.object(agent, '_publish_contagion_alert'):
            agent._handle_contagion(contagion_report)

            # Should execute intervention for each affected agent
            assert mock_haven_coordinator.execute_intervention.call_count == 2


class TestRiskManagerAgentMonitoring:
    """Test monitoring operations."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Create test risk manager agent."""
        return RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

    def test_perform_risk_monitoring(self, agent, mock_haven_coordinator):
        """Test performing risk monitoring on all agents."""
        # Register some agents
        for i in range(3):
            agent.register_agent_for_monitoring(
                f"agent_{i}",
                {"performance_history": [0.8], "policy_parameters": {}}
            )

        mock_risk_assessment = RiskAssessment(
            agent_id="agent_0",
            risk_level=RiskLevel.LOW,
            risk_score=0.3,
            contributing_factors={},
            recommended_intervention=InterventionType.NONE,
            timestamp=1234567890.0,
            confidence=0.90
        )
        mock_haven_coordinator.assess_agent_risk.return_value = mock_risk_assessment

        agent._perform_risk_monitoring()

        # Should assess all agents
        assert agent.total_assessments == 3
        assert mock_haven_coordinator.assess_agent_risk.call_count == 3

    def test_perform_risk_monitoring_handles_high_risk(self, agent, mock_haven_coordinator):
        """Test monitoring handles high-risk agents."""
        agent.register_agent_for_monitoring(
            "high_risk_agent",
            {"performance_history": [0.2], "policy_parameters": {}}
        )

        high_risk_assessment = RiskAssessment(
            agent_id="high_risk_agent",
            risk_level=RiskLevel.HIGH,
            risk_score=0.88,
            contributing_factors={"poor_performance": 0.7},
            recommended_intervention=InterventionType.MONITORING,
            timestamp=1234567890.0,
            confidence=0.88
        )
        mock_haven_coordinator.assess_agent_risk.return_value = high_risk_assessment

        with patch.object(agent, '_handle_high_risk_agent') as mock_handle:
            agent._perform_risk_monitoring()

            assert agent.high_risk_detections == 1
            mock_handle.assert_called_once()

    def test_perform_risk_monitoring_skips_isolated_agents(self, agent, mock_haven_coordinator):
        """Test monitoring skips isolated agents."""
        agent.register_agent_for_monitoring(
            "isolated_agent",
            {"is_isolated": True, "performance_history": []}
        )

        agent._perform_risk_monitoring()

        # Should skip isolated agent
        mock_haven_coordinator.assess_agent_risk.assert_not_called()
        assert agent.total_assessments == 0


class TestRiskManagerAgentAlerting:
    """Test alerting functionality."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Create test risk manager agent."""
        return RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

    def test_publish_risk_alert(self, agent, mock_redis_client):
        """Test publishing risk alert."""
        risk_assessment = RiskAssessment(
            agent_id="problem_agent",
            risk_level=RiskLevel.CRITICAL,
            risk_score=0.92,
            contributing_factors={"data_corruption": 0.9},
            recommended_intervention=InterventionType.ISOLATION,
            timestamp=1234567890.0,
            confidence=0.92
        )

        agent._publish_risk_alert(
            "problem_agent",
            risk_assessment,
            InterventionType.ISOLATION
        )

        mock_redis_client.publish.assert_called_once()
        call_args = mock_redis_client.publish.call_args
        assert call_args[0][0] == "abm:risk_alerts"
        assert "problem_agent" in agent.active_alerts

    def test_publish_contagion_alert(self, agent, mock_redis_client):
        """Test publishing contagion alert."""
        contagion_report = ContagionReport(
            contagion_status=ContagionStatus.SPREADING,
            contagion_score=0.8,
            affected_agents={"a1", "a2", "a3"},
            source_agents={"source_1", "source_2"},
            spread_rate=0.65,
            containment_actions=[InterventionType.ISOLATION],
            timestamp=1234567890.0
        )
        sources = [("source_1", 0.95), ("source_2", 0.85)]

        agent._publish_contagion_alert(contagion_report, sources)

        mock_redis_client.publish.assert_called_once()

    def test_update_alert_level_minimal(self, agent):
        """Test alert level update for minimal risk."""
        agent._update_alert_level(0.1)

        assert agent.current_alert_level == RiskLevel.MINIMAL

    def test_update_alert_level_low(self, agent):
        """Test alert level update for low risk."""
        agent._update_alert_level(0.3)

        assert agent.current_alert_level == RiskLevel.LOW

    def test_update_alert_level_moderate(self, agent):
        """Test alert level update for moderate risk."""
        agent._update_alert_level(0.5)

        assert agent.current_alert_level == RiskLevel.MODERATE

    def test_update_alert_level_high(self, agent):
        """Test alert level update for high risk."""
        agent._update_alert_level(0.7)

        assert agent.current_alert_level == RiskLevel.HIGH

    def test_update_alert_level_critical(self, agent):
        """Test alert level update for critical risk."""
        agent._update_alert_level(0.9)

        assert agent.current_alert_level == RiskLevel.CRITICAL


class TestRiskManagerAgentLogging:
    """Test event logging."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Create test risk manager agent."""
        return RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

    def test_log_risk_event(self, agent):
        """Test logging a risk event."""
        risk_assessment = RiskAssessment(
            agent_id="test_agent",
            risk_level=RiskLevel.MODERATE,
            risk_score=0.55,
            contributing_factors={"factor1": 0.3, "factor2": 0.25},
            recommended_intervention=InterventionType.MONITORING,
            timestamp=1234567890.0,
            confidence=0.85
        )

        agent._log_risk_event("test_agent", risk_assessment)

        assert len(agent.risk_events) == 1
        event = agent.risk_events[0]
        assert event["agent_id"] == "test_agent"
        assert event["risk_level"] == "moderate"
        assert event["risk_score"] == 0.55

    def test_log_risk_event_size_limit(self, agent):
        """Test risk event log size limiting."""
        # Log many events
        for i in range(11000):
            risk_assessment = RiskAssessment(
                agent_id=f"agent_{i}",
                risk_level=RiskLevel.LOW,
                risk_score=0.2,
                contributing_factors={},
                recommended_intervention=InterventionType.NONE,
                timestamp=1234567890.0,
                confidence=0.95
            )
            agent._log_risk_event(f"agent_{i}", risk_assessment)

        # Should limit to 10000
        assert len(agent.risk_events) == 10000

    def test_log_intervention(self, agent):
        """Test logging an intervention."""
        agent._log_intervention(
            "agent_x",
            InterventionType.ISOLATION,
            {"reason": "high_risk"}
        )

        assert len(agent.intervention_history) == 1
        intervention = agent.intervention_history[0]
        assert intervention["agent_id"] == "agent_x"
        assert intervention["intervention_type"] == "isolation"

    def test_log_intervention_size_limit(self, agent):
        """Test intervention history size limiting."""
        # Log many interventions
        for i in range(6000):
            agent._log_intervention(f"agent_{i}", InterventionType.NONE, {})

        # Should limit to 5000
        assert len(agent.intervention_history) == 5000


class TestRiskManagerAgentMetrics:
    """Test metrics and reporting."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Create test risk manager agent."""
        return RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator
        )

    def test_get_risk_statistics(self, agent):
        """Test getting risk statistics."""
        agent.monitored_agents = {"a1": {}, "a2": {}, "a3": {}}
        agent.total_assessments = 100
        agent.high_risk_detections = 15
        agent.interventions_executed = 10
        agent.contagions_detected = 2
        agent.risk_score = 0.45

        stats = agent.get_risk_statistics()

        assert stats["monitored_agents"] == 3
        assert stats["total_assessments"] == 100
        assert stats["high_risk_detections"] == 15
        assert stats["interventions_executed"] == 10
        assert stats["contagions_detected"] == 2
        assert stats["system_risk_score"] == 0.45

    def test_calculate_intervention_rate(self, agent):
        """Test intervention rate calculation."""
        agent.total_assessments = 100
        agent.interventions_executed = 15

        rate = agent._calculate_intervention_rate()

        assert rate == 0.15

    def test_calculate_intervention_rate_no_assessments(self, agent):
        """Test intervention rate with no assessments."""
        rate = agent._calculate_intervention_rate()

        assert rate == 0.0

    def test_get_recent_risk_events(self, agent):
        """Test getting recent risk events."""
        # Add some events
        for i in range(20):
            risk_assessment = RiskAssessment(
                agent_id=f"agent_{i}",
                risk_level=RiskLevel.LOW,
                risk_score=0.2,
                contributing_factors={},
                recommended_intervention=InterventionType.NONE,
                timestamp=1234567890.0,
                confidence=0.95
            )
            agent._log_risk_event(f"agent_{i}", risk_assessment)

        recent = agent.get_recent_risk_events(count=5)

        assert len(recent) == 5
        # Should be most recent
        assert recent[-1]["agent_id"] == "agent_19"

    def test_get_recent_interventions(self, agent):
        """Test getting recent interventions."""
        # Add some interventions
        for i in range(15):
            agent._log_intervention(f"agent_{i}", InterventionType.MONITORING, {})

        recent = agent.get_recent_interventions(count=3)

        assert len(recent) == 3

    def test_publish_health_report(self, agent, mock_haven_coordinator, mock_redis_client):
        """Test publishing system health report."""
        mock_haven_coordinator.get_system_health_report.return_value = {
            "system_status": "healthy"
        }

        agent._publish_health_report()

        mock_redis_client.publish.assert_called_once()
        call_args = mock_redis_client.publish.call_args
        assert call_args[0][0] == "abm:health_reports"


class TestRiskManagerAgentStep:
    """Test step execution."""

    @pytest.fixture
    def agent(self, mock_redis_client, mock_mesa_model, mock_haven_coordinator):
        """Create test risk manager agent."""
        return RiskManagerAgent(model=mock_mesa_model,
            redis_client=mock_redis_client,
            haven_coordinator=mock_haven_coordinator,
            agent_config={"monitoring_interval": 5}
        )

    def test_step_increments_count(self, agent):
        """Test that step increments step count."""
        initial_count = agent.step_count

        with patch.object(agent, '_assess_system_risk', return_value=0.0):
            agent.step()

        assert agent.step_count == initial_count + 1

    def test_step_performs_monitoring_periodically(self, agent):
        """Test periodic monitoring execution."""
        with patch.object(agent, '_perform_risk_monitoring') as mock_monitor:
            with patch.object(agent, '_assess_system_risk', return_value=0.0):
                # Run 5 steps (monitoring_interval = 5)
                for _ in range(5):
                    agent.step()

                # Should have monitored once (at step 5)
                assert mock_monitor.call_count == 1

    def test_step_checks_contagion_periodically(self, agent):
        """Test periodic contagion checking."""
        with patch.object(agent, '_check_for_contagion') as mock_check:
            with patch.object(agent, '_assess_system_risk', return_value=0.0):
                # Run 10 steps (contagion check interval = monitoring_interval * 2)
                for _ in range(10):
                    agent.step()

                # Should have checked once (at step 10)
                assert mock_check.call_count == 1

    def test_step_updates_risk_score(self, agent):
        """Test that step updates risk score."""
        with patch.object(agent, '_assess_system_risk', return_value=0.65):
            agent.step()

        assert agent.risk_score == 0.65

    def test_step_calculates_reward(self, agent):
        """Test reward calculation based on system stability."""
        with patch.object(agent, '_assess_system_risk', return_value=0.3):
            initial_reward = agent.cumulative_reward

            agent.step()

            # Reward should be 1.0 - risk_score = 0.7
            assert agent.last_reward == 0.7
            assert agent.cumulative_reward > initial_reward
