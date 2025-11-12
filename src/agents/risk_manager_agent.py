"""
Risk Manager Agent for the Mycelial ABM Framework

This agent implements the HAVEN (Hierarchical Adversarial Value Estimation Network)
framework to monitor the agent swarm for toxic policies, assess system-wide risk,
and coordinate interventions to maintain system stability and safety.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
import logging
import numpy as np

from agents.base_agent import MycelialAgent
from core.haven_base import (
    HavenRiskCoordinator,
    RiskLevel,
    ContagionStatus,
    InterventionType,
    RiskAssessment,
    ContagionReport
)

logger = logging.getLogger(__name__)


class RiskManagerAgent(MycelialAgent):
    """
    Risk Manager Agent that monitors and coordinates system-wide safety.

    This agent serves as the HAVEN oversight layer that:
    - Monitors all specialist agents for risky behavior
    - Detects policy contagion (bad policies spreading)
    - Assesses system-wide risk levels
    - Recommends and executes interventions
    - Maintains audit trail of risk events
    - Provides system health reporting

    The Risk Manager operates with minimal overhead and preserves agent
    autonomy while preventing catastrophic failures.

    Typical responsibilities:
    - Real-time risk assessment
    - Policy contagion detection
    - Automated intervention execution
    - System health monitoring
    - Adversarial scenario testing
    - Safety constraint verification
    """

    def __init__(
        self,
        unique_id: int,
        model,
        redis_client,
        haven_coordinator: HavenRiskCoordinator,
        team_id: str = "risk_managers",
        agent_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Risk Manager Agent.

        Args:
            unique_id: Unique identifier for this agent
            model: The Mesa model this agent belongs to
            redis_client: Redis client for data operations
            haven_coordinator: HAVEN risk coordinator instance
            team_id: Team identifier (default: "risk_managers")
            agent_config: Optional configuration dictionary
        """
        super().__init__(unique_id, model, redis_client, team_id, agent_config)

        # HAVEN coordinator for risk management
        self.haven_coordinator = haven_coordinator

        # Also register with model's HAVEN coordinator if available
        if hasattr(model, 'haven_coordinator') and model.haven_coordinator:
            model.haven_coordinator = haven_coordinator

        # Risk monitoring configuration
        self.monitoring_interval = agent_config.get("monitoring_interval", 5) if agent_config else 5
        self.risk_threshold = agent_config.get("risk_threshold", 0.7) if agent_config else 0.7
        self.auto_intervention = agent_config.get("auto_intervention", True) if agent_config else True

        # Channels for monitoring
        self.agent_state_channel = "abm:agent_states"
        self.alert_channel = "abm:risk_alerts"

        # Subscribe to agent state updates
        self.subscribe_to_channel(self.agent_state_channel)

        # Tracked agents
        self.monitored_agents: Dict[str, Dict[str, Any]] = {}

        # Risk event history
        self.risk_events: List[Dict[str, Any]] = []
        self.intervention_history: List[Dict[str, Any]] = []

        # System metrics
        self.total_assessments: int = 0
        self.high_risk_detections: int = 0
        self.interventions_executed: int = 0
        self.contagions_detected: int = 0

        # Alert state
        self.current_alert_level: RiskLevel = RiskLevel.MINIMAL
        self.active_alerts: List[str] = []

        logger.info("RiskManagerAgent %s initialized with HAVEN coordinator",
                   self.agent_id)

    def step(self):
        """
        Execute one step of risk management behavior.

        Flow:
        1. Collect agent state updates from monitored agents
        2. Assess risk for each agent
        3. Detect policy contagion patterns
        4. Assess overall system risk
        5. Recommend and execute interventions if needed
        6. Publish risk alerts
        7. Update metrics and logs
        """
        self.step_count += 1

        # Perform monitoring if it's time
        if self.step_count % self.monitoring_interval == 0:
            self._perform_risk_monitoring()

        # Check for contagion periodically
        if self.step_count % (self.monitoring_interval * 2) == 0:
            self._check_for_contagion()

        # Assess system-wide risk
        system_risk = self._assess_system_risk()
        self.risk_score = system_risk

        # Update alert level
        self._update_alert_level(system_risk)

        # Publish health report periodically
        if self.step_count % 50 == 0:
            self._publish_health_report()

        # Reward based on system stability (low risk is good)
        self.last_reward = 1.0 - system_risk
        self.cumulative_reward += self.last_reward

        # Update performance metrics
        self._update_performance_metrics(self.last_reward)

        # Save state periodically
        if self.step_count % 20 == 0:
            self._save_state_to_redis()

        logger.debug("%s completed monitoring step %d (system risk: %.3f)",
                    self.agent_id, self.step_count, system_risk)

    def _perform_risk_monitoring(self):
        """
        Monitor all agents and assess their risk levels.
        """
        logger.debug("%s performing risk monitoring on %d agents",
                    self.agent_id, len(self.monitored_agents))

        for agent_id, agent_data in self.monitored_agents.items():
            # Skip if agent is already isolated
            if agent_data.get("is_isolated", False):
                continue

            # Assess agent risk
            risk_assessment = self._assess_agent_risk(agent_id, agent_data)

            # Store assessment
            self.total_assessments += 1

            # Handle high-risk agents
            if risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self.high_risk_detections += 1
                self._handle_high_risk_agent(agent_id, risk_assessment)

    def _assess_agent_risk(
        self,
        agent_id: str,
        agent_data: Dict[str, Any]
    ) -> RiskAssessment:
        """
        Assess risk level for a specific agent.

        Args:
            agent_id: ID of the agent to assess
            agent_data: Agent's state data

        Returns:
            RiskAssessment object
        """
        # Extract metrics from agent data
        policy_state = agent_data.get("policy_parameters", {})
        performance_history = agent_data.get("performance_history", [])
        behavioral_metrics = {
            "cumulative_reward": agent_data.get("cumulative_reward", 0.0),
            "policy_version": agent_data.get("policy_version", 0),
            "exploration_rate": agent_data.get("exploration_rate", 0.0)
        }

        # Use HAVEN coordinator to assess risk
        risk_assessment = self.haven_coordinator.assess_agent_risk(
            agent_id=agent_id,
            policy_state=policy_state,
            recent_performance=performance_history,
            behavioral_metrics=behavioral_metrics
        )

        logger.debug("%s assessed risk for %s: %s (%.3f)",
                    self.agent_id, agent_id,
                    risk_assessment.risk_level.value,
                    risk_assessment.risk_score)

        return risk_assessment

    def _handle_high_risk_agent(self, agent_id: str, risk_assessment: RiskAssessment):
        """
        Handle an agent identified as high risk.

        Args:
            agent_id: ID of the high-risk agent
            risk_assessment: The risk assessment
        """
        # Log risk event
        self._log_risk_event(agent_id, risk_assessment)

        # Get intervention recommendation
        intervention = self.haven_coordinator.recommend_intervention(risk_assessment)

        # Execute intervention if auto-intervention is enabled
        if self.auto_intervention and intervention != InterventionType.NONE:
            success = self._execute_intervention(agent_id, intervention, risk_assessment)

            if success:
                # Publish alert
                self._publish_risk_alert(agent_id, risk_assessment, intervention)

    def _check_for_contagion(self):
        """
        Check for policy contagion in the system.
        """
        logger.debug("%s checking for policy contagion", self.agent_id)

        # Use HAVEN coordinator to detect contagion
        contagion_report = self.haven_coordinator.detect_policy_contagion(
            time_window=self.monitoring_interval * 5
        )

        # Handle contagion if detected
        if contagion_report.contagion_status not in [ContagionStatus.HEALTHY, ContagionStatus.EARLY_WARNING]:
            self.contagions_detected += 1
            self._handle_contagion(contagion_report)

    def _handle_contagion(self, contagion_report: ContagionReport):
        """
        Handle detected policy contagion.

        Args:
            contagion_report: The contagion analysis report
        """
        logger.warning("%s detected policy contagion: %s (score: %.3f)",
                      self.agent_id,
                      contagion_report.contagion_status.value,
                      contagion_report.contagion_score)

        # Identify source agents
        sources = self.haven_coordinator.identify_contagion_source(
            contagion_report.affected_agents
        )

        logger.info("%s identified %d potential contagion sources",
                   self.agent_id, len(sources))

        # Execute containment actions
        for intervention_type in contagion_report.containment_actions:
            for agent_id in contagion_report.affected_agents:
                self._execute_intervention(
                    agent_id,
                    intervention_type,
                    {"reason": "contagion_containment", "sources": sources}
                )

        # Publish contagion alert
        self._publish_contagion_alert(contagion_report, sources)

    def _assess_system_risk(self) -> float:
        """
        Assess overall system risk.

        Returns:
            System risk score (0.0 to 1.0)
        """
        if not self.monitored_agents:
            return 0.0

        # Use HAVEN coordinator to assess system risk
        system_risk = self.haven_coordinator.assess_system_risk()

        return system_risk

    def _execute_intervention(
        self,
        agent_id: str,
        intervention: InterventionType,
        metadata: Any
    ) -> bool:
        """
        Execute a risk mitigation intervention.

        Args:
            agent_id: Agent to intervene on
            intervention: Type of intervention
            metadata: Additional context

        Returns:
            True if intervention was successful
        """
        logger.info("%s executing %s intervention on %s",
                   self.agent_id, intervention.value, agent_id)

        # Use HAVEN coordinator to execute intervention
        success = self.haven_coordinator.execute_intervention(
            agent_id=agent_id,
            intervention=intervention,
            metadata=metadata if isinstance(metadata, dict) else {"assessment": str(metadata)}
        )

        if success:
            self.interventions_executed += 1
            self._log_intervention(agent_id, intervention, metadata)

        return success

    # ==========================================
    # Agent registration and tracking
    # ==========================================

    def register_agent_for_monitoring(self, agent_id: str, agent_data: Dict[str, Any]):
        """
        Register an agent for risk monitoring.

        Args:
            agent_id: ID of the agent to monitor
            agent_data: Initial agent state data
        """
        self.monitored_agents[agent_id] = agent_data
        self.haven_coordinator.register_agent(agent_id)

        logger.info("%s registered agent for monitoring: %s", self.agent_id, agent_id)

    def unregister_agent(self, agent_id: str):
        """
        Unregister an agent from monitoring.

        Args:
            agent_id: ID of the agent to unregister
        """
        self.monitored_agents.pop(agent_id, None)
        self.haven_coordinator.unregister_agent(agent_id)

        logger.info("%s unregistered agent: %s", self.agent_id, agent_id)

    def update_agent_state(self, agent_id: str, agent_data: Dict[str, Any]):
        """
        Update the state data for a monitored agent.

        Args:
            agent_id: ID of the agent
            agent_data: Updated state data
        """
        if agent_id in self.monitored_agents:
            self.monitored_agents[agent_id].update(agent_data)

    # ==========================================
    # Logging and alerting
    # ==========================================

    def _log_risk_event(self, agent_id: str, risk_assessment: RiskAssessment):
        """
        Log a risk event for audit trail.

        Args:
            agent_id: Agent that triggered the event
            risk_assessment: The risk assessment
        """
        event = {
            "timestamp": self.step_count,
            "agent_id": agent_id,
            "risk_level": risk_assessment.risk_level.value,
            "risk_score": risk_assessment.risk_score,
            "contributing_factors": risk_assessment.contributing_factors,
            "recommended_intervention": risk_assessment.recommended_intervention.value
        }

        self.risk_events.append(event)

        # Limit event history size
        if len(self.risk_events) > 10000:
            self.risk_events = self.risk_events[-10000:]

    def _log_intervention(
        self,
        agent_id: str,
        intervention: InterventionType,
        metadata: Any
    ):
        """
        Log an intervention for audit trail.

        Args:
            agent_id: Agent that was intervened on
            intervention: Type of intervention
            metadata: Additional context
        """
        intervention_record = {
            "timestamp": self.step_count,
            "agent_id": agent_id,
            "intervention_type": intervention.value,
            "metadata": metadata,
            "executed_by": self.agent_id
        }

        self.intervention_history.append(intervention_record)

        # Limit history size
        if len(self.intervention_history) > 5000:
            self.intervention_history = self.intervention_history[-5000:]

    def _publish_risk_alert(
        self,
        agent_id: str,
        risk_assessment: RiskAssessment,
        intervention: InterventionType
    ):
        """
        Publish a risk alert to the alert channel.

        Args:
            agent_id: Agent that triggered the alert
            risk_assessment: The risk assessment
            intervention: Intervention taken
        """
        alert = {
            "type": "risk_alert",
            "timestamp": self.step_count,
            "manager_id": self.agent_id,
            "agent_id": agent_id,
            "risk_level": risk_assessment.risk_level.value,
            "risk_score": risk_assessment.risk_score,
            "intervention": intervention.value,
            "factors": risk_assessment.contributing_factors
        }

        self.publish_message(self.alert_channel, alert)
        self.active_alerts.append(agent_id)

    def _publish_contagion_alert(
        self,
        contagion_report: ContagionReport,
        sources: List[Tuple[str, float]]
    ):
        """
        Publish a policy contagion alert.

        Args:
            contagion_report: The contagion report
            sources: List of (agent_id, confidence) tuples for sources
        """
        alert = {
            "type": "contagion_alert",
            "timestamp": self.step_count,
            "manager_id": self.agent_id,
            "contagion_status": contagion_report.contagion_status.value,
            "contagion_score": contagion_report.contagion_score,
            "affected_agents": list(contagion_report.affected_agents),
            "suspected_sources": sources,
            "spread_rate": contagion_report.spread_rate,
            "containment_actions": [action.value for action in contagion_report.containment_actions]
        }

        self.publish_message(self.alert_channel, alert)

    def _update_alert_level(self, system_risk: float):
        """
        Update the current alert level based on system risk.

        Args:
            system_risk: Current system risk score
        """
        if system_risk < 0.2:
            self.current_alert_level = RiskLevel.MINIMAL
        elif system_risk < 0.4:
            self.current_alert_level = RiskLevel.LOW
        elif system_risk < 0.6:
            self.current_alert_level = RiskLevel.MODERATE
        elif system_risk < 0.8:
            self.current_alert_level = RiskLevel.HIGH
        else:
            self.current_alert_level = RiskLevel.CRITICAL

    def _publish_health_report(self):
        """
        Publish a system health report.
        """
        health_report = self.haven_coordinator.get_system_health_report()

        # Add risk manager metrics
        health_report["manager_metrics"] = {
            "manager_id": self.agent_id,
            "total_assessments": self.total_assessments,
            "high_risk_detections": self.high_risk_detections,
            "interventions_executed": self.interventions_executed,
            "contagions_detected": self.contagions_detected,
            "current_alert_level": self.current_alert_level.value,
            "active_alerts": len(self.active_alerts)
        }

        # Publish to health report channel
        self.publish_message("abm:health_reports", health_report)

        logger.info("%s published health report (alert level: %s)",
                   self.agent_id, self.current_alert_level.value)

    # ==========================================
    # Metrics and reporting
    # ==========================================

    def get_risk_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about risk management operations.

        Returns:
            Dictionary with risk management metrics
        """
        return {
            "agent_id": self.agent_id,
            "monitored_agents": len(self.monitored_agents),
            "total_assessments": self.total_assessments,
            "high_risk_detections": self.high_risk_detections,
            "interventions_executed": self.interventions_executed,
            "contagions_detected": self.contagions_detected,
            "current_alert_level": self.current_alert_level.value,
            "system_risk_score": self.risk_score,
            "active_alerts": len(self.active_alerts),
            "intervention_rate": self._calculate_intervention_rate()
        }

    def _calculate_intervention_rate(self) -> float:
        """
        Calculate the rate of interventions per assessment.

        Returns:
            Intervention rate
        """
        if self.total_assessments == 0:
            return 0.0
        return self.interventions_executed / self.total_assessments

    def get_recent_risk_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent risk events.

        Args:
            count: Number of recent events to retrieve

        Returns:
            List of recent risk events
        """
        return self.risk_events[-count:] if self.risk_events else []

    def get_recent_interventions(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent interventions.

        Args:
            count: Number of recent interventions to retrieve

        Returns:
            List of recent interventions
        """
        return self.intervention_history[-count:] if self.intervention_history else []

    def export_risk_analytics(self) -> Dict[str, Any]:
        """
        Export comprehensive risk analytics data.

        Returns:
            Dictionary with detailed analytics
        """
        return self.haven_coordinator.export_risk_analytics()

    def _save_state_to_redis(self):
        """
        Save risk manager state to Redis.
        """
        # Call parent method for basic state
        super()._save_state_to_redis()

        # Save additional risk manager state
        manager_state = {
            "monitored_agents": list(self.monitored_agents.keys()),
            "total_assessments": self.total_assessments,
            "high_risk_detections": self.high_risk_detections,
            "interventions_executed": self.interventions_executed,
            "contagions_detected": self.contagions_detected,
            "current_alert_level": self.current_alert_level.value,
            "system_risk_score": self.risk_score
        }

        key = f"agent:risk_manager_state:{self.agent_id}"
        self.redis_client.set_key_value(key, manager_state)

    def reset(self):
        """
        Reset the risk manager agent to initial state.
        """
        super().reset()

        # Reset metrics
        self.total_assessments = 0
        self.high_risk_detections = 0
        self.interventions_executed = 0
        self.contagions_detected = 0
        self.current_alert_level = RiskLevel.MINIMAL
        self.active_alerts.clear()
        self.risk_events.clear()
        self.intervention_history.clear()

        logger.info("%s has been reset", self.agent_id)
