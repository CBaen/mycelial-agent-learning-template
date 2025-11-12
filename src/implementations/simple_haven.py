"""
SimpleHAVEN - Reference Implementation of HAVEN Risk Coordinator

This module provides a simple implementation of the HAVEN framework for
risk assessment, contagion detection, and intervention in multi-agent systems.

Use this as:
1. A starting point for your own HAVEN implementation
2. A reference for understanding risk coordination
3. A functional engine for prototyping and testing

For production systems, extend with ML-based anomaly detection, graph neural
networks for contagion tracking, and sophisticated intervention strategies.
"""

import sys
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.haven_base import (
    HavenRiskCoordinator,
    RiskAssessment,
    ContagionReport,
    RiskLevel,
    ContagionStatus,
    InterventionType
)
from src.connectors.redis_client import RedisClient
from src.connectors.sql_logger import SQLiteLogger

logger = logging.getLogger(__name__)


class SimpleHAVEN(HavenRiskCoordinator):
    """
    Simple HAVEN Risk Coordinator Implementation.

    Features:
    - Threshold-based risk assessment
    - Graph-based contagion tracking
    - Automated interventions
    - Performance monitoring

    This implementation uses:
    - Statistical anomaly detection for risk assessment
    - Policy influence graph for contagion detection
    - Rule-based intervention logic
    """

    def __init__(
        self,
        coordinator_id: str,
        redis_client: RedisClient,
        sql_logger: Optional[SQLiteLogger] = None,
        risk_threshold: float = 0.7,
        contagion_threshold: float = 0.3,
        performance_window: int = 50,
        intervention_cooldown: int = 10,
        auto_intervention: bool = True
    ):
        """
        Initialize SimpleHAVEN coordinator.

        Args:
            coordinator_id: Unique coordinator ID
            redis_client: Redis client for communication
            sql_logger: SQLite logger for events
            risk_threshold: Threshold for high risk
            contagion_threshold: Threshold for contagion alarm
            performance_window: Window size for performance tracking
            intervention_cooldown: Steps between interventions on same agent
            auto_intervention: Enable automatic interventions
        """
        self.coordinator_id = coordinator_id
        self.redis_client = redis_client
        self.sql_logger = sql_logger
        self.risk_threshold = risk_threshold
        self.contagion_threshold = contagion_threshold
        self.performance_window = performance_window
        self.intervention_cooldown = intervention_cooldown
        self.auto_intervention = auto_intervention

        # Agent tracking
        self.monitored_agents: Dict[str, Dict[str, Any]] = {}  # agent_id -> metadata
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=performance_window))
        self.risk_scores: Dict[str, float] = {}  # agent_id -> current risk score
        self.last_intervention: Dict[str, int] = {}  # agent_id -> step number

        # Contagion tracking
        self.policy_influence_graph: Dict[str, Set[str]] = defaultdict(set)  # who influences whom
        self.policy_versions: Dict[str, int] = {}  # agent_id -> version
        self.infected_agents: Set[str] = set()  # Agents with high risk

        # Intervention tracking
        self.active_interventions: Dict[str, InterventionType] = {}  # agent_id -> intervention type
        self.intervention_history: List[Dict[str, Any]] = []

        # System state
        self.current_step = 0
        self.system_risk_history = deque(maxlen=100)
        self.contagion_detected = False
        self.contagion_detection_step: Optional[int] = None

        # Statistics
        self.total_assessments = 0
        self.total_interventions = 0
        self.high_risk_count = 0

        logger.info("SimpleHAVEN initialized (coordinator: %s)", coordinator_id)

    # =========================================================================
    # Agent Risk Assessment
    # =========================================================================

    def assess_agent_risk(
        self,
        agent_id: str,
        policy_state: Dict[str, Any],
        recent_performance: List[float],
        behavioral_metrics: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        Assess risk level of a specific agent.

        Risk factors:
        1. Performance degradation
        2. Extreme policy values
        3. Behavioral anomalies
        4. Policy sharing patterns

        Returns:
            RiskAssessment with risk level and score
        """
        self.total_assessments += 1

        # Initialize risk components
        risk_components = {}

        # 1. Performance Risk
        performance_risk = self._assess_performance_risk(agent_id, recent_performance)
        risk_components["performance"] = performance_risk

        # 2. Policy Value Risk
        policy_risk = self._assess_policy_risk(policy_state)
        risk_components["policy_values"] = policy_risk

        # 3. Behavioral Risk
        behavioral_risk = self._assess_behavioral_risk(agent_id, behavioral_metrics or {})
        risk_components["behavioral"] = behavioral_risk

        # 4. Contagion Risk
        contagion_risk = self._assess_contagion_risk(agent_id)
        risk_components["contagion"] = contagion_risk

        # Compute overall risk score (weighted average)
        overall_risk = (
            0.4 * performance_risk +
            0.3 * policy_risk +
            0.2 * behavioral_risk +
            0.1 * contagion_risk
        )

        # Determine risk level
        risk_level = self._risk_score_to_level(overall_risk)

        # Store risk score
        self.risk_scores[agent_id] = overall_risk

        # Track high-risk agents
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            self.infected_agents.add(agent_id)
            self.high_risk_count += 1
        else:
            self.infected_agents.discard(agent_id)

        # Update performance history
        if recent_performance:
            self.performance_history[agent_id].extend(recent_performance)

        # Log to SQLite
        if self.sql_logger:
            self.sql_logger.log_risk_event(
                agent_id=agent_id,
                risk_level=risk_level.value,
                risk_score=overall_risk,
                contributing_factors=risk_components,
                intervention=self.active_interventions.get(agent_id)
            )

        return RiskAssessment(
            agent_id=agent_id,
            risk_level=risk_level,
            risk_score=overall_risk,
            risk_components=risk_components,
            timestamp=time.time(),
            recommended_intervention=self._recommend_intervention(risk_level)
        )

    def _assess_performance_risk(self, agent_id: str, recent_performance: List[float]) -> float:
        """Assess risk based on performance degradation."""
        if not recent_performance:
            return 0.5  # Neutral risk if no data

        recent_perf = np.array(recent_performance[-self.performance_window:])

        # Check for declining performance
        if len(recent_perf) >= 5:
            # Linear regression to detect trend
            x = np.arange(len(recent_perf))
            slope, _ = np.polyfit(x, recent_perf, 1)

            # Negative slope indicates declining performance
            if slope < -0.01:
                decline_risk = min(1.0, abs(slope) * 10)
            else:
                decline_risk = 0.0
        else:
            decline_risk = 0.0

        # Check for low absolute performance
        avg_performance = np.mean(recent_perf)
        low_performance_risk = max(0.0, (0.5 - avg_performance) * 2)  # High risk if perf < 0.5

        # Combined risk
        performance_risk = 0.6 * decline_risk + 0.4 * low_performance_risk

        return float(np.clip(performance_risk, 0.0, 1.0))

    def _assess_policy_risk(self, policy_state: Dict[str, Any]) -> float:
        """Assess risk based on policy parameter values."""
        risk_factors = []

        for key, value in policy_state.items():
            if isinstance(value, (int, float)):
                # Check for extreme values
                if abs(value) > 100:
                    risk_factors.append(0.8)
                elif abs(value) > 10:
                    risk_factors.append(0.5)
                else:
                    risk_factors.append(0.1)

            elif isinstance(value, (list, tuple, np.ndarray)):
                arr = np.array(value)
                # Check for NaN or Inf
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    risk_factors.append(1.0)
                # Check for extreme values
                elif np.any(np.abs(arr) > 100):
                    risk_factors.append(0.8)
                else:
                    risk_factors.append(0.1)

        if not risk_factors:
            return 0.3  # Moderate risk if no policy data

        return float(np.clip(np.mean(risk_factors), 0.0, 1.0))

    def _assess_behavioral_risk(self, agent_id: str, behavioral_metrics: Dict[str, Any]) -> float:
        """Assess risk based on behavioral patterns."""
        risk_score = 0.0

        # Check policy update frequency
        update_freq = behavioral_metrics.get("update_frequency", 1.0)
        if update_freq > 10:  # Too frequent updates
            risk_score += 0.3
        elif update_freq < 0.1:  # Too infrequent
            risk_score += 0.2

        # Check exploration rate
        exploration = behavioral_metrics.get("exploration_rate", 0.1)
        if exploration > 0.5:  # Too random
            risk_score += 0.3

        # Check sharing behavior
        share_rate = behavioral_metrics.get("share_rate", 0.1)
        if share_rate > 0.8:  # Over-sharing (potential poisoning)
            risk_score += 0.4

        return float(np.clip(risk_score, 0.0, 1.0))

    def _assess_contagion_risk(self, agent_id: str) -> float:
        """Assess risk based on contagion graph position."""
        # How many high-risk peers does this agent connect to?
        influenced_by = self.policy_influence_graph.get(agent_id, set())

        high_risk_neighbors = sum(
            1 for peer_id in influenced_by
            if self.risk_scores.get(peer_id, 0.0) > self.risk_threshold
        )

        if not influenced_by:
            return 0.0

        contagion_risk = high_risk_neighbors / len(influenced_by)

        return float(contagion_risk)

    def _risk_score_to_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to risk level enum."""
        if risk_score >= 0.9:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.7:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        elif risk_score >= 0.15:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _recommend_intervention(self, risk_level: RiskLevel) -> Optional[InterventionType]:
        """Recommend intervention based on risk level."""
        if risk_level == RiskLevel.CRITICAL:
            return InterventionType.EMERGENCY_STOP
        elif risk_level == RiskLevel.HIGH:
            return InterventionType.ISOLATION
        elif risk_level == RiskLevel.MEDIUM:
            return InterventionType.FREEZE_POLICY
        else:
            return None

    # =========================================================================
    # System Risk Assessment
    # =========================================================================

    def assess_system_risk(self) -> float:
        """
        Assess overall system risk level.

        Returns:
            System-wide risk score (0.0 to 1.0)
        """
        if not self.risk_scores:
            return 0.0

        # Average of all agent risks
        avg_risk = np.mean(list(self.risk_scores.values()))

        # Weight by proportion of high-risk agents
        high_risk_proportion = len(self.infected_agents) / len(self.risk_scores)

        # System risk
        system_risk = 0.7 * avg_risk + 0.3 * high_risk_proportion

        self.system_risk_history.append(system_risk)

        return float(system_risk)

    # =========================================================================
    # Contagion Detection
    # =========================================================================

    def detect_policy_contagion(
        self,
        time_window: Optional[int] = None
    ) -> ContagionReport:
        """
        Detect if toxic policies are spreading through the network.

        Contagion indicators:
        1. Increasing number of high-risk agents
        2. Graph connectivity between high-risk agents
        3. Rapid spread pattern

        Returns:
            ContagionReport with detection status
        """
        time_window = time_window or self.performance_window

        # Count infected agents
        num_infected = len(self.infected_agents)
        num_monitored = len(self.monitored_agents)

        if num_monitored == 0:
            return ContagionReport(
                contagion_detected=False,
                contagion_status=ContagionStatus.NO_CONTAGION,
                infection_rate=0.0,
                infected_agents=[],
                contagion_source=[],
                spread_pattern={},
                containment_actions=[],
                timestamp=time.time()
            )

        infection_rate = num_infected / num_monitored

        # Detect contagion based on infection rate
        if infection_rate > self.contagion_threshold:
            contagion_status = ContagionStatus.ACTIVE_CONTAGION
            contagion_detected = True

            if not self.contagion_detected:
                self.contagion_detected = True
                self.contagion_detection_step = self.current_step
                logger.warning("CONTAGION DETECTED at step %d (infection rate: %.2f%%)",
                             self.current_step, infection_rate * 100)

        elif infection_rate > self.contagion_threshold * 0.5:
            contagion_status = ContagionStatus.EARLY_DETECTION
            contagion_detected = True
        else:
            contagion_status = ContagionStatus.NO_CONTAGION
            contagion_detected = False

        # Identify contagion source (highest risk agents)
        contagion_sources = sorted(
            self.infected_agents,
            key=lambda a: self.risk_scores.get(a, 0.0),
            reverse=True
        )[:3]

        # Analyze spread pattern
        spread_pattern = self._analyze_spread_pattern()

        # Determine containment actions
        containment_actions = self._plan_containment(contagion_status, list(self.infected_agents))

        return ContagionReport(
            contagion_detected=contagion_detected,
            contagion_status=contagion_status,
            infection_rate=infection_rate,
            infected_agents=list(self.infected_agents),
            contagion_source=contagion_sources,
            spread_pattern=spread_pattern,
            containment_actions=containment_actions,
            timestamp=time.time()
        )

    def _analyze_spread_pattern(self) -> Dict[str, Any]:
        """Analyze how contagion is spreading through the network."""
        # Compute graph connectivity among infected agents
        infected_edges = 0
        for agent_id in self.infected_agents:
            influences = self.policy_influence_graph.get(agent_id, set())
            infected_neighbors = influences.intersection(self.infected_agents)
            infected_edges += len(infected_neighbors)

        return {
            "infected_count": len(self.infected_agents),
            "infected_edges": infected_edges,
            "connectivity": infected_edges / max(1, len(self.infected_agents)),
            "detection_step": self.contagion_detection_step
        }

    def _plan_containment(
        self,
        contagion_status: ContagionStatus,
        infected_agents: List[str]
    ) -> List[str]:
        """Plan containment actions based on contagion status."""
        actions = []

        if contagion_status == ContagionStatus.ACTIVE_CONTAGION:
            actions.append("ISOLATE_ALL_INFECTED")
            actions.append("FREEZE_POLICIES")
            actions.append("ROLLBACK_TO_SAFE_STATE")

        elif contagion_status == ContagionStatus.EARLY_DETECTION:
            actions.append("MONITOR_CLOSELY")
            actions.append("FREEZE_HIGH_RISK_POLICIES")

        return actions

    # =========================================================================
    # Interventions
    # =========================================================================

    def execute_intervention(
        self,
        agent_id: str,
        intervention: InterventionType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Execute intervention on a specific agent.

        Args:
            agent_id: Target agent
            intervention: Type of intervention
            metadata: Additional context

        Returns:
            True if intervention successful
        """
        # Check cooldown
        last_intervention_step = self.last_intervention.get(agent_id, -1000)
        if self.current_step - last_intervention_step < self.intervention_cooldown:
            logger.debug("Intervention cooldown active for %s", agent_id)
            return False

        try:
            # Execute based on intervention type
            if intervention == InterventionType.MONITORING:
                success = self._intervention_monitoring(agent_id, metadata)

            elif intervention == InterventionType.FREEZE_POLICY:
                success = self._intervention_freeze_policy(agent_id, metadata)

            elif intervention == InterventionType.ISOLATION:
                success = self._intervention_isolation(agent_id, metadata)

            elif intervention == InterventionType.ROLLBACK_POLICY:
                success = self._intervention_rollback(agent_id, metadata)

            elif intervention == InterventionType.POLICY_RESET:
                success = self._intervention_reset(agent_id, metadata)

            elif intervention == InterventionType.EMERGENCY_STOP:
                success = self._intervention_emergency_stop(agent_id, metadata)

            else:
                logger.warning("Unknown intervention type: %s", intervention)
                success = False

            if success:
                self.active_interventions[agent_id] = intervention
                self.last_intervention[agent_id] = self.current_step
                self.total_interventions += 1

                # Log intervention
                self.intervention_history.append({
                    "agent_id": agent_id,
                    "intervention": intervention.value,
                    "step": self.current_step,
                    "metadata": metadata or {},
                    "success": success
                })

                if self.sql_logger:
                    self.sql_logger.log_system_event(
                        event_type="intervention",
                        severity="WARNING",
                        description=f"Intervention {intervention.value} on {agent_id}",
                        data={"agent_id": agent_id, "intervention": intervention.value}
                    )

                logger.info("Intervention %s executed on %s", intervention.value, agent_id)

            return success

        except Exception as e:
            logger.error("Intervention failed on %s: %s", agent_id, e)
            return False

    def _intervention_monitoring(self, agent_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Increase monitoring frequency for agent."""
        # Set flag in Redis for agent to see
        self.redis_client.set_key_value(
            f"haven:monitoring:{agent_id}",
            {"enabled": True, "frequency": "high"},
            ex=3600
        )
        return True

    def _intervention_freeze_policy(self, agent_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Freeze agent's policy (no updates allowed)."""
        self.redis_client.set_key_value(
            f"haven:freeze:{agent_id}",
            {"frozen": True, "timestamp": time.time()},
            ex=3600
        )
        return True

    def _intervention_isolation(self, agent_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Isolate agent from policy sharing network."""
        # Remove from influence graph
        self.policy_influence_graph.pop(agent_id, None)

        # Notify agent
        self.redis_client.publish(
            f"haven:intervention:{agent_id}",
            {"type": "isolation", "reason": "high_risk"}
        )
        return True

    def _intervention_rollback(self, agent_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Rollback agent to previous safe policy."""
        # Signal agent to rollback
        self.redis_client.publish(
            f"haven:intervention:{agent_id}",
            {"type": "rollback", "target_version": metadata.get("version", -1) if metadata else -1}
        )
        return True

    def _intervention_reset(self, agent_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Reset agent policy to initial state."""
        self.redis_client.publish(
            f"haven:intervention:{agent_id}",
            {"type": "reset"}
        )
        return True

    def _intervention_emergency_stop(self, agent_id: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Emergency stop for agent (halt all operations)."""
        self.redis_client.set_key_value(
            f"haven:emergency_stop:{agent_id}",
            {"stopped": True, "timestamp": time.time()},
            ex=3600
        )

        self.redis_client.publish(
            f"haven:intervention:{agent_id}",
            {"type": "emergency_stop"}
        )

        logger.critical("EMERGENCY STOP executed on %s", agent_id)
        return True

    # =========================================================================
    # Agent Management
    # =========================================================================

    def register_agent(
        self,
        agent_id: str,
        agent_metadata: Dict[str, Any]
    ):
        """Register an agent for monitoring."""
        self.monitored_agents[agent_id] = agent_metadata
        self.risk_scores[agent_id] = 0.0
        logger.info("Agent %s registered with HAVEN", agent_id)

    def deregister_agent(self, agent_id: str):
        """Remove agent from monitoring."""
        self.monitored_agents.pop(agent_id, None)
        self.risk_scores.pop(agent_id, None)
        self.infected_agents.discard(agent_id)
        self.active_interventions.pop(agent_id, None)
        logger.info("Agent %s deregistered from HAVEN", agent_id)

    def record_policy_influence(
        self,
        source_agent_id: str,
        target_agent_id: str
    ):
        """Record that source agent influenced target agent's policy."""
        self.policy_influence_graph[target_agent_id].add(source_agent_id)

    # =========================================================================
    # Statistics and Reporting
    # =========================================================================

    def get_agent_health_report(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive health report for agent."""
        return {
            "agent_id": agent_id,
            "risk_score": self.risk_scores.get(agent_id, 0.0),
            "risk_level": self._risk_score_to_level(self.risk_scores.get(agent_id, 0.0)).value,
            "is_infected": agent_id in self.infected_agents,
            "active_intervention": self.active_interventions.get(agent_id),
            "performance_history": list(self.performance_history.get(agent_id, [])),
            "influenced_by": list(self.policy_influence_graph.get(agent_id, set())),
            "last_intervention_step": self.last_intervention.get(agent_id)
        }

    def get_system_health_report(self) -> Dict[str, Any]:
        """Get system-wide health report."""
        system_risk = self.assess_system_risk()

        return {
            "current_step": self.current_step,
            "system_risk": system_risk,
            "monitored_agents": len(self.monitored_agents),
            "infected_agents": len(self.infected_agents),
            "infection_rate": len(self.infected_agents) / max(1, len(self.monitored_agents)),
            "contagion_detected": self.contagion_detected,
            "active_interventions": len(self.active_interventions),
            "total_interventions": self.total_interventions,
            "total_assessments": self.total_assessments,
            "avg_risk_score": np.mean(list(self.risk_scores.values())) if self.risk_scores else 0.0
        }

    def step(self):
        """Advance coordinator by one time step."""
        self.current_step += 1

        # Auto-intervention if enabled
        if self.auto_intervention:
            for agent_id in list(self.infected_agents):
                risk_level = self._risk_score_to_level(self.risk_scores.get(agent_id, 0.0))
                recommended = self._recommend_intervention(risk_level)

                if recommended and agent_id not in self.active_interventions:
                    self.execute_intervention(agent_id, recommended)


# =========================================================================
# Factory Function
# =========================================================================

def create_haven_coordinator(
    coordinator_id: str,
    redis_client: RedisClient,
    sql_logger: Optional[SQLiteLogger] = None,
    config: Optional[Dict[str, Any]] = None
) -> SimpleHAVEN:
    """
    Factory function to create SimpleHAVEN coordinator.

    Args:
        coordinator_id: Coordinator identifier
        redis_client: Redis client instance
        sql_logger: SQLite logger (optional)
        config: Optional configuration dictionary

    Returns:
        SimpleHAVEN coordinator instance
    """
    config = config or {}

    return SimpleHAVEN(
        coordinator_id=coordinator_id,
        redis_client=redis_client,
        sql_logger=sql_logger,
        risk_threshold=config.get("risk_threshold", 0.7),
        contagion_threshold=config.get("contagion_threshold", 0.3),
        performance_window=config.get("performance_window", 50),
        intervention_cooldown=config.get("intervention_cooldown", 10),
        auto_intervention=config.get("auto_intervention", True)
    )
