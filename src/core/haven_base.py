"""
HAVEN (Hierarchical Adversarial Value Estimation Network) Framework Abstract Base Class

This module defines the abstract interface for a lightweight coordination layer
that manages system-wide risk and prevents 'policy contagion' in multi-agent
reinforcement learning systems. HAVEN provides adversarial risk assessment and
coordination mechanisms to ensure system stability and safety.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for system and agent policies."""
    MINIMAL = "minimal"  # Risk < 0.2
    LOW = "low"  # Risk 0.2-0.4
    MODERATE = "moderate"  # Risk 0.4-0.6
    HIGH = "high"  # Risk 0.6-0.8
    CRITICAL = "critical"  # Risk > 0.8


class ContagionStatus(Enum):
    """Status of policy contagion in the system."""
    HEALTHY = "healthy"  # No contagion detected
    EARLY_WARNING = "early_warning"  # Early indicators present
    SPREADING = "spreading"  # Contagion actively spreading
    CONTAINED = "contained"  # Contagion detected and isolated
    SYSTEM_WIDE = "system_wide"  # Widespread contagion


class InterventionType(Enum):
    """Types of interventions for risk mitigation."""
    NONE = "none"  # No intervention needed
    MONITORING = "monitoring"  # Increase monitoring frequency
    POLICY_FREEZE = "policy_freeze"  # Freeze policy updates for affected agents
    ISOLATION = "isolation"  # Isolate high-risk agents from network
    ROLLBACK = "rollback"  # Rollback to previous safe policy
    EMERGENCY_STOP = "emergency_stop"  # Stop all learning system-wide


@dataclass
class RiskAssessment:
    """Container for risk assessment results."""
    agent_id: str
    risk_level: RiskLevel
    risk_score: float  # 0.0 to 1.0
    contributing_factors: Dict[str, float]
    recommended_intervention: InterventionType
    timestamp: float
    confidence: float  # Confidence in assessment (0.0 to 1.0)


@dataclass
class ContagionReport:
    """Container for policy contagion analysis."""
    contagion_status: ContagionStatus
    affected_agents: Set[str]
    source_agents: Set[str]  # Suspected sources of bad policy
    contagion_score: float  # 0.0 to 1.0
    spread_rate: float  # Rate of spread per time unit
    containment_actions: List[InterventionType]
    timestamp: float


class HavenRiskCoordinator(ABC):
    """
    Abstract base class for HAVEN Risk Coordination.

    This coordinator acts as a lightweight oversight layer that:
    - Monitors agent policies for adversarial or unstable behavior
    - Detects and prevents policy contagion (bad policies spreading)
    - Coordinates risk mitigation across the multi-agent system
    - Maintains system stability while preserving agent autonomy
    - Provides hierarchical risk assessment at agent and system levels

    Key Principles:
    - Minimal overhead (lightweight coordination)
    - Decentralized monitoring with centralized risk aggregation
    - Adversarial value estimation for robustness
    - Proactive intervention before catastrophic failure
    - Transparency and explainability in risk assessment
    """

    def __init__(
        self,
        coordinator_id: str,
        redis_client: Any,
        risk_threshold: float = 0.7,
        contagion_threshold: float = 0.5,
        intervention_enabled: bool = True
    ):
        """
        Initialize the HAVEN Risk Coordinator.

        Args:
            coordinator_id: Unique identifier for this coordinator
            redis_client: Redis client for system-wide communication
            risk_threshold: Threshold above which interventions are triggered
            contagion_threshold: Threshold for detecting policy contagion
            intervention_enabled: Whether coordinator can intervene or only monitor
        """
        self.coordinator_id = coordinator_id
        self.redis_client = redis_client
        self.risk_threshold = risk_threshold
        self.contagion_threshold = contagion_threshold
        self.intervention_enabled = intervention_enabled

        # Track all agents in the system
        self.monitored_agents: Set[str] = set()

        # Risk assessments for each agent
        self.agent_risk_assessments: Dict[str, RiskAssessment] = {}

        # Historical risk data for trend analysis
        self.risk_history: Dict[str, List[float]] = {}

        # Policy contagion tracking
        self.contagion_graph: Dict[str, Set[str]] = {}  # Agent influence graph

        # Intervention history
        self.intervention_log: List[Dict[str, Any]] = []

        # System-wide metrics
        self.system_risk_score: float = 0.0
        self.active_interventions: Dict[str, InterventionType] = {}

    @abstractmethod
    def assess_agent_risk(
        self,
        agent_id: str,
        policy_state: Dict[str, Any],
        recent_performance: List[float],
        behavioral_metrics: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        Assess the risk level of a specific agent's policy.

        Evaluates:
        - Policy stability (variance, sudden changes)
        - Performance degradation
        - Adversarial behavior indicators
        - Deviation from expected behavior
        - Potential for causing system instability

        Args:
            agent_id: ID of the agent to assess
            policy_state: Current policy parameters/state
            recent_performance: Recent performance history
            behavioral_metrics: Additional behavioral indicators

        Returns:
            RiskAssessment object with detailed risk analysis
        """
        pass

    @abstractmethod
    def assess_system_risk(self) -> float:
        """
        Assess overall system-level risk.

        Aggregates individual agent risks and evaluates system-wide
        stability, performance, and safety metrics.

        Returns:
            System risk score (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def detect_policy_contagion(
        self,
        time_window: Optional[int] = None
    ) -> ContagionReport:
        """
        Detect if bad policies are spreading through the agent network.

        Analyzes:
        - Temporal correlation of performance degradation
        - Policy similarity increases among poorly performing agents
        - Spatial/network patterns of risk escalation
        - Federated learning propagation patterns

        Args:
            time_window: Optional time window for analysis (in time steps)

        Returns:
            ContagionReport with detailed contagion analysis
        """
        pass

    @abstractmethod
    def identify_contagion_source(
        self,
        affected_agents: Set[str]
    ) -> List[Tuple[str, float]]:
        """
        Identify likely source(s) of policy contagion.

        Uses network analysis and temporal tracking to find agents
        that likely introduced bad policies into the system.

        Args:
            affected_agents: Set of agents showing contagion symptoms

        Returns:
            List of (agent_id, confidence) tuples for suspected sources
        """
        pass

    @abstractmethod
    def recommend_intervention(
        self,
        risk_assessment: RiskAssessment
    ) -> InterventionType:
        """
        Recommend intervention strategy based on risk assessment.

        Maps risk levels and patterns to appropriate intervention actions.

        Args:
            risk_assessment: The risk assessment to base recommendation on

        Returns:
            Recommended intervention type
        """
        pass

    @abstractmethod
    def execute_intervention(
        self,
        agent_id: str,
        intervention: InterventionType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Execute an intervention to mitigate risk.

        Args:
            agent_id: Agent to intervene on
            intervention: Type of intervention to execute
            metadata: Additional context for the intervention

        Returns:
            True if intervention was successfully executed
        """
        pass

    @abstractmethod
    def isolate_agent(
        self,
        agent_id: str,
        reason: str
    ):
        """
        Isolate an agent from the federated learning network.

        Prevents the agent from sharing or receiving policy updates
        until it's deemed safe.

        Args:
            agent_id: Agent to isolate
            reason: Explanation for isolation
        """
        pass

    @abstractmethod
    def restore_agent(
        self,
        agent_id: str,
        safe_policy: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Restore an isolated agent to normal operation.

        Args:
            agent_id: Agent to restore
            safe_policy: Optional safe policy to restore agent to

        Returns:
            True if restoration successful
        """
        pass

    @abstractmethod
    def compute_adversarial_value(
        self,
        state: Dict[str, Any],
        joint_action: Dict[str, Any],
        adversarial_scenario: str
    ) -> float:
        """
        Compute value under adversarial conditions.

        Evaluates how policies perform under worst-case scenarios,
        adversarial attacks, or edge cases.

        Args:
            state: Current state
            joint_action: Joint action being evaluated
            adversarial_scenario: Type of adversarial condition to test

        Returns:
            Value estimate under adversarial conditions
        """
        pass

    @abstractmethod
    def evaluate_policy_robustness(
        self,
        agent_id: str,
        policy_state: Dict[str, Any],
        test_scenarios: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate robustness of an agent's policy.

        Tests policy against various scenarios including:
        - Distribution shift
        - Adversarial perturbations
        - Edge cases
        - Novel situations

        Args:
            agent_id: Agent whose policy to evaluate
            policy_state: The policy to evaluate
            test_scenarios: List of test scenarios

        Returns:
            Robustness score (0.0 to 1.0, higher is more robust)
        """
        pass

    @abstractmethod
    def track_policy_influence(
        self,
        source_agent: str,
        target_agent: str,
        influence_weight: float
    ):
        """
        Track policy influence between agents in federated learning.

        Builds a graph of which agents influence which others, enabling
        contagion source tracking.

        Args:
            source_agent: Agent whose policy influenced another
            target_agent: Agent that was influenced
            influence_weight: Strength of influence (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def get_influence_graph(self) -> Dict[str, Set[str]]:
        """
        Get the policy influence graph.

        Returns:
            Dictionary mapping each agent to the set of agents it influences
        """
        pass

    @abstractmethod
    def compute_risk_propagation(
        self,
        source_agent: str,
        max_hops: int = 3
    ) -> Dict[str, float]:
        """
        Compute how risk from a source agent might propagate.

        Simulates risk spreading through the federated learning network.

        Args:
            source_agent: Source of potential risk
            max_hops: Maximum propagation distance to simulate

        Returns:
            Dictionary mapping agent_id to propagated risk score
        """
        pass

    @abstractmethod
    def establish_safety_constraints(
        self,
        constraints: Dict[str, Any]
    ):
        """
        Establish system-wide safety constraints.

        Defines boundaries that agent policies must not violate.

        Args:
            constraints: Dictionary of constraint definitions
        """
        pass

    @abstractmethod
    def verify_safety_constraints(
        self,
        agent_id: str,
        policy_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Verify that a policy satisfies safety constraints.

        Args:
            agent_id: Agent whose policy to verify
            policy_state: Policy to verify

        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        pass

    @abstractmethod
    def get_system_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system health report.

        Returns:
            Dictionary containing:
            - system_risk_score: Overall risk level
            - agent_risk_summary: Statistics on agent risks
            - contagion_status: Current contagion state
            - active_interventions: List of ongoing interventions
            - performance_metrics: System performance indicators
            - recommendations: Suggested actions
        """
        pass

    @abstractmethod
    def export_risk_analytics(
        self,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Export risk analytics data for analysis.

        Args:
            time_range: Optional (start_time, end_time) tuple

        Returns:
            Dictionary with detailed analytics data
        """
        pass

    @abstractmethod
    def calibrate_risk_thresholds(
        self,
        historical_data: Dict[str, Any]
    ):
        """
        Calibrate risk thresholds based on historical data.

        Auto-tunes risk_threshold and contagion_threshold based on
        observed system behavior.

        Args:
            historical_data: Historical performance and risk data
        """
        pass

    @abstractmethod
    def simulate_intervention_impact(
        self,
        intervention: InterventionType,
        affected_agents: Set[str]
    ) -> Dict[str, Any]:
        """
        Simulate the impact of an intervention before executing it.

        Args:
            intervention: Type of intervention to simulate
            affected_agents: Agents that would be affected

        Returns:
            Dictionary with projected outcomes:
            - risk_reduction: Expected reduction in risk
            - performance_impact: Impact on system performance
            - side_effects: Potential negative consequences
            - confidence: Confidence in predictions
        """
        pass

    def register_agent(self, agent_id: str):
        """Register an agent for monitoring."""
        self.monitored_agents.add(agent_id)
        self.risk_history[agent_id] = []
        self.contagion_graph[agent_id] = set()
        logger.info("HAVEN coordinator registered agent %s", agent_id)

    def unregister_agent(self, agent_id: str):
        """Unregister an agent from monitoring."""
        self.monitored_agents.discard(agent_id)
        self.agent_risk_assessments.pop(agent_id, None)
        self.risk_history.pop(agent_id, None)
        self.contagion_graph.pop(agent_id, None)
        logger.info("HAVEN coordinator unregistered agent %s", agent_id)

    def get_monitored_agent_count(self) -> int:
        """Get the number of monitored agents."""
        return len(self.monitored_agents)

    def get_high_risk_agents(self) -> List[str]:
        """Get list of agents with high or critical risk levels."""
        high_risk = []
        for agent_id, assessment in self.agent_risk_assessments.items():
            if assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                high_risk.append(agent_id)
        return high_risk

    def log_intervention(
        self,
        agent_id: str,
        intervention: InterventionType,
        reason: str,
        outcome: str
    ):
        """Log an intervention for audit trail."""
        self.intervention_log.append({
            "agent_id": agent_id,
            "intervention": intervention.value,
            "reason": reason,
            "outcome": outcome,
            "timestamp": logger.time()  # Use appropriate timestamp
        })
        logger.info("Logged intervention: %s on agent %s", intervention.value, agent_id)

    def is_intervention_active(self, agent_id: str) -> bool:
        """Check if an intervention is currently active for an agent."""
        return agent_id in self.active_interventions

    def get_agent_intervention(self, agent_id: str) -> Optional[InterventionType]:
        """Get the current intervention for an agent, if any."""
        return self.active_interventions.get(agent_id)
