"""
Adversarial Simulation Runner

This script runs safety tests by:
1. Launching the adversarial model
2. Injecting toxic agents into a population of healthy specialists
3. Running the simulation for a specified number of steps
4. Generating a comprehensive report on policy contagion and system response
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors.redis_client import RedisClient
from simulation.adversarial_model import AdversarialModel
from simulation.toxic_agent import ToxicAgent, ToxicAgentFactory, ToxicBehaviorType
from agents.specialist_agent import SpecialistAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.data_miner_agent import DataMinerAgent

# Import HAVEN coordinator base (you'll need to implement a concrete version)
# For this demo, we'll create a mock implementation
from core.haven_base import HavenRiskCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockHavenCoordinator(HavenRiskCoordinator):
    """
    Mock implementation of HAVEN coordinator for testing.

    In production, you would implement the full abstract methods.
    This provides basic functionality for the adversarial simulation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assessments = {}

    def assess_agent_risk(self, agent_id, policy_state, recent_performance, behavioral_metrics=None):
        from core.haven_base import RiskAssessment, RiskLevel, InterventionType
        import numpy as np

        # Simple risk assessment based on performance
        if not recent_performance:
            risk_score = 0.5
        else:
            avg_performance = np.mean(recent_performance[-10:]) if len(recent_performance) >= 10 else np.mean(recent_performance)
            # Low performance = high risk
            risk_score = max(0.0, 1.0 - avg_performance) if avg_performance > 0 else 0.8

        # Determine risk level
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
        return assessment

    def assess_system_risk(self):
        if not self.assessments:
            return 0.0
        risk_scores = [a.risk_score for a in self.assessments.values()]
        return sum(risk_scores) / len(risk_scores)

    def detect_policy_contagion(self, time_window=None):
        from core.haven_base import ContagionReport, ContagionStatus, InterventionType

        # Simple contagion detection
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
            source_agents=set(),
            contagion_score=len(high_risk_agents) / max(1, len(self.monitored_agents)),
            spread_rate=0.0,
            containment_actions=[InterventionType.ISOLATION] if status != ContagionStatus.HEALTHY else [],
            timestamp=0.0
        )

    def identify_contagion_source(self, affected_agents):
        return []

    def recommend_intervention(self, risk_assessment):
        return risk_assessment.recommended_intervention

    def execute_intervention(self, agent_id, intervention, metadata=None):
        from core.haven_base import InterventionType
        logger.info(f"Executing {intervention.value} on {agent_id}")
        return True

    def isolate_agent(self, agent_id, reason):
        logger.warning(f"Isolating agent {agent_id}: {reason}")

    def restore_agent(self, agent_id, safe_policy=None):
        return True

    def compute_adversarial_value(self, state, joint_action, adversarial_scenario):
        return 0.0

    def evaluate_policy_robustness(self, agent_id, policy_state, test_scenarios):
        return 0.5

    def track_policy_influence(self, source_agent, target_agent, influence_weight):
        pass

    def get_influence_graph(self):
        return {}

    def compute_risk_propagation(self, source_agent, max_hops=3):
        return {}

    def establish_safety_constraints(self, constraints):
        pass

    def verify_safety_constraints(self, agent_id, policy_state):
        return (True, [])

    def get_system_health_report(self):
        return {
            "system_risk_score": self.assess_system_risk(),
            "monitored_agents": len(self.monitored_agents),
            "high_risk_agents": sum(1 for a in self.assessments.values() if a.risk_score > 0.7)
        }

    def export_risk_analytics(self, time_range=None):
        return {}

    def calibrate_risk_thresholds(self, historical_data):
        pass

    def simulate_intervention_impact(self, intervention, affected_agents):
        return {}


class SimulationRunner:
    """
    Runner for adversarial simulations with comprehensive reporting.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 1,  # Use separate DB for testing
        output_dir: str = "simulation_results"
    ):
        """
        Initialize the simulation runner.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            output_dir: Directory to save simulation results
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Redis client
        self.redis_client: Optional[RedisClient] = None

        # Model and agents
        self.model: Optional[AdversarialModel] = None
        self.haven_coordinator: Optional[HavenRiskCoordinator] = None

        logger.info("SimulationRunner initialized (output: %s)", self.output_dir)

    def setup_redis(self):
        """
        Set up Redis connection.
        """
        try:
            logger.info("Connecting to Redis at %s:%d (db=%d)",
                       self.redis_host, self.redis_port, self.redis_db)

            self.redis_client = RedisClient(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db
            )

            # Clear test database
            logger.warning("Clearing test database...")
            self.redis_client.flushdb()

            logger.info("Redis connection established")

        except Exception as e:
            logger.error("Failed to connect to Redis: %s", e)
            raise

    def setup_model(
        self,
        num_healthy_agents: int = 10,
        num_toxic_agents: int = 2,
        test_scenario: str = "policy_contagion"
    ):
        """
        Set up the adversarial model and HAVEN coordinator.

        Args:
            num_healthy_agents: Number of healthy specialist agents
            num_toxic_agents: Number of toxic agents to inject
            test_scenario: Type of test scenario
        """
        logger.info("Setting up adversarial model...")

        # Create HAVEN coordinator
        self.haven_coordinator = MockHavenCoordinator(
            coordinator_id="haven_test",
            redis_client=self.redis_client,
            risk_threshold=0.7,
            contagion_threshold=0.5,
            intervention_enabled=True
        )

        # Create adversarial model
        self.model = AdversarialModel(
            redis_client=self.redis_client,
            num_healthy_agents=num_healthy_agents,
            num_toxic_agents=num_toxic_agents,
            test_scenario=test_scenario,
            haven_coordinator=self.haven_coordinator,
            random_seed=42
        )

        logger.info("Adversarial model created")

    def create_agents(
        self,
        num_healthy: int = 10,
        num_toxic: int = 2,
        toxic_behavior: str = ToxicBehaviorType.POLICY_POISONING
    ):
        """
        Create and add agents to the model.

        Args:
            num_healthy: Number of healthy specialist agents
            num_toxic: Number of toxic agents
            toxic_behavior: Type of toxic behavior
        """
        logger.info("Creating %d healthy agents and %d toxic agents",
                   num_healthy, num_toxic)

        # Create data miner (provides data to specialists)
        data_miner = DataMinerAgent(
            unique_id=0,
            model=self.model,
            redis_client=self.redis_client,
            source_streams=["test_data_stream"],
            output_channel="processed_data"
        )
        self.model.schedule.add(data_miner)
        self.model.data_miner = data_miner

        # Create healthy specialist agents
        for i in range(num_healthy):
            agent = SpecialistAgent(
                unique_id=i + 1,
                model=self.model,
                redis_client=self.redis_client,
                data_channel="processed_data",
                specialization="normal",
                agent_config={
                    "learning_rate": 0.01,
                    "exploration_rate": 0.1
                }
            )
            self.model.add_healthy_agent(agent)

        # Create toxic agents
        for i in range(num_toxic):
            toxic_agent = ToxicAgent(
                unique_id=num_healthy + i + 1,
                model=self.model,
                redis_client=self.redis_client,
                toxic_behavior=toxic_behavior,
                toxicity_level=0.8,
                data_channel="processed_data",
                agent_config={"evasion_enabled": False}
            )
            self.model.add_toxic_agent(toxic_agent)

        # Create risk manager with HAVEN coordinator
        risk_manager = RiskManagerAgent(
            unique_id=num_healthy + num_toxic + 1,
            model=self.model,
            redis_client=self.redis_client,
            haven_coordinator=self.haven_coordinator,
            agent_config={
                "monitoring_interval": 5,
                "auto_intervention": True
            }
        )
        self.model.schedule.add(risk_manager)
        self.model.risk_manager = risk_manager

        # Register all agents with risk manager
        for agent in self.model.healthy_agents + self.model.toxic_agents:
            risk_manager.register_agent_for_monitoring(
                agent.agent_id,
                agent.get_state_summary()
            )

        # Setup model
        self.model.setup()

        logger.info("Created %d total agents", len(self.model.schedule.agents))

    def run_simulation(self, num_steps: int = 1000) -> Dict[str, Any]:
        """
        Run the adversarial simulation.

        Args:
            num_steps: Number of simulation steps

        Returns:
            Test results dictionary
        """
        logger.info("=" * 60)
        logger.info("Starting adversarial simulation for %d steps", num_steps)
        logger.info("=" * 60)

        # Run the model
        results = self.model.run(num_steps=num_steps)

        logger.info("=" * 60)
        logger.info("Simulation complete")
        logger.info("=" * 60)

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable report from results.

        Args:
            results: Test results dictionary

        Returns:
            Formatted report string
        """
        # Use model's summary method
        summary = self.model.get_summary()

        # Add additional analysis
        additional_info = f"""

=== Detailed Analysis ===

Toxic Agent Behavior:
"""
        for toxic_agent in self.model.toxic_agents:
            stats = toxic_agent.get_toxic_statistics()
            detection = toxic_agent.get_detection_status()

            additional_info += f"""
Agent: {toxic_agent.agent_id}
- Behavior: {stats['toxic_behavior']}
- Toxicity Level: {stats['toxicity_level']:.2f}
- Toxic Actions: {stats['toxic_actions_taken']}
- Policies Poisoned: {stats['policies_poisoned']}
- System Damage: {stats['system_damage_caused']:.3f}
- Detected: {'Yes' if detection['is_detected'] else 'No'}
- Isolated: {'Yes' if detection['is_isolated'] else 'No'}
- Risk Score: {detection['risk_score']:.3f}
"""

        full_report = summary + additional_info

        return full_report

    def save_results(self, results: Dict[str, Any], report: str):
        """
        Save results to files.

        Args:
            results: Test results dictionary
            report: Formatted report string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = self.output_dir / f"results_{timestamp}.json"
        self.model.export_results(str(json_path))
        logger.info("Results saved to %s", json_path)

        # Save text report
        report_path = self.output_dir / f"report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info("Report saved to %s", report_path)

    def cleanup(self):
        """
        Clean up resources.
        """
        if self.redis_client:
            logger.info("Closing Redis connection")
            self.redis_client.close()

    def run_full_test(
        self,
        num_healthy: int = 10,
        num_toxic: int = 2,
        toxic_behavior: str = ToxicBehaviorType.POLICY_POISONING,
        num_steps: int = 1000
    ) -> Dict[str, Any]:
        """
        Run a complete adversarial test.

        Args:
            num_healthy: Number of healthy agents
            num_toxic: Number of toxic agents
            toxic_behavior: Type of toxic behavior
            num_steps: Number of simulation steps

        Returns:
            Test results dictionary
        """
        try:
            # Setup
            self.setup_redis()
            self.setup_model(
                num_healthy_agents=num_healthy,
                num_toxic_agents=num_toxic,
                test_scenario="policy_contagion"
            )
            self.create_agents(
                num_healthy=num_healthy,
                num_toxic=num_toxic,
                toxic_behavior=toxic_behavior
            )

            # Run simulation
            results = self.run_simulation(num_steps=num_steps)

            # Generate and save report
            report = self.generate_report(results)
            print("\n" + report)
            self.save_results(results, report)

            return results

        finally:
            self.cleanup()


def main():
    """
    Main entry point for running adversarial simulations.
    """
    logger.info("Adversarial Simulation Runner")
    logger.info("=" * 60)

    # Create runner
    runner = SimulationRunner(
        redis_host="localhost",
        redis_port=6379,
        redis_db=1,
        output_dir="simulation_results"
    )

    # Run test
    results = runner.run_full_test(
        num_healthy=10,
        num_toxic=2,
        toxic_behavior=ToxicBehaviorType.POLICY_POISONING,
        num_steps=1000
    )

    # Print final result
    if results["test_passed"]:
        logger.info("✓ TEST PASSED - System successfully contained contagion")
        return 0
    else:
        logger.warning("✗ TEST FAILED - Contagion was not properly contained")
        return 1


if __name__ == "__main__":
    sys.exit(main())
