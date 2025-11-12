"""
Adversarial Model for Safety Testing and Risk Validation

This model is separate from the main production model and is designed to:
- Test the robustness of the HAVEN risk management system
- Validate policy contagion detection mechanisms
- Assess the resilience of federated learning against toxic agents
- Generate safety reports and metrics
"""

from mesa import Model
from mesa.time import RandomActivation
from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class AdversarialModel(Model):
    """
    Specialized Mesa model for adversarial testing and safety validation.

    This model creates a controlled environment to test:
    - Policy contagion detection and containment
    - Risk assessment accuracy
    - Intervention effectiveness
    - System resilience to malicious/buggy agents
    - Recovery mechanisms

    Unlike the production model, this model:
    - Has deterministic seeding for reproducibility
    - Includes safety monitoring hooks
    - Generates detailed test reports
    - Supports scenario injection
    """

    def __init__(
        self,
        redis_client,
        num_healthy_agents: int = 10,
        num_toxic_agents: int = 2,
        test_scenario: str = "policy_contagion",
        haven_coordinator=None,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize the Adversarial Testing Model.

        Args:
            redis_client: Redis client for data operations
            num_healthy_agents: Number of normal specialist agents
            num_toxic_agents: Number of toxic agents to inject
            test_scenario: Type of adversarial scenario to test
            haven_coordinator: Optional HAVEN coordinator instance
            random_seed: Random seed for reproducibility
        """
        super().__init__()

        # Set random seed for reproducibility
        if random_seed is not None:
            self.random.seed(random_seed)

        self.redis_client = redis_client
        self.haven_coordinator = haven_coordinator

        # Test configuration
        self.num_healthy_agents = num_healthy_agents
        self.num_toxic_agents = num_toxic_agents
        self.test_scenario = test_scenario

        # Initialize scheduler
        self.schedule = RandomActivation(self)

        # Agent tracking
        self.healthy_agents: List[Any] = []
        self.toxic_agents: List[Any] = []
        self.risk_manager = None
        self.data_miner = None

        # Simulation metrics
        self.current_step = 0
        self.test_start_step = 0

        # Safety metrics tracking
        self.contagion_detected: bool = False
        self.contagion_detection_step: Optional[int] = None
        self.infected_agents: List[str] = []
        self.interventions_triggered: int = 0
        self.system_risk_history: List[float] = []
        self.agent_performance_history: Dict[str, List[float]] = {}

        # Test results
        self.test_passed: bool = False
        self.test_results: Dict[str, Any] = {}

        logger.info("AdversarialModel initialized: scenario=%s, healthy=%d, toxic=%d",
                   test_scenario, num_healthy_agents, num_toxic_agents)

    def setup(self):
        """
        Set up the adversarial testing environment.

        This method should be called after agents have been added to the model.
        """
        # Initialize tracking for all agents
        for agent in self.schedule.agents:
            agent_id = agent.agent_id
            self.agent_performance_history[agent_id] = []

            # Register with HAVEN if available
            if self.haven_coordinator and hasattr(agent, 'agent_type'):
                if agent.agent_type == "RiskManagerAgent":
                    self.risk_manager = agent
                elif agent.agent_type != "DataMinerAgent":
                    self.haven_coordinator.register_agent(agent_id)

        self.test_start_step = self.current_step

        logger.info("AdversarialModel setup complete with %d agents", len(self.schedule.agents))

    def step(self):
        """
        Execute one step of the adversarial simulation.

        Extends the base step to include safety monitoring and metrics collection.
        """
        self.current_step += 1

        # Run all agents
        self.schedule.step()

        # Collect metrics
        self._collect_step_metrics()

        # Check for contagion
        if not self.contagion_detected:
            self._check_contagion_spread()

        # Assess system state
        self._assess_system_state()

        # Log progress periodically
        if self.current_step % 100 == 0:
            logger.info("Adversarial simulation step %d complete (contagion: %s)",
                       self.current_step, self.contagion_detected)

    def _collect_step_metrics(self):
        """
        Collect metrics for this simulation step.
        """
        # Track individual agent performance
        for agent in self.schedule.agents:
            if hasattr(agent, 'agent_id') and hasattr(agent, 'last_reward'):
                agent_id = agent.agent_id
                if agent_id in self.agent_performance_history:
                    self.agent_performance_history[agent_id].append(agent.last_reward)

        # Track system-wide risk if risk manager exists
        if self.risk_manager:
            self.system_risk_history.append(self.risk_manager.risk_score)

            # Track interventions
            if hasattr(self.risk_manager, 'interventions_executed'):
                current_interventions = self.risk_manager.interventions_executed
                if current_interventions > self.interventions_triggered:
                    self.interventions_triggered = current_interventions

    def _check_contagion_spread(self):
        """
        Check if toxic policy has spread to healthy agents.
        """
        # Get toxic agent IDs
        toxic_ids = {agent.agent_id for agent in self.toxic_agents}

        # Check healthy agents for performance degradation
        infected_count = 0
        for agent in self.healthy_agents:
            if self._is_agent_infected(agent):
                if agent.agent_id not in self.infected_agents:
                    self.infected_agents.append(agent.agent_id)
                    logger.warning("Contagion detected: agent %s shows toxic behavior", agent.agent_id)
                infected_count += 1

        # Mark contagion detected if threshold reached
        if infected_count > 0 and not self.contagion_detected:
            self.contagion_detected = True
            self.contagion_detection_step = self.current_step
            logger.warning("CONTAGION DETECTED at step %d (%d agents infected)",
                          self.current_step, infected_count)

    def _is_agent_infected(self, agent) -> bool:
        """
        Determine if a healthy agent has been infected by toxic policy.

        Args:
            agent: Agent to check

        Returns:
            True if agent shows signs of toxic behavior
        """
        agent_id = agent.agent_id

        # Need sufficient history to assess
        if agent_id not in self.agent_performance_history:
            return False

        history = self.agent_performance_history[agent_id]
        if len(history) < 20:
            return False

        # Check for sudden performance degradation
        recent_avg = sum(history[-10:]) / 10
        baseline_avg = sum(history[5:15]) / 10 if len(history) > 15 else recent_avg

        # Infected if performance dropped significantly
        if baseline_avg > 0 and recent_avg < baseline_avg * 0.5:
            return True

        # Check for high risk score
        if hasattr(agent, 'risk_score') and agent.risk_score > 0.7:
            return True

        # Check if agent is isolated (detected by risk manager)
        if hasattr(agent, 'is_isolated') and agent.is_isolated:
            return True

        return False

    def _assess_system_state(self):
        """
        Assess the overall state of the system.
        """
        # Calculate infection rate
        infection_rate = len(self.infected_agents) / max(1, len(self.healthy_agents))

        # Check system risk
        system_risk = self.system_risk_history[-1] if self.system_risk_history else 0.0

        # Determine if test is passing
        # Test passes if contagion is detected AND contained (low infection rate)
        if self.contagion_detected:
            # If interventions were triggered and infection rate is low, system is working
            if self.interventions_triggered > 0 and infection_rate < 0.3:
                self.test_passed = True

    def run(self, num_steps: int = 1000) -> Dict[str, Any]:
        """
        Run the adversarial simulation for a specified number of steps.

        Args:
            num_steps: Number of simulation steps to execute

        Returns:
            Dictionary containing test results and metrics
        """
        logger.info("Starting adversarial simulation for %d steps", num_steps)

        for step in range(num_steps):
            self.step()

        # Generate final report
        self.test_results = self._generate_test_report()

        logger.info("Adversarial simulation completed: %s",
                   "PASSED" if self.test_passed else "FAILED")

        return self.test_results

    def _generate_test_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive test report.

        Returns:
            Dictionary with detailed test results
        """
        # Calculate metrics
        infection_rate = len(self.infected_agents) / max(1, len(self.healthy_agents))
        containment_rate = 1.0 - infection_rate if self.contagion_detected else 1.0

        # Assess response time
        response_time = None
        if self.contagion_detected and self.interventions_triggered > 0:
            # Response time is when interventions started after contagion
            response_time = self.contagion_detection_step

        # Performance degradation analysis
        performance_degradation = self._calculate_performance_degradation()

        # Generate report
        report = {
            "test_scenario": self.test_scenario,
            "test_passed": self.test_passed,
            "simulation_steps": self.current_step,

            # Agent composition
            "num_healthy_agents": self.num_healthy_agents,
            "num_toxic_agents": self.num_toxic_agents,

            # Contagion metrics
            "contagion_detected": self.contagion_detected,
            "contagion_detection_step": self.contagion_detection_step,
            "infected_agents": self.infected_agents,
            "infection_rate": infection_rate,
            "containment_rate": containment_rate,

            # System response
            "interventions_triggered": self.interventions_triggered,
            "response_time": response_time,
            "final_system_risk": self.system_risk_history[-1] if self.system_risk_history else 0.0,
            "max_system_risk": max(self.system_risk_history) if self.system_risk_history else 0.0,

            # Performance analysis
            "performance_degradation": performance_degradation,
            "system_recovered": self._check_system_recovery(),

            # Detailed metrics
            "agent_performance_summary": self._summarize_agent_performance(),
            "risk_timeline": self._generate_risk_timeline(),
        }

        # Add scenario-specific metrics
        if self.test_scenario == "policy_contagion":
            report["contagion_spread_rate"] = self._calculate_spread_rate()
            report["containment_effectiveness"] = containment_rate

        return report

    def _calculate_performance_degradation(self) -> Dict[str, float]:
        """
        Calculate performance degradation metrics.

        Returns:
            Dictionary with degradation statistics
        """
        total_degradation = 0.0
        degraded_count = 0

        for agent in self.healthy_agents:
            agent_id = agent.agent_id
            if agent_id not in self.agent_performance_history:
                continue

            history = self.agent_performance_history[agent_id]
            if len(history) < 20:
                continue

            baseline = sum(history[:10]) / 10
            final = sum(history[-10:]) / 10

            if baseline > 0:
                degradation = (baseline - final) / baseline
                if degradation > 0.2:  # More than 20% degradation
                    total_degradation += degradation
                    degraded_count += 1

        return {
            "average_degradation": total_degradation / max(1, degraded_count),
            "agents_degraded": degraded_count,
            "degradation_rate": degraded_count / max(1, len(self.healthy_agents))
        }

    def _check_system_recovery(self) -> bool:
        """
        Check if the system recovered from the attack.

        Returns:
            True if system performance recovered
        """
        if not self.system_risk_history or len(self.system_risk_history) < 100:
            return False

        # Check if risk decreased in final quarter
        final_quarter_start = len(self.system_risk_history) * 3 // 4
        max_risk = max(self.system_risk_history)
        final_risk = self.system_risk_history[-1]

        # Recovered if risk dropped significantly from peak
        return final_risk < max_risk * 0.5

    def _summarize_agent_performance(self) -> Dict[str, Any]:
        """
        Summarize performance for each agent category.

        Returns:
            Dictionary with performance summaries
        """
        def get_avg_performance(agents):
            total = 0.0
            count = 0
            for agent in agents:
                if agent.agent_id in self.agent_performance_history:
                    history = self.agent_performance_history[agent.agent_id]
                    if history:
                        total += sum(history) / len(history)
                        count += 1
            return total / max(1, count)

        return {
            "healthy_agents_avg": get_avg_performance(self.healthy_agents),
            "toxic_agents_avg": get_avg_performance(self.toxic_agents),
            "overall_avg": get_avg_performance(self.healthy_agents + self.toxic_agents)
        }

    def _generate_risk_timeline(self) -> List[Dict[str, Any]]:
        """
        Generate a timeline of key risk events.

        Returns:
            List of risk event dictionaries
        """
        timeline = []

        # Add contagion detection event
        if self.contagion_detected:
            timeline.append({
                "step": self.contagion_detection_step,
                "event": "contagion_detected",
                "details": f"{len(self.infected_agents)} agents infected"
            })

        # Add intervention milestones
        if self.interventions_triggered > 0:
            timeline.append({
                "step": self.contagion_detection_step or 0,
                "event": "interventions_started",
                "details": f"{self.interventions_triggered} total interventions"
            })

        # Add peak risk event
        if self.system_risk_history:
            peak_risk = max(self.system_risk_history)
            peak_step = self.system_risk_history.index(peak_risk)
            timeline.append({
                "step": peak_step,
                "event": "peak_risk",
                "details": f"Risk score: {peak_risk:.3f}"
            })

        return timeline

    def _calculate_spread_rate(self) -> float:
        """
        Calculate the rate at which contagion spread.

        Returns:
            Spread rate (agents infected per step)
        """
        if not self.contagion_detected or not self.contagion_detection_step:
            return 0.0

        steps_elapsed = self.current_step - self.contagion_detection_step
        if steps_elapsed == 0:
            return 0.0

        return len(self.infected_agents) / steps_elapsed

    def export_results(self, filepath: str):
        """
        Export test results to a JSON file.

        Args:
            filepath: Path to save the results file
        """
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)

        logger.info("Test results exported to %s", filepath)

    def add_healthy_agent(self, agent):
        """
        Add a healthy agent to the simulation.

        Args:
            agent: Agent instance to add
        """
        self.schedule.add(agent)
        self.healthy_agents.append(agent)
        logger.debug("Added healthy agent: %s", agent.agent_id)

    def add_toxic_agent(self, agent):
        """
        Add a toxic agent to the simulation.

        Args:
            agent: Toxic agent instance to add
        """
        self.schedule.add(agent)
        self.toxic_agents.append(agent)
        logger.warning("Added toxic agent: %s", agent.agent_id)

    def get_summary(self) -> str:
        """
        Get a human-readable summary of the test.

        Returns:
            Summary string
        """
        if not self.test_results:
            return "Test not yet completed"

        results = self.test_results

        summary = f"""
=== Adversarial Simulation Report ===
Scenario: {results['test_scenario']}
Result: {'PASSED ✓' if results['test_passed'] else 'FAILED ✗'}

Agents:
- Healthy: {results['num_healthy_agents']}
- Toxic: {results['num_toxic_agents']}

Contagion Detection:
- Detected: {'Yes' if results['contagion_detected'] else 'No'}
- Step: {results['contagion_detection_step'] or 'N/A'}
- Infection Rate: {results['infection_rate']:.1%}
- Containment Rate: {results['containment_rate']:.1%}

System Response:
- Interventions: {results['interventions_triggered']}
- Response Time: {results['response_time'] or 'N/A'} steps
- Final Risk: {results['final_system_risk']:.3f}
- Max Risk: {results['max_system_risk']:.3f}
- System Recovered: {'Yes' if results['system_recovered'] else 'No'}

Performance:
- Healthy Agents Avg: {results['agent_performance_summary']['healthy_agents_avg']:.3f}
- Toxic Agents Avg: {results['agent_performance_summary']['toxic_agents_avg']:.3f}
"""
        return summary
