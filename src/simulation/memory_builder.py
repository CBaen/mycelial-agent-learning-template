"""
Memory Builder - Pre-Training & Knowledge Base Generator

This module creates the initial knowledge base for agents by:
1. Running adversarial simulations against historical/synthetic data
2. Generating successful and failed pattern examples
3. Storing pattern embeddings in the Vector DB
4. Creating a rich initial memory for agent learning

This is the "self-healing testbed" that enables agents to start with
domain knowledge instead of learning from scratch.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import json
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectors.redis_client import RedisClient
from connectors.vector_db import VectorDBInterface, create_vector_db
from connectors.sql_logger import SQLiteLogger
from simulation.adversarial_model import AdversarialModel
from simulation.toxic_agent import ToxicAgent, ToxicBehaviorType
from agents.specialist_agent import SpecialistAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.data_miner_agent import DataMinerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PatternType:
    """Types of patterns to generate and store."""
    SUCCESSFUL_POLICY = "successful_policy"
    FAILED_POLICY = "failed_policy"
    TOXIC_DETECTION = "toxic_detection"
    CONTAGION_PATTERN = "contagion_pattern"
    RECOVERY_PATTERN = "recovery_pattern"
    OPTIMAL_COORDINATION = "optimal_coordination"
    RISK_MITIGATION = "risk_mitigation"


class MemoryBuilder:
    """
    Memory Builder for creating initial knowledge base from simulations.

    This class runs multiple adversarial simulations with varying parameters
    to generate a diverse set of successful and failed patterns, which are
    then stored as embeddings in the Vector DB for agent pre-training.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        vector_db: VectorDBInterface,
        sql_logger: SQLiteLogger,
        output_dir: str = "memory_building_results"
    ):
        """
        Initialize the Memory Builder.

        Args:
            redis_client: Redis client for data operations
            vector_db: Vector database for pattern storage
            sql_logger: SQLite logger for event tracking
            output_dir: Directory to save memory building results
        """
        self.redis_client = redis_client
        self.vector_db = vector_db
        self.sql_logger = sql_logger
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Pattern collection
        self.patterns: List[Dict[str, Any]] = []
        self.pattern_embeddings: List[np.ndarray] = []

        # Statistics
        self.total_simulations_run: int = 0
        self.successful_patterns_generated: int = 0
        self.failed_patterns_generated: int = 0
        self.patterns_stored: int = 0

        # Configuration
        self.embedding_dim = 128  # Standard embedding dimension

        logger.info("MemoryBuilder initialized (output: %s)", self.output_dir)

    # =========================================================================
    # Simulation Running
    # =========================================================================

    def run_pattern_generation_campaign(
        self,
        num_simulations: int = 10,
        steps_per_simulation: int = 500,
        scenario_variations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run multiple simulations with variations to generate diverse patterns.

        Args:
            num_simulations: Number of simulation runs
            steps_per_simulation: Steps per simulation
            scenario_variations: List of scenario configurations to test

        Returns:
            Campaign results summary
        """
        logger.info("=" * 70)
        logger.info("Starting Pattern Generation Campaign")
        logger.info("Simulations: %d | Steps per sim: %d", num_simulations, steps_per_simulation)
        logger.info("=" * 70)

        # Default scenario variations
        if scenario_variations is None:
            scenario_variations = self._generate_default_scenarios()

        campaign_results = {
            "total_simulations": 0,
            "successful_simulations": 0,
            "failed_simulations": 0,
            "patterns_generated": 0,
            "scenarios_tested": len(scenario_variations),
            "simulation_details": []
        }

        # Run each scenario
        for i, scenario_config in enumerate(scenario_variations):
            logger.info("\n" + "=" * 70)
            logger.info("Running Scenario %d/%d: %s",
                       i + 1, len(scenario_variations), scenario_config.get("name", "unnamed"))
            logger.info("=" * 70)

            sim_result = self._run_single_simulation(
                scenario_config=scenario_config,
                num_steps=steps_per_simulation,
                simulation_id=f"sim_{i:03d}"
            )

            # Extract patterns from simulation
            patterns = self._extract_patterns_from_simulation(sim_result, scenario_config)

            campaign_results["total_simulations"] += 1
            campaign_results["patterns_generated"] += len(patterns)
            campaign_results["simulation_details"].append({
                "simulation_id": sim_result["simulation_id"],
                "scenario": scenario_config.get("name"),
                "test_passed": sim_result["test_passed"],
                "patterns_extracted": len(patterns)
            })

            if sim_result["test_passed"]:
                campaign_results["successful_simulations"] += 1
            else:
                campaign_results["failed_simulations"] += 1

            self.total_simulations_run += 1

            logger.info("Extracted %d patterns from simulation", len(patterns))

        # Store all patterns in Vector DB
        logger.info("\n" + "=" * 70)
        logger.info("Storing patterns in Vector DB...")
        logger.info("=" * 70)

        stored_count = self._store_patterns_in_vector_db()
        campaign_results["patterns_stored"] = stored_count

        # Generate summary report
        self._save_campaign_results(campaign_results)

        logger.info("\n" + "=" * 70)
        logger.info("Pattern Generation Campaign Complete")
        logger.info("Total Patterns Generated: %d", campaign_results["patterns_generated"])
        logger.info("Patterns Stored in Vector DB: %d", stored_count)
        logger.info("=" * 70)

        return campaign_results

    def _generate_default_scenarios(self) -> List[Dict[str, Any]]:
        """
        Generate default scenario variations for testing.

        Returns:
            List of scenario configurations
        """
        scenarios = []

        # Scenario 1: Policy Poisoning with High Containment
        scenarios.append({
            "name": "policy_poisoning_high_containment",
            "num_healthy": 15,
            "num_toxic": 1,
            "toxic_behavior": ToxicBehaviorType.POLICY_POISONING,
            "toxicity_level": 0.7,
            "risk_threshold": 0.6,
            "monitoring_interval": 3
        })

        # Scenario 2: Multiple Toxic Agents
        scenarios.append({
            "name": "multiple_toxic_agents",
            "num_healthy": 12,
            "num_toxic": 3,
            "toxic_behavior": ToxicBehaviorType.HIGH_RISK,
            "toxicity_level": 0.8,
            "risk_threshold": 0.7,
            "monitoring_interval": 5
        })

        # Scenario 3: Buggy Agent Detection
        scenarios.append({
            "name": "buggy_agent_detection",
            "num_healthy": 10,
            "num_toxic": 2,
            "toxic_behavior": ToxicBehaviorType.BUGGY,
            "toxicity_level": 0.9,
            "risk_threshold": 0.65,
            "monitoring_interval": 4
        })

        # Scenario 4: Byzantine Behavior
        scenarios.append({
            "name": "byzantine_behavior",
            "num_healthy": 8,
            "num_toxic": 2,
            "toxic_behavior": ToxicBehaviorType.BYZANTINE,
            "toxicity_level": 0.85,
            "risk_threshold": 0.7,
            "monitoring_interval": 5
        })

        # Scenario 5: Sabotage Attack
        scenarios.append({
            "name": "sabotage_attack",
            "num_healthy": 10,
            "num_toxic": 1,
            "toxic_behavior": ToxicBehaviorType.SABOTAGE,
            "toxicity_level": 0.95,
            "risk_threshold": 0.6,
            "monitoring_interval": 3
        })

        # Scenario 6: Low Toxicity (Subtle Attack)
        scenarios.append({
            "name": "subtle_attack",
            "num_healthy": 15,
            "num_toxic": 2,
            "toxic_behavior": ToxicBehaviorType.MANIPULATIVE,
            "toxicity_level": 0.4,
            "risk_threshold": 0.5,
            "monitoring_interval": 7
        })

        return scenarios

    def _run_single_simulation(
        self,
        scenario_config: Dict[str, Any],
        num_steps: int,
        simulation_id: str
    ) -> Dict[str, Any]:
        """
        Run a single simulation with the given configuration.

        Args:
            scenario_config: Scenario parameters
            num_steps: Number of simulation steps
            simulation_id: Unique ID for this simulation

        Returns:
            Simulation results dictionary
        """
        from simulation.simulation_runner import MockHavenCoordinator

        # Extract parameters
        num_healthy = scenario_config.get("num_healthy", 10)
        num_toxic = scenario_config.get("num_toxic", 2)
        toxic_behavior = scenario_config.get("toxic_behavior", ToxicBehaviorType.POLICY_POISONING)
        toxicity_level = scenario_config.get("toxicity_level", 0.8)
        risk_threshold = scenario_config.get("risk_threshold", 0.7)
        monitoring_interval = scenario_config.get("monitoring_interval", 5)

        # Create HAVEN coordinator
        haven_coordinator = MockHavenCoordinator(
            coordinator_id=f"haven_{simulation_id}",
            redis_client=self.redis_client,
            risk_threshold=risk_threshold,
            contagion_threshold=0.5,
            intervention_enabled=True
        )

        # Create adversarial model
        model = AdversarialModel(
            redis_client=self.redis_client,
            num_healthy_agents=num_healthy,
            num_toxic_agents=num_toxic,
            test_scenario="policy_contagion",
            haven_coordinator=haven_coordinator,
            random_seed=None  # Different seed each time for diversity
        )

        # Create data miner
        data_miner = DataMinerAgent(
            unique_id=0,
            model=model,
            redis_client=self.redis_client,
            source_streams=["test_stream"],
            output_channel="processed_data"
        )
        model.schedule.add(data_miner)

        # Create healthy agents
        healthy_agents = []
        for i in range(num_healthy):
            agent = SpecialistAgent(
                unique_id=i + 1,
                model=model,
                redis_client=self.redis_client,
                data_channel="processed_data",
                team_id=f"team_{i % 3}",  # Distribute across 3 teams
                specialization="normal"
            )
            model.add_healthy_agent(agent)
            healthy_agents.append(agent)

        # Create toxic agents
        toxic_agents = []
        for i in range(num_toxic):
            toxic_agent = ToxicAgent(
                unique_id=num_healthy + i + 1,
                model=model,
                redis_client=self.redis_client,
                toxic_behavior=toxic_behavior,
                toxicity_level=toxicity_level,
                data_channel="processed_data"
            )
            model.add_toxic_agent(toxic_agent)
            toxic_agents.append(toxic_agent)

        # Create risk manager
        risk_manager = RiskManagerAgent(
            unique_id=num_healthy + num_toxic + 1,
            model=model,
            redis_client=self.redis_client,
            haven_coordinator=haven_coordinator,
            agent_config={
                "monitoring_interval": monitoring_interval,
                "risk_threshold": risk_threshold,
                "auto_intervention": True
            }
        )
        model.schedule.add(risk_manager)
        model.risk_manager = risk_manager

        # Register agents with risk manager
        for agent in healthy_agents + toxic_agents:
            risk_manager.register_agent_for_monitoring(
                agent.agent_id,
                agent.get_state_summary()
            )

        # Setup and run
        model.setup()
        results = model.run(num_steps=num_steps)

        # Add metadata
        results["simulation_id"] = simulation_id
        results["scenario_config"] = scenario_config

        return results

    # =========================================================================
    # Pattern Extraction
    # =========================================================================

    def _extract_patterns_from_simulation(
        self,
        sim_result: Dict[str, Any],
        scenario_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract learning patterns from a completed simulation.

        Args:
            sim_result: Simulation results
            scenario_config: Scenario configuration

        Returns:
            List of extracted patterns
        """
        patterns = []

        # Extract successful policy patterns (if test passed)
        if sim_result.get("test_passed", False):
            successful_patterns = self._extract_successful_patterns(sim_result, scenario_config)
            patterns.extend(successful_patterns)
            self.successful_patterns_generated += len(successful_patterns)

        # Extract failed policy patterns
        failed_patterns = self._extract_failed_patterns(sim_result, scenario_config)
        patterns.extend(failed_patterns)
        self.failed_patterns_generated += len(failed_patterns)

        # Extract toxic detection patterns
        detection_patterns = self._extract_detection_patterns(sim_result, scenario_config)
        patterns.extend(detection_patterns)

        # Extract recovery patterns
        if sim_result.get("system_recovered", False):
            recovery_patterns = self._extract_recovery_patterns(sim_result, scenario_config)
            patterns.extend(recovery_patterns)

        # Store patterns for later embedding
        self.patterns.extend(patterns)

        return patterns

    def _extract_successful_patterns(
        self,
        sim_result: Dict[str, Any],
        scenario_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract patterns from successful containment."""
        patterns = []

        # Pattern: Successful Contagion Containment
        pattern = {
            "pattern_id": f"success_{self.total_simulations_run}_{len(self.patterns)}",
            "pattern_type": PatternType.SUCCESSFUL_POLICY,
            "scenario": scenario_config.get("name"),
            "description": "Successfully contained policy contagion",
            "metrics": {
                "containment_rate": sim_result.get("containment_rate", 0.0),
                "response_time": sim_result.get("response_time", 0),
                "final_system_risk": sim_result.get("final_system_risk", 0.0),
                "interventions": sim_result.get("interventions_triggered", 0)
            },
            "context": {
                "num_healthy": scenario_config.get("num_healthy"),
                "num_toxic": scenario_config.get("num_toxic"),
                "toxic_behavior": scenario_config.get("toxic_behavior"),
                "monitoring_interval": scenario_config.get("monitoring_interval")
            },
            "outcome": "success",
            "confidence": 0.9
        }

        patterns.append(pattern)

        # Pattern: Optimal Coordination
        if sim_result.get("containment_rate", 0) > 0.8:
            coordination_pattern = {
                "pattern_id": f"coord_{self.total_simulations_run}_{len(self.patterns) + 1}",
                "pattern_type": PatternType.OPTIMAL_COORDINATION,
                "scenario": scenario_config.get("name"),
                "description": "High-quality team coordination during crisis",
                "metrics": {
                    "containment_rate": sim_result.get("containment_rate"),
                    "healthy_avg_performance": sim_result.get("agent_performance_summary", {}).get("healthy_agents_avg", 0)
                },
                "context": scenario_config,
                "outcome": "success",
                "confidence": 0.85
            }
            patterns.append(coordination_pattern)

        return patterns

    def _extract_failed_patterns(
        self,
        sim_result: Dict[str, Any],
        scenario_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract patterns from failed policies/scenarios."""
        patterns = []

        # If test failed, create failure pattern
        if not sim_result.get("test_passed", False):
            pattern = {
                "pattern_id": f"fail_{self.total_simulations_run}_{len(self.patterns)}",
                "pattern_type": PatternType.FAILED_POLICY,
                "scenario": scenario_config.get("name"),
                "description": "Failed to contain policy contagion",
                "metrics": {
                    "infection_rate": sim_result.get("infection_rate", 0.0),
                    "containment_rate": sim_result.get("containment_rate", 0.0),
                    "max_system_risk": sim_result.get("max_system_risk", 0.0)
                },
                "context": scenario_config,
                "outcome": "failure",
                "confidence": 0.9
            }
            patterns.append(pattern)

        # Extract toxic behavior patterns
        if sim_result.get("contagion_detected", False):
            contagion_pattern = {
                "pattern_id": f"contagion_{self.total_simulations_run}_{len(self.patterns) + 1}",
                "pattern_type": PatternType.CONTAGION_PATTERN,
                "scenario": scenario_config.get("name"),
                "description": f"Contagion spread pattern: {scenario_config.get('toxic_behavior')}",
                "metrics": {
                    "spread_rate": sim_result.get("contagion_spread_rate", 0.0),
                    "infection_rate": sim_result.get("infection_rate", 0.0),
                    "detection_step": sim_result.get("contagion_detection_step", 0)
                },
                "context": {
                    "toxic_behavior": scenario_config.get("toxic_behavior"),
                    "toxicity_level": scenario_config.get("toxicity_level")
                },
                "outcome": "contagion",
                "confidence": 0.8
            }
            patterns.append(contagion_pattern)

        return patterns

    def _extract_detection_patterns(
        self,
        sim_result: Dict[str, Any],
        scenario_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract patterns related to toxic agent detection."""
        patterns = []

        if sim_result.get("contagion_detected", False):
            detection_pattern = {
                "pattern_id": f"detect_{self.total_simulations_run}_{len(self.patterns)}",
                "pattern_type": PatternType.TOXIC_DETECTION,
                "scenario": scenario_config.get("name"),
                "description": "Toxic agent detection signature",
                "metrics": {
                    "detection_time": sim_result.get("contagion_detection_step", 0),
                    "infected_agents": len(sim_result.get("infected_agents", [])),
                    "interventions": sim_result.get("interventions_triggered", 0)
                },
                "context": {
                    "toxic_behavior": scenario_config.get("toxic_behavior"),
                    "monitoring_interval": scenario_config.get("monitoring_interval"),
                    "risk_threshold": scenario_config.get("risk_threshold")
                },
                "outcome": "detected",
                "confidence": 0.85
            }
            patterns.append(detection_pattern)

        return patterns

    def _extract_recovery_patterns(
        self,
        sim_result: Dict[str, Any],
        scenario_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract patterns from system recovery."""
        patterns = []

        recovery_pattern = {
            "pattern_id": f"recovery_{self.total_simulations_run}_{len(self.patterns)}",
            "pattern_type": PatternType.RECOVERY_PATTERN,
            "scenario": scenario_config.get("name"),
            "description": "System recovery after attack",
            "metrics": {
                "max_risk": sim_result.get("max_system_risk", 0.0),
                "final_risk": sim_result.get("final_system_risk", 0.0),
                "recovery_ratio": 1.0 - (sim_result.get("final_system_risk", 1.0) / max(0.01, sim_result.get("max_system_risk", 1.0)))
            },
            "context": scenario_config,
            "outcome": "recovered",
            "confidence": 0.8
        }
        patterns.append(recovery_pattern)

        return patterns

    # =========================================================================
    # Embedding Generation & Storage
    # =========================================================================

    def _create_pattern_embedding(self, pattern: Dict[str, Any]) -> np.ndarray:
        """
        Create a vector embedding for a pattern.

        Args:
            pattern: Pattern dictionary

        Returns:
            Embedding vector (128-dim)
        """
        # Extract numeric features
        features = []

        # Add metrics as features
        metrics = pattern.get("metrics", {})
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, (int, float)):
                features.append(float(value))

        # Add context features
        context = pattern.get("context", {})
        for key in sorted(context.keys()):
            value = context[key]
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Hash string to numeric value
                features.append(float(hash(value) % 1000) / 1000.0)

        # Add outcome encoding
        outcome_encoding = {
            "success": 1.0,
            "failure": -1.0,
            "contagion": -0.5,
            "detected": 0.5,
            "recovered": 0.8
        }
        features.append(outcome_encoding.get(pattern.get("outcome", ""), 0.0))

        # Add confidence
        features.append(pattern.get("confidence", 0.5))

        # Add pattern type encoding
        pattern_type_encoding = {
            PatternType.SUCCESSFUL_POLICY: 1.0,
            PatternType.FAILED_POLICY: -1.0,
            PatternType.TOXIC_DETECTION: 0.5,
            PatternType.CONTAGION_PATTERN: -0.5,
            PatternType.RECOVERY_PATTERN: 0.8,
            PatternType.OPTIMAL_COORDINATION: 0.9,
            PatternType.RISK_MITIGATION: 0.7
        }
        features.append(pattern_type_encoding.get(pattern.get("pattern_type", ""), 0.0))

        # Pad or truncate to embedding_dim
        while len(features) < self.embedding_dim:
            features.append(0.0)

        embedding = np.array(features[:self.embedding_dim])

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _store_patterns_in_vector_db(self) -> int:
        """
        Store all collected patterns in the Vector DB.

        Returns:
            Number of patterns stored
        """
        stored_count = 0

        logger.info("Creating embeddings for %d patterns...", len(self.patterns))

        for pattern in self.patterns:
            try:
                # Create embedding
                embedding = self._create_pattern_embedding(pattern)

                # Prepare metadata
                metadata = {
                    "pattern_id": pattern.get("pattern_id"),
                    "pattern_type": pattern.get("pattern_type"),
                    "scenario": pattern.get("scenario", "unknown"),
                    "outcome": pattern.get("outcome", "unknown"),
                    "confidence": pattern.get("confidence", 0.5),
                    "description": pattern.get("description", "")
                }

                # Store in Vector DB
                self.vector_db.add_policy_embedding(
                    policy_id=pattern["pattern_id"],
                    agent_id="memory_builder",
                    embedding=embedding.tolist(),
                    metadata=metadata
                )

                stored_count += 1

                # Log to SQL
                self.sql_logger.log_system_event(
                    event_type="pattern_stored",
                    severity="INFO",
                    description=f"Stored pattern: {pattern['pattern_type']}",
                    data=metadata
                )

            except Exception as e:
                logger.error("Failed to store pattern %s: %s",
                            pattern.get("pattern_id"), e)

        self.patterns_stored = stored_count

        logger.info("Successfully stored %d patterns in Vector DB", stored_count)

        return stored_count

    # =========================================================================
    # Synthetic Data Generation
    # =========================================================================

    def generate_synthetic_patterns(
        self,
        num_patterns: int = 100,
        pattern_distribution: Optional[Dict[str, float]] = None
    ) -> int:
        """
        Generate synthetic patterns without running full simulations.

        Useful for quickly populating the Vector DB with diverse examples.

        Args:
            num_patterns: Number of synthetic patterns to generate
            pattern_distribution: Distribution of pattern types

        Returns:
            Number of patterns generated
        """
        logger.info("Generating %d synthetic patterns...", num_patterns)

        if pattern_distribution is None:
            pattern_distribution = {
                PatternType.SUCCESSFUL_POLICY: 0.4,
                PatternType.FAILED_POLICY: 0.2,
                PatternType.TOXIC_DETECTION: 0.15,
                PatternType.CONTAGION_PATTERN: 0.1,
                PatternType.RECOVERY_PATTERN: 0.1,
                PatternType.OPTIMAL_COORDINATION: 0.05
            }

        generated_count = 0

        for i in range(num_patterns):
            # Select pattern type based on distribution
            pattern_type = np.random.choice(
                list(pattern_distribution.keys()),
                p=list(pattern_distribution.values())
            )

            # Generate synthetic pattern
            pattern = self._generate_synthetic_pattern(pattern_type, i)
            self.patterns.append(pattern)
            generated_count += 1

            if pattern.get("outcome") == "success":
                self.successful_patterns_generated += 1
            else:
                self.failed_patterns_generated += 1

        logger.info("Generated %d synthetic patterns", generated_count)

        # Store in Vector DB
        stored_count = self._store_patterns_in_vector_db()

        return generated_count

    def _generate_synthetic_pattern(
        self,
        pattern_type: str,
        index: int
    ) -> Dict[str, Any]:
        """Generate a single synthetic pattern."""

        if pattern_type == PatternType.SUCCESSFUL_POLICY:
            return {
                "pattern_id": f"synthetic_success_{index}",
                "pattern_type": pattern_type,
                "scenario": "synthetic",
                "description": "Synthetic successful policy pattern",
                "metrics": {
                    "containment_rate": np.random.uniform(0.7, 1.0),
                    "response_time": np.random.randint(5, 50),
                    "final_system_risk": np.random.uniform(0.0, 0.3)
                },
                "context": {
                    "num_healthy": np.random.randint(8, 20),
                    "num_toxic": np.random.randint(1, 4),
                    "monitoring_interval": np.random.randint(3, 10)
                },
                "outcome": "success",
                "confidence": np.random.uniform(0.7, 0.95)
            }

        elif pattern_type == PatternType.FAILED_POLICY:
            return {
                "pattern_id": f"synthetic_fail_{index}",
                "pattern_type": pattern_type,
                "scenario": "synthetic",
                "description": "Synthetic failed policy pattern",
                "metrics": {
                    "infection_rate": np.random.uniform(0.4, 1.0),
                    "containment_rate": np.random.uniform(0.0, 0.5),
                    "max_system_risk": np.random.uniform(0.6, 1.0)
                },
                "context": {
                    "num_healthy": np.random.randint(5, 15),
                    "num_toxic": np.random.randint(2, 6)
                },
                "outcome": "failure",
                "confidence": np.random.uniform(0.7, 0.9)
            }

        else:
            # Generic pattern
            return {
                "pattern_id": f"synthetic_{pattern_type}_{index}",
                "pattern_type": pattern_type,
                "scenario": "synthetic",
                "description": f"Synthetic {pattern_type} pattern",
                "metrics": {
                    "value": np.random.uniform(0.0, 1.0)
                },
                "context": {},
                "outcome": "neutral",
                "confidence": np.random.uniform(0.5, 0.8)
            }

    # =========================================================================
    # Querying & Validation
    # =========================================================================

    def query_similar_patterns(
        self,
        query_pattern: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the Vector DB for similar patterns.

        Args:
            query_pattern: Pattern to match against
            top_k: Number of similar patterns to retrieve

        Returns:
            List of similar patterns
        """
        # Create embedding for query
        query_embedding = self._create_pattern_embedding(query_pattern)

        # Search Vector DB
        results = self.vector_db.search_similar_policies(
            query_embedding=query_embedding.tolist(),
            top_k=top_k
        )

        logger.info("Found %d similar patterns", len(results))

        return results

    def validate_memory_quality(self) -> Dict[str, Any]:
        """
        Validate the quality of the generated memory.

        Returns:
            Quality metrics dictionary
        """
        logger.info("Validating memory quality...")

        # Get all patterns from Vector DB
        total_patterns = self.patterns_stored

        # Calculate diversity
        pattern_types = {}
        for pattern in self.patterns:
            ptype = pattern.get("pattern_type", "unknown")
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1

        # Calculate balance
        success_ratio = self.successful_patterns_generated / max(1, total_patterns)
        failure_ratio = self.failed_patterns_generated / max(1, total_patterns)

        quality_report = {
            "total_patterns": total_patterns,
            "successful_patterns": self.successful_patterns_generated,
            "failed_patterns": self.failed_patterns_generated,
            "pattern_type_distribution": pattern_types,
            "success_failure_ratio": success_ratio / max(0.01, failure_ratio),
            "diversity_score": len(pattern_types) / max(1, len(PatternType.__dict__)),
            "quality_grade": self._assess_quality_grade(success_ratio, failure_ratio, len(pattern_types))
        }

        logger.info("Memory Quality Report:")
        logger.info("  Total Patterns: %d", total_patterns)
        logger.info("  Pattern Types: %d", len(pattern_types))
        logger.info("  Quality Grade: %s", quality_report["quality_grade"])

        return quality_report

    def _assess_quality_grade(
        self,
        success_ratio: float,
        failure_ratio: float,
        num_types: int
    ) -> str:
        """Assess overall quality grade."""
        # Good balance of success/failure
        balance_score = 1.0 - abs(success_ratio - failure_ratio)

        # Good diversity
        diversity_score = num_types / 7.0  # 7 pattern types total

        # Overall score
        overall_score = (balance_score * 0.4) + (diversity_score * 0.6)

        if overall_score > 0.8:
            return "A (Excellent)"
        elif overall_score > 0.6:
            return "B (Good)"
        elif overall_score > 0.4:
            return "C (Adequate)"
        else:
            return "D (Needs Improvement)"

    # =========================================================================
    # Results & Reporting
    # =========================================================================

    def _save_campaign_results(self, campaign_results: Dict[str, Any]):
        """Save campaign results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_path = self.output_dir / f"campaign_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(campaign_results, f, indent=2)

        logger.info("Campaign results saved to %s", json_path)

        # Save patterns
        patterns_path = self.output_dir / f"patterns_{timestamp}.pkl"
        with open(patterns_path, 'wb') as f:
            pickle.dump(self.patterns, f)

        logger.info("Patterns saved to %s", patterns_path)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory builder statistics."""
        return {
            "total_simulations_run": self.total_simulations_run,
            "successful_patterns": self.successful_patterns_generated,
            "failed_patterns": self.failed_patterns_generated,
            "total_patterns": len(self.patterns),
            "patterns_stored": self.patterns_stored
        }


def main():
    """
    Main entry point for memory building.
    """
    logger.info("=" * 70)
    logger.info("Memory Builder - Knowledge Base Pre-Training")
    logger.info("=" * 70)

    # Initialize infrastructure
    redis_client = RedisClient(host="localhost", port=6379, db=2)
    vector_db = create_vector_db(
        backend="chromadb",
        collection_name="mae_patterns",
        embedding_dim=128,
        persist_directory="data/pattern_memory"
    )
    vector_db.initialize()

    sql_logger = SQLiteLogger(db_path="data/memory_building.db")

    # Create memory builder
    builder = MemoryBuilder(
        redis_client=redis_client,
        vector_db=vector_db,
        sql_logger=sql_logger,
        output_dir="memory_building_results"
    )

    # Run pattern generation campaign
    campaign_results = builder.run_pattern_generation_campaign(
        num_simulations=6,  # One per scenario
        steps_per_simulation=300
    )

    # Generate additional synthetic patterns for diversity
    builder.generate_synthetic_patterns(num_patterns=50)

    # Validate quality
    quality_report = builder.validate_memory_quality()

    # Print final statistics
    stats = builder.get_statistics()
    logger.info("\n" + "=" * 70)
    logger.info("Memory Building Complete!")
    logger.info("=" * 70)
    logger.info("Total Simulations: %d", stats["total_simulations_run"])
    logger.info("Total Patterns: %d", stats["total_patterns"])
    logger.info("Patterns Stored: %d", stats["patterns_stored"])
    logger.info("Quality Grade: %s", quality_report["quality_grade"])
    logger.info("=" * 70)

    # Cleanup
    sql_logger.stop()
    vector_db.close()
    redis_client.close()


if __name__ == "__main__":
    main()
