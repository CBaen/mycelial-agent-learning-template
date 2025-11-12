"""
Settings Module for Mycelial Agent Engine (MAE)

This module loads and validates all system configuration from config.yaml
and environment variables. It provides a centralized settings object for
the entire application.

Usage:
    from config.settings import settings

    redis_host = settings.redis.host
    num_agents = settings.agents.num_specialists
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RedisSettings:
    """Redis configuration settings."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    decode_responses: bool = True


@dataclass
class VectorDBSettings:
    """Vector database configuration settings."""
    backend: str = "chromadb"  # chromadb, milvus, qdrant
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "mae_policies"
    embedding_dim: int = 128
    persist_directory: str = "data/chromadb"
    similarity_metric: str = "cosine"  # cosine, euclidean, dot_product


@dataclass
class SQLiteSettings:
    """SQLite logger configuration settings."""
    db_path: str = "data/mae_system.db"
    queue_size: int = 10000
    batch_size: int = 100
    flush_interval: float = 1.0
    cleanup_days: int = 30


@dataclass
class AgentSettings:
    """Agent configuration settings."""
    # Agent counts
    num_specialists: int = 10
    num_data_miners: int = 1
    num_risk_managers: int = 1

    # Specialist agent settings
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    buffer_size: int = 1000

    # Data miner settings
    batch_size: int = 10
    validation_enabled: bool = True

    # Risk manager settings
    monitoring_interval: int = 5
    risk_threshold: float = 0.7
    contagion_threshold: float = 0.5
    auto_intervention: bool = True


@dataclass
class FederatedLearningSettings:
    """Federated Reinforcement Learning (FRL) settings."""
    enabled: bool = True
    update_strategy: str = "selective"  # broadcast, selective, neighborhood, performance_based, random
    aggregation_method: str = "weighted_average"  # weighted_average, median, federated_averaging, trust_weighted
    share_frequency: int = 10  # Share policy every N steps
    min_peer_trust: float = 0.3
    max_peers: int = 5


@dataclass
class ValueDecompositionSettings:
    """Value-Decomposition Networks (VDN) settings."""
    enabled: bool = True
    decomposition_method: str = "additive"  # additive, monotonic, weighted, context_dependent
    credit_strategy: str = "difference_rewards"  # difference_rewards, shapley_value, counterfactual
    discount_factor: float = 0.99


@dataclass
class HAVENSettings:
    """HAVEN Risk Coordination settings."""
    enabled: bool = True
    risk_threshold: float = 0.7
    contagion_threshold: float = 0.5
    intervention_enabled: bool = True
    max_isolation_time: int = 100  # steps
    risk_history_window: int = 50


@dataclass
class SimulationSettings:
    """Simulation configuration settings."""
    mode: str = "simulation"  # simulation, live
    num_steps: int = 1000
    random_seed: Optional[int] = 42
    log_level: str = "INFO"
    step_delay: float = 0.0  # seconds between steps (0 = no delay)


@dataclass
class StreamSettings:
    """Redis Stream configuration."""
    data_ingestion_stream: str = "mae:data_ingestion"
    max_stream_length: int = 10000
    read_batch_size: int = 100
    block_timeout_ms: int = 1000


@dataclass
class PubSubSettings:
    """Redis Pub/Sub configuration."""
    agent_messages_channel: str = "mae:agent_messages"
    system_events_channel: str = "mae:system_events"
    risk_alerts_channel: str = "mae:risk_alerts"
    health_reports_channel: str = "mae:health_reports"


@dataclass
class APISettings:
    """External API configuration."""
    # Add your API keys and endpoints here
    data_source_api_key: Optional[str] = None
    data_source_endpoint: Optional[str] = None
    enable_external_apis: bool = False


@dataclass
class LoggingSettings:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_file_path: str = "logs/mae.log"
    max_log_file_size: int = 10485760  # 10 MB
    backup_count: int = 5


@dataclass
class PerformanceSettings:
    """Performance and optimization settings."""
    enable_profiling: bool = False
    enable_metrics_collection: bool = True
    metrics_collection_interval: int = 10  # steps
    enable_caching: bool = True
    cache_ttl: int = 300  # seconds


@dataclass
class Settings:
    """
    Main settings container for MAE system.

    All configuration is loaded from config.yaml and can be overridden
    by environment variables.
    """
    redis: RedisSettings = field(default_factory=RedisSettings)
    vector_db: VectorDBSettings = field(default_factory=VectorDBSettings)
    sqlite: SQLiteSettings = field(default_factory=SQLiteSettings)
    agents: AgentSettings = field(default_factory=AgentSettings)
    frl: FederatedLearningSettings = field(default_factory=FederatedLearningSettings)
    vdn: ValueDecompositionSettings = field(default_factory=ValueDecompositionSettings)
    haven: HAVENSettings = field(default_factory=HAVENSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    streams: StreamSettings = field(default_factory=StreamSettings)
    pubsub: PubSubSettings = field(default_factory=PubSubSettings)
    api: APISettings = field(default_factory=APISettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)

    def __post_init__(self):
        """Apply environment variable overrides after initialization."""
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides to settings."""
        # Redis overrides
        if os.getenv("REDIS_HOST"):
            self.redis.host = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            self.redis.port = int(os.getenv("REDIS_PORT"))
        if os.getenv("REDIS_PASSWORD"):
            self.redis.password = os.getenv("REDIS_PASSWORD")
        if os.getenv("REDIS_DB"):
            self.redis.db = int(os.getenv("REDIS_DB"))

        # Vector DB overrides
        if os.getenv("VECTOR_DB_BACKEND"):
            self.vector_db.backend = os.getenv("VECTOR_DB_BACKEND")
        if os.getenv("VECTOR_DB_HOST"):
            self.vector_db.host = os.getenv("VECTOR_DB_HOST")
        if os.getenv("VECTOR_DB_PORT"):
            self.vector_db.port = int(os.getenv("VECTOR_DB_PORT"))

        # API keys
        if os.getenv("DATA_SOURCE_API_KEY"):
            self.api.data_source_api_key = os.getenv("DATA_SOURCE_API_KEY")

        # Log level
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL")
            self.simulation.log_level = os.getenv("LOG_LEVEL")

    @classmethod
    def from_yaml(cls, config_path: str = "config/config.yaml") -> "Settings":
        """
        Load settings from YAML file.

        Args:
            config_path: Path to config.yaml file

        Returns:
            Settings object
        """
        config_file = Path(config_path)

        if not config_file.exists():
            logger.warning("Config file not found at %s, using defaults", config_path)
            return cls()

        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                logger.warning("Empty config file, using defaults")
                return cls()

            # Create settings object
            settings_obj = cls()

            # Load each section
            if "redis" in config_data:
                settings_obj.redis = RedisSettings(**config_data["redis"])

            if "vector_db" in config_data:
                settings_obj.vector_db = VectorDBSettings(**config_data["vector_db"])

            if "sqlite" in config_data:
                settings_obj.sqlite = SQLiteSettings(**config_data["sqlite"])

            if "agents" in config_data:
                settings_obj.agents = AgentSettings(**config_data["agents"])

            if "frl" in config_data:
                settings_obj.frl = FederatedLearningSettings(**config_data["frl"])

            if "vdn" in config_data:
                settings_obj.vdn = ValueDecompositionSettings(**config_data["vdn"])

            if "haven" in config_data:
                settings_obj.haven = HAVENSettings(**config_data["haven"])

            if "simulation" in config_data:
                settings_obj.simulation = SimulationSettings(**config_data["simulation"])

            if "streams" in config_data:
                settings_obj.streams = StreamSettings(**config_data["streams"])

            if "pubsub" in config_data:
                settings_obj.pubsub = PubSubSettings(**config_data["pubsub"])

            if "api" in config_data:
                settings_obj.api = APISettings(**config_data["api"])

            if "logging" in config_data:
                settings_obj.logging = LoggingSettings(**config_data["logging"])

            if "performance" in config_data:
                settings_obj.performance = PerformanceSettings(**config_data["performance"])

            # Apply environment variable overrides
            settings_obj._apply_env_overrides()

            logger.info("Settings loaded from %s", config_path)
            return settings_obj

        except Exception as e:
            logger.error("Error loading config file: %s", e, exc_info=True)
            logger.warning("Falling back to default settings")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.

        Returns:
            Dictionary representation of settings
        """
        return {
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "max_connections": self.redis.max_connections,
            },
            "vector_db": {
                "backend": self.vector_db.backend,
                "collection_name": self.vector_db.collection_name,
                "embedding_dim": self.vector_db.embedding_dim,
            },
            "sqlite": {
                "db_path": self.sqlite.db_path,
                "queue_size": self.sqlite.queue_size,
            },
            "agents": {
                "num_specialists": self.agents.num_specialists,
                "num_data_miners": self.agents.num_data_miners,
                "num_risk_managers": self.agents.num_risk_managers,
            },
            "frl": {
                "enabled": self.frl.enabled,
                "update_strategy": self.frl.update_strategy,
                "aggregation_method": self.frl.aggregation_method,
            },
            "vdn": {
                "enabled": self.vdn.enabled,
                "decomposition_method": self.vdn.decomposition_method,
            },
            "haven": {
                "enabled": self.haven.enabled,
                "risk_threshold": self.haven.risk_threshold,
            },
            "simulation": {
                "mode": self.simulation.mode,
                "num_steps": self.simulation.num_steps,
            },
        }

    def validate(self) -> bool:
        """
        Validate settings for consistency and correctness.

        Returns:
            True if settings are valid

        Raises:
            ValueError: If settings are invalid
        """
        # Validate Redis settings
        if self.redis.port < 1 or self.redis.port > 65535:
            raise ValueError(f"Invalid Redis port: {self.redis.port}")

        # Validate agent counts
        if self.agents.num_specialists < 1:
            raise ValueError("Must have at least 1 specialist agent")

        # Validate thresholds
        if not 0.0 <= self.agents.risk_threshold <= 1.0:
            raise ValueError("Risk threshold must be between 0 and 1")

        if not 0.0 <= self.agents.learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0 and 1")

        # Validate simulation settings
        if self.simulation.num_steps < 1:
            raise ValueError("Number of steps must be at least 1")

        logger.info("Settings validation passed")
        return True


# Global settings instance
# Load from config.yaml on import
_config_path = os.getenv("MAE_CONFIG_PATH", "config/config.yaml")
settings = Settings.from_yaml(_config_path)

# Validate settings
try:
    settings.validate()
except ValueError as e:
    logger.error("Invalid settings: %s", e)
    raise


def reload_settings(config_path: Optional[str] = None):
    """
    Reload settings from config file.

    Args:
        config_path: Optional path to config file
    """
    global settings

    if config_path:
        settings = Settings.from_yaml(config_path)
    else:
        settings = Settings.from_yaml(_config_path)

    settings.validate()
    logger.info("Settings reloaded")


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Returns:
        Settings object
    """
    return settings
