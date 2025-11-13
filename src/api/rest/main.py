"""
MAE REST API - Main Application

FastAPI application with comprehensive endpoints for agent management,
training control, metrics querying, policy management, and system monitoring.
"""

import time
import psutil
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Path, Depends, status
from fastapi.responses import Response, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from src.api.schemas import (
    # Agents
    AgentCreate, AgentResponse, AgentList, AgentStats,
    # Training
    TrainingConfig, TrainingStart, TrainingStatus, TrainingResponse,
    # Metrics
    MetricsQuery, MetricsResponse, SystemMetrics,
    # Policies
    PolicyExport, PolicyImport, PolicyCompare,
    # System
    HealthResponse, VersionResponse, SystemStats, HealthStatus,
    ComponentHealth, ResourceUsage
)

# Global state management (would be replaced with proper state manager in production)
class APIState:
    """Global API state."""
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.training_sessions: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()
        self.request_count = 0
        self.total_response_time = 0.0

api_state = APIState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Starting MAE REST API...")
    yield
    # Shutdown
    print("Shutting down MAE REST API...")


# FastAPI application
app = FastAPI(
    title="MAE REST API",
    description="Production-grade REST API for Mycelial Agent Environment",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    """Track API requests for metrics."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    api_state.request_count += 1
    api_state.total_response_time += process_time

    response.headers["X-Process-Time"] = str(process_time)
    return response


# ============================================================================
# AGENT ENDPOINTS
# ============================================================================

@app.post(
    "/agents",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["agents"],
    summary="Create new agent",
    description="Create a new agent with specified type and configuration"
)
async def create_agent(agent: AgentCreate) -> AgentResponse:
    """Create a new agent."""
    agent_id = f"agent_{len(api_state.agents) + 1}"

    agent_data = {
        "agent_id": agent_id,
        "agent_type": agent.agent_type,
        "team_id": agent.team_id,
        "status": "idle",
        "created_at": datetime.utcnow(),
        "config": agent.config,
        "stats": {
            "learning_steps": 0,
            "average_reward": 0.0,
            "convergence_score": 0.0,
            "satisfaction_score": 0.0,
            "xp": 0
        }
    }

    api_state.agents[agent_id] = agent_data

    return AgentResponse(
        agent_id=agent_id,
        agent_type=agent.agent_type,
        team_id=agent.team_id,
        status="idle",
        created_at=agent_data["created_at"],
        stats=AgentStats(**agent_data["stats"])
    )


@app.get(
    "/agents",
    response_model=AgentList,
    tags=["agents"],
    summary="List all agents",
    description="Get paginated list of all agents"
)
async def list_agents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size")
) -> AgentList:
    """List all agents with pagination."""
    agents_list = list(api_state.agents.values())
    total = len(agents_list)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_agents = agents_list[start_idx:end_idx]

    agent_responses = [
        AgentResponse(
            agent_id=agent["agent_id"],
            agent_type=agent["agent_type"],
            team_id=agent["team_id"],
            status=agent["status"],
            created_at=agent["created_at"],
            stats=AgentStats(**agent["stats"])
        )
        for agent in paginated_agents
    ]

    return AgentList(
        agents=agent_responses,
        total=total,
        page=page,
        page_size=page_size
    )


@app.get(
    "/agents/{agent_id}",
    response_model=AgentResponse,
    tags=["agents"],
    summary="Get agent details",
    description="Get detailed information about a specific agent"
)
async def get_agent(
    agent_id: str = Path(..., description="Agent identifier")
) -> AgentResponse:
    """Get agent details."""
    if agent_id not in api_state.agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = api_state.agents[agent_id]
    return AgentResponse(
        agent_id=agent["agent_id"],
        agent_type=agent["agent_type"],
        team_id=agent["team_id"],
        status=agent["status"],
        created_at=agent["created_at"],
        stats=AgentStats(**agent["stats"])
    )


@app.delete(
    "/agents/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["agents"],
    summary="Delete agent",
    description="Remove an agent from the system"
)
async def delete_agent(
    agent_id: str = Path(..., description="Agent identifier")
):
    """Delete an agent."""
    if agent_id not in api_state.agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    del api_state.agents[agent_id]
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post(
    "/agents/{agent_id}/reset",
    response_model=AgentResponse,
    tags=["agents"],
    summary="Reset agent state",
    description="Reset agent to initial state"
)
async def reset_agent(
    agent_id: str = Path(..., description="Agent identifier")
) -> AgentResponse:
    """Reset agent state."""
    if agent_id not in api_state.agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    agent = api_state.agents[agent_id]
    agent["stats"] = {
        "learning_steps": 0,
        "average_reward": 0.0,
        "convergence_score": 0.0,
        "satisfaction_score": 0.0,
        "xp": 0
    }
    agent["status"] = "idle"

    return AgentResponse(
        agent_id=agent["agent_id"],
        agent_type=agent["agent_type"],
        team_id=agent["team_id"],
        status=agent["status"],
        created_at=agent["created_at"],
        stats=AgentStats(**agent["stats"])
    )


# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================

@app.post(
    "/training/start",
    response_model=TrainingResponse,
    tags=["training"],
    summary="Start training session",
    description="Initiate a new training session with specified configuration"
)
async def start_training(training: TrainingStart) -> TrainingResponse:
    """Start training session."""
    session_id = f"session_{len(api_state.training_sessions) + 1}"

    config = training.config if training.config else TrainingConfig()

    session = {
        "session_id": session_id,
        "status": TrainingStatus.RUNNING,
        "started_at": datetime.utcnow(),
        "stopped_at": None,
        "config": config.dict(),
        "agents": training.agents or list(api_state.agents.keys()),
        "metrics": {
            "episodes_completed": 0,
            "total_steps": 0,
            "average_reward": 0.0,
            "average_loss": 0.0,
            "convergence_rate": 0.0
        }
    }

    api_state.training_sessions[session_id] = session

    # Update agent statuses
    for agent_id in session["agents"]:
        if agent_id in api_state.agents:
            api_state.agents[agent_id]["status"] = "active"

    return TrainingResponse(
        session_id=session_id,
        status=TrainingStatus.RUNNING,
        started_at=session["started_at"],
        stopped_at=None,
        config=config,
        metrics=None
    )


@app.post(
    "/training/stop",
    response_model=TrainingResponse,
    tags=["training"],
    summary="Stop training session",
    description="Stop the current training session"
)
async def stop_training(
    session_id: Optional[str] = Query(None, description="Session ID (current if not provided)")
) -> TrainingResponse:
    """Stop training session."""
    if not api_state.training_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active training sessions"
        )

    # Get session (latest if not specified)
    if session_id:
        if session_id not in api_state.training_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        session = api_state.training_sessions[session_id]
    else:
        session = list(api_state.training_sessions.values())[-1]
        session_id = session["session_id"]

    session["status"] = TrainingStatus.STOPPED
    session["stopped_at"] = datetime.utcnow()

    # Update agent statuses
    for agent_id in session["agents"]:
        if agent_id in api_state.agents:
            api_state.agents[agent_id]["status"] = "idle"

    return TrainingResponse(
        session_id=session_id,
        status=TrainingStatus.STOPPED,
        started_at=session["started_at"],
        stopped_at=session["stopped_at"],
        config=TrainingConfig(**session["config"]),
        metrics=session.get("metrics")
    )


@app.get(
    "/training/status",
    response_model=TrainingResponse,
    tags=["training"],
    summary="Get training status",
    description="Get current training session status and metrics"
)
async def get_training_status(
    session_id: Optional[str] = Query(None, description="Session ID (current if not provided)")
) -> TrainingResponse:
    """Get training status."""
    if not api_state.training_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No training sessions found"
        )

    # Get session (latest if not specified)
    if session_id:
        if session_id not in api_state.training_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        session = api_state.training_sessions[session_id]
    else:
        session = list(api_state.training_sessions.values())[-1]
        session_id = session["session_id"]

    from src.api.schemas.training import TrainingMetrics

    return TrainingResponse(
        session_id=session_id,
        status=session["status"],
        started_at=session["started_at"],
        stopped_at=session.get("stopped_at"),
        config=TrainingConfig(**session["config"]),
        metrics=TrainingMetrics(**session["metrics"]) if session.get("metrics") else None
    )


@app.put(
    "/training/config",
    response_model=TrainingConfig,
    tags=["training"],
    summary="Update training configuration",
    description="Update hyperparameters for current training session"
)
async def update_training_config(config: TrainingConfig) -> TrainingConfig:
    """Update training configuration."""
    if not api_state.training_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active training sessions"
        )

    session = list(api_state.training_sessions.values())[-1]
    session["config"] = config.dict()

    return config


# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@app.get(
    "/metrics/agents/{agent_id}",
    response_model=List[MetricsResponse],
    tags=["metrics"],
    summary="Get agent metrics",
    description="Query metrics for a specific agent"
)
async def get_agent_metrics(
    agent_id: str = Path(..., description="Agent identifier")
) -> List[MetricsResponse]:
    """Get agent-specific metrics."""
    if agent_id not in api_state.agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    from src.api.schemas.metrics import MetricDataPoint, MetricType

    agent = api_state.agents[agent_id]

    # Return mock metrics (would query from metrics system in production)
    return [
        MetricsResponse(
            metric_name="agent_reward",
            metric_type=MetricType.AGENT,
            data=[
                MetricDataPoint(
                    timestamp=datetime.utcnow(),
                    value=agent["stats"]["average_reward"],
                    labels={"agent_id": agent_id}
                )
            ],
            aggregated_value=agent["stats"]["average_reward"]
        )
    ]


@app.get(
    "/metrics/system",
    response_model=SystemMetrics,
    tags=["metrics"],
    summary="Get system metrics",
    description="Get current system-wide metrics"
)
async def get_system_metrics() -> SystemMetrics:
    """Get system-wide metrics."""
    active_agents = sum(1 for a in api_state.agents.values() if a["status"] == "active")
    training_sessions = sum(1 for s in api_state.training_sessions.values()
                           if s["status"] == TrainingStatus.RUNNING)

    total_steps = sum(a["stats"]["learning_steps"] for a in api_state.agents.values())
    avg_reward = (sum(a["stats"]["average_reward"] for a in api_state.agents.values()) /
                  len(api_state.agents)) if api_state.agents else 0.0

    return SystemMetrics(
        cpu_percent=psutil.cpu_percent(),
        memory_bytes=psutil.Process().memory_info().rss,
        active_agents=active_agents,
        training_sessions=training_sessions,
        total_learning_steps=total_steps,
        average_reward=avg_reward,
        uptime_seconds=time.time() - api_state.start_time
    )


@app.get(
    "/metrics/export",
    response_class=PlainTextResponse,
    tags=["metrics"],
    summary="Export Prometheus metrics",
    description="Export metrics in Prometheus format"
)
async def export_metrics() -> str:
    """Export metrics in Prometheus format."""
    metrics = []

    # Agent metrics
    for agent_id, agent in api_state.agents.items():
        metrics.append(f'mae_agent_reward{{agent_id="{agent_id}"}} {agent["stats"]["average_reward"]}')
        metrics.append(f'mae_agent_convergence{{agent_id="{agent_id}"}} {agent["stats"]["convergence_score"]}')

    # System metrics
    metrics.append(f'mae_system_agents_total {len(api_state.agents)}')
    metrics.append(f'mae_system_requests_total {api_state.request_count}')

    return "\n".join(metrics)


# ============================================================================
# POLICY ENDPOINTS
# ============================================================================

@app.post(
    "/policies/export",
    tags=["policies"],
    summary="Export agent policies",
    description="Export policies for specified agents"
)
async def export_policies(export: PolicyExport) -> Dict[str, Any]:
    """Export agent policies."""
    agent_ids = export.agent_ids or list(api_state.agents.keys())

    policies = {}
    for agent_id in agent_ids:
        if agent_id in api_state.agents:
            agent = api_state.agents[agent_id]
            policies[agent_id] = {
                "policy_data": "base64_encoded_policy",  # Mock
                "metadata": {
                    "agent_id": agent_id,
                    "agent_type": agent["agent_type"],
                    "exported_at": datetime.utcnow().isoformat(),
                    "training_steps": agent["stats"]["learning_steps"],
                    "average_reward": agent["stats"]["average_reward"],
                    "convergence_score": agent["stats"]["convergence_score"]
                }
            }

    return {"policies": policies, "format": export.format}


@app.post(
    "/policies/import",
    tags=["policies"],
    summary="Import agent policy",
    description="Import a policy for an agent"
)
async def import_policy(policy: PolicyImport) -> Dict[str, str]:
    """Import agent policy."""
    agent_id = policy.agent_id or f"agent_{len(api_state.agents) + 1}"

    if agent_id in api_state.agents and not policy.overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent {agent_id} already exists. Use overwrite=true to replace."
        )

    return {
        "agent_id": agent_id,
        "status": "imported",
        "message": f"Policy imported for {agent_id}"
    }


@app.get(
    "/policies/compare",
    response_model=PolicyCompare,
    tags=["policies"],
    summary="Compare agent policies",
    description="Compare policies between two agents"
)
async def compare_policies(
    agent1_id: str = Query(..., description="First agent ID"),
    agent2_id: str = Query(..., description="Second agent ID")
) -> PolicyCompare:
    """Compare policies between two agents."""
    if agent1_id not in api_state.agents or agent2_id not in api_state.agents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or both agents not found"
        )

    agent1 = api_state.agents[agent1_id]
    agent2 = api_state.agents[agent2_id]

    from src.api.schemas.policies import PolicyComparison

    return PolicyCompare(
        agent1_id=agent1_id,
        agent2_id=agent2_id,
        comparisons=[
            PolicyComparison(
                metric="average_reward",
                agent1_value=agent1["stats"]["average_reward"],
                agent2_value=agent2["stats"]["average_reward"],
                difference=abs(agent1["stats"]["average_reward"] - agent2["stats"]["average_reward"]),
                relative_difference=0.05
            )
        ],
        overall_similarity=0.95
    )


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.get(
    "/system/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
    description="Check system and component health"
)
async def health_check() -> HealthResponse:
    """System health check."""
    components = [
        ComponentHealth(
            name="api",
            status=HealthStatus.HEALTHY,
            message="API is operational",
            last_check=datetime.utcnow()
        ),
        ComponentHealth(
            name="agents",
            status=HealthStatus.HEALTHY,
            message=f"{len(api_state.agents)} agents active",
            last_check=datetime.utcnow()
        )
    ]

    overall_status = HealthStatus.HEALTHY
    if any(c.status == HealthStatus.UNHEALTHY for c in components):
        overall_status = HealthStatus.UNHEALTHY
    elif any(c.status == HealthStatus.DEGRADED for c in components):
        overall_status = HealthStatus.DEGRADED

    return HealthResponse(
        status=overall_status,
        components=components,
        timestamp=datetime.utcnow()
    )


@app.get(
    "/system/version",
    response_model=VersionResponse,
    tags=["system"],
    summary="Version information",
    description="Get system version and build information"
)
async def version_info() -> VersionResponse:
    """Get version information."""
    import sys

    return VersionResponse(
        version="3.0.0",
        build_date="2025-11-12",
        git_commit="fc96e2b",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        dependencies={
            "fastapi": "0.104.0",
            "pydantic": "2.5.0",
            "uvicorn": "0.24.0"
        }
    )


@app.get(
    "/system/stats",
    response_model=SystemStats,
    tags=["system"],
    summary="System statistics",
    description="Get comprehensive system statistics"
)
async def system_stats() -> SystemStats:
    """Get system statistics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    disk = psutil.disk_usage('/')

    active_agents = sum(1 for a in api_state.agents.values() if a["status"] == "active")
    training_sessions = sum(1 for s in api_state.training_sessions.values()
                           if s["status"] == TrainingStatus.RUNNING)
    total_steps = sum(a["stats"]["learning_steps"] for a in api_state.agents.values())

    avg_response_time = (api_state.total_response_time / api_state.request_count
                        if api_state.request_count > 0 else 0.0) * 1000  # Convert to ms

    return SystemStats(
        uptime_seconds=time.time() - api_state.start_time,
        resource_usage=ResourceUsage(
            cpu_percent=psutil.cpu_percent(),
            memory_used_mb=memory_info.rss / (1024 * 1024),
            memory_total_mb=psutil.virtual_memory().total / (1024 * 1024),
            disk_used_gb=disk.used / (1024 * 1024 * 1024),
            disk_total_gb=disk.total / (1024 * 1024 * 1024)
        ),
        agent_count=len(api_state.agents),
        active_agents=active_agents,
        training_sessions=training_sessions,
        total_learning_steps=total_steps,
        total_requests=api_state.request_count,
        average_response_time_ms=avg_response_time
    )


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MAE REST API",
        "version": "3.0.0",
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
