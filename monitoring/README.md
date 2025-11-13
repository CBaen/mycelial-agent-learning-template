# MAE Observability Stack

Comprehensive monitoring, logging, and tracing infrastructure for the Mycelial Agent Environment.

## Overview

This monitoring stack provides production-grade observability through:
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Metrics visualization with pre-built dashboards
- **Jaeger**: Distributed tracing for multi-agent workflows
- **Alertmanager**: Alert routing and notification management

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Port availability: 3000 (Grafana), 9090 (Prometheus), 16686 (Jaeger), 9093 (Alertmanager)

### Starting the Stack

```bash
cd monitoring
docker-compose up -d
```

### Accessing Services

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | N/A |
| Jaeger UI | http://localhost:16686 | N/A |
| Alertmanager | http://localhost:9093 | N/A |

### Stopping the Stack

```bash
docker-compose down
```

To remove all data volumes:
```bash
docker-compose down -v
```

## Dashboards

### 1. Agent Performance Dashboard
**Path**: `grafana/dashboards/agent_performance.json`

Monitors agent learning and performance metrics:
- **Learning Rate**: Current learning rates per agent
- **Rewards Distribution**: Average reward trends
- **Learning Steps**: Steps per second throughput
- **Convergence Score**: Agent convergence progress (0-1)
- **Satisfaction Score**: Agent satisfaction levels (Big Rock 4)
- **Experience Points**: Gamification XP tracking
- **Top Performers**: Agents ranked by average reward
- **Summary Stats**: Agent count, total steps, averages

**Key Metrics**:
- `mae_agent_learning_rate`
- `mae_agent_reward`
- `mae_agent_learning_steps_total`
- `mae_agent_convergence_score`
- `mae_agent_satisfaction`
- `mae_agent_xp`

### 2. System Health Dashboard
**Path**: `grafana/dashboards/system_health.json`

Monitors system resource utilization and health:
- **CPU Usage**: Real-time CPU percentage
- **Memory Usage**: Memory consumption in MB
- **Request Latency**: p50, p95, p99 latency percentiles
- **Error Rate**: Errors per second by type
- **Request Throughput**: Requests per second by operation
- **Error Distribution**: Pie chart of error types
- **System Uptime**: Total uptime in seconds
- **Summary Stats**: Total errors, avg CPU/memory

**Key Metrics**:
- `mae_system_cpu_percent`
- `mae_system_memory_bytes`
- `mae_system_request_latency_seconds`
- `mae_system_errors_total`

**Alerts**:
- High/Critical CPU usage (80%/95%)
- High memory usage (>8GB)
- High/Critical error rate (>1/sec, >10/sec)
- High request latency (p99 >500ms)

### 3. Communication Metrics Dashboard
**Path**: `grafana/dashboards/communication_metrics.json`

Monitors inter-agent communication:
- **Messages Sent/Received**: Rate by type and agent
- **Message Latency**: p50, p95, p99 by message type
- **GNN Routing Efficiency**: Routing effectiveness (0-1)
- **Message Volume**: Breakdown by BROADCAST/UNICAST/MULTICAST
- **Top Communicators**: Most active agents
- **Summary Stats**: Total messages, avg latency, current efficiency

**Key Metrics**:
- `mae_communication_messages_sent_total`
- `mae_communication_messages_received_total`
- `mae_communication_message_latency_seconds`
- `mae_communication_gnn_routing_efficiency`

**Alerts**:
- High message latency (p99 >10ms)
- Low/Critical routing efficiency (<60%/<40%)
- No messages flowing (0 msg/sec for 2min)

### 4. Memory Subsystem Dashboard
**Path**: `grafana/dashboards/memory_subsystem.json`

Monitors episodic memory performance:
- **Buffer Utilization**: Size/capacity ratio per agent
- **Buffer Size vs Capacity**: Absolute values
- **Replay Operations**: Memory replay rate
- **Consolidation Operations**: Consolidation frequency
- **Utilization Heatmap**: Visual buffer pressure overview
- **Operations by Agent**: Table of replay/consolidation rates
- **Summary Stats**: Total operations, avg utilization, full buffers

**Key Metrics**:
- `mae_memory_episodic_buffer_size`
- `mae_memory_episodic_buffer_capacity`
- `mae_memory_replay_total`
- `mae_memory_consolidation_total`

**Alerts**:
- Buffer nearly full (>90% for 5min)
- Buffer full (100% for 2min)
- No replay activity (0 replays for 5min)
- High memory pressure (>5 agents >80%)

## Alerting Rules

### Alert Severities
- **Critical**: Immediate attention required, system degradation
- **Warning**: Potential issue, monitor closely

### Alert Groups

#### Agent Alerts
- `AgentLowConvergence`: Convergence <30% for 5min
- `AgentLowSatisfaction`: Satisfaction <50% for 5min
- `AgentStoppedLearning`: No learning steps for 2min

#### System Alerts
- `HighCPUUsage`: CPU >80% for 5min (warning)
- `CriticalCPUUsage`: CPU >95% for 2min (critical)
- `HighMemoryUsage`: Memory >8GB for 5min
- `HighErrorRate`: >1 error/sec for 2min (warning)
- `CriticalErrorRate`: >10 errors/sec for 1min (critical)
- `HighRequestLatency`: p99 >500ms for 5min

#### Communication Alerts
- `HighMessageLatency`: p99 >10ms for 5min
- `LowRoutingEfficiency`: Efficiency <60% for 5min (warning)
- `CriticalRoutingEfficiency`: Efficiency <40% for 2min (critical)
- `NoMessagesFlowing`: 0 msg/sec for 2min

#### Memory Alerts
- `BufferNearlyFull`: >90% utilization for 5min
- `BufferFull`: 100% utilization for 2min
- `NoReplayActivity`: No replays for 5min
- `HighMemoryPressure`: >5 agents >80% for 5min

## Integration with MAE Application

### Exposing Metrics

```python
from src.observability.metrics import MetricsCollector, MetricsConfig

# Initialize collector
collector = MetricsCollector(
    namespace="mae",
    subsystem="agents"
)

# Record metrics
collector.record_agent_reward("agent_1", 10.5)
collector.record_learning_step("agent_1", 0.001)

# Expose metrics endpoint (FastAPI example)
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/metrics")
def metrics():
    metrics_data = collector.export_metrics()
    return Response(
        content=metrics_data,
        media_type="text/plain"
    )
```

### Structured Logging

```python
from src.observability.structured_logger import get_logger

logger = get_logger("mae.agents")

# Log with correlation ID
with logger.correlation_id() as cid:
    logger.info("Agent started", agent_id="agent_1")
    # All logs in this context share the correlation ID
```

### Distributed Tracing

```python
from src.observability.tracing import TracingProvider, TracingConfig, ExporterType

# Initialize tracing
provider = TracingProvider(config=TracingConfig(
    service_name="mae.agents",
    exporter_type=ExporterType.JAEGER,
    jaeger_endpoint="http://localhost:14268/api/traces"
))

# Create spans
with provider.agent_span("agent_1", "learn", step=1):
    # Agent learning logic
    pass
```

## Architecture

```
┌─────────────┐
│   MAE App   │──┐
│  (Port 8000)│  │ Metrics via /metrics endpoint
└─────────────┘  │
                 ▼
            ┌─────────────┐     ┌─────────────┐
            │ Prometheus  │────▶│ Alertmanager│
            │  (Port 9090)│     │  (Port 9093)│
            └─────────────┘     └─────────────┘
                 │
                 │ Data source
                 ▼
            ┌─────────────┐
            │   Grafana   │
            │  (Port 3000)│
            └─────────────┘

            ┌─────────────┐
            │    Jaeger   │◀──── Traces from MAE App
            │ (Port 16686)│
            └─────────────┘
```

## Configuration

### Prometheus
- **Config**: `prometheus/prometheus.yml`
- **Alert Rules**: `prometheus/alert_rules.yml`
- **Scrape Interval**: 5 seconds
- **Evaluation Interval**: 5 seconds

### Grafana
- **Provisioning**: `grafana/provisioning/`
- **Dashboards**: `grafana/dashboards/`
- **Data Sources**: Auto-configured Prometheus
- **Refresh**: 5 seconds

### Jaeger
- **UI Port**: 16686
- **Collector Port**: 14268
- **Agent Port**: 6831 (UDP)

## Troubleshooting

### Dashboards Not Appearing
1. Check Grafana logs: `docker logs mae-grafana`
2. Verify dashboard files in `grafana/dashboards/`
3. Restart Grafana: `docker-compose restart grafana`

### No Metrics in Prometheus
1. Check MAE app is exposing `/metrics` endpoint
2. Verify Prometheus can reach app: `http://localhost:9090/targets`
3. Check Prometheus logs: `docker logs mae-prometheus`

### Alerts Not Firing
1. Verify alert rules: `http://localhost:9090/alerts`
2. Check Alertmanager status: `http://localhost:9093`
3. Review alert rule syntax in `prometheus/alert_rules.yml`

### High Resource Usage
1. Reduce scrape frequency in `prometheus.yml`
2. Decrease dashboard refresh rates
3. Limit retention period: Add `--storage.tsdb.retention.time=7d` to Prometheus command

## Performance Targets

| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| Metrics Collection | <100ms overhead | N/A |
| Metrics Throughput | >10,000 metrics/sec | N/A |
| Metric Accuracy | >99.9% | N/A |
| Alert Latency | <30s | N/A |
| Dashboard Load Time | <2s | N/A |
| Trace Overhead | <5ms per span | N/A |

## Maintenance

### Backup Dashboards
```bash
docker exec mae-grafana grafana-cli admin export-dashboard > backup.json
```

### Update Alert Rules
1. Edit `prometheus/alert_rules.yml`
2. Reload Prometheus: `curl -X POST http://localhost:9090/-/reload`

### Clean Old Data
```bash
# Remove Prometheus data older than 7 days (auto-configured)
# Manual cleanup if needed:
docker exec mae-prometheus promtool tsdb cleanup /prometheus
```

## Support

For issues or questions:
1. Check logs: `docker-compose logs [service-name]`
2. Review configuration files
3. Consult MAE documentation
4. Report issues: https://github.com/anthropics/claude-code/issues
