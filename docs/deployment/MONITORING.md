# MAE Monitoring Integration Guide

Integration guide for deploying the monitoring stack (Big Rock 12) with MAE in production.

## Overview

MAE includes a comprehensive monitoring stack built in Big Rock 12:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **Alert Manager**: Alert routing and notifications

---

## Quick Start

### Docker Compose (Development)

```bash
# Start MAE with monitoring
docker-compose --profile monitoring up -d

# Access dashboards
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
open http://localhost:16686  # Jaeger
```

### Kubernetes (Production)

```bash
# Deploy MAE with monitoring enabled
helm install mae ./helm/mae-chart \
  --namespace mae-prod \
  --set monitoring.prometheus.enabled=true \
  --set monitoring.grafana.enabled=true \
  --set monitoring.jaeger.enabled=true

# Access via port-forward
kubectl port-forward -n mae-prod svc/grafana 3000:3000
```

---

## Prometheus Setup

### ServiceMonitor

Create ServiceMonitor for automatic scraping:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mae-api
  namespace: mae-prod
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: mae
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
```

### Prometheus Rules

Alert rules for MAE:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: mae-alerts
  namespace: mae-prod
spec:
  groups:
  - name: mae
    interval: 30s
    rules:
    # Agent Performance
    - alert: AgentNotConverging
      expr: agent_convergence_status == 0
      for: 30m
      labels:
        severity: warning
      annotations:
        summary: "Agent {{ $labels.agent_id }} not converging"

    - alert: LowAgentSatisfaction
      expr: agent_satisfaction_score < 0.3
      for: 15m
      labels:
        severity: warning
      annotations:
        summary: "Agent {{ $labels.agent_id }} has low satisfaction"

    # System Health
    - alert: HighCPUUsage
      expr: system_cpu_percent > 80
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High CPU usage: {{ $value }}%"

    - alert: HighMemoryUsage
      expr: system_memory_percent > 85
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High memory usage: {{ $value }}%"

    - alert: HighErrorRate
      expr: rate(api_requests_failed_total[5m]) > 0.05
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High API error rate: {{ $value }}"
```

---

## Grafana Dashboards

### Installation

The monitoring stack includes 4 pre-built dashboards:

1. **Agent Performance** (`monitoring/grafana/dashboards/agent_performance.json`)
2. **System Health** (`monitoring/grafana/dashboards/system_health.json`)
3. **Communication Metrics** (`monitoring/grafana/dashboards/communication_metrics.json`)
4. **Memory Subsystem** (`monitoring/grafana/dashboards/memory_subsystem.json`)

### Import Dashboards in Kubernetes

```bash
# Create ConfigMap with dashboards
kubectl create configmap mae-dashboards \
  --from-file=monitoring/grafana/dashboards/ \
  -n mae-prod

# Update Grafana deployment to mount dashboards
kubectl patch deployment grafana -n mae-prod -p '
spec:
  template:
    spec:
      containers:
      - name: grafana
        volumeMounts:
        - name: dashboards
          mountPath: /etc/grafana/provisioning/dashboards
      volumes:
      - name: dashboards
        configMap:
          name: mae-dashboards
'
```

### Dashboard Configuration

Configure Grafana to auto-provision:

```yaml
# grafana-dashboard-provider.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-provider
  namespace: mae-prod
data:
  dashboards.yaml: |
    apiVersion: 1
    providers:
    - name: 'MAE Dashboards'
      orgId: 1
      folder: 'MAE'
      type: file
      disableDeletion: false
      updateIntervalSeconds: 10
      allowUiUpdates: true
      options:
        path: /etc/grafana/provisioning/dashboards
```

---

## Jaeger Tracing

### Configuration

Enable tracing in MAE:

```yaml
# values.yaml
monitoring:
  jaeger:
    enabled: true
    endpoint: "http://jaeger-collector:14268/api/traces"
    samplingRate: 0.1  # 10% sampling
```

### Viewing Traces

```bash
# Port-forward Jaeger UI
kubectl port-forward -n mae-prod svc/jaeger-query 16686:16686

# Open in browser
open http://localhost:16686
```

### Trace Search

Common searches:
- Service: `mae-api`
- Operation: `agent_step`, `policy_sharing`, `credit_assignment`
- Tags: `agent_id`, `error=true`, `http.status_code=500`

---

## Alert Manager

### Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: mae-prod
data:
  alertmanager.yml: |
    global:
      resolve_timeout: 5m
      slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

    route:
      group_by: ['alertname', 'cluster']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 12h
      receiver: 'slack'
      routes:
      - match:
          severity: critical
        receiver: 'pagerduty'
      - match:
          severity: warning
        receiver: 'slack'

    receivers:
    - name: 'slack'
      slack_configs:
      - channel: '#mae-alerts'
        title: 'MAE Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}'

    - name: 'pagerduty'
      pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

### Testing Alerts

```bash
# Trigger test alert
kubectl exec -it -n mae-prod deployment/prometheus -- \
  curl -X POST http://localhost:9090/-/reload

# Send test notification
kubectl exec -it -n mae-prod deployment/alertmanager -- \
  amtool alert add test_alert
```

---

## Metrics Reference

### Agent Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `agent_learning_rate` | Gauge | Current learning rate |
| `agent_reward_total` | Counter | Cumulative reward |
| `agent_episode_length` | Histogram | Episode duration |
| `agent_convergence_status` | Gauge | Convergence indicator (0/1) |
| `agent_satisfaction_score` | Gauge | Agent satisfaction (0-1) |

### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `system_cpu_percent` | Gauge | CPU utilization % |
| `system_memory_percent` | Gauge | Memory utilization % |
| `system_network_bytes_sent` | Counter | Network bytes sent |
| `system_network_bytes_recv` | Counter | Network bytes received |

### Communication Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `comm_messages_sent_total` | Counter | Messages sent |
| `comm_messages_received_total` | Counter | Messages received |
| `comm_message_latency_seconds` | Histogram | Message latency |
| `comm_routing_efficiency` | Gauge | Routing efficiency (0-1) |

### Memory Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `memory_buffer_size` | Gauge | Replay buffer size |
| `memory_buffer_utilization` | Gauge | Buffer utilization % |
| `memory_samples_total` | Counter | Total samples stored |
| `memory_replay_batch_size` | Histogram | Replay batch sizes |

---

## Custom Queries

### PromQL Examples

```promql
# Average reward per agent
avg(agent_reward_total) by (agent_id)

# Request rate (5-minute average)
rate(api_requests_total[5m])

# Error rate
rate(api_requests_failed_total[5m]) / rate(api_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, api_request_duration_seconds_bucket)

# Memory buffer pressure
memory_buffer_utilization > 0.9

# Top 5 agents by reward
topk(5, agent_reward_total)
```

---

## Performance Tuning

### Prometheus

```yaml
# Increase retention
prometheus:
  retention: 30d
  retentionSize: 50GB

# Adjust scrape interval
scrapeInterval: 15s  # Default
# Use 30s for large deployments to reduce load
```

### Grafana

```yaml
# Enable caching
grafana:
  config:
    caching:
      enabled: true
      backend: redis

# Increase query timeout
    explore:
      query_timeout: 60s
```

---

## Troubleshooting

### Prometheus Not Scraping

```bash
# Check ServiceMonitor
kubectl get servicemonitor -n mae-prod

# Check Prometheus targets
kubectl port-forward -n mae-prod svc/prometheus 9090:9090
# Visit http://localhost:9090/targets

# Check pod labels
kubectl get pods -n mae-prod --show-labels
```

### Grafana Dashboards Not Loading

```bash
# Check ConfigMap
kubectl get configmap mae-dashboards -n mae-prod

# Check Grafana logs
kubectl logs -n mae-prod deployment/grafana

# Restart Grafana
kubectl rollout restart deployment/grafana -n mae-prod
```

### Missing Metrics

```bash
# Check if metrics endpoint is accessible
kubectl exec -it -n mae-prod deployment/mae-api -- \
  curl http://localhost:8080/metrics

# Check Prometheus configuration
kubectl exec -it -n mae-prod deployment/prometheus -- \
  cat /etc/prometheus/prometheus.yml
```

---

## Best Practices

1. **Use appropriate scrape intervals**: 15s for production, 30s for cost optimization
2. **Set retention policies**: Balance storage cost vs. historical data needs
3. **Configure alert thresholds**: Tune based on normal operating ranges
4. **Use recording rules**: Pre-compute expensive queries
5. **Enable high availability**: Run multiple Prometheus replicas
6. **Implement federation**: For multi-cluster deployments
7. **Regular backups**: Backup Prometheus data and Grafana dashboards
8. **Monitor the monitoring**: Alert on Prometheus/Grafana issues

---

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
