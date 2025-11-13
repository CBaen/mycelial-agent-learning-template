# MAE Helm Chart

This Helm chart deploys the Mycelial Agent Engine (MAE) to a Kubernetes cluster.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.0+
- PV provisioner support in the underlying infrastructure (for persistent volumes)
- NGINX Ingress Controller (if using Ingress)
- Cert-Manager (if using TLS)

## Installation

### Quick Start

```bash
# Install with default values
helm install mae ./helm/mae-chart --namespace mae --create-namespace

# Or using Make
make helm-install
```

### Custom Configuration

Create a `custom-values.yaml` file:

```yaml
api:
  replicaCount: 3
  resources:
    requests:
      memory: "1Gi"
      cpu: "1000m"

security:
  secretKey: "your-generated-secret-key-here"

ingress:
  enabled: true
  hosts:
    - host: api.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
```

Install with custom values:

```bash
helm install mae ./helm/mae-chart \
  --namespace mae \
  --create-namespace \
  -f custom-values.yaml
```

## Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api.replicaCount` | Number of API pod replicas | `2` |
| `api.image.repository` | API container image repository | `mae-api` |
| `api.image.tag` | API container image tag | `latest` |
| `api.resources.requests.memory` | Memory request for API pods | `512Mi` |
| `api.resources.requests.cpu` | CPU request for API pods | `500m` |
| `api.resources.limits.memory` | Memory limit for API pods | `2Gi` |
| `api.resources.limits.cpu` | CPU limit for API pods | `2000m` |
| `api.autoscaling.enabled` | Enable horizontal pod autoscaling | `true` |
| `api.autoscaling.minReplicas` | Minimum number of replicas | `2` |
| `api.autoscaling.maxReplicas` | Maximum number of replicas | `10` |
| `redis.enabled` | Deploy Redis | `true` |
| `redis.persistence.enabled` | Enable Redis persistence | `true` |
| `redis.persistence.size` | Redis PVC size | `10Gi` |
| `chromadb.enabled` | Deploy ChromaDB | `true` |
| `chromadb.persistence.enabled` | Enable ChromaDB persistence | `true` |
| `chromadb.persistence.size` | ChromaDB PVC size | `20Gi` |
| `ingress.enabled` | Enable Ingress | `true` |
| `ingress.className` | Ingress class name | `nginx` |
| `ingress.hosts[0].host` | Hostname for API access | `api.mae.example.com` |
| `security.secretKey` | JWT secret key (CHANGE IN PRODUCTION) | `change-me-in-production...` |
| `security.jwt.algorithm` | JWT algorithm | `HS256` |
| `security.serviceAccount.create` | Create service account | `true` |
| `security.rbac.create` | Create RBAC resources | `true` |
| `monitoring.prometheus.enabled` | Enable Prometheus metrics | `true` |

### Full Configuration Reference

See `values.yaml` for all available parameters.

## Upgrading

### Upgrade Release

```bash
# Upgrade with new values
helm upgrade mae ./helm/mae-chart -f custom-values.yaml

# Or using Make
make helm-upgrade
```

### View Differences

```bash
# Show what would change
helm diff upgrade mae ./helm/mae-chart -f custom-values.yaml
```

## Uninstallation

```bash
# Uninstall the release
helm uninstall mae --namespace mae

# Or using Make
make helm-uninstall
```

To delete persistent data:

```bash
kubectl delete pvc -n mae --all
```

## Usage Examples

### Production Deployment

```yaml
# production-values.yaml
api:
  replicaCount: 5
  resources:
    requests:
      memory: "1Gi"
      cpu: "1000m"
    limits:
      memory: "4Gi"
      cpu: "4000m"
  autoscaling:
    minReplicas: 5
    maxReplicas: 20

security:
  secretKey: "your-strong-random-secret-key-32-bytes"

ingress:
  enabled: true
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: api.production.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mae-prod-tls
      hosts:
        - api.production.com

redis:
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
  persistence:
    size: 50Gi

chromadb:
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "2000m"
  persistence:
    size: 100Gi

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000

networkPolicy:
  enabled: true
```

Deploy:

```bash
helm install mae ./helm/mae-chart \
  --namespace mae-prod \
  --create-namespace \
  -f production-values.yaml
```

### Development Deployment

```yaml
# dev-values.yaml
api:
  replicaCount: 1
  image:
    tag: dev
    pullPolicy: Always
  autoscaling:
    enabled: false

redis:
  persistence:
    enabled: false

chromadb:
  persistence:
    enabled: false

ingress:
  enabled: false

monitoring:
  prometheus:
    enabled: false
```

Deploy:

```bash
helm install mae-dev ./helm/mae-chart \
  --namespace mae-dev \
  --create-namespace \
  -f dev-values.yaml
```

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics` on port 8080.

To scrape metrics, add this annotation to pods:

```yaml
podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8080"
  prometheus.io/path: "/metrics"
```

### View Metrics

```bash
# Port-forward to Prometheus (if deployed)
kubectl port-forward -n mae svc/prometheus 9090:9090

# Open http://localhost:9090
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n mae
kubectl describe pod <pod-name> -n mae
kubectl logs <pod-name> -n mae
```

### Check Services

```bash
kubectl get svc -n mae
kubectl describe svc mae-api -n mae
```

### Check Ingress

```bash
kubectl get ingress -n mae
kubectl describe ingress mae-ingress -n mae
```

### Check PVCs

```bash
kubectl get pvc -n mae
kubectl describe pvc <pvc-name> -n mae
```

### Common Issues

#### Pods not starting

1. Check resource availability:
   ```bash
   kubectl describe nodes
   ```

2. Check events:
   ```bash
   kubectl get events -n mae --sort-by='.lastTimestamp'
   ```

#### Ingress not working

1. Verify Ingress controller is installed:
   ```bash
   kubectl get pods -n ingress-nginx
   ```

2. Check Ingress configuration:
   ```bash
   kubectl describe ingress -n mae
   ```

#### Database connection issues

1. Verify Redis is running:
   ```bash
   kubectl exec -it <api-pod> -n mae -- nc -zv redis-service 6379
   ```

2. Verify ChromaDB is running:
   ```bash
   kubectl exec -it <api-pod> -n mae -- curl http://chromadb-service:8000/api/v1/heartbeat
   ```

## Testing the Chart

### Lint the Chart

```bash
helm lint ./helm/mae-chart
```

### Template Output

```bash
# See what would be deployed
helm template mae ./helm/mae-chart

# With custom values
helm template mae ./helm/mae-chart -f custom-values.yaml
```

### Dry Run

```bash
helm install mae ./helm/mae-chart --dry-run --debug
```

## Advanced Configuration

### Using External Redis

Disable built-in Redis and point to external instance:

```yaml
redis:
  enabled: false

api:
  env:
    REDIS_HOST: "external-redis.example.com"
    REDIS_PORT: "6379"
    REDIS_PASSWORD: "password"
```

### Using External Vector Database

```yaml
chromadb:
  enabled: false

api:
  env:
    VECTOR_DB_BACKEND: "milvus"
    MILVUS_HOST: "external-milvus.example.com"
    MILVUS_PORT: "19530"
```

### Custom Init Containers

```yaml
initContainers:
  - name: wait-for-redis
    image: busybox:1.36
    command:
      - sh
      - -c
      - |
        until nc -z redis-service 6379; do
          echo "Waiting for Redis..."
          sleep 2
        done
```

### Custom Sidecar Containers

```yaml
sidecarContainers:
  - name: log-forwarder
    image: fluent/fluent-bit:latest
    volumeMounts:
      - name: logs
        mountPath: /app/logs
```

## Development

### Local Testing with Kind

```bash
# Create Kind cluster
kind create cluster --name mae-test

# Build and load image
docker build -f docker/Dockerfile.api -t mae-api:dev .
kind load docker-image mae-api:dev --name mae-test

# Install chart
helm install mae ./helm/mae-chart \
  --set api.image.tag=dev \
  --set api.image.pullPolicy=Never
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.

## License

See [LICENSE](../../LICENSE) for license information.
