# MAE Deployment Guide

Complete guide for deploying the Mycelial Agent Engine (MAE) to various environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Production Deployment](#production-deployment)
- [Monitoring Setup](#monitoring-setup)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Local Development (Docker Compose)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/mae.git
cd mae

# 2. Set up environment
cp .env.example .env
# Edit .env with your configuration

# 3. Start services
make docker-up

# 4. Verify deployment
make docker-logs
curl http://localhost:8080/system/health
```

### Kubernetes (Helm)

```bash
# 1. Install with Helm
helm install mae ./helm/mae-chart \
  --namespace mae \
  --create-namespace

# 2. Check status
kubectl get pods -n mae

# 3. Access API
kubectl port-forward -n mae svc/mae-api 8080:8080
```

---

## Deployment Options

| Environment | Tool | Persistence | Monitoring | Autoscaling | Use Case |
|-------------|------|-------------|------------|-------------|----------|
| Local Dev | Docker Compose | Optional | Optional | No | Development, Testing |
| Staging | Kubernetes + Helm | Yes | Yes | Yes | Pre-production Testing |
| Production | Kubernetes + Helm | Yes | Yes | Yes | Live System |

---

## Prerequisites

### All Deployments
- Docker 24.0+
- Git

### Kubernetes Deployments
- kubectl 1.24+
- Helm 3.0+
- Kubernetes cluster (1.24+)
- Persistent volume provisioner

### Optional
- NGINX Ingress Controller (for external access)
- Cert-Manager (for TLS certificates)
- Prometheus + Grafana (for monitoring)

---

## Local Development

### Using Docker Compose

#### 1. Basic Setup

```bash
# Start core services only
docker-compose up -d

# Start with monitoring stack
docker-compose --profile monitoring up -d

# Start with debug tools
docker-compose --profile debug up -d
```

#### 2. Configuration

Edit `.env` file:

```env
# API Configuration
API_WORKERS=4
LOG_LEVEL=INFO

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# ChromaDB Configuration
CHROMA_SERVER_HOST=chromadb
CHROMA_SERVER_HTTP_PORT=8000

# Security
SECRET_KEY=your-development-secret-key-here
```

#### 3. Access Services

- **API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
- **Prometheus**: http://localhost:9090 (with monitoring profile)
- **Grafana**: http://localhost:3000 (with monitoring profile)
  - Default login: admin/admin
- **Jaeger**: http://localhost:16686 (with monitoring profile)
- **Redis Commander**: http://localhost:8081 (with debug profile)

#### 4. Development Workflow

```bash
# View logs
make docker-logs

# Restart services
make docker-restart

# Run tests in container
make docker-test

# Stop all services
make docker-down

# Clean up everything
make docker-clean
```

---

## Docker Deployment

### Building Images

```bash
# Build API image
make docker-build

# Build with custom tag
docker build -f docker/Dockerfile.api -t mae-api:v1.0.0 .

# Multi-platform build
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile.api \
  -t mae-api:latest \
  --push .
```

### Running Standalone Containers

```bash
# Run Redis
docker run -d \
  --name mae-redis \
  -p 6379:6379 \
  -v mae-redis-data:/data \
  redis:7-alpine

# Run ChromaDB
docker run -d \
  --name mae-chromadb \
  -p 8000:8000 \
  -v mae-chroma-data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  chromadb/chroma:latest

# Run MAE API
docker run -d \
  --name mae-api \
  -p 8080:8080 \
  --env-file .env \
  --link mae-redis:redis \
  --link mae-chromadb:chromadb \
  mae-api:latest
```

---

## Kubernetes Deployment

### Using Helm (Recommended)

#### 1. Install MAE

```bash
# Basic installation
helm install mae ./helm/mae-chart \
  --namespace mae \
  --create-namespace

# Custom values
helm install mae ./helm/mae-chart \
  --namespace mae \
  --create-namespace \
  -f custom-values.yaml

# Development environment
helm install mae-dev ./helm/mae-chart \
  --namespace mae-dev \
  --create-namespace \
  --set api.replicaCount=1 \
  --set api.autoscaling.enabled=false \
  --set redis.persistence.enabled=false \
  --set chromadb.persistence.enabled=false
```

#### 2. Configuration

Create `custom-values.yaml`:

```yaml
# API Configuration
api:
  replicaCount: 3
  image:
    repository: mae-api
    tag: v1.0.0
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "2000m"
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10

# Redis Configuration
redis:
  persistence:
    enabled: true
    size: 10Gi
  resources:
    requests:
      memory: "512Mi"
      cpu: "250m"

# ChromaDB Configuration
chromadb:
  persistence:
    enabled: true
    size: 20Gi
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"

# Ingress Configuration
ingress:
  enabled: true
  hosts:
    - host: api.mae.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mae-tls-cert
      hosts:
        - api.mae.example.com

# Security
security:
  secretKey: "your-production-secret-key-here"
```

#### 3. Upgrade Deployment

```bash
# Upgrade with new values
helm upgrade mae ./helm/mae-chart \
  -f custom-values.yaml

# Rollback to previous version
helm rollback mae

# View release history
helm history mae
```

#### 4. Uninstall

```bash
# Uninstall release
helm uninstall mae --namespace mae

# Delete namespace and all resources
kubectl delete namespace mae
```

### Using Raw Manifests

```bash
# Apply all manifests
kubectl apply -f k8s/

# Apply specific components
kubectl apply -f k8s/deployment-api.yaml
kubectl apply -f k8s/service-api.yaml

# Delete all resources
kubectl delete -f k8s/
```

---

## Production Deployment

### Prerequisites

1. **Kubernetes Cluster**: EKS, GKE, AKS, or self-managed
2. **Persistent Storage**: StorageClass configured
3. **Ingress Controller**: NGINX Ingress installed
4. **TLS Certificates**: Cert-Manager or manual certs
5. **Secrets Management**: External secrets or sealed secrets

### Production Checklist

- [ ] Configure persistent storage for Redis and ChromaDB
- [ ] Set strong `SECRET_KEY` for JWT authentication
- [ ] Configure TLS certificates for Ingress
- [ ] Set up horizontal pod autoscaling
- [ ] Configure resource requests and limits
- [ ] Enable monitoring (Prometheus + Grafana)
- [ ] Set up alerting rules
- [ ] Configure backup strategy
- [ ] Test disaster recovery procedures
- [ ] Document runbooks
- [ ] Set up log aggregation
- [ ] Configure network policies
- [ ] Enable pod security policies
- [ ] Set up CI/CD pipeline

### Production Values Example

See [production-values.yaml](./production-values.yaml) for complete configuration.

### Blue-Green Deployment

```bash
# Deploy green version
helm install mae-green ./helm/mae-chart \
  --namespace mae-prod \
  --set podLabels.color=green \
  --set api.image.tag=v2.0.0

# Test green deployment
kubectl run -it --rm test --image=curlimages/curl --restart=Never \
  -- curl http://mae-green-api:8080/system/health

# Switch traffic
kubectl patch service mae-api -n mae-prod \
  -p '{"spec":{"selector":{"color":"green"}}}'

# Scale down blue
kubectl scale deployment mae-blue-api -n mae-prod --replicas=0
```

### Canary Deployment

```bash
# Deploy canary with 10% traffic
helm install mae-canary ./helm/mae-chart \
  --namespace mae-prod \
  --set api.replicaCount=1 \
  --set podLabels.version=canary

# Monitor metrics for 30 minutes
# If successful, gradually increase canary traffic

# Full rollout
helm upgrade mae ./helm/mae-chart \
  --set api.image.tag=v2.0.0
```

---

## Monitoring Setup

See [MONITORING.md](./MONITORING.md) for complete monitoring setup guide.

### Quick Setup

```bash
# Deploy monitoring stack
kubectl apply -f monitoring/kubernetes/

# Or with Docker Compose
docker-compose -f monitoring/docker-compose.yml up -d

# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000
# Open http://localhost:3000 (admin/admin)
```

### Pre-built Dashboards

1. **Agent Performance**: Learning metrics, rewards, convergence
2. **System Health**: CPU, memory, latency, errors
3. **Communication**: Message flow, routing efficiency
4. **Memory Subsystem**: Buffer utilization, replay activity

---

## Security

### Secrets Management

#### Using Kubernetes Secrets

```bash
# Create secret for JWT key
kubectl create secret generic mae-secrets \
  --from-literal=secret-key=$(python -c "import secrets; print(secrets.token_hex(32))") \
  -n mae

# Update Helm values
helm upgrade mae ./helm/mae-chart \
  --set security.secretKey="" \
  --set security.existingSecret=mae-secrets
```

#### Using External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: mae-secrets
  namespace: mae
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: mae-secrets
  data:
  - secretKey: secret-key
    remoteRef:
      key: mae/production/secret-key
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mae-network-policy
  namespace: mae
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: mae
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: chromadb
    ports:
    - protocol: TCP
      port: 8000
```

---

## Troubleshooting

### Common Issues

#### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n mae

# Describe pod for events
kubectl describe pod <pod-name> -n mae

# Check logs
kubectl logs <pod-name> -n mae

# Check resource availability
kubectl describe nodes
```

#### Database Connection Issues

```bash
# Test Redis connection
kubectl exec -it <api-pod> -n mae -- redis-cli -h redis-service ping

# Test ChromaDB connection
kubectl exec -it <api-pod> -n mae -- curl http://chromadb-service:8000/api/v1/heartbeat
```

#### Ingress Not Working

```bash
# Check ingress status
kubectl get ingress -n mae
kubectl describe ingress mae-ingress -n mae

# Verify ingress controller
kubectl get pods -n ingress-nginx

# Check TLS certificate
kubectl get certificate -n mae
```

#### High Memory Usage

```bash
# Check resource usage
kubectl top pods -n mae

# View detailed metrics
kubectl describe pod <pod-name> -n mae

# Adjust resource limits in values.yaml
```

### Debug Commands

```bash
# Get all resources in namespace
kubectl get all -n mae

# View recent events
kubectl get events -n mae --sort-by='.lastTimestamp'

# Shell into running pod
kubectl exec -it <pod-name> -n mae -- /bin/bash

# Port forward for local access
kubectl port-forward -n mae svc/mae-api 8080:8080

# View Helm values
helm get values mae -n mae

# View rendered templates
helm template mae ./helm/mae-chart
```

---

## Additional Resources

- [Helm Chart Documentation](../../helm/mae-chart/README.md)
- [Kubernetes Manifests](../../k8s/README.md)
- [Monitoring Guide](./MONITORING.md)
- [Security Guide](./SECURITY.md)
- [Performance Tuning](./PERFORMANCE.md)
- [Backup and Recovery](./BACKUP.md)

---

## Support

- GitHub Issues: https://github.com/yourusername/mae/issues
- Documentation: https://mae.readthedocs.io
- Community: https://discord.gg/mae
