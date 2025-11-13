# MAE Kubernetes Deployment

This directory contains Kubernetes manifests for deploying MAE to a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- nginx-ingress-controller (for Ingress)
- cert-manager (for TLS certificates, optional)

## Quick Start

### 1. Deploy to Kubernetes

```bash
# Create namespace and deploy all resources
kubectl apply -f namespace.yaml
kubectl apply -k base/

# Or use the Makefile
make k8s-deploy
```

### 2. Check Deployment Status

```bash
kubectl get all -n mae
kubectl get pods -n mae --watch
```

### 3. Access the API

```bash
# Port-forward for local access
kubectl port-forward -n mae svc/mae-api-service 8080:8080

# Access API at http://localhost:8080
# API docs at http://localhost:8080/docs
```

## Directory Structure

```
k8s/
├── namespace.yaml          # MAE namespace
├── base/                   # Base Kubernetes manifests
│   ├── configmap.yaml     # Non-sensitive configuration
│   ├── secret.yaml        # Sensitive configuration
│   ├── rbac.yaml          # Service account and permissions
│   ├── redis.yaml         # Redis deployment + service
│   ├── chromadb.yaml      # ChromaDB deployment + service
│   ├── mae-api.yaml       # MAE API deployment + HPA
│   ├── ingress.yaml       # Ingress for external access
│   └── kustomization.yaml # Kustomize configuration
├── overlays/              # Environment-specific overlays
│   ├── development/       # Development environment
│   └── production/        # Production environment
└── README.md              # This file
```

## Configuration

### Update Secrets

Before deploying to production, update the secrets:

```bash
# Generate a new SECRET_KEY
python3 -c "import secrets; print(secrets.token_hex(32))"

# Create the secret
kubectl create secret generic mae-secrets \
  --from-literal=SECRET_KEY=your-generated-key \
  --namespace mae
```

### Update Ingress

Edit `base/ingress.yaml` to use your domain:

```yaml
spec:
  tls:
    - hosts:
        - api.yourdomain.com  # Change this
      secretName: mae-tls-cert
  rules:
    - host: api.yourdomain.com  # Change this
```

### Configure Storage

Update storage class in `base/redis.yaml` and `base/chromadb.yaml`:

```yaml
spec:
  storageClassName: standard  # Change to your storage class
```

## Scaling

### Manual Scaling

```bash
# Scale API pods
kubectl scale deployment mae-api -n mae --replicas=5
```

### Horizontal Pod Autoscaler

HPA is already configured in `base/mae-api.yaml`:

- Min replicas: 2
- Max replicas: 10
- Target CPU: 70%
- Target Memory: 80%

View HPA status:

```bash
kubectl get hpa -n mae
kubectl describe hpa mae-api-hpa -n mae
```

## Monitoring

### View Logs

```bash
# All pods
kubectl logs -n mae -l app=mae-api --tail=100 -f

# Specific pod
kubectl logs -n mae <pod-name> -f

# Using Makefile
make k8s-logs
```

### Pod Status

```bash
kubectl get pods -n mae
kubectl describe pod <pod-name> -n mae
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n mae

# Check logs
kubectl logs <pod-name> -n mae

# Check resource usage
kubectl top pods -n mae
```

### Service Not Accessible

```bash
# Check services
kubectl get svc -n mae

# Check endpoints
kubectl get endpoints -n mae

# Test internal connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -n mae -- sh
# Inside the pod:
wget -O- http://mae-api-service:8080/system/health
```

### Database Connection Issues

```bash
# Check Redis
kubectl exec -it <redis-pod> -n mae -- redis-cli ping

# Check ChromaDB
kubectl exec -it <chromadb-pod> -n mae -- curl http://localhost:8000/api/v1/heartbeat
```

## Cleanup

```bash
# Delete all resources
kubectl delete -k base/
kubectl delete namespace mae

# Or use Makefile
make k8s-delete
```

## Production Checklist

Before deploying to production:

- [ ] Update SECRET_KEY in secrets
- [ ] Configure proper domain in Ingress
- [ ] Set up TLS certificates (cert-manager)
- [ ] Configure appropriate resource limits
- [ ] Set up proper storage class
- [ ] Configure backup for persistent volumes
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Set up log aggregation
- [ ] Configure network policies
- [ ] Enable pod security policies
- [ ] Set up proper RBAC
- [ ] Configure rate limiting
- [ ] Set up alerting

## Further Reading

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kustomize Documentation](https://kustomize.io/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [Cert-Manager](https://cert-manager.io/)
