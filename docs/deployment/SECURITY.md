# Kubernetes Security Hardening Guide

Complete guide for securing MAE deployments in Kubernetes.

## Security Checklist

- [ ] Use specific image tags (not `latest`)
- [ ] Run containers as non-root user
- [ ] Enable read-only root filesystem where possible
- [ ] Drop all unnecessary Linux capabilities
- [ ] Enable Pod Security Standards
- [ ] Configure Network Policies
- [ ] Use secrets management (External Secrets Operator)
- [ ] Enable TLS for all ingress traffic
- [ ] Implement RBAC with least privilege
- [ ] Enable audit logging
- [ ] Regular security scanning of images
- [ ] Configure resource limits
- [ ] Enable Pod Disruption Budgets
- [ ] Use service mesh for mTLS (optional)

---

## Pod Security

### Security Context

```yaml
# Pod-level security context
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault

# Container-level security context
securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
      - ALL
  runAsNonRoot: true
  runAsUser: 1000
```

### Pod Security Standards

Apply Pod Security Standards to namespace:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mae-prod
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

---

## Network Security

### Network Policies

Restrict network traffic between pods:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mae-api-network-policy
  namespace: mae-prod
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: mae
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from ingress controller only
    - from:
      - namespaceSelector:
          matchLabels:
            name: ingress-nginx
      ports:
      - protocol: TCP
        port: 8080
  egress:
    # Allow DNS
    - to:
      - namespaceSelector:
          matchLabels:
            name: kube-system
        podSelector:
          matchLabels:
            k8s-app: kube-dns
      ports:
      - protocol: UDP
        port: 53
    # Allow Redis
    - to:
      - podSelector:
          matchLabels:
            app: redis
      ports:
      - protocol: TCP
        port: 6379
    # Allow ChromaDB
    - to:
      - podSelector:
          matchLabels:
            app: chromadb
      ports:
      - protocol: TCP
        port: 8000
```

---

## Secrets Management

### Using External Secrets Operator

Install External Secrets Operator:

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets \
  external-secrets/external-secrets \
  -n external-secrets-system \
  --create-namespace
```

Configure AWS Secrets Manager:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: mae-prod
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: mae-service-account
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: mae-secrets
  namespace: mae-prod
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: mae-secrets
    creationPolicy: Owner
  data:
  - secretKey: secret-key
    remoteRef:
      key: mae/production/jwt-secret
  - secretKey: database-password
    remoteRef:
      key: mae/production/db-password
```

---

## RBAC (Role-Based Access Control)

### Service Account

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mae-service-account
  namespace: mae-prod
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/mae-prod
```

### Role and RoleBinding

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mae-role
  namespace: mae-prod
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mae-rolebinding
  namespace: mae-prod
subjects:
- kind: ServiceAccount
  name: mae-service-account
  namespace: mae-prod
roleRef:
  kind: Role
  name: mae-role
  apiGroup: rbac.authorization.k8s.io
```

---

## TLS/SSL Configuration

### Cert-Manager

Install Cert-Manager:

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

Create ClusterIssuer:

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

---

## Image Security

### Image Scanning

Scan images with Trivy:

```bash
# Scan local image
trivy image mae-api:latest

# Scan with severity filter
trivy image --severity CRITICAL,HIGH mae-api:latest

# Generate SARIF report for GitHub
trivy image --format sarif --output trivy-results.sarif mae-api:latest
```

### Image Signing

Sign images with Cosign:

```bash
# Generate key pair
cosign generate-key-pair

# Sign image
cosign sign --key cosign.key ghcr.io/org/mae-api:v1.0.0

# Verify signature
cosign verify --key cosign.pub ghcr.io/org/mae-api:v1.0.0
```

---

## Runtime Security

### Falco

Install Falco for runtime security:

```bash
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm install falco falcosecurity/falco \
  --namespace falco-system \
  --create-namespace
```

Custom Falco rules for MAE:

```yaml
- rule: Unauthorized Process in MAE Container
  desc: Detect unexpected processes in MAE containers
  condition: >
    spawned_process and
    container.image.repository = "mae-api" and
    not proc.name in (python3, gunicorn, uvicorn)
  output: >
    Unauthorized process in MAE container
    (user=%user.name command=%proc.cmdline container=%container.name)
  priority: WARNING
```

---

## Audit Logging

Enable Kubernetes audit logging:

```yaml
apiVersion: v1
kind: Policy
rules:
- level: Metadata
  resources:
  - group: ""
    resources: ["secrets", "configmaps"]
  namespaces: ["mae-prod"]
- level: RequestResponse
  resources:
  - group: "apps"
    resources: ["deployments", "statefulsets"]
  namespaces: ["mae-prod"]
```

---

## Compliance

### CIS Kubernetes Benchmark

Run kube-bench:

```bash
kubectl apply -f https://raw.githubusercontent.com/aquasecurity/kube-bench/main/job.yaml

# View results
kubectl logs job/kube-bench
```

### OWASP Top 10

MAE addresses OWASP Top 10:

1. **Broken Access Control**: JWT authentication, RBAC
2. **Cryptographic Failures**: TLS everywhere, encrypted secrets
3. **Injection**: Input validation with Pydantic
4. **Insecure Design**: Security by design principles
5. **Security Misconfiguration**: Secure defaults, hardening guide
6. **Vulnerable Components**: Regular scanning, updates
7. **Authentication Failures**: Strong password policies, JWT
8. **Software and Data Integrity**: Image signing, checksums
9. **Logging Failures**: Comprehensive audit logs
10. **SSRF**: Network policies, egress filtering

---

## Incident Response

### Security Incident Runbook

1. **Detection**: Alert from Falco/monitoring
2. **Containment**: Isolate affected pods
3. **Investigation**: Collect logs and metrics
4. **Eradication**: Remove compromised components
5. **Recovery**: Restore from known-good state
6. **Lessons Learned**: Update security controls

### Emergency Commands

```bash
# Isolate pod (remove from service)
kubectl label pod <pod-name> -n mae-prod security=quarantine

# Drain node
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Rollback deployment
helm rollback mae -n mae-prod

# Force delete pod
kubectl delete pod <pod-name> -n mae-prod --force --grace-period=0
```

---

## Monitoring Security

### Security Metrics

Monitor these security metrics:

- Failed authentication attempts
- Unauthorized API access attempts
- Pod restart counts (potential crashes/exploits)
- Network policy violations
- Certificate expiration dates
- Secret access patterns

### Alerts

Configure alerts for:

```yaml
groups:
- name: security
  rules:
  - alert: HighAuthFailureRate
    expr: rate(auth_failures_total[5m]) > 10
    for: 5m
    annotations:
      summary: High authentication failure rate detected

  - alert: UnauthorizedAPIAccess
    expr: rate(http_requests_total{code="403"}[5m]) > 5
    for: 5m
    annotations:
      summary: Multiple unauthorized access attempts

  - alert: CertificateExpiringSoon
    expr: (cert_expiry_timestamp - time()) < 86400 * 7
    for: 1h
    annotations:
      summary: TLS certificate expiring in less than 7 days
```

---

## Regular Security Tasks

### Weekly
- Review security logs
- Check for security updates
- Scan images for vulnerabilities

### Monthly
- Rotate secrets and keys
- Review RBAC policies
- Update security documentation

### Quarterly
- Penetration testing
- Security audit
- Disaster recovery drill

---

## Additional Resources

- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)
- [OWASP Kubernetes Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Kubernetes_Security_Cheat_Sheet.html)
- [NSA/CISA Kubernetes Hardening Guide](https://media.defense.gov/2022/Aug/29/2003066362/-1/-1/0/CTR_KUBERNETES_HARDENING_GUIDANCE_1.2_20220829.PDF)
