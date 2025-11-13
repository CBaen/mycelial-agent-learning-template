# MAE White-Label Roadmap

**Status**: Deferred until after personal validation
**Current Readiness**: 80% (architecture is white-label friendly)
**Estimated Work**: 1 week to reach 100%

---

## Current White-Label Readiness ✅

MAE's architecture is already 80% white-label ready:

1. **Multi-User Authentication** ✅
   - JWT-based authentication
   - 3 roles: Admin, Operator, Viewer
   - Password hashing with bcrypt
   - Token-based sessions

2. **Basic Tenant Isolation** ✅
   - `team_id` field in all agent schemas
   - Separate data per team/tenant
   - User-to-team mapping possible

3. **Clean API Separation** ✅
   - RESTful API with 25 endpoints
   - Frontend-agnostic design
   - OpenAPI documentation
   - No UI coupling

4. **Self-Hostable** ✅
   - No vendor lock-in
   - Deploy on any infrastructure
   - Full data control
   - Docker-ready

5. **Environment-Configurable** ✅
   - SECRET_KEY via environment
   - Database URLs configurable
   - API settings in code (easily externalized)

---

## Remaining 20% (Deferred Features)

### Priority 1: Essential for First Customer (3 days)

#### 1. Multi-Tenancy Middleware
```python
# Add to src/api/middleware/tenant.py

@app.middleware("http")
async def tenant_context(request: Request, call_next):
    """Extract and validate tenant context from subdomain or API key."""
    tenant_id = extract_tenant(request)  # From subdomain, JWT, or API key
    request.state.tenant_id = tenant_id

    # Validate tenant exists and is active
    if not await validate_tenant(tenant_id):
        raise HTTPException(status_code=403, detail="Invalid tenant")

    return await call_next(request)
```

**Impact**: Automatic tenant isolation on every request

#### 2. Environment-Based Configuration
```python
# Add to src/config/settings.py

from pydantic import BaseSettings

class Settings(BaseSettings):
    # Branding
    PRODUCT_NAME: str = "MAE"
    API_TITLE: str = "MAE REST API"
    LOGO_URL: str = ""

    # Security
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Database
    DATABASE_URL: str
    REDIS_URL: str

    class Config:
        env_file = ".env"

settings = Settings()
```

**Impact**: Per-customer configuration without code changes

#### 3. Tenant Branding Configuration
```python
# Add to src/api/schemas/tenant.py

class TenantBranding(BaseModel):
    tenant_id: str
    product_name: str
    logo_url: str
    primary_color: str
    secondary_color: str
    custom_domain: Optional[str] = None

class TenantLimits(BaseModel):
    max_agents: int = 100
    max_requests_per_minute: int = 1000
    max_storage_gb: int = 10
    features_enabled: List[str] = []
```

**Impact**: Customer-specific branding and limits

---

### Priority 2: Important for Scaling (2 days)

#### 4. Rate Limiting Per Tenant
```python
# Add to src/api/middleware/rate_limit.py

from slowapi import Limiter
from slowapi.util import get_remote_address

def get_tenant_id(request: Request) -> str:
    return request.state.tenant_id

limiter = Limiter(key_func=get_tenant_id)

@app.post("/agents")
@limiter.limit(lambda: get_tenant_limits().rate_limit)
async def create_agent(...):
    ...
```

**Impact**: Tier-based pricing support, abuse prevention

#### 5. Audit Logging
```python
# Add to src/api/middleware/audit.py

class AuditLog(BaseModel):
    tenant_id: str
    user_id: str
    action: str  # "agent.created", "policy.exported"
    resource_id: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    request_body: Optional[Dict] = None
    response_status: int

async def log_audit_event(request, response, tenant_id, user_id):
    log = AuditLog(
        tenant_id=tenant_id,
        user_id=user_id,
        action=f"{request.method} {request.url.path}",
        ...
    )
    await audit_db.insert(log)
```

**Impact**: Compliance (SOC2, HIPAA, GDPR), debugging

#### 6. API Key Authentication
```python
# Add to src/api/auth/api_keys.py

class APIKey(BaseModel):
    key_id: str
    tenant_id: str
    name: str  # "Production App", "Dev Environment"
    key_hash: str
    scopes: List[str]
    expires_at: Optional[datetime]
    created_at: datetime
    last_used: Optional[datetime]

async def validate_api_key(api_key: str) -> Tuple[str, List[str]]:
    """Returns (tenant_id, scopes) or raises 401."""
    ...
```

**Impact**: Service-to-service authentication, integrations

---

### Priority 3: Enterprise Features (1 week)

#### 7. HashiCorp Vault Integration
```python
# Add to src/security/vault_client.py

import hvac

class VaultClient:
    def __init__(self, vault_url: str, token: str):
        self.client = hvac.Client(url=vault_url, token=token)

    async def get_secret(self, tenant_id: str, key: str) -> str:
        """Get tenant-specific secret from Vault."""
        path = f"tenants/{tenant_id}/{key}"
        return self.client.secrets.kv.v2.read_secret_version(path=path)
```

**Impact**: Enterprise security requirement, secret rotation

#### 8. Advanced Tenant Isolation
```python
# Database-level isolation

# Option A: Separate database per tenant
def get_tenant_db(tenant_id: str):
    return create_engine(f"postgresql://db-{tenant_id}")

# Option B: Row-level security (PostgreSQL)
CREATE POLICY tenant_isolation ON agents
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant')::uuid);
```

**Impact**: Data isolation guarantee, regulatory compliance

#### 9. Usage Analytics Per Tenant
```python
# Add to src/observability/usage_tracker.py

class UsageMetrics(BaseModel):
    tenant_id: str
    date: date
    agents_created: int
    training_sessions: int
    api_requests: int
    storage_used_gb: float
    compute_minutes: float

async def track_usage(tenant_id: str, metric: str, value: float):
    """Track usage for billing."""
    ...
```

**Impact**: Usage-based billing, capacity planning

---

## Implementation Timeline

### Before First Customer (1 week)
1. ✅ Day 1-2: Multi-tenancy middleware + tenant config service
2. ✅ Day 3: Environment-based configuration + branding
3. ✅ Day 4-5: Rate limiting + audit logging (basic)

### Before 5th Customer (1 week)
4. ✅ API key authentication
5. ✅ Enhanced audit logging (compliance-ready)
6. ✅ Usage analytics

### Before Enterprise Deals (1-2 weeks)
7. ✅ Vault integration
8. ✅ Database-level tenant isolation
9. ✅ Advanced compliance features

---

## Business Model Implications

### Self-Hosted License Model
**MAE is perfect for this today** ✅
- Customer deploys on their infrastructure
- They control all data
- You provide updates/support
- No multi-tenancy needed

**Pricing Ideas**:
- $10k-50k/year per deployment
- Support tiers (community, business, enterprise)
- Custom feature development

### SaaS Multi-Tenant Model
**Needs Priority 1-2 features** (1 week)
- All customers share infrastructure
- Automatic tenant isolation
- Usage-based pricing
- Centralized management

**Pricing Ideas**:
- $500-5k/month per tenant
- Tiered by agents, requests, storage
- Annual discounts

### Hybrid Model (Recommended)
- **Base**: Self-hosted (current architecture works)
- **Premium**: Managed SaaS (needs 1 week of work)
- **Enterprise**: Dedicated deployment + support

---

## White-Label Customization Levels

### Level 1: Basic Rebranding (Ready Now)
- Customer logo in UI
- Custom API title/description
- Custom domain name
- Environment variables for branding

**Customer Example**: "Acme AgentOS powered by MAE"

### Level 2: Full White-Label (Priority 1-2)
- Complete UI rebrand
- Custom pricing tiers
- Tenant-specific features
- Rate limiting per tier

**Customer Example**: "Acme AgentOS" (no MAE mention)

### Level 3: Private-Label Enterprise (Priority 3)
- Dedicated infrastructure
- Custom feature development
- On-premises deployment
- Source code access

**Customer Example**: "Acme Internal AI Platform"

---

## Database Schema Changes Needed

```sql
-- Add tenants table
CREATE TABLE tenants (
    tenant_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    subdomain VARCHAR(63) UNIQUE,
    custom_domain VARCHAR(255),
    branding JSONB,  -- logo, colors, etc.
    limits JSONB,    -- rate limits, quotas
    features JSONB,  -- enabled features
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Add tenant_id to existing tables
ALTER TABLE agents ADD COLUMN tenant_id UUID REFERENCES tenants(tenant_id);
ALTER TABLE users ADD COLUMN tenant_id UUID REFERENCES tenants(tenant_id);

-- Add API keys table
CREATE TABLE api_keys (
    key_id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(tenant_id),
    name VARCHAR(255),
    key_hash VARCHAR(255),
    scopes TEXT[],
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    last_used TIMESTAMP
);

-- Add audit logs table
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(tenant_id),
    user_id UUID,
    action VARCHAR(255),
    resource_id UUID,
    timestamp TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    request_body JSONB,
    response_status INT
);
```

---

## Testing Strategy

### Unit Tests (2 days)
```python
# tests/unit/api/test_multitenancy.py
def test_tenant_isolation():
    """Verify tenant A cannot access tenant B's data."""
    ...

def test_rate_limiting_per_tenant():
    """Verify different tenants have different limits."""
    ...

# tests/unit/api/test_branding.py
def test_tenant_branding():
    """Verify branding configuration works."""
    ...
```

### Integration Tests (1 day)
```python
# tests/integration/test_white_label.py
def test_complete_tenant_lifecycle():
    """Test creating tenant, adding users, isolating data."""
    ...

def test_multi_tenant_concurrent_access():
    """Verify concurrent access from different tenants."""
    ...
```

---

## Security Considerations

1. **Tenant Isolation** ✅
   - Database-level separation (row-level security)
   - API-level validation (middleware)
   - No cross-tenant data leaks

2. **Secret Management** ⏸️
   - Vault for production secrets
   - Per-tenant secret namespaces
   - Secret rotation policies

3. **Audit Trail** ⏸️
   - All tenant actions logged
   - Immutable audit logs
   - Compliance-ready format

4. **Rate Limiting** ⏸️
   - Per-tenant limits
   - DDoS protection
   - Fair usage enforcement

---

## Decision: Why Defer?

1. **Product Validation First**
   - Use MAE personally to validate it works
   - Learn what features are actually valuable
   - Avoid over-engineering before product-market fit

2. **Customer-Driven Development**
   - First customer will reveal real needs
   - Different industries need different features
   - Build what customers will pay for, not guesses

3. **Architecture Already Ready**
   - `team_id` field provides basic isolation
   - Multi-user auth exists
   - Can add multi-tenancy in ~1 week when needed

4. **Faster Time to Value**
   - Focus on Big Rock 14 (Cloud deployment)
   - Get MAE production-ready for personal use
   - Prove value before selling

---

## When to Implement

### Trigger: First Paying Customer
- They express interest in white-label
- They need multi-tenancy
- They require custom branding

### Timing: 1 Week Before Customer Launch
- Implement Priority 1 features (3 days)
- Implement Priority 2 features (2 days)
- Testing and validation (2 days)

### Cost: ~$10k of development time
Worth it if customer LTV > $50k

---

## Reference Implementation

See conversation with Claude Code on 2025-11-12 for detailed implementation examples and code snippets.

**Key Files to Create**:
- `src/api/middleware/tenant.py`
- `src/config/settings.py`
- `src/api/schemas/tenant.py`
- `src/api/middleware/rate_limit.py`
- `src/api/middleware/audit.py`
- `src/api/auth/api_keys.py`
- `src/security/vault_client.py`

**Test Files**:
- `tests/unit/api/test_multitenancy.py`
- `tests/integration/test_white_label.py`

---

**Status**: Document created for future reference
**Next Action**: Proceed with Big Rock 14 (Cloud-Native Deployment)
**Review**: When first customer expresses interest in white-label
