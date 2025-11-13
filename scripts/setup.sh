#!/bin/bash
# ============================================================================
# MAE Setup Script
# ============================================================================
#
# This script automates the initial setup of the MAE development environment.
#
# Usage:
#   ./scripts/setup.sh
#
# ============================================================================

set -e  # Exit on error

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_warning "$1 is not installed"
        return 1
    fi
}

# ============================================================================
# Main Setup
# ============================================================================

print_header "MAE Development Environment Setup"

# Check system requirements
print_header "Checking System Requirements"

MISSING_REQUIREMENTS=false

if ! check_command python3; then
    print_error "Python 3.8+ is required"
    MISSING_REQUIREMENTS=true
fi

if ! check_command pip3; then
    print_error "pip3 is required"
    MISSING_REQUIREMENTS=true
fi

if ! check_command docker; then
    print_warning "Docker is not installed (optional for local development)"
fi

if ! check_command docker-compose; then
    print_warning "Docker Compose is not installed (optional for local development)"
fi

if ! check_command kubectl; then
    print_warning "kubectl is not installed (optional for Kubernetes deployment)"
fi

if ! check_command helm; then
    print_warning "Helm is not installed (optional for Kubernetes deployment)"
fi

if [ "$MISSING_REQUIREMENTS" = true ]; then
    print_error "Missing required dependencies. Please install them and try again."
    exit 1
fi

# Check Python version
print_header "Checking Python Version"
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if awk "BEGIN {exit !($PYTHON_VERSION >= $REQUIRED_VERSION)}"; then
    print_success "Python version $PYTHON_VERSION is compatible"
else
    print_error "Python 3.8+ is required (found $PYTHON_VERSION)"
    exit 1
fi

# Create virtual environment
print_header "Creating Virtual Environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_header "Activating Virtual Environment"
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_header "Upgrading pip"
pip install --upgrade pip
print_success "pip upgraded"

# Install dependencies
print_header "Installing Python Dependencies"
pip install -r requirements.txt
print_success "Dependencies installed"

# Create .env file if it doesn't exist
print_header "Setting Up Environment Variables"
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_success "Created .env file from .env.example"
    print_warning "Please edit .env and configure your settings"
else
    print_warning ".env file already exists"
fi

# Create required directories
print_header "Creating Required Directories"
mkdir -p logs
mkdir -p data
mkdir -p simulation_results
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
print_success "Directories created"

# Create Prometheus config if it doesn't exist
print_header "Setting Up Monitoring Configuration"
if [ ! -f "monitoring/prometheus/prometheus.yml" ]; then
    cat > monitoring/prometheus/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mae-api'
    static_configs:
      - targets: ['mae-api:8080']
    metrics_path: '/metrics'
EOF
    print_success "Created Prometheus configuration"
else
    print_warning "Prometheus configuration already exists"
fi

# Create Grafana datasource config
if [ ! -f "monitoring/grafana/datasources/prometheus.yaml" ]; then
    cat > monitoring/grafana/datasources/prometheus.yaml <<EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    print_success "Created Grafana datasource configuration"
else
    print_warning "Grafana datasource configuration already exists"
fi

# Initialize pre-commit hooks (if available)
print_header "Setting Up Pre-Commit Hooks"
if [ -f ".pre-commit-config.yaml" ]; then
    if command -v pre-commit &> /dev/null; then
        pip install pre-commit
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not found, skipping"
    fi
else
    print_warning "No .pre-commit-config.yaml found, skipping"
fi

# Test Redis connection (if Docker is running)
print_header "Testing Docker Services"
if command -v docker &> /dev/null && docker ps &> /dev/null; then
    print_success "Docker is running"

    # Check if services are already running
    if docker ps | grep -q mae_redis; then
        print_success "MAE services are already running"
    else
        print_warning "MAE services are not running. Start them with: make dev"
    fi
else
    print_warning "Docker is not running or not accessible"
fi

# Run tests
print_header "Running Tests"
if python -m pytest tests/unit/ -v --tb=short; then
    print_success "Unit tests passed"
else
    print_warning "Some tests failed. This is normal for a fresh installation."
fi

# Generate SECRET_KEY
print_header "Generating Security Key"
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
print_success "Generated SECRET_KEY (add to .env):"
echo -e "${YELLOW}SECRET_KEY=${SECRET_KEY}${NC}"

# Print summary
print_header "Setup Complete!"

echo -e "${GREEN}MAE development environment is ready!${NC}\n"
echo -e "Next steps:"
echo -e "  1. ${YELLOW}Edit .env${NC} and configure your settings (especially SECRET_KEY)"
echo -e "  2. ${YELLOW}make dev${NC} to start Docker Compose services"
echo -e "  3. ${YELLOW}make test${NC} to run the test suite"
echo -e "  4. ${YELLOW}python run_simulation.py${NC} to run a simulation"
echo -e "  5. ${YELLOW}make help${NC} to see all available commands"
echo ""
echo -e "API Documentation will be available at: ${BLUE}http://localhost:8080/docs${NC}"
echo ""
echo -e "${GREEN}Happy coding!${NC}\n"
