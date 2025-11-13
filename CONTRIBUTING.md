# Contributing to MAE

Thank you for your interest in contributing to the Mycelial Agent Engine (MAE)! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment. We expect all contributors to:

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the project
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for local development)
- Git
- kubectl and Helm (for Kubernetes deployment, optional)

### Setup Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mae.git
   cd mae
   ```

2. **Run the setup script**:
   ```bash
   ./scripts/setup.sh
   ```

   Or manually:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

3. **Start services**:
   ```bash
   make dev
   ```

4. **Run tests**:
   ```bash
   make test
   ```

---

## Development Workflow

### 1. Create a Branch

Always create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### 2. Make Your Changes

- Write clean, readable code following our [Coding Standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed
- Keep commits atomic and well-described

### 3. Test Your Changes

Before submitting, ensure all tests pass:

```bash
# Run all tests
make test

# Run specific test suite
make test-unit
make test-integration
make test-api

# Check code quality
make check  # Runs format, lint, type-check, and test
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description

- Detailed point 1
- Detailed point 2
- Closes #issue_number"
```

Commit message format:
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description (wrap at 72 chars)
- Reference related issues

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters max
- **Formatting**: Use `black` for automatic formatting
- **Imports**: Use `isort` for import sorting
- **Type hints**: Required for all function signatures
- **Docstrings**: Google-style docstrings for all public functions

Example:

```python
from typing import List, Optional

from src.agents.base_agent import BaseAgent


class MyAgent(BaseAgent):
    """Agent that performs specific task.

    Args:
        agent_id: Unique identifier for the agent
        config: Configuration dictionary

    Attributes:
        state: Current agent state
        actions: Available actions
    """

    def __init__(self, agent_id: str, config: dict) -> None:
        super().__init__(agent_id)
        self.config = config

    def step(self, observation: dict) -> Optional[str]:
        """Process observation and select action.

        Args:
            observation: Current environment observation

        Returns:
            Selected action or None if no action available
        """
        # Implementation
        pass
```

### Code Organization

- **File structure**: Follow the existing project structure
- **Module imports**: Use absolute imports from `src/`
- **Constants**: Define at module level in UPPER_CASE
- **Private methods**: Prefix with underscore `_method_name`

### Comments

- Write self-documenting code when possible
- Add comments for complex logic
- Keep comments up-to-date with code changes
- Use TODO comments for future improvements:
  ```python
  # TODO(username): Description of what needs to be done
  ```

---

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
│   ├── test_agents.py
│   ├── test_memory.py
│   └── api/
│       └── test_auth.py
├── integration/       # Integration tests (slower, uses real services)
│   ├── test_redis_integration.py
│   └── test_vector_db.py
└── conftest.py       # Shared fixtures
```

### Writing Tests

1. **Unit Tests**: Test individual functions/methods in isolation
   ```python
   def test_agent_initialization():
       """Test that agent initializes with correct state."""
       agent = BaseAgent(agent_id="test-001")
       assert agent.agent_id == "test-001"
       assert agent.state == AgentState.IDLE
   ```

2. **Integration Tests**: Test interaction between components
   ```python
   @pytest.mark.integration
   def test_agent_redis_persistence(redis_client):
       """Test that agent state persists to Redis."""
       agent = BaseAgent(agent_id="test-001")
       agent.save_state(redis_client)

       restored_agent = BaseAgent.load_state(redis_client, "test-001")
       assert restored_agent.state == agent.state
   ```

3. **API Tests**: Test REST API endpoints
   ```python
   def test_create_agent(client, admin_token):
       """Test creating agent via API."""
       response = client.post(
           "/agents",
           json={"agent_type": "specialist", "config": {}},
           headers={"Authorization": f"Bearer {admin_token}"}
       )
       assert response.status_code == 201
   ```

### Test Requirements

- **Coverage**: Aim for >80% code coverage
- **Assertions**: Use descriptive assertion messages
- **Fixtures**: Use pytest fixtures for reusable test setup
- **Mocking**: Mock external dependencies (APIs, databases) in unit tests
- **Naming**: Test functions should start with `test_` and be descriptive

---

## Submitting Changes

### Pull Request Process

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**:
   - Go to GitHub and create a new Pull Request
   - Fill out the PR template completely
   - Link related issues
   - Request review from maintainers

3. **PR Requirements**:
   - [ ] All tests pass
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated (for significant changes)
   - [ ] No merge conflicts

4. **Review Process**:
   - Maintainers will review your code
   - Address any feedback promptly
   - Make changes in response to reviews
   - Once approved, maintainers will merge

### PR Title Format

```
[Type] Brief description

Types: Feature, Fix, Docs, Refactor, Test, Chore
Examples:
- [Feature] Add episodic memory for agents
- [Fix] Resolve Redis connection timeout
- [Docs] Update API documentation
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How was this tested?

## Screenshots (if applicable)

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] All tests passing
```

---

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Example: `v1.2.3`

### Creating a Release

1. **Update version** in `src/__init__.py`
2. **Update CHANGELOG.md** with release notes
3. **Create git tag**:
   ```bash
   git tag -a v1.2.3 -m "Release v1.2.3"
   git push origin v1.2.3
   ```
4. **Create GitHub Release** with release notes

---

## Additional Resources

### Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Documentation](http://localhost:8080/docs)
- [Deployment Guide](docs/DEPLOYMENT.md)

### Communication

- **Issues**: Report bugs or request features on GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: For security issues, email security@mae-project.com

### Useful Commands

```bash
# Development
make dev              # Start development environment
make stop             # Stop services
make logs             # View logs

# Testing
make test             # Run all tests
make test-watch       # Run tests in watch mode
make test-coverage    # Generate coverage report

# Code Quality
make format           # Format code with black
make lint             # Run linters
make type-check       # Run type checking
make check            # Run all quality checks

# Docker
make build            # Build Docker images
make clean-docker     # Clean Docker volumes

# Kubernetes
make k8s-deploy       # Deploy to Kubernetes
make k8s-status       # Check deployment status
```

---

## Questions?

If you have questions not covered here, please:

1. Check existing documentation in `docs/`
2. Search GitHub Issues for similar questions
3. Open a new GitHub Discussion
4. Reach out to maintainers

Thank you for contributing to MAE!
