# Monorepo Restructuring Plan - RAG Application

**Created:** 2024-12-10  
**Author:** GitHub Copilot for Marcelo Prates  
**Status:** Planning Phase

## Executive Summary

This document outlines a comprehensive restructuring plan for the RAG (Retrieval-Augmented Generation) application monorepo to follow industry best practices for Python monorepos with multiple services.

## Current Issues

### 1. **Mixed Responsibilities**

- `src/` contains shared library code AND the backend API
- `backend/` only has Docker/requirements, not the actual backend code
- Unclear separation between library code and service code

### 2. **Configuration Management**

- Configs are inside the `src/` package (should be service-specific or root-level)
- No clear environment-based configuration strategy
- Missing `.env.example` files

### 3. **Data Management**

- `data/` at root with minimal structure
- No clear separation between raw data, processed data, indices, and artifacts
- Backend mounts generic `data/` without clear purpose

### 4. **Testing & Quality**

- No visible test structure
- Missing CI/CD configuration (`.github/workflows/`)
- No code quality tools (linters, formatters)

### 5. **Documentation**

- Empty README.md
- No architectural documentation
- No API documentation
- Missing setup/contribution guides

### 6. **Dependency Management**

- Multiple `requirements.txt` files without clear hierarchy
- `pyproject.toml` underutilized (should be the source of truth)
- No dependency groups (dev, prod, test)

### 7. **Build & Deployment**

- Docker setup mixes concerns (backend Dockerfile references src/)
- No clear build process for shared library
- Missing production-ready configurations

## Proposed Structure

```
rag/
├── .github/
│   ├── workflows/              # CI/CD pipelines
│   │   ├── backend-ci.yml
│   │   ├── frontend-ci.yml
│   │   └── shared-ci.yml
│   └── docs/                   # Project documentation
│       ├── architecture/
│       ├── todo/
│       └── old/
│
├── apps/                       # Application/Service layer
│   ├── backend/
│   │   ├── src/
│   │   │   └── rag_backend/
│   │   │       ├── __init__.py
│   │   │       ├── main.py     # FastAPI app
│   │   │       ├── api/        # API routes
│   │   │       ├── services/   # Business logic
│   │   │       └── schemas/    # Pydantic models
│   │   ├── tests/
│   │   ├── configs/
│   │   │   ├── default.yaml
│   │   │   ├── production.yaml
│   │   │   └── development.yaml
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── .env.example
│   │   └── README.md
│   │
│   └── frontend/
│       ├── src/
│       │   └── rag_frontend/
│       │       ├── __init__.py
│       │       ├── app.py
│       │       ├── pages/      # Multi-page Streamlit
│       │       └── components/ # Reusable components
│       ├── tests/
│       ├── configs/
│       ├── Dockerfile
│       ├── pyproject.toml
│       ├── .env.example
│       └── README.md
│
├── packages/                   # Shared libraries
│   └── rag-core/
│       ├── src/
│       │   └── rag_core/
│       │       ├── __init__.py
│       │       ├── agents/
│       │       ├── ingestion/
│       │       ├── llm/
│       │       ├── prompt_engineering/
│       │       ├── retrieval/
│       │       ├── vectorstore/
│       │       └── utils/
│       ├── tests/
│       │   ├── unit/
│       │   ├── integration/
│       │   └── conftest.py
│       ├── pyproject.toml
│       └── README.md
│
├── data/                       # Data directory (gitignored except structure)
│   ├── raw/                    # Original documents
│   ├── processed/              # Processed/chunked data
│   ├── indices/                # Vector store indices
│   │   ├── faiss/
│   │   └── whoosh/
│   ├── cache/                  # Temporary cache
│   └── .gitkeep
│
├── scripts/                    # Utility scripts
│   ├── setup_dev.sh
│   ├── run_tests.sh
│   ├── build_all.sh
│   └── seed_data.py
│
├── docs/                       # User/developer documentation
│   ├── getting-started.md
│   ├── architecture.md
│   ├── api-reference.md
│   └── deployment.md
│
├── .github/
├── .gitignore
├── .pre-commit-config.yaml     # Code quality hooks
├── docker-compose.yml
├── docker-compose.prod.yml
├── Makefile                    # Common commands
├── pyproject.toml              # Workspace root config
├── README.md                   # Comprehensive overview
└── LICENSE
```

## Restructuring Steps

### Phase 1: Setup Infrastructure (Non-Breaking)

- [ ] Create directory structure for `apps/`, `packages/`, `scripts/`, `docs/`
- [ ] Set up proper `.gitignore` for each service and package
- [ ] Create `.env.example` files for backend and frontend
- [ ] Set up `Makefile` with common commands
- [ ] Create `.pre-commit-config.yaml` for code quality
- [ ] Add comprehensive `README.md` at root

### Phase 2: Extract Shared Library

- [ ] Create `packages/rag-core/` structure
- [ ] Move shared code from `src/` to `packages/rag-core/src/rag_core/`:
  - [ ] `agents/`
  - [ ] `ingestion/`
  - [ ] `llm/`
  - [ ] `prompt_engineering/`
  - [ ] `retrieval/`
  - [ ] `vectorstore/`
  - [ ] `utils/` (except API-specific utils)
- [ ] Create `packages/rag-core/pyproject.toml` with dependencies
- [ ] Set up as editable install in workspace

### Phase 3: Restructure Backend

- [ ] Create `apps/backend/src/rag_backend/`
- [ ] Move `src/api/main.py` to `apps/backend/src/rag_backend/main.py`
- [ ] Refactor API into modular structure:
  - [ ] Create `api/` directory for routes
  - [ ] Create `services/` for business logic
  - [ ] Create `schemas/` for Pydantic models
- [ ] Move configs from `src/configs/` to `apps/backend/configs/`
- [ ] Update imports to use `rag_core` package
- [ ] Create `apps/backend/pyproject.toml`
- [ ] Update `apps/backend/Dockerfile`
- [ ] Create backend tests structure

### Phase 4: Restructure Frontend

- [ ] Create `apps/frontend/src/rag_frontend/`
- [ ] Move `frontend/app.py` to `apps/frontend/src/rag_frontend/app.py`
- [ ] Organize into multi-page structure if needed
- [ ] Extract reusable components
- [ ] Create `apps/frontend/pyproject.toml`
- [ ] Update `apps/frontend/Dockerfile`
- [ ] Add frontend configuration management

### Phase 5: Data & Configuration Management

- [ ] Restructure `data/` directory:
  - [ ] Create subdirectories: `raw/`, `processed/`, `indices/`, `cache/`
  - [ ] Add `.gitkeep` files
  - [ ] Update `.gitignore`
- [ ] Create environment-specific configs for backend
- [ ] Update Docker Compose volume mounts
- [ ] Create data seeding scripts

### Phase 6: Testing & Quality

- [ ] Set up pytest structure for all packages
- [ ] Create `conftest.py` files
- [ ] Add unit tests for `rag-core`
- [ ] Add integration tests for backend
- [ ] Set up coverage reporting
- [ ] Configure linters (ruff/flake8, mypy)
- [ ] Configure formatters (black, isort)
- [ ] Set up pre-commit hooks

### Phase 7: CI/CD & Automation

- [ ] Create GitHub Actions workflows:
  - [ ] `shared-ci.yml` - Test rag-core package
  - [ ] `backend-ci.yml` - Test and build backend
  - [ ] `frontend-ci.yml` - Test and build frontend
- [ ] Set up Docker image building and pushing
- [ ] Add automated versioning
- [ ] Configure branch protection rules

### Phase 8: Documentation

- [ ] Write comprehensive root `README.md`
- [ ] Create `docs/getting-started.md`
- [ ] Create `docs/architecture.md` with diagrams
- [ ] Document API endpoints
- [ ] Create deployment guide
- [ ] Add inline code documentation
- [ ] Generate API docs with Sphinx/MkDocs

### Phase 9: Deployment & Production

- [ ] Create `docker-compose.prod.yml`
- [ ] Add health checks to services
- [ ] Configure logging and monitoring
- [ ] Add environment variable validation
- [ ] Create Kubernetes manifests (if needed)
- [ ] Set up secrets management

### Phase 10: Cleanup

- [ ] Remove old `src/` directory structure
- [ ] Remove old `backend/` stub directory
- [ ] Clean up duplicate requirements files
- [ ] Update all documentation
- [ ] Create migration guide for team
- [ ] Archive this plan to `.github/docs/old/`

## Key Benefits

1. **Clear Separation of Concerns**: Apps, packages, and infrastructure are clearly separated
2. **Reusability**: Shared code in `rag-core` can be used by multiple services
3. **Independent Deployment**: Each app can be deployed independently
4. **Better Testing**: Clear test structure for each component
5. **Scalability**: Easy to add new apps or packages
6. **Developer Experience**: Clear structure, good documentation, automated quality checks
7. **Production Ready**: Proper configuration management, CI/CD, monitoring

## Migration Strategy

### Approach: Incremental Migration

- Keep both old and new structure temporarily
- Update imports gradually
- Use symlinks if needed during transition
- Maintain backward compatibility until migration complete
- Run tests at each phase

### Rollback Plan

- Keep git history clean with atomic commits per phase
- Tag releases before major changes
- Document rollback procedures
- Maintain old structure in separate branch until confident

## Timeline Estimate

- **Phase 1-2**: 1-2 days (Setup + Extract shared library)
- **Phase 3-4**: 2-3 days (Restructure services)
- **Phase 5**: 1 day (Data & configs)
- **Phase 6**: 2-3 days (Testing & quality)
- **Phase 7**: 1-2 days (CI/CD)
- **Phase 8**: 1-2 days (Documentation)
- **Phase 9**: 1-2 days (Production readiness)
- **Phase 10**: 1 day (Cleanup)

**Total: 10-16 days** (can be parallelized with team)

## Dependencies & Tools

### Python Package Management

- **pyproject.toml**: Modern Python packaging standard
- **pip-tools** or **poetry**: Dependency resolution
- Workspace/editable installs for local development

### Code Quality

- **ruff**: Fast linter and formatter
- **mypy**: Static type checking
- **pre-commit**: Git hooks for quality checks
- **pytest**: Testing framework
- **coverage**: Code coverage

### CI/CD

- **GitHub Actions**: CI/CD pipelines
- **Docker**: Containerization
- **docker-compose**: Local development

### Documentation

- **MkDocs** or **Sphinx**: Documentation generation
- **Mermaid**: Architecture diagrams
- **OpenAPI/Swagger**: API documentation

## Current Status

**Phase**: Planning Complete  
**Next Steps**: Review plan with team, then begin Phase 1

---

## Progress Notes

### 2024-12-10 11:30

- Initial plan created
- Analyzed current structure
- Identified key issues and proposed comprehensive solution
- Awaiting review and approval to proceed
