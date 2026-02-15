# AGENTS.md

Guidance for coding agents working in this repository.

## Project Snapshot

- Stack: FastAPI service for image comparison metrics (LPIPS, DISTS).
- Runtime: Python 3.12.
- App entrypoint: `app.main:app`.
- Main packages: `app/api`, `app/core`, `app/metrics`, `app/schemas`.
- Tests live in `tests/` and run with `pytest`.

## Source of Truth for Rules

- No `.cursorrules` file found.
- No files found under `.cursor/rules/`.
- No `.github/copilot-instructions.md` found.
- Therefore, follow this file plus repository configs (`pyproject.toml`, `pytest.ini`, `.pre-commit-config.yaml`, `Makefile`).

## Environment Setup

- Create virtualenv: `python -m venv .venv`.
- Activate: `source .venv/bin/activate`.
- Upgrade pip: `pip install --upgrade pip`.
- Install runtime deps: `pip install -r requirements.txt`.
- Install dev deps: `pip install -r requirements-dev.txt`.

## Build / Run Commands

- Run API (dev): `uvicorn app.main:app --reload`.
- Run API (prod-like): `uvicorn app.main:app --host 0.0.0.0 --port 8080`.
- Health check: `curl http://127.0.0.1:8080/health`.

## Test Commands

- Run all tests (preferred short): `pytest -q`.
- Equivalent Make target: `make test`.
- Run a single file: `pytest -q tests/test_api.py`.
- Run one test function: `pytest -q tests/test_api.py::test_health`.
- Run one test class/test node: `pytest -q tests/test_api.py::test_compare_dists_returns_only_score`.
- Run tests by expression: `pytest -q -k "compare and not contracts"`.
- Stop after first failure: `pytest -q -x`.

## Lint / Format / Type Commands

- Lint: `ruff check .`.
- Lint with auto-fix: `ruff check . --fix`.
- Format: `black .`.
- Type-check: `mypy .`.
- Run pre-commit hooks: `pre-commit run --all-files`.
- Make shortcuts:
  - `make lint`
  - `make format`
  - `make type`
  - `make precommit`

## Full Validation Before PR

- Use `make full-check` before finalizing significant changes.
- `make full-check` runs: tests, ruff, mypy, pip-audit, pre-commit.

## Formatting and Imports

- Black is authoritative formatter.
- Max line length is 100 (`black` + `ruff` configured to 100).
- Ruff lint rules enabled: `E`, `F`, `I`, `UP`, `B`, `SIM`.
- `E501` is ignored in Ruff, but still aim for readable <=100 char lines.
- Import order should satisfy Ruff/isort (`I` rules).
- Prefer stdlib -> third-party -> local import grouping.
- Avoid unused imports and dead assignments.

## Typing Guidelines

- Annotate public functions and non-trivial internals.
- Use modern Python 3.12 typing (`X | None`, built-in generics).
- Prefer concrete return types (`dict[str, Any]`, `tuple[...]`, etc.).
- Use `Literal` for constrained API config options (pattern already used in schemas).
- Mypy settings include:
  - `no_implicit_optional = true`
  - `check_untyped_defs = true`
  - `warn_return_any = true`
- `disallow_untyped_defs = false`, but new code should still be typed where practical.

## Naming Conventions

- Modules and files: `snake_case`.
- Functions/variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Metric identifiers are stable lowercase strings (`"lpips"`, `"dists"`).
- Keep API field names consistent with existing schema names (do not rename casually).

## FastAPI / API Patterns

- Define request/response contracts with Pydantic models in `app/schemas`.
- In routes, set `response_model=...` explicitly.
- Keep route handlers thin: parse request, delegate to metric/core logic, map exceptions.
- Preserve current endpoint shapes:
  - `POST /compare`
  - `POST /compare/lpips`
  - `POST /compare/dists`
  - `GET /health`

## Error Handling Conventions

- Map domain/runtime errors to `HTTPException` in route layer.
- Existing mappings to keep consistent:
  - `PermissionError` -> 403
  - `ValueError` / `NotImplementedError` / `KeyError` -> 400
  - `FileNotFoundError` -> 404
- Respect `API_DEBUG` behavior:
  - `API_DEBUG=1`: JSON 500 with details via global handler.
  - `API_DEBUG=0`: default FastAPI plain `Internal Server Error` response.

## Metrics and Core Logic Patterns

- Metric implementations extend `Metric` and return `MetricResult`.
- Cache heavy models per config/device (see LPIPS and DISTS classes).
- Use `resolve_device()` for CPU/CUDA selection.
- Keep tensor range contracts explicit:
  - LPIPS tensors in `[-1, 1]`.
  - DISTS tensors in `[0, 1]`.
- Keep path safety checks through `resolve_input_path()` and `IMAGE_BASE_DIR`.

## Testing Conventions

- Use `fastapi.testclient.TestClient` for API tests.
- Use `monkeypatch` to stub registry/metric behavior in endpoint tests.
- Validate response shape and status codes, not only happy-path values.
- Keep tests deterministic and independent of GPU availability when possible.
- Place new tests in `tests/` with `test_*.py` naming.

## Security and Dependency Hygiene

- Run `pip-audit` for vulnerability checks (`make audit`).
- Large files are blocked by pre-commit (`check-added-large-files`, max ~2 MB).
- Do not bypass path sandboxing (`IMAGE_BASE_DIR`) in application code.

## Agent Workflow Recommendations

- Prefer minimal, targeted changes over broad refactors.
- Update/add tests in the same change when behavior changes.
- Run focused checks during iteration, then run `make full-check`.
- Keep commits small and descriptive if asked to commit.
- Avoid introducing new tooling unless requested.

## Agent Do / Don't

- Do: read relevant files first and follow existing patterns before editing.
- Do: keep route handlers thin and push logic to `app/core` or `app/metrics`.
- Do: preserve public API contracts unless the task explicitly requires a contract change.
- Do: add or update tests for behavior changes, especially endpoint status/shape.
- Do: run the smallest useful validation command first, then broader checks.

- Don't: bypass path safety (`resolve_input_path`, `IMAGE_BASE_DIR`) or weaken permission checks.
- Don't: introduce breaking field renames in schemas without coordinated API updates/tests.
- Don't: add broad refactors, new frameworks, or dependency churn for small fixes.
- Don't: hardcode device assumptions (CUDA/CPU); always use `resolve_device()` flow.
- Don't: silence exceptions in metrics/routes; map them to the existing HTTP semantics.

## Quick Command Cheat Sheet

- Install dev env: `pip install -r requirements-dev.txt`.
- Run app: `uvicorn app.main:app --reload`.
- Run all tests: `pytest -q`.
- Run one test: `pytest -q tests/test_api.py::test_health`.
- Lint: `ruff check .`.
- Format: `black .`.
- Types: `mypy .`.
- Full gate: `make full-check`.
