# Client Contract Changelog

This document summarizes API contract updates relevant for client developers.

## 2026-02-15

### Endpoints covered

- `POST /compare`
- `POST /compare/lpips`
- `POST /compare/dists`

### Request contract changes

- Added optional base64 image inputs:
  - `ref_image_base64`
  - `test_image_base64`
- Existing path-based inputs remain supported:
  - `ref_path`
  - `test_path`
- Supported source combinations are now explicit:
  - `paths`: `ref_path` + `test_path`
  - `mixed`: `ref_path` + `test_image_base64`
  - `api_upload`: `ref_image_base64` + `test_image_base64`
- Invalid combinations return validation errors (`422`).

### Response contract changes

- Added top-level field `integration_mode` to all compare responses.
- Added top-level field `analyzed_models` to all compare responses.
- Existing metric payloads remain unchanged:
  - `lpips` and `dists` keep shape `{ value, meta }`.
  - `lpips_heatmap_png_base64` remains base64-encoded PNG.

### Compatibility notes

- Path-only clients continue to work without changes.
- Clients that validate exact response key sets must be updated to accept:
  - `integration_mode`
  - `analyzed_models`

### Recommended client migration

1. Regenerate client SDK from latest OpenAPI contract.
2. Update response models to include `integration_mode` and `analyzed_models`.
3. Add request builders for `mixed` and `api_upload` modes.
4. Add tests for invalid input combinations expecting `422`.

### Contract artifacts

- Live OpenAPI: `/openapi.json`
- Snapshot in repo: `docs/openapi.json`
