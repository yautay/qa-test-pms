# Perceptual Metrics Service / Serwis Metryk Percepcyjnych

FastAPI service for comparing two images using perceptual similarity metrics (LPIPS), with optional difference heatmaps.

Serwis FastAPI do porownywania dwoch obrazow przy uzyciu metryk percepcyjnych (LPIPS), z opcjonalnym generowaniem mapy roznic.

## English

### What it does

- Compares two images available on the server filesystem
- Returns LPIPS + DISTS scores and LPIPS heatmap in one response (`/compare`)
- Supports dedicated endpoints for LPIPS (`/compare/lpips`) and DISTS (`/compare/dists`)
- Accepts image input in 3 integration modes: paths, mixed (ref path + uploaded test), full API upload

### Requirements

- Python + pip
- Packages from `requirements.txt` (includes `fastapi`, `uvicorn`, `torch`, `lpips`, `pillow`)

### Install

```bash
pip install -r requirements.txt
```

### Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Open Swagger UI:

- `http://127.0.0.1:8080/docs`

### API

Health:

```bash
curl http://127.0.0.1:8080/health
```

Example response:

```json
{
  "status": "ok",
  "message": "Service is healthy",
  "device": "cpu",
  "metrics": ["lpips", "dists"],
  "build": {
    "version": "v1.2",
    "tag": "v1.2",
    "commit_hash": "b21e4c2",
    "commit_date": "2026-02-24T13:57:15+01:00",
    "branch": "feature/NN-22777-tech-ai-wpiecie-lpips-w-pipeline"
  }
}
```

Compare all (JSON -> JSON):

```bash
curl -sS -X POST "http://127.0.0.1:8080/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_path": "tests/assets/ref_1.png",
    "test_path": "tests/assets/test_1.png",
    "config": {
      "lpips_net": "vgg",
      "force_device": null,
      "max_side": 1024,
      "overlay_on": "test",
      "alpha": 0.45
    }
  }'
```

Compare LPIPS in mixed mode (ref path + test image base64):

```bash
curl -sS -X POST "http://127.0.0.1:8080/compare/lpips" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_path": "tests/assets/ref_1.png",
    "test_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR42mP4DwABAQEAG7buVgAAAABJRU5ErkJggg==",
    "config": {
      "net": "vgg",
      "force_device": "cpu"
    }
  }'
```

Compare DISTS in API upload mode (both images as base64):

```bash
curl -sS -X POST "http://127.0.0.1:8080/compare/dists" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR42mP4DwABAQEAG7buVgAAAABJRU5ErkJggg==",
    "test_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR42mP4DwABAQEAG7buVgAAAABJRU5ErkJggg==",
    "config": {
      "force_device": "cpu"
    }
  }'
```


LPIPS only (JSON -> JSON):

```bash
curl -sS -X POST "http://127.0.0.1:8080/compare/lpips" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_path": "tests/assets/ref_1.png",
    "test_path": "tests/assets/test_1.png",
    "config": {
      "net": "vgg",
      "force_device": null,
      "max_side": 1024,
      "overlay_on": "test",
      "alpha": 0.45
    }
  }'
```

DISTS only (JSON -> JSON):

```bash
curl -sS -X POST "http://127.0.0.1:8080/compare/dists" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_path": "tests/assets/ref_1.png",
    "test_path": "tests/assets/test_1.png",
    "config": {
      "force_device": null
    }
  }'
```

### Notes

- You can send image sources in one of 3 modes:
  - `paths`: `ref_path` + `test_path`
  - `mixed`: `ref_path` + `test_image_base64`
  - `api_upload`: `ref_image_base64` + `test_image_base64`
- `ref_image_base64` and `test_image_base64` must be valid image bytes encoded as base64.
- Responses include:
  - `integration_mode` (`paths`, `mixed`, `api_upload`)
  - `analyzed_models` (metric + model used)
- `lpips_heatmap_png_base64` is returned as base64 PNG.
- `ref_path`/`test_path` must resolve inside `IMAGE_BASE_DIR` (default: current working directory).
- GPU: you need a CUDA-enabled PyTorch build; then set `config.force_device` to `"cuda"`.

### Runtime configuration (TOML + ENV)

Configuration precedence is:

- `default < pyproject.toml:[tool.ai_corner] < env`

Supported variables:

- `API_DEBUG` (bool): enables detailed JSON 500 response (`true/false`, `1/0`, `yes/no`, `on/off`).
- `IMAGE_BASE_DIR` (string): base directory used for safe path resolution (`ref_path`, `test_path`).
- `LOG_LEVEL` (string): logger threshold (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- `AI_CORNER_CONFIG_TOML` (string, env-only): path to custom TOML file to read instead of default `pyproject.toml`.

Build metadata for `/health` (`version`, `tag`, `commit_hash`, `commit_date`, `branch`) is read from Git.

Default TOML section in this repo:

```toml
[tool.ai_corner]
api_debug = true
image_base_dir = "."
log_level = "INFO"
```

Set variables in shell (Linux/macOS/WSL):

```bash
export API_DEBUG=0
export IMAGE_BASE_DIR=/srv/images
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload
```

Set variables for one command only:

```bash
API_DEBUG=0 LOG_LEVEL=ERROR uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Use a custom TOML file:

```bash
AI_CORNER_CONFIG_TOML=/etc/ai-corner/config.toml uvicorn app.main:app --reload
```

On startup, API logs the effective values and their source (`default`, `toml`, `env`).

### Metrics

- `lpips`: implemented (scalar + heatmap). Nets: `vgg`, `alex`, `squeeze`.
- `dists`: implemented for scalar scoring.

### Contract for client developers

- OpenAPI JSON contract: `http://127.0.0.1:8080/openapi.json`
- Swagger UI: `http://127.0.0.1:8080/docs`
- ReDoc: `http://127.0.0.1:8080/redoc`
- Repository snapshot of contract: `docs/openapi.json`
- Client-facing change notes: `docs/CLIENT_CONTRACT_CHANGELOG.md`

Download contract:

```bash
curl -sS http://127.0.0.1:8080/openapi.json -o openapi.json
```

Generate TypeScript client (example with OpenAPI Generator):

```bash
openapi-generator-cli generate -i openapi.json -g typescript-fetch -o client-ts
```

Decode LPIPS heatmap base64 to PNG (Python):

```python
import base64

heatmap_base64 = response_json["lpips_heatmap_png_base64"]
with open("heatmap.png", "wb") as f:
    f.write(base64.b64decode(heatmap_base64))
```

Decode LPIPS heatmap base64 to PNG (JavaScript / Node.js):

```javascript
const fs = require("fs");

const heatmapBase64 = responseJson.lpips_heatmap_png_base64;
fs.writeFileSync("heatmap.png", Buffer.from(heatmapBase64, "base64"));
```
