# Integration Testing Guide

This guide covers opt-in integration tests that require external services.

## Redis Auth Integration Suite

The Redis auth suite lives in `tests/test_redis_auth_integration.py` and is marked with `redis_integration`.

- Default `pytest -q` remains the standard fast path.
- Redis integration tests are opt-in and require explicit environment setup.

Required env vars:

```bash
export RUN_REDIS_INTEGRATION=1
export PMS_REDIS_AUTH_URL='redis://:pms-secret@127.0.0.1:6380/0'
```

`PMS_REDIS_AUTH_URL` must point to an auth-enabled Redis instance.

Run integration tests against your auth-enabled Redis instance:

```bash
pytest -q -m redis_integration tests/test_redis_auth_integration.py
```

Or run scenarios individually:

```bash
pytest -q -m redis_integration tests/test_redis_auth_integration.py -k url
pytest -q -m redis_integration tests/test_redis_auth_integration.py -k split
pytest -q -m redis_integration tests/test_redis_auth_integration.py -k invalid
```

## Expected Behavior

- URL mode and split-vars mode both pass full API flow (`/health`, create job, read job back).
- Invalid credentials fail during app startup.
- Failure output must not expose raw Redis password values.
