from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.core.logging import ApiLogSink


class _DummyMessage:
    def __init__(self):
        self.record = {
            "time": datetime.now(),
            "level": SimpleNamespace(name="ERROR"),
            "message": "boom",
            "file": SimpleNamespace(name="main.py"),
            "module": "main",
            "function": "handler",
            "line": 42,
            "exception": RuntimeError("failure"),
            "extra": {
                "job_id": "123",
                "branch": "feature/test",
                "class_name": "CompareService",
                "method_name": "handle_request",
            },
        }


def test_api_log_sink_posts_expected_payload(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeResponse:
        status = 200

        def read(self) -> bytes:
            return b""

    class _FakeHTTPConnection:
        def __init__(self, host, port, timeout=None):
            captured["host"] = host
            captured["port"] = port
            captured["timeout"] = timeout

        def request(self, method, path, body=None, headers=None):
            captured["method"] = method
            captured["path"] = path
            captured["body"] = json.loads(body)
            captured["headers"] = dict(headers or {})

        def getresponse(self):
            return _FakeResponse()

        def close(self):
            pass

    monkeypatch.setattr("app.core.logging.http.client.HTTPConnection", _FakeHTTPConnection)

    sink = ApiLogSink(
        "http://logs.local/ingest",
        service_name="svc",
        timeout_ms=1500,
        token="topsecret",
    )
    sink.write(_DummyMessage())

    assert captured["host"] == "logs.local"
    assert captured["path"] == "/ingest"
    payload = captured["body"]
    assert isinstance(payload, dict)
    assert payload["service"] == "svc"
    assert payload["level"] == "ERROR"
    assert payload["message"] == "boom"
    assert payload["branch"] == "feature/test"
    assert payload["file"] == "main.py"
    assert payload["class"] == "CompareService"
    assert payload["method"] == "handle_request"
    assert payload["exception"] == "failure"
    assert payload["extra"] == {
        "job_id": "123",
        "branch": "feature/test",
        "class_name": "CompareService",
        "method_name": "handle_request",
    }
    headers = captured["headers"]
    assert isinstance(headers, dict)
    headers_lower = {k.lower(): v for k, v in headers.items()}
    assert headers_lower["content-type"] == "application/json"
    assert headers_lower["authorization"] == "Bearer topsecret"


def test_api_log_sink_swallows_post_errors(monkeypatch):
    class _RaisingHTTPConnection:
        def __init__(self, host, port, timeout=None):
            pass

        def request(self, method, path, body=None, headers=None):
            raise TimeoutError("network timeout")

        def getresponse(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("app.core.logging.http.client.HTTPConnection", _RaisingHTTPConnection)

    sink = ApiLogSink("http://logs.local/ingest", service_name="svc", timeout_ms=1500)
    sink.write(_DummyMessage())


def test_api_log_sink_rejects_non_http_url():
    with pytest.raises(ValueError, match="http or https"):
        ApiLogSink("file:///etc/passwd", service_name="svc", timeout_ms=1500)

    with pytest.raises(ValueError, match="http or https"):
        ApiLogSink("ftp://example.com/logs", service_name="svc", timeout_ms=1500)


def test_api_log_sink_uses_https_connection(monkeypatch):
    used_class: list[str] = []

    class _FakeResponse:
        status = 200

        def read(self) -> bytes:
            return b""

    class _FakeConn:
        def __init__(self, host, port, timeout=None):
            pass

        def request(self, method, path, body=None, headers=None):
            pass

        def getresponse(self):
            return _FakeResponse()

        def close(self):
            pass

    class _FakeHTTPSConnection(_FakeConn):
        def __init__(self, host, port, timeout=None):
            used_class.append("https")

    monkeypatch.setattr("app.core.logging.http.client.HTTPSConnection", _FakeHTTPSConnection)

    sink = ApiLogSink("https://logs.secure/api", service_name="svc", timeout_ms=1000)
    sink.write(_DummyMessage())

    assert used_class == ["https"]
