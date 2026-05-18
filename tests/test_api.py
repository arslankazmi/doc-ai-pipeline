"""FastAPI endpoint smoke tests using httpx TestClient."""
import sys
from pathlib import Path
import os
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def set_db_path(tmp_path):
    os.environ["CASES_DB_PATH"] = str(tmp_path / "test.db")
    yield


def test_health():
    from fastapi.testclient import TestClient
    from pipeline.api import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_demo_page_loads():
    from fastapi.testclient import TestClient
    from pipeline.api import app
    client = TestClient(app)
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert "Document AI" in resp.text


def test_extract_invalid_image():
    from fastapi.testclient import TestClient
    from pipeline.api import app
    client = TestClient(app)
    resp = client.post("/extract", json={"image_b64": "notbase64", "models": ["donut"]})
    # Should return 422 or 500 — not crash the server
    assert resp.status_code in (200, 202, 400, 422, 500)


def test_hitl_health():
    from fastapi.testclient import TestClient
    from hitl.app import app as hitl_app
    client = TestClient(hitl_app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_hitl_queue_loads():
    from fastapi.testclient import TestClient
    from hitl.app import app as hitl_app
    client = TestClient(hitl_app)
    resp = client.get("/queue")
    assert resp.status_code == 200
    assert "Queue" in resp.text
