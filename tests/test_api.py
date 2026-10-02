"""Tests for FastAPI REST service."""

import pytest
from fastapi.testclient import TestClient
from agent_decision_router.api import app, get_api_engine
from agent_decision_router.backend.mock import MockBackend


@pytest.fixture
def client():
    engine = get_api_engine()
    engine.backend = MockBackend(canned_action="read_code_file", canned_confidence=0.88)
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "registered_tools_count" in data


def test_route_endpoint(client):
    payload = {
        "state": "User wants to inspect the contents of main.py",
        "skip_cache": True,
    }
    response = client.post("/v1/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["decision"]["action"] == "read_code_file"
    assert data["decision"]["confidence"] == 0.88


def test_tools_list_and_register_endpoint(client):
    # List initial tools
    resp = client.get("/v1/tools")
    assert resp.status_code == 200
    initial_count = len(resp.json())

    # Register new tool
    new_tool = {
        "name": "docker_compose_up",
        "description": "Start containers using docker compose",
        "criteria": "Spin up backend database and redis services with docker",
        "parameters": {},
        "is_destructive": False,
    }
    reg_resp = client.post("/v1/register", json=new_tool)
    assert reg_resp.status_code == 200
    assert reg_resp.json()["success"] is True

    # Check updated tools
    resp_updated = client.get("/v1/tools")
    assert len(resp_updated.json()) == initial_count + 1

    # Delete registered tool
    del_resp = client.delete("/v1/tools/docker_compose_up")
    assert del_resp.status_code == 200
    assert del_resp.json()["unregistered"] == "docker_compose_up"


def test_cache_stats_and_clear_endpoint(client):
    stats_resp = client.get("/v1/cache/stats")
    assert stats_resp.status_code == 200
    assert "hits" in stats_resp.json()

    clear_resp = client.post("/v1/cache/clear")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["success"] is True
