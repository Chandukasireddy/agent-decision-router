"""Tests for Decision Backends (Cloudflare, Ollama, Mock)."""

import pytest
import httpx
from unittest.mock import patch, MagicMock

from agent_decision_router.models import (
    DecisionPayload,
    QuestionDefinition,
    QuestionType,
)
from agent_decision_router.backend.cloudflare import CloudflareBackend, CloudflareAPIError
from agent_decision_router.backend.ollama import OllamaBackend, OllamaAPIError
from agent_decision_router.backend.mock import MockBackend


@pytest.fixture
def sample_payload():
    return DecisionPayload(
        state="User requested: Run the unit tests and fix any assertion errors in auth_test.py. Current git status: 1 modified file.",
        questions={
            "next_tool": QuestionDefinition(
                type=QuestionType.CHOICE.value,
                instructions="Which tool should the agent run next?",
                criteria={
                    "run_terminal_command": "Execute shell commands, pytest, or build tools",
                    "read_code_file": "Inspect contents of a file on disk",
                    "apply_code_patch": "Write or modify existing source files",
                    "ask_user_clarification": "Task is ambiguous or requires human input",
                }
            ),
            "is_destructive": QuestionDefinition(
                type=QuestionType.NOUL.value,
                instructions="Will this planned action delete data, overwrite uncommitted changes, or terminate processes?",
            ),
            "task_completion": QuestionDefinition(
                type=QuestionType.NOUL.value,
                instructions="Has the user's task or objective been completely satisfied and no further tool execution is needed?",
            )
        }
    )


def test_cloudflare_backend_endpoint_and_headers():
    backend = CloudflareBackend(
        account_id="test_account_123",
        api_token="test_token_xyz",
        model="@cf/cloudflare/clef-flash",
    )

    assert backend.endpoint_url == "https://api.cloudflare.com/client/v4/accounts/test_account_123/ai/run/@cf/cloudflare/clef-flash"
    headers = backend._get_headers()
    assert headers["Authorization"] == "Bearer test_token_xyz"
    assert headers["Content-Type"] == "application/json"


def test_cloudflare_backend_missing_credentials():
    backend = CloudflareBackend(account_id="", api_token="")
    with pytest.raises(CloudflareAPIError):
        _ = backend.endpoint_url

    backend2 = CloudflareBackend(account_id="123", api_token="")
    with pytest.raises(CloudflareAPIError):
        _ = backend2._get_headers()


def test_cloudflare_backend_response_parsing(sample_payload):
    backend = CloudflareBackend(account_id="acc", api_token="tok")

    mock_cf_response = {
        "result": {
            "next_tool": {
                "choice": "run_terminal_command",
                "probabilities": {
                    "run_terminal_command": 0.85,
                    "read_code_file": 0.10,
                    "apply_code_patch": 0.05,
                }
            },
            "is_destructive": {
                "noul": False,
                "confidence": 0.05,
            },
            "task_completion": {
                "noul": False,
                "confidence": 0.02,
            }
        },
        "success": True,
        "errors": [],
        "messages": [],
    }

    mock_http_resp = MagicMock(spec=httpx.Response)
    mock_http_resp.status_code = 200
    mock_http_resp.json.return_value = mock_cf_response

    parsed = backend._process_response(mock_http_resp, sample_payload)
    assert parsed.selected_action == "run_terminal_command"
    assert parsed.confidence == 0.85
    assert parsed.is_destructive is False
    assert parsed.task_completion is False
    assert parsed.probabilities["run_terminal_command"] == 0.85


def test_ollama_backend_endpoint_and_payload(sample_payload):
    backend = OllamaBackend(base_url="http://localhost:11434", model="nimble")
    assert backend.endpoint_url == "http://localhost:11434/v1/systemone"

    payload_dict = backend._prepare_payload(sample_payload)
    assert payload_dict["model"] == "nimble"
    assert "next_tool" in payload_dict["questions"]
    assert payload_dict["questions"]["next_tool"]["type"] == "choice"


def test_mock_backend_canned():
    backend = MockBackend(
        canned_action="read_code_file",
        canned_confidence=0.95,
        canned_destructive=False,
        canned_completion=False,
    )
    payload = DecisionPayload(
        state="inspect file",
        questions={},
    )
    parsed = backend.query_decision_sync(payload)
    assert parsed.selected_action == "read_code_file"
    assert parsed.confidence == 0.95
    assert parsed.is_destructive is False
    assert parsed.task_completion is False


def test_mock_backend_heuristic_completion_and_destructive(sample_payload):
    backend = MockBackend(auto_heuristic=True)

    # Destructive test
    destructive_payload = DecisionPayload(
        state="User said: run rm -rf on the old build folder and git reset --hard",
        questions=sample_payload.questions,
    )
    res_dest = backend.query_decision_sync(destructive_payload)
    assert res_dest.is_destructive is True

    # Completion test
    complete_payload = DecisionPayload(
        state="All tests passed successfully and output verified. Task is complete.",
        questions=sample_payload.questions,
    )
    res_comp = backend.query_decision_sync(complete_payload)
    assert res_comp.task_completion is True
