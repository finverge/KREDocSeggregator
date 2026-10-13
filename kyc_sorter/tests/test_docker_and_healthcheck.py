"""
Tests for Docker deployment configuration, healthcheck verification, and container environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from kyc_sorter.main import parse_args
from kyc_sorter.utils.config_loader import load_config
from kyc_sorter.utils.healthcheck import extract_base_url, verify_ollama_connectivity


class TestHealthcheck:
    def test_extract_base_url(self):
        assert extract_base_url("http://localhost:11434/api/chat") == "http://localhost:11434"
        assert extract_base_url("http://host.docker.internal:11434/api/chat") == "http://host.docker.internal:11434"
        assert extract_base_url("http://ollama:11434/api") == "http://ollama:11434"
        assert extract_base_url("http://192.168.1.100:11434") == "http://192.168.1.100:11434"
        assert extract_base_url("https://remote-gpu.cloud.org/api/chat") == "https://remote-gpu.cloud.org"

    @patch("kyc_sorter.utils.healthcheck.requests.get")
    def test_verify_ollama_connectivity_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen3-vl:4b-instruct", "size": 3295637504, "modified_at": "2026-10-08T12:00:00Z"},
                {"name": "llama3:latest", "size": 4000000000, "modified_at": "2026-10-08T10:00:00Z"},
            ]
        }
        mock_get.return_value = mock_resp

        is_healthy, msg = verify_ollama_connectivity(
            endpoint="http://host.docker.internal:11434/api/chat",
            model_name="qwen3-vl:4b-instruct",
            timeout=5,
        )

        assert is_healthy is True
        assert "is reachable" in msg
        assert "qwen3-vl:4b-instruct" in msg
        mock_get.assert_called_once_with("http://host.docker.internal:11434/api/tags", timeout=5)

    @patch("kyc_sorter.utils.healthcheck.requests.get")
    def test_verify_ollama_connectivity_missing_model(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3:latest", "size": 4000000000},
            ]
        }
        mock_get.return_value = mock_resp

        is_healthy, msg = verify_ollama_connectivity(
            endpoint="http://localhost:11434/api/chat",
            model_name="qwen3-vl:4b-instruct",
            timeout=5,
        )

        assert is_healthy is False
        assert "NOT found" in msg
        assert "ollama pull qwen3-vl:4b-instruct" in msg

    @patch("kyc_sorter.utils.healthcheck.requests.get")
    def test_verify_ollama_connectivity_network_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        is_healthy, msg = verify_ollama_connectivity(
            endpoint="http://localhost:11434/api/chat",
            model_name="qwen3-vl:4b-instruct",
            timeout=5,
        )

        assert is_healthy is False
        assert "Cannot connect to Ollama" in msg
        assert "host.docker.internal" in msg


class TestDockerConfiguration:
    def test_check_connection_cli_argument(self):
        args = parse_args(["--check-connection", "--endpoint", "http://my-ollama:11434", "--model", "qwen3-vl:4b-instruct"])
        assert args.check_connection is True
        assert args.ollama_endpoint == "http://my-ollama:11434"
        assert args.model == "qwen3-vl:4b-instruct"

    def test_container_environment_variable_overrides(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "container_in"
        out_dir = tmp_path / "container_out"
        in_dir.mkdir()
        out_dir.mkdir()

        monkeypatch.setenv("KYC_INPUT_DIR", str(in_dir))
        monkeypatch.setenv("KYC_OUTPUT_DIR", str(out_dir))
        monkeypatch.setenv("KYC_JOURNAL_PATH", str(out_dir / "journal.db"))
        monkeypatch.setenv("KYC_VLM_ENDPOINT", "http://host.docker.internal:11434/api/chat")
        monkeypatch.setenv("KYC_VLM_MODEL", "qwen3-vl:4b-instruct")
        monkeypatch.setenv("KYC_MAX_WORKERS", "6")
        monkeypatch.setenv("KYC_BATCH_SIZE", "50")

        cfg = load_config()

        assert cfg.input.resolved_path == in_dir.resolve()
        assert cfg.output.resolved_path == out_dir.resolve()
        assert cfg.processing.journal_path == str(out_dir / "journal.db")
        assert cfg.vlm.endpoint == "http://host.docker.internal:11434/api/chat"
        assert cfg.vlm.model == "qwen3-vl:4b-instruct"
        assert cfg.processing.max_workers == 6
        assert cfg.processing.batch_size == 50

    def test_watcher_polling_defaults_off(self, tmp_path):
        cfg = load_config(base_dir=tmp_path)
        assert cfg.watcher.use_polling is False
        assert cfg.watcher.polling_interval_sec == 5.0

    def test_watcher_polling_environment_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KYC_WATCHER_USE_POLLING", "true")
        monkeypatch.setenv("KYC_WATCHER_POLLING_INTERVAL", "2.5")

        cfg = load_config(base_dir=tmp_path)

        assert cfg.watcher.use_polling is True
        assert cfg.watcher.polling_interval_sec == 2.5

    def test_watcher_polling_interval_must_be_positive(self, tmp_path, monkeypatch):
        from kyc_sorter.utils.config_loader import ConfigurationError

        monkeypatch.setenv("KYC_WATCHER_POLLING_INTERVAL", "0")
        with pytest.raises(ConfigurationError):
            load_config(base_dir=tmp_path)
