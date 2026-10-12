"""
Unit tests for configuration loader and path resolution.
"""

import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import yaml

from kyc_sorter.utils.config_loader import (
    load_config,
    Config,
    ConfigurationError,
    normalize_ollama_endpoint,
)


class TestConfigLoader(unittest.TestCase):
    def test_load_default_config(self):
        config = load_config()
        self.assertIsInstance(config, Config)
        self.assertEqual(config.application.name, "KYC Document Sorter")
        self.assertIn(config.output.operation, ("copy", "move"))
        self.assertAlmostEqual(config.processing.confidence_threshold, 0.80)
        self.assertEqual(config.processing.supported_extensions, {".jpg"})
        self.assertFalse(config.security.log_sensitive_text)
        self.assertEqual(config.get_destination_subfolder("AADHAAR_CARD"), "AADHAAR CARD")
        self.assertEqual(config.get_destination_subfolder("PAN_CARD"), "PAN CARD")
        self.assertEqual(config.get_destination_subfolder("BANK_PASSBOOK"), "BANK STATEMENT")
        self.assertEqual(config.get_destination_subfolder("NON_EXISTENT"), "NON KYC")
        self.assertEqual(config.unknown_folder, "NON KYC")
        # Filename suffixes test
        self.assertEqual(config.get_filename_suffix("AADHAAR_CARD"), "AADHAAR_CARD")
        self.assertEqual(config.get_filename_suffix("PAN_CARD"), "PAN_CARD")
        self.assertEqual(config.get_filename_suffix("VOTER_ID"), "VOTER_ID")
        self.assertEqual(config.get_filename_suffix("BANK_PASSBOOK"), "BANK_STATEMENT")
        self.assertEqual(config.get_filename_suffix("PASSPORT"), "PASSPORT")
        self.assertEqual(config.get_filename_suffix("UNKNOWN"), "UNKNOWN")
        self.assertEqual(config.get_filename_suffix("NON_EXISTENT"), "UNKNOWN")
        # Watcher config test
        self.assertEqual(config.watcher.stability_delay_sec, 1.0)
        self.assertEqual(config.watcher.settle_timeout_sec, 30.0)
        # New processing configs
        self.assertEqual(config.processing.cpu_limit_percent, 80.0)
        self.assertTrue(config.processing.adaptive_workers)
        self.assertAlmostEqual(config.processing.multiple_document_score_margin, 0.10)
        # VLM configs
        self.assertEqual(config.vlm.model, "qwen3-vl:4b-instruct")
        self.assertEqual(config.vlm.endpoint, "http://localhost:11434/api/chat")
        self.assertEqual(config.vlm.timeout, 300)

    def test_overrides_and_validation(self):
        overrides = {
            "output": {"operation": "MOVE"},
            "processing": {
                "confidence_threshold": 0.95,
                "dry_run": True,
                "cpu_limit_percent": 75.0,
                "multiple_document_score_margin": 0.15,
                "supported_extensions": ["JPG", "pdf", ".TIFF"],
            },
            "security": {"log_sensitive_text": True},
        }
        config = load_config(overrides=overrides)
        self.assertEqual(config.output.operation, "move")
        self.assertAlmostEqual(config.processing.confidence_threshold, 0.95)
        self.assertTrue(config.processing.dry_run)
        self.assertEqual(config.processing.cpu_limit_percent, 75.0)
        self.assertAlmostEqual(config.processing.multiple_document_score_margin, 0.15)
        self.assertTrue(config.security.log_sensitive_text)
        self.assertIn(".jpg", config.processing.supported_extensions)
        self.assertIn(".pdf", config.processing.supported_extensions)
        self.assertIn(".tiff", config.processing.supported_extensions)

    def test_invalid_operation_fallback(self):
        overrides = {"output": {"operation": "delete_all"}}
        config = load_config(overrides=overrides)
        # Should fallback to safe default "copy"
        self.assertEqual(config.output.operation, "copy")

    def test_confidence_threshold_clamping(self):
        config_low = load_config(overrides={"processing": {"confidence_threshold": -0.5}})
        self.assertEqual(config_low.processing.confidence_threshold, 0.0)

        config_high = load_config(overrides={"processing": {"confidence_threshold": 1.5}})
        self.assertEqual(config_high.processing.confidence_threshold, 1.0)

    def test_max_workers_configuration(self):
        config = load_config()
        self.assertEqual(config.processing.max_workers, 4)

        custom = load_config(overrides={
            "processing": {"max_workers": 8},
            "vlm": {"model": "custom-vlm", "endpoint": "http://10.0.0.1:11434/api/chat", "timeout": 120}
        })
        self.assertEqual(custom.processing.max_workers, 8)
        self.assertEqual(custom.vlm.model, "custom-vlm")
        self.assertEqual(custom.vlm.endpoint, "http://10.0.0.1:11434/api/chat")
        self.assertEqual(custom.vlm.timeout, 120)

    def test_adaptive_worker_calculation(self):
        from kyc_sorter.sorter.batch_processor import compute_adaptive_workers
        self.assertEqual(compute_adaptive_workers(4, total_cores=2), 1)
        self.assertEqual(compute_adaptive_workers(4, total_cores=4), 2)
        self.assertEqual(compute_adaptive_workers(4, total_cores=6), 3)
        self.assertEqual(compute_adaptive_workers(4, total_cores=8), 4)
        self.assertEqual(compute_adaptive_workers(4, total_cores=12), 4)
        self.assertEqual(compute_adaptive_workers(6, total_cores=12), 6)
        self.assertEqual(compute_adaptive_workers(8, total_cores=4), 2)

    def test_yaml_only_configuration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg_file = tmp_path / "custom_config.yaml"
            input_dir = tmp_path / "incoming"
            output_dir = tmp_path / "sorted"
            input_dir.mkdir()
            output_dir.mkdir()

            yaml_content = {
                "input": {"directory": "incoming"},
                "output": {"directory": "sorted", "operation": "move"},
                "processing": {"max_workers": 2, "confidence_threshold": 0.85},
                "vlm": {"model": "custom-vlm-model", "endpoint": "http://vlm-server:11434/api/chat", "timeout": 90},
            }
            with open(cfg_file, "w", encoding="utf-8") as f:
                yaml.dump(yaml_content, f)

            config = load_config(config_path=cfg_file)
            self.assertEqual(config.input.resolved_path, input_dir.resolve())
            self.assertEqual(config.output.resolved_path, output_dir.resolve())
            self.assertEqual(config.output.operation, "move")
            self.assertEqual(config.processing.max_workers, 2)
            self.assertAlmostEqual(config.processing.confidence_threshold, 0.85)
            self.assertEqual(config.vlm.model, "custom-vlm-model")
            self.assertEqual(config.vlm.endpoint, "http://vlm-server:11434/api/chat")
            self.assertEqual(config.vlm.timeout, 90)

    def test_env_variable_overrides(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            in_dir = tmp_path / "env_in"
            out_dir = tmp_path / "env_out"
            in_dir.mkdir()
            out_dir.mkdir()

            env_vars = {
                "KYC_INPUT_DIR": str(in_dir),
                "KYC_OUTPUT_DIR": str(out_dir),
                "KYC_OPERATION": "move",
                "KYC_DRY_RUN": "true",
                "KYC_WORKERS": "6",
                "KYC_CONFIDENCE_THRESHOLD": "0.92",
                "KYC_CPU_LIMIT_PERCENT": "70",
                "KYC_CPU_THROTTLE_PERCENT": "55",
                "KYC_LIMIT": "25",
                "KYC_SUPPORTED_EXTENSIONS": ".jpg,.png",
                "KYC_LOG_SENSITIVE_TEXT": "1",
                "KYC_VLM_MODEL": "env-model-vlm",
                "KYC_VLM_ENDPOINT": "http://env-endpoint:11434",
                "KYC_VLM_TIMEOUT": "150",
            }
            with patch.dict(os.environ, env_vars, clear=False):
                config = load_config(base_dir=tmp_path)
                self.assertEqual(config.input.resolved_path, in_dir.resolve())
                self.assertEqual(config.output.resolved_path, out_dir.resolve())
                self.assertEqual(config.output.operation, "move")
                self.assertTrue(config.processing.dry_run)
                self.assertEqual(config.processing.max_workers, 6)
                self.assertAlmostEqual(config.processing.confidence_threshold, 0.92)
                self.assertEqual(config.processing.cpu_limit_percent, 70.0)
                self.assertEqual(config.processing.throttle_cpu_percent, 55.0)
                self.assertEqual(config.processing.limit, 25)
                self.assertEqual(config.processing.supported_extensions, {".jpg", ".png"})
                self.assertTrue(config.security.log_sensitive_text)
                self.assertEqual(config.vlm.model, "env-model-vlm")
                self.assertEqual(config.vlm.endpoint, "http://env-endpoint:11434/api/chat")
                self.assertEqual(config.vlm.timeout, 150)

    def test_ollama_host_and_endpoint_env(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            in_dir = tmp_path / "in"
            out_dir = tmp_path / "out"
            in_dir.mkdir()
            out_dir.mkdir()

            # Test OLLAMA_HOST and OLLAMA_MODEL standard env vars
            env_vars = {
                "KYC_INPUT_DIR": str(in_dir),
                "KYC_OUTPUT_DIR": str(out_dir),
                "OLLAMA_HOST": "http://remote-ollama:11434",
                "OLLAMA_MODEL": "qwen3-vl:8b",
                "OLLAMA_TIMEOUT": "240",
            }
            with patch.dict(os.environ, env_vars, clear=False):
                config = load_config(base_dir=tmp_path)
                self.assertEqual(config.vlm.endpoint, "http://remote-ollama:11434/api/chat")
                self.assertEqual(config.vlm.model, "qwen3-vl:8b")
                self.assertEqual(config.vlm.timeout, 240)

            # Test KYC_VLM_ENDPOINT precedence over OLLAMA_HOST
            env_vars_both = {
                "KYC_INPUT_DIR": str(in_dir),
                "KYC_OUTPUT_DIR": str(out_dir),
                "KYC_VLM_ENDPOINT": "http://primary-host:11434",
                "OLLAMA_HOST": "http://secondary-host:11434",
            }
            with patch.dict(os.environ, env_vars_both, clear=False):
                config = load_config(base_dir=tmp_path)
                self.assertEqual(config.vlm.endpoint, "http://primary-host:11434/api/chat")

    def test_precedence_cli_over_env_over_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            in_yaml = tmp_path / "yaml_in"
            out_yaml = tmp_path / "yaml_out"
            in_cli = tmp_path / "cli_in"
            out_cli = tmp_path / "cli_out"
            in_env = tmp_path / "env_in"
            out_env = tmp_path / "env_out"
            for d in (in_yaml, out_yaml, in_cli, out_cli, in_env, out_env):
                d.mkdir()

            cfg_file = tmp_path / "precedence.yaml"
            with open(cfg_file, "w", encoding="utf-8") as f:
                yaml.dump({
                    "input": {"directory": str(in_yaml)},
                    "output": {"directory": str(out_yaml)},
                    "processing": {"max_workers": 2, "confidence_threshold": 0.70},
                    "vlm": {"model": "yaml-model", "endpoint": "http://yaml-host:11434", "timeout": 100},
                }, f)

            env_vars = {
                "KYC_INPUT_DIR": str(in_env),
                "KYC_OUTPUT_DIR": str(out_env),
                "KYC_WORKERS": "4",
                "KYC_CONFIDENCE_THRESHOLD": "0.85",
                "OLLAMA_MODEL": "env-model",
                "OLLAMA_HOST": "http://env-host:11434",
            }

            cli_overrides = {
                "input": {"directory": str(in_cli)},
                "output": {"directory": str(out_cli)},
                "processing": {"max_workers": 8},
                "vlm": {"model": "cli-model"},
            }

            with patch.dict(os.environ, env_vars, clear=False):
                # When CLI overrides are provided: CLI wins over Env and YAML
                config = load_config(config_path=cfg_file, overrides=cli_overrides)
                self.assertEqual(config.input.resolved_path, in_cli.resolve())
                self.assertEqual(config.output.resolved_path, out_cli.resolve())
                self.assertEqual(config.processing.max_workers, 8)  # CLI
                self.assertEqual(config.vlm.model, "cli-model")  # CLI
                self.assertAlmostEqual(config.processing.confidence_threshold, 0.85)  # Env won over YAML
                self.assertEqual(config.vlm.endpoint, "http://env-host:11434/api/chat")  # Env won over YAML
                self.assertEqual(config.vlm.timeout, 100)  # YAML retained

    def test_relative_path_resolution_independent_of_cwd(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg_dir = tmp_path / "project_root"
            cfg_dir.mkdir()
            cfg_file = cfg_dir / "config.yaml"

            in_dir = cfg_dir / "rel_in"
            out_dir = cfg_dir / "rel_out"
            in_dir.mkdir()
            out_dir.mkdir()

            with open(cfg_file, "w", encoding="utf-8") as f:
                yaml.dump({
                    "input": {"directory": "rel_in"},
                    "output": {"directory": "rel_out"},
                }, f)

            unrelated_dir = tmp_path / "unrelated_cwd"
            unrelated_dir.mkdir()

            # Even if CWD is changed to an unrelated directory, relative paths
            # MUST resolve against cfg_file.parent, not CWD!
            orig_cwd = os.getcwd()
            try:
                os.chdir(unrelated_dir)
                config = load_config(config_path=cfg_file)
                self.assertEqual(config.input.resolved_path, in_dir.resolve())
                self.assertEqual(config.output.resolved_path, out_dir.resolve())
                self.assertNotEqual(config.input.resolved_path, (unrelated_dir / "rel_in").resolve())
            finally:
                os.chdir(orig_cwd)

    def test_absolute_and_mounted_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            in_mounted = tmp_path / "data" / "incoming"
            out_mounted = tmp_path / "data" / "sorted"
            in_mounted.mkdir(parents=True)
            out_mounted.mkdir(parents=True)

            overrides = {
                "input": {"directory": str(in_mounted.resolve())},
                "output": {"directory": str(out_mounted.resolve())},
            }
            config = load_config(overrides=overrides)
            self.assertEqual(config.input.resolved_path, in_mounted.resolve())
            self.assertEqual(config.output.resolved_path, out_mounted.resolve())

    def test_endpoint_normalization_variants(self):
        # Base url
        self.assertEqual(
            normalize_ollama_endpoint("http://localhost:11434"),
            "http://localhost:11434/api/chat",
        )
        # Trailing slash
        self.assertEqual(
            normalize_ollama_endpoint("http://localhost:11434/"),
            "http://localhost:11434/api/chat",
        )
        # Already has /api/chat
        self.assertEqual(
            normalize_ollama_endpoint("http://ollama-server:11434/api/chat"),
            "http://ollama-server:11434/api/chat",
        )
        # Trailing slash after /api/chat
        self.assertEqual(
            normalize_ollama_endpoint("http://ollama-server:11434/api/chat/"),
            "http://ollama-server:11434/api/chat",
        )
        # HTTPS
        self.assertEqual(
            normalize_ollama_endpoint("https://secure.ollama.lan:8443"),
            "https://secure.ollama.lan:8443/api/chat",
        )
        # No scheme defaults to http
        self.assertEqual(
            normalize_ollama_endpoint("ollama:11434"),
            "http://ollama:11434/api/chat",
        )
        # Invalid scheme
        with self.assertRaises(ConfigurationError):
            normalize_ollama_endpoint("ftp://localhost:11434")
        # Empty
        with self.assertRaises(ConfigurationError):
            normalize_ollama_endpoint("")
        with self.assertRaises(ConfigurationError):
            normalize_ollama_endpoint("   ")
        # Missing host
        with self.assertRaises(ConfigurationError):
            normalize_ollama_endpoint("http://")

    def test_invalid_setting_values_raise_configuration_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            in_dir = tmp_path / "valid_in"
            out_dir = tmp_path / "valid_out"
            in_dir.mkdir()
            out_dir.mkdir()

            base_overrides = {
                "input": {"directory": str(in_dir)},
                "output": {"directory": str(out_dir)},
            }

            # Empty input directory
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(overrides={"input": {"directory": ""}, "output": {"directory": str(out_dir)}})
            self.assertIn("input directory", str(ctx.exception).lower())

            # Empty output directory
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(overrides={"input": {"directory": str(in_dir)}, "output": {"directory": "  "}})
            self.assertIn("output directory", str(ctx.exception).lower())

            # Zero or negative workers
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(overrides={**base_overrides, "processing": {"max_workers": 0}})
            self.assertIn("worker count", str(ctx.exception).lower())

            # Invalid confidence
            with self.assertRaises(ConfigurationError) as ctx:
                cfg = load_config(overrides=base_overrides)
                cfg.processing.confidence_threshold = 1.5
                from kyc_sorter.utils.config_loader import validate_config
                validate_config(cfg)
            self.assertIn("confidence threshold", str(ctx.exception).lower())

            # Invalid CPU limit
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(overrides={**base_overrides, "processing": {"cpu_limit_percent": 150.0}})
            self.assertIn("cpu_limit_percent", str(ctx.exception).lower())

            # Invalid CPU throttle > CPU limit
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(overrides={**base_overrides, "processing": {"cpu_limit_percent": 60.0, "throttle_cpu_percent": 80.0}})
            self.assertIn("throttle", str(ctx.exception).lower())

            # Timeout <= 0
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(overrides={**base_overrides, "vlm": {"timeout": 0}})
            self.assertIn("timeout", str(ctx.exception).lower())

            # Empty VLM model
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(overrides={**base_overrides, "vlm": {"model": ""}})
            self.assertIn("model", str(ctx.exception).lower())

    def test_directory_conflicts_raise_configuration_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            same_dir = tmp_path / "same"
            same_dir.mkdir()

            # Same directory
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(overrides={"input": {"directory": str(same_dir)}, "output": {"directory": str(same_dir)}})
            self.assertIn("cannot be the same directory", str(ctx.exception))

            # Output inside input
            parent_in = tmp_path / "parent_in"
            nested_out = parent_in / "nested_out"
            parent_in.mkdir()
            nested_out.mkdir()
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(overrides={"input": {"directory": str(parent_in)}, "output": {"directory": str(nested_out)}})
            self.assertIn("cannot be inside Input directory", str(ctx.exception))

            # Input inside output
            parent_out = tmp_path / "parent_out"
            nested_in = parent_out / "nested_in"
            parent_out.mkdir()
            nested_in.mkdir()
            with self.assertRaises(ConfigurationError) as ctx:
                load_config(overrides={"input": {"directory": str(nested_in)}, "output": {"directory": str(parent_out)}})
            self.assertIn("cannot be inside Output directory", str(ctx.exception))

    def test_missing_config_file_safety(self):
        with self.assertRaises(ConfigurationError) as ctx:
            load_config(config_path="non_existent_file_path_12345.yaml")
        self.assertIn("does not exist", str(ctx.exception))

    def test_prevention_of_accidental_access_to_legacy_projects(self):
        forbidden_path_1 = "D:\\KYC_Document_sorter\\Source Files\\Input"
        forbidden_path_2 = "D:\\KYC_Document_sorter_OCR_VLM\\Source Files\\Input"

        with self.assertRaises(ConfigurationError) as ctx:
            load_config(overrides={"input": {"directory": forbidden_path_1}})
        self.assertIn("points to forbidden", str(ctx.exception).lower())

        with self.assertRaises(ConfigurationError) as ctx:
            load_config(overrides={"output": {"directory": forbidden_path_2}})
        self.assertIn("points to forbidden", str(ctx.exception).lower())

        # Explicit safe override should allow redirection
        redirected_cfg = load_config(
            overrides={"input": {"directory": forbidden_path_1}},
            allow_legacy_paths=True,
        )
        self.assertIn("KYC_Document_sorter_VLM_ONLY", str(redirected_cfg.input.resolved_path))

    def test_vlm_single_pass_and_image_options_defaults_and_overrides(self):
        cfg = load_config()
        self.assertFalse(cfg.vlm.single_pass_mode)
        self.assertEqual(cfg.vlm.image_max_dimension, 1200)
        self.assertEqual(cfg.vlm.image_quality, 85)

        # Override via dictionary
        cfg_override = load_config(overrides={
            "vlm": {
                "single_pass_mode": True,
                "image_max_dimension": 1600,
                "image_quality": 90,
            }
        })
        self.assertTrue(cfg_override.vlm.single_pass_mode)
        self.assertEqual(cfg_override.vlm.image_max_dimension, 1600)
        self.assertEqual(cfg_override.vlm.image_quality, 90)

    def test_vlm_invalid_image_dimension_and_quality_raises_error(self):
        with self.assertRaises(ConfigurationError) as ctx1:
            load_config(overrides={"vlm": {"image_max_dimension": 50}})
        self.assertIn("image_max_dimension", str(ctx1.exception))

        with self.assertRaises(ConfigurationError) as ctx2:
            load_config(overrides={"vlm": {"image_quality": 120}})
        self.assertIn("image_quality", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
