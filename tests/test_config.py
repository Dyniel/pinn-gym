"""Tests for the YAML config schema and the CLI surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pinn_gym.cli import build_parser
from pinn_gym.config import (
    KNOWN_FIGURES,
    KNOWN_METHODS,
    KNOWN_PRESETS,
    GymConfig,
    dump_default_yaml,
    load_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


@pytest.mark.parametrize("name", ["default.yaml", "smoke.yaml", "sr_full.yaml"])
def test_shipped_configs_validate(name: str) -> None:
    cfg = load_config(CONFIGS_DIR / name)
    assert isinstance(cfg, GymConfig)
    assert cfg.run.stages, "every shipped config must declare at least one stage"
    for preset in cfg.materials.presets:
        assert isinstance(preset, str) and preset


def test_default_yaml_roundtrip(tmp_path: Path) -> None:
    path = dump_default_yaml(tmp_path / "default.yaml")
    cfg = load_config(path)
    again = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert again["run"]["stages"] == cfg.run.stages


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("run:\n  name: x\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError) as exc:
        load_config(path)
    assert "bogus" in str(exc.value).lower()


def test_unknown_method_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "train:\n  methods: [not_a_real_method]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_unknown_figure_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "plots:\n  figures: [made_up_figure]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_curve_limit_must_fit_in_envelope(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "candidate_pool:\n  max_displacement_mm: 30.0\n"
        "evaluate:\n  curve_limit_mm: 40.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_target_energy_auto_or_float(tmp_path: Path) -> None:
    path = tmp_path / "ok.yaml"
    path.write_text("evaluate:\n  target_energy_j: 12.5\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.evaluate.target_energy_j == 12.5

    path.write_text("evaluate:\n  target_energy_j: auto\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.evaluate.target_energy_j == "auto"


def test_precision_ks_must_be_positive(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("evaluate:\n  precision_ks: [0, 1, 5]\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_known_constants_are_aligned() -> None:
    assert "pa12" in KNOWN_PRESETS
    assert "pinn_full" in KNOWN_METHODS
    assert "curves_overlay" in KNOWN_FIGURES


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/file.yaml")


def test_cli_validate_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(["validate", str(CONFIGS_DIR / "smoke.yaml")])
    rc = args.func(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["presets"] == ["pa12"]


def _smoke_config_with_output_root(tmp_path: Path) -> Path:
    """Copy the shipped smoke config and inject run.output_root → tmp_path/runs."""
    data = yaml.safe_load((CONFIGS_DIR / "smoke.yaml").read_text(encoding="utf-8"))
    data.setdefault("run", {})["output_root"] = str(tmp_path / "runs")
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_cli_run_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg_path = _smoke_config_with_output_root(tmp_path)

    parser = build_parser()
    args = parser.parse_args(["run", str(cfg_path), "--dry-run"])
    rc = args.func(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["dry_run"] is True

    run_dir = Path(payload["run_dir"])
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
    for sub in ("datasets", "checkpoints", "figures", "tables", "logs"):
        assert (run_dir / sub).is_dir(), f"missing {sub}/ in run dir"


def test_cli_stage_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg_path = _smoke_config_with_output_root(tmp_path)

    parser = build_parser()
    args = parser.parse_args(["run", str(cfg_path), "--stage", "build", "--dry-run"])
    rc = args.func(args)
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    run_dir = Path(payload["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["run"]["stages"] == ["build"]


def test_cli_unknown_stage_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text((CONFIGS_DIR / "smoke.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(["run", str(cfg_path), "--stage", "totally_invented"])
    rc = args.func(args)
    assert rc == 2
