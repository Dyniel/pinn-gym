"""Command-line entry point for pinn_gym.

Surface area:

* ``pinn-gym run <config.yaml> [--stage ...] [--dry-run]``
* ``pinn-gym validate <config.yaml>``
* ``pinn-gym status <run_dir>``
* ``pinn-gym materials --list``
* ``pinn-gym write-default <path.yaml>``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import KNOWN_PRESETS, KNOWN_STAGES, GymConfig, dump_default_yaml, load_config


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run as run_pipeline

    cfg = load_config(args.config)
    if args.stage:
        requested = [s.strip() for s in args.stage.split(",") if s.strip()]
        unknown = sorted(set(requested) - set(KNOWN_STAGES))
        if unknown:
            print(f"error: unknown stage(s): {unknown}", file=sys.stderr)
            return 2
        cfg = cfg.model_copy(update={"run": cfg.run.model_copy(update={"stages": requested})})
    paths = run_pipeline(cfg, dry_run=args.dry_run)
    print(json.dumps({"run_dir": str(paths.root), "dry_run": args.dry_run}, indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(json.dumps({"status": "ok", "stages": cfg.run.stages, "presets": cfg.materials.presets}, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        return 2
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        print(f"error: no manifest.json in {run_dir}", file=sys.stderr)
        return 2
    artefacts = {
        sub.name: sorted(p.name for p in sub.iterdir()) if sub.is_dir() else None
        for sub in sorted(run_dir.iterdir())
        if sub.is_dir()
    }
    payload = {
        "run_dir": str(run_dir),
        "manifest": json.loads(manifest.read_text(encoding="utf-8")),
        "artefacts": artefacts,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_materials(args: argparse.Namespace) -> int:
    if args.list:
        print(json.dumps({"presets": list(KNOWN_PRESETS)}, indent=2))
        return 0
    print("usage: pinn-gym materials --list", file=sys.stderr)
    return 2


def _cmd_write_default(args: argparse.Namespace) -> int:
    out = dump_default_yaml(args.out)
    print(json.dumps({"out": str(out)}, indent=2))
    return 0


def _cmd_reviewer_pack(args: argparse.Namespace) -> int:
    from .core.reviewer_pack import build_pack

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "publication_pack"
    payload = build_pack(run_dir, out_dir)
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_verify_pack(args: argparse.Namespace) -> int:
    from .core.reviewer_pack import verify_pack

    payload = verify_pack(Path(args.pack_dir))
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinn-gym",
        description="Open-source numerical gym for physics-informed lattice design.",
    )
    parser.add_argument("--version", action="version", version=f"pinn-gym {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Execute the configured pipeline.")
    p_run.add_argument("config", help="Path to a YAML config (see configs/).")
    p_run.add_argument(
        "--stage",
        help="Comma-separated subset of stages to run (default: all stages from config).",
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare run dir and manifest but skip the scientific stages.",
    )
    p_run.set_defaults(func=_cmd_run)

    p_val = sub.add_parser("validate", help="Schema-check a YAML config without running it.")
    p_val.add_argument("config")
    p_val.set_defaults(func=_cmd_validate)

    p_status = sub.add_parser("status", help="Summarise a previous run directory.")
    p_status.add_argument("run_dir")
    p_status.set_defaults(func=_cmd_status)

    p_mat = sub.add_parser("materials", help="Inspect built-in material presets.")
    p_mat.add_argument("--list", action="store_true", help="List preset names.")
    p_mat.set_defaults(func=_cmd_materials)

    p_def = sub.add_parser(
        "write-default",
        help="Write a YAML with every field populated to its default value.",
    )
    p_def.add_argument("out", help="Destination YAML path.")
    p_def.set_defaults(func=_cmd_write_default)

    p_pack = sub.add_parser(
        "reviewer-pack",
        help="Build a publication-ready reviewer pack from a reviewer_experiments run.",
    )
    p_pack.add_argument("--run-dir", required=True, help="reviewer_experiments_* run directory.")
    p_pack.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <run-dir>/publication_pack).",
    )
    p_pack.set_defaults(func=_cmd_reviewer_pack)

    p_verify = sub.add_parser("verify-pack", help="Verify reviewer pack files and SHA256SUMS.")
    p_verify.add_argument("pack_dir", help="Reviewer pack directory.")
    p_verify.set_defaults(func=_cmd_verify_pack)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
