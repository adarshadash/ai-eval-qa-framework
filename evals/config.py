"""Loads configs/eval_config.yaml (+ .env). EVAL_MODE / EVAL_CONFIG env vars override."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict:
    path = Path(os.getenv("EVAL_CONFIG", ROOT / "configs" / "eval_config.yaml"))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = os.getenv("EVAL_MODE", cfg.get("mode", "mock")).lower()
    if cfg["mode"] not in ("mock", "live"):
        raise ValueError(f"EVAL_MODE must be 'mock' or 'live', got {cfg['mode']!r}")
    if cfg["mode"] == "live" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("EVAL_MODE=live needs ANTHROPIC_API_KEY (set it in .env)")
    return cfg


def thresholds(suite: str) -> dict:
    return load_config()["thresholds"][suite]


def is_live() -> bool:
    return load_config()["mode"] == "live"


def load_dataset(name: str) -> list[dict]:
    """Load datasets/<name>.jsonl. Lines starting with // are comments."""
    return load_jsonl(DATASETS_DIR / f"{name}.jsonl")


def load_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    name = path.stem
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name}.jsonl line {line_no}: {exc}") from exc
    ids = [r["id"] for r in rows]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{name}.jsonl has duplicate ids: {dupes}")
    return rows
