"""Zero-hardcoded-path locators for bundled REVO3 assets, plans, and pretrained results."""
from __future__ import annotations

from pathlib import Path


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """Locate repository root containing assets/, plans/, and results/."""
    cur = package_dir()
    for candidate in (cur.parent.parent, cur.parent, Path.cwd()):
        if (candidate / "assets" / "revo3_right").is_dir():
            return candidate.resolve()
    return cur.parent.parent.resolve()


def default_assets_dir(side: str = "right") -> Path:
    if side != "right":
        raise ValueError(f"Only 'right' bundled assets are currently packaged (got {side!r})")
    p = repo_root() / "assets" / f"revo3_{side}"
    if not p.is_dir():
        raise FileNotFoundError(f"Bundled REVO3 assets directory not found at {p}")
    return p


def default_xml_path(side: str = "right") -> Path:
    p = default_assets_dir(side) / f"revo3_{side}.xml"
    if not p.is_file():
        raise FileNotFoundError(f"Bundled MuJoCo MJCF not found at {p}")
    return p


def default_usd_path(side: str = "right") -> Path:
    p = default_assets_dir(side) / f"revo3_{side}.usd"
    if not p.is_file():
        raise FileNotFoundError(f"Bundled Isaac Lab USD not found at {p}")
    return p


def default_plans_dir() -> Path:
    p = repo_root() / "plans"
    if not p.is_dir():
        raise FileNotFoundError(f"Bundled trajectory plans directory not found at {p}")
    return p


def default_pretrained_dir() -> Path:
    return repo_root() / "results" / "fresh50hz_v1"


def default_pretrained_json() -> Path:
    return default_pretrained_dir() / "revo3_right_identified.json"
