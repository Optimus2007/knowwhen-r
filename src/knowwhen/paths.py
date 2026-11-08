"""Project locations. Every write goes through these so nothing lands off the SSD."""
from __future__ import annotations

import os
from pathlib import Path

# Hosted runs set KW_REMOTE=1 and their own locations. Locally the project root is the folder
# that holds this code, so a stray KW_ROOT cannot move writes somewhere else.
REMOTE = os.environ.get("KW_REMOTE") == "1"
LOCAL_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["KW_ROOT"]) if REMOTE else LOCAL_ROOT
ENV = Path(os.environ["KW_ENV"]) if REMOTE else Path("/Volumes/KWenv")
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
LOGS = ROOT / "experiments" / "logs"
WIKI_PARQUET = ENV / "data" / "wikipedia" / "20231101.en"
WIKI_INDEX = ENV / "data" / "wiki_tantivy"


def drive_of(path: Path) -> Path:
    """The mount point that holds path."""
    p = Path(path).resolve()
    while not os.path.ismount(p):
        p = p.parent
    return p


def check_storage() -> None:
    """Stop early if the SSD or environment image is missing."""
    if REMOTE:
        if not ROOT.is_dir() or not ENV.is_dir():
            raise RuntimeError(f"Remote run folders are missing: {ROOT}, {ENV}")
        return
    if not ROOT.is_dir() or drive_of(ROOT) == Path("/"):
        raise RuntimeError("Project drive is not mounted, or the project is on the internal disk")
    if not os.path.ismount(str(ENV)):
        raise RuntimeError("Environment image is not mounted. Run: source env.sh")


def inside_project(path: Path) -> Path:
    """Return the resolved path, refusing anything outside the SSD project or image."""
    resolved = Path(path).resolve()
    if not (resolved.is_relative_to(ROOT) or resolved.is_relative_to(ENV)):
        raise ValueError(f"Refusing to write outside the project: {resolved}")
    return resolved
