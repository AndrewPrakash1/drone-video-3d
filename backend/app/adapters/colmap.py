"""Optional COLMAP SfM / MVS adapter. Never fail the parent job if missing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def colmap_available() -> bool:
    return shutil.which("colmap") is not None


def run_sfm(image_dir: Path, work_dir: Path) -> dict[str, Any] | None:
    binary = shutil.which("colmap")
    if not binary:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    sparse = work_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    try:
        subprocess.run(
            [binary, "feature_extractor", "--database_path", str(db), "--image_path", str(image_dir)],
            check=True,
            capture_output=True,
            timeout=180,
        )
        subprocess.run(
            [binary, "exhaustive_matcher", "--database_path", str(db)],
            check=True,
            capture_output=True,
            timeout=180,
        )
        subprocess.run(
            [
                binary,
                "mapper",
                "--database_path",
                str(db),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(sparse),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return {"sparse_dir": str(sparse), "status": "ok"}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return {"status": "skipped", "reason": str(exc)}
