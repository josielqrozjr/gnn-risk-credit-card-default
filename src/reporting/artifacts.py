"""Atomic writing and integrity manifests for MVP artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactStore:
    """Write sanitized run evidence into one canonical directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def write_json(self, name: str, payload: Any) -> Path:
        destination = self.path(name)
        content = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        self._atomic_write(destination, content.encode("utf-8"))
        return destination

    def write_csv(self, name: str, frame: pd.DataFrame) -> Path:
        destination = self.path(name)
        self._atomic_write(destination, frame.to_csv(index=False).encode("utf-8"))
        return destination

    def write_manifest(self, metadata: dict[str, Any] | None = None) -> Path:
        """Hash every artifact except the manifest itself."""
        files = []
        for path in sorted(self.root.iterdir()):
            if not path.is_file() or path.name == "manifest.json":
                continue
            files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        manifest = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "files": files,
        }
        if metadata:
            manifest["run"] = metadata
        return self.write_json("manifest.json", manifest)

    @staticmethod
    def _atomic_write(destination: Path, content: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}."
        )
        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()