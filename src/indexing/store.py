import dataclasses
import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from src.core.chunker import Chunk

STATE_FILE = "index_state.json"
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"


@dataclass
class IndexState:
    last_indexed: dict[str, datetime] = field(default_factory=dict)


class IndexStore:
    """
    Persists the built index (chunks + embeddings) and per-doc timestamps to a directory.

    Point `directory` at a folder inside iCloud Drive (see `sync.icloud_subdir`
    in config.yaml) and iCloud's own file sync propagates the index to every
    other device signed into that account — no custom networking needed.
    A second device picks up already-embedded chunks instead of recomputing
    them. Conflicting concurrent writes from two devices are resolved the
    way iCloud resolves any file conflict (last writer wins, with a
    "filename 2" conflict copy if edits truly race).
    """

    def __init__(self, directory: pathlib.Path):
        self.directory = directory

    def load(self) -> tuple[IndexState, list[Chunk], Optional[np.ndarray]]:
        return self._load_state(), self._load_chunks(), self._load_embeddings()

    def save(self, state: IndexState, chunks: list[Chunk], embeddings: Optional[np.ndarray]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._save_state(state)
        self._save_chunks(chunks)
        self._save_embeddings(embeddings)

    def _load_state(self) -> IndexState:
        path = self.directory / STATE_FILE
        if not path.exists():
            return IndexState()
        with open(path) as f:
            data = json.load(f)
        return IndexState(last_indexed={k: datetime.fromisoformat(v) for k, v in data.items()})

    def _save_state(self, state: IndexState) -> None:
        data = {doc_id: ts.isoformat() for doc_id, ts in state.last_indexed.items()}
        self._atomic_write_json(self.directory / STATE_FILE, data)

    def _load_chunks(self) -> list[Chunk]:
        path = self.directory / CHUNKS_FILE
        if not path.exists():
            return []
        with open(path) as f:
            data = json.load(f)
        return [Chunk(**c) for c in data]

    def _save_chunks(self, chunks: list[Chunk]) -> None:
        data = [dataclasses.asdict(c) for c in chunks]
        self._atomic_write_json(self.directory / CHUNKS_FILE, data)

    def _load_embeddings(self) -> Optional[np.ndarray]:
        path = self.directory / EMBEDDINGS_FILE
        if not path.exists():
            return None
        return np.load(path)

    def _save_embeddings(self, embeddings: Optional[np.ndarray]) -> None:
        path = self.directory / EMBEDDINGS_FILE
        if embeddings is None:
            path.unlink(missing_ok=True)
            return
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, embeddings)
        tmp.replace(path)

    @staticmethod
    def _atomic_write_json(path: pathlib.Path, data) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)


def resolve_store_dir(config: dict, config_path: pathlib.Path) -> pathlib.Path:
    """Pick where the index lives: inside iCloud Drive if sync is enabled (so it
    propagates to every device on that iCloud account), otherwise next to the config."""
    sync_cfg = config.get("sync", {})
    icloud_cfg = config.get("icloud", {})
    if sync_cfg.get("enabled") and icloud_cfg.get("enabled") and icloud_cfg.get("root"):
        return pathlib.Path(icloud_cfg["root"]) / sync_cfg.get("icloud_subdir", ".cntxt-sync")
    return config_path.parent / ".cntxt-index"
