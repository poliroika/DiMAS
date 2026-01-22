import json
from pathlib import Path
from typing import Any

__all__ = ["InMemoryStateStorage", "FileStateStorage"]


class InMemoryStateStorage:
    """Простое in-memory хранилище состояний узлов."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def save(self, node_id: str, state: dict[str, Any]) -> None:
        self._store[node_id] = state

    def load(self, node_id: str) -> dict[str, Any] | None:
        return self._store.get(node_id)

    def delete(self, node_id: str) -> None:
        self._store.pop(node_id, None)

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()


class FileStateStorage:
    """Файловое JSON-хранилище состояний узлов."""

    def __init__(self, directory: Path | str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, node_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in node_id)
        return self._dir / f"{safe_id}.json"

    def save(self, node_id: str, state: dict[str, Any]) -> None:
        path = self._path(node_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"node_id": node_id, "state": state}, f, ensure_ascii=False, indent=2)

    def load(self, node_id: str) -> dict[str, Any] | None:
        path = self._path(node_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("state")

    def delete(self, node_id: str) -> None:
        path = self._path(node_id)
        if path.exists():
            path.unlink()

    def keys(self) -> list[str]:
        result = []
        for p in self._dir.glob("*.json"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "node_id" in data:
                        result.append(data["node_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        return result

    def clear(self) -> None:
        for p in self._dir.glob("*.json"):
            p.unlink()
