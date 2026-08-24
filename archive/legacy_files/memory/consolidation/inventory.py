"""Enterprise‑grade in‑memory inventory engine.

The engine stores arbitrary items identified by a UUID. It provides a simple
CRUD API plus optional JSON persistence to a file path supplied at
construction time. The implementation is deliberately lightweight – it does
not depend on any external database – but includes thorough error handling,
thread‑safety and type hints to satisfy production quality standards.

Typical usage::

    from ai.memory.consolidation.inventory import InventoryEngine

    engine = InventoryEngine()
    item = engine.add_item(name="patient-record", metadata={"risk": "high"})
    fetched = engine.get_item(item.id)
    engine.update_item(item.id, metadata={"risk": "low"})
    engine.remove_item(item.id)

The engine can also be pointed at a JSON file for durable storage:

    engine = InventoryEngine(storage_path="/tmp/inventory.json")
    engine.load()   # loads existing data if the file exists
    ...
    engine.save()   # writes current state back to the file

Thread‑safety is achieved via a ``threading.RLock`` – all public mutating
operations acquire the lock before touching the internal dictionary.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InventoryItem:
    """Represent a single inventory entry.

    Attributes
    ----------
    id:
        UUID string uniquely identifying the item.
    name:
        Human‑readable name for the item.
    metadata:
        Optional free‑form mapping with additional details.
    """

    id: str
    name: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON‑serialisable representation.
        ``asdict`` works for dataclasses but does not guarantee that ``metadata``
        is a plain ``dict``; we explicitly cast it for safety.
        """
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data


class InventoryEngine:
    """Manage a collection of :class:`InventoryItem` objects.

    The engine is deliberately agnostic of any persistence layer; callers can
    decide whether to keep the data in‑memory only or persist it to a JSON file
    via :meth:`save`/:meth:`load`.
    """

    def __init__(self, *, storage_path: Path | str | None = None) -> None:
        self._items: dict[str, InventoryItem] = {}
        self._lock = RLock()
        self._storage_path: Path | None = Path(storage_path) if storage_path else None
        if self._storage_path and self._storage_path.is_file():
            self.load()

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------
    def load(self) -> None:
        """Load inventory state from ``self._storage_path``.

        Raises
        ------
        FileNotFoundError
            If ``self._storage_path`` is not set or the file does not exist.
        json.JSONDecodeError
            If the file contents are not valid JSON.
        """
        if not self._storage_path:
            raise FileNotFoundError("No storage_path configured for InventoryEngine")
        with self._storage_path.open("r", encoding="utf-8") as f:
            raw_items: list[dict[str, object]] = json.load(f)
        with self._lock:
            for item in raw_items:
                meta = item.get("metadata", {})
                if not isinstance(meta, dict):
                    meta = {}
                self._items[str(item["id"])] = InventoryItem(
                    id=str(item["id"]),
                    name=str(item["name"]),
                    metadata=dict(meta),
                )
        log.debug("Inventory loaded from %s – %d items", self._storage_path, len(self._items))

    def save(self) -> None:
        """Serialise the current inventory to ``self._storage_path``.

        If no path was supplied during construction a ``FileNotFoundError`` is
        raised.  The output format is a JSON array where each element is the
        dictionary representation of an :class:`InventoryItem`.
        """
        if not self._storage_path:
            raise FileNotFoundError("No storage_path configured for InventoryEngine")
        with self._lock, self._storage_path.open("w", encoding="utf-8") as f:
            json.dump([item.to_dict() for item in self._items.values()], f, indent=2)
        log.debug("Inventory saved to %s – %d items", self._storage_path, len(self._items))

    # ---------------------------------------------------------------------
    # CRUD API
    # ---------------------------------------------------------------------
    def add_item(self, *, name: str, metadata: Mapping[str, object] | None = None) -> InventoryItem:
        """Create a new :class:`InventoryItem` and store it.

        Parameters
        ----------
        name:
            Descriptive name of the item.
        metadata:
            Optional mapping of additional attributes.
        """
        item_id = uuid.uuid4().hex
        item = InventoryItem(id=item_id, name=name, metadata=dict(metadata or {}))
        with self._lock:
            self._items[item_id] = item
        log.info("Added inventory item %s (%s)", item_id, name)
        return item

    def get_item(self, item_id: str) -> InventoryItem:
        """Return the item identified by ``item_id``.

        Raises
        ------
        KeyError
            If the ID is unknown.
        """
        with self._lock:
            try:
                return self._items[item_id]
            except KeyError as exc:
                raise KeyError(f"Inventory item {item_id!r} not found") from exc

    def update_item(
        self,
        item_id: str,
        *,
        name: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> InventoryItem:
        """Update ``name`` and/or ``metadata`` of an existing item.

        Returns the updated :class:`InventoryItem`.
        """
        with self._lock:
            existing = self.get_item(item_id)
            new_name = name if name is not None else existing.name
            new_meta: dict[str, object] = dict(existing.metadata)
            if metadata:
                new_meta.update(metadata)
            updated = InventoryItem(id=item_id, name=new_name, metadata=new_meta)
            self._items[item_id] = updated
        log.info("Updated inventory item %s", item_id)
        return updated

    def remove_item(self, item_id: str) -> None:
        """Delete the item identified by ``item_id``.

        Raises ``KeyError`` if the item does not exist.
        """
        with self._lock:
            try:
                del self._items[item_id]
            except KeyError as exc:
                raise KeyError(f"Inventory item {item_id!r} not found") from exc
        log.info("Removed inventory item %s", item_id)

    def list_items(self) -> list[InventoryItem]:
        """Return a list of all stored items (order is undefined)."""
        with self._lock:
            return list(self._items.values())

    def count(self) -> int:
        """Return the number of items currently stored."""
        with self._lock:
            return len(self._items)

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover – simple delegation
        return self.count()

    def __iter__(self) -> Iterable[InventoryItem]:  # pragma: no cover
        return iter(self.list_items())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<InventoryEngine items={len(self)}>"
