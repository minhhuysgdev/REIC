"""Hierarchical ontology cho intent (3P / 1P verticals)."""

import json
from pathlib import Path
from typing import Iterator

from reic.models import IntentDefinition


class IntentOntology:
    """
    Ontology phân cấp intent theo structure.md:
    - Level 1: Order-related, Product, ...
    - Level 2: Shipping, Returns, ...
    - Level 3: ChangeShippingAddress, DeliveryStatus, ...
    """

    def __init__(self, intents: list[IntentDefinition]):
        self._intents = {i.id: i for i in intents}
        self._by_vertical: dict[str, list[IntentDefinition]] = {}
        for i in intents:
            self._by_vertical.setdefault(i.vertical, []).append(i)

    def get_intent(self, intent_id: str) -> IntentDefinition | None:
        return self._intents.get(intent_id)

    def list_intents(self, vertical: str | None = None) -> list[IntentDefinition]:
        if vertical is None:
            return list(self._intents.values())
        return self._by_vertical.get(vertical, [])

    def get_intents_in_branch(self, path_prefix: list[str]) -> list[IntentDefinition]:
        """Lấy các intent nằm trong nhánh hierarchy (cùng path prefix)."""
        return [
            i
            for i in self._intents.values()
            if len(i.path) >= len(path_prefix)
            and i.path[: len(path_prefix)] == path_prefix
        ]

    def iter_leaf_intents(self) -> Iterator[IntentDefinition]:
        """Chỉ lấy leaf intents (có examples)."""
        for i in self._intents.values():
            if i.examples:
                yield i

    @classmethod
    def from_json(cls, path: str | Path) -> "IntentOntology":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        intents = []
        for item in data.get("intents", []):
            intents.append(
                IntentDefinition(
                    id=item["id"],
                    name=item["name"],
                    description=item["description"],
                    path=item["path"],
                    examples=item.get("examples", []),
                    vertical=item.get("vertical", "3P"),
                )
            )
        return cls(intents)
