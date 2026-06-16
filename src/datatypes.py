# src/datatypes.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)