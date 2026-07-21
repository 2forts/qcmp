from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class GateProfile:
    """Resource cost assigned to one logical use of a gate/instruction.

    ``num_qubits`` may be ``None`` to match every arity with the same name.
    Delay and T-depth are local critical-path lengths for the gate's chosen
    decomposition, not total gate counts.
    """

    name: str
    quantum_cost: int
    delay: int
    t_count: int
    t_depth: int
    num_qubits: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        for field_name in ("quantum_cost", "delay", "t_count", "t_depth"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.num_qubits is not None and self.num_qubits < 1:
            raise ValueError("num_qubits must be >= 1 or None")

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> "GateProfile":
        return cls(
            name=name,
            num_qubits=data.get("num_qubits"),
            quantum_cost=int(data["quantum_cost"]),
            delay=int(data["delay"]),
            t_count=int(data["t_count"]),
            t_depth=int(data["t_depth"]),
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "quantum_cost": self.quantum_cost,
            "delay": self.delay,
            "t_count": self.t_count,
            "t_depth": self.t_depth,
            "description": self.description,
        }


class GateProfileRegistry:
    """Mutable registry of transparent resource profiles for large gates."""

    def __init__(self, profiles: list[GateProfile] | None = None) -> None:
        self._profiles: dict[tuple[str, int | None], GateProfile] = {}
        for profile in profiles or []:
            self.register(profile)

    def copy(self) -> "GateProfileRegistry":
        return GateProfileRegistry(list(self._profiles.values()))

    def register(self, profile: GateProfile, *, replace: bool = False) -> None:
        key = (profile.name, profile.num_qubits)
        if key in self._profiles and not replace:
            raise KeyError(
                f"Gate profile {profile.name!r}/{profile.num_qubits!r} already exists"
            )
        self._profiles[key] = profile

    def get(self, name: str, num_qubits: int) -> GateProfile | None:
        return self._profiles.get((name, num_qubits)) or self._profiles.get((name, None))

    def to_dict(self) -> dict[str, dict[str, Any]]:
        items: dict[str, dict[str, Any]] = {}
        for (_, arity), profile in sorted(
            self._profiles.items(), key=lambda item: (item[0][0], item[0][1] or -1)
        ):
            key = profile.name if arity is None else f"{profile.name}/{arity}"
            value = profile.to_dict()
            value.pop("name")
            items[key] = value
        return items

    @classmethod
    def from_json(cls, path: str | Path) -> "GateProfileRegistry":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Gate-profile JSON must contain an object")

        registry = cls()
        for raw_key, data in raw.items():
            if not isinstance(data, dict):
                raise ValueError(f"Profile {raw_key!r} must be an object")
            if "/" in raw_key:
                name, arity_text = raw_key.rsplit("/", 1)
                data = dict(data)
                data.setdefault("num_qubits", int(arity_text))
            else:
                name = raw_key
            registry.register(GateProfile.from_mapping(name, data), replace=True)
        return registry


# This is a convention, not a hardware claim. It can be replaced from Python
# or JSON. It is included because Toffoli is the only opaque >2-qubit primitive
# used directly by the current comparator implementations.
DEFAULT_GATE_PROFILES = GateProfileRegistry(
    [
        GateProfile(
            name="ccx",
            num_qubits=3,
            quantum_cost=5,
            delay=5,
            t_count=7,
            t_depth=3,
            description=(
                "Default textbook Toffoli convention. Override this profile "
                "to match the decomposition used in an experiment."
            ),
        )
    ]
)


def load_gate_profiles(
    path: str | Path | None,
    *,
    include_defaults: bool = True,
) -> GateProfileRegistry:
    registry = DEFAULT_GATE_PROFILES.copy() if include_defaults else GateProfileRegistry()
    if path is None:
        return registry
    custom = GateProfileRegistry.from_json(path)
    for profile in custom._profiles.values():
        registry.register(profile, replace=True)
    return registry
