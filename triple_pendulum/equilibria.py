from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Equilibrium:
    name: str
    angles: np.ndarray

    def state(self, cart_position: float = 0.0) -> np.ndarray:
        q = np.array([*self.angles, cart_position], dtype=float)
        return np.concatenate((q, np.zeros(4)))


def all_equilibria() -> dict[str, Equilibrium]:
    equilibria: dict[str, Equilibrium] = {}
    for bits in product((0, 1), repeat=3):
        name = "".join("D" if bit else "U" for bit in bits)
        angles = np.array([np.pi if bit else 0.0 for bit in bits], dtype=float)
        equilibria[name] = Equilibrium(name=name, angles=angles)
    return equilibria


def get_equilibrium(name: str) -> Equilibrium:
    key = name.upper()
    equilibria = all_equilibria()
    if key not in equilibria:
        raise KeyError(f"unknown equilibrium {name!r}; expected one of {', '.join(equilibria)}")
    return equilibria[key]
