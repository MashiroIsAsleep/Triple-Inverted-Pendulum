from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .model import TriplePendulumModel

Array = np.ndarray


@dataclass
class SimulationResult:
    time: Array
    states: Array
    controls: Array
    mode: str


def _rk4_step(
    model: TriplePendulumModel,
    state: Array,
    t: float,
    dt: float,
    control: Callable[[float, Array], float],
    mode: str,
) -> tuple[Array, float]:
    u1 = float(control(t, state))
    k1 = model.derivative(t, state, u1, mode=mode)
    s2 = state + 0.5 * dt * k1
    k2 = model.derivative(t + 0.5 * dt, s2, float(control(t + 0.5 * dt, s2)), mode=mode)
    s3 = state + 0.5 * dt * k2
    k3 = model.derivative(t + 0.5 * dt, s3, float(control(t + 0.5 * dt, s3)), mode=mode)
    s4 = state + dt * k3
    k4 = model.derivative(t + dt, s4, float(control(t + dt, s4)), mode=mode)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), u1


def simulate(
    model: TriplePendulumModel,
    initial_state: Array,
    control: Callable[[float, Array], float],
    duration: float,
    dt: float = 0.002,
    mode: str = "acceleration",
) -> SimulationResult:
    if mode not in {"acceleration", "force"}:
        raise ValueError("mode must be 'acceleration' or 'force'")
    if dt <= 0.0 or duration <= 0.0:
        raise ValueError("dt and duration must be positive")
    steps = int(np.ceil(duration / dt))
    time = np.linspace(0.0, steps * dt, steps + 1)
    states = np.zeros((steps + 1, model.state_size))
    controls = np.zeros(steps + 1)
    states[0] = np.asarray(initial_state, dtype=float)
    state = states[0].copy()
    for i in range(steps):
        state, command = _rk4_step(model, state, time[i], dt, control, mode)
        states[i + 1] = state
        controls[i] = command
    controls[-1] = float(control(time[-1], states[-1]))
    return SimulationResult(time=time, states=states, controls=controls, mode=mode)
