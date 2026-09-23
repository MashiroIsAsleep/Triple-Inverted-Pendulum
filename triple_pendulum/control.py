from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .equilibria import get_equilibrium
from .model import TriplePendulumModel

Array = np.ndarray


def wrap_to_pi(values: Array) -> Array:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def state_error(state: Array, target: Array) -> Array:
    error = np.asarray(state, dtype=float) - np.asarray(target, dtype=float)
    error[:3] = wrap_to_pi(error[:3])
    return error


def linearize_acceleration_input(
    model: TriplePendulumModel,
    state: Array,
    control: float = 0.0,
    step_state: float = 1e-6,
    step_input: float = 1e-6,
) -> tuple[Array, Array]:
    state = np.asarray(state, dtype=float)
    n = state.size
    a = np.zeros((n, n))
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = step_state
        fp = model.derivative(0.0, state + delta, control, mode="acceleration")
        fm = model.derivative(0.0, state - delta, control, mode="acceleration")
        a[:, i] = (fp - fm) / (2.0 * step_state)
    fp = model.derivative(0.0, state, control + step_input, mode="acceleration")
    fm = model.derivative(0.0, state, control - step_input, mode="acceleration")
    b = ((fp - fm) / (2.0 * step_input)).reshape(n, 1)
    return a, b


@dataclass
class ConstantReference:
    target_state: Array
    feedforward_acceleration: float = 0.0

    def state(self, t: float) -> Array:
        return self.target_state

    def acceleration(self, t: float) -> float:
        return self.feedforward_acceleration


class LQRFeedback:
    def __init__(
        self,
        model: TriplePendulumModel,
        reference_state: Array,
        q_weights: tuple[float, ...] = (200.0, 200.0, 200.0, 20.0, 30.0, 30.0, 30.0, 12.0),
        r_weight: float = 0.5,
    ):
        self.model = model
        self.reference_state = np.asarray(reference_state, dtype=float)
        self.a, self.b = linearize_acceleration_input(model, self.reference_state)
        self.gain = self._solve(q_weights, r_weight)

    def _solve(self, q_weights: tuple[float, ...], r_weight: float) -> Array:
        from scipy.linalg import solve_continuous_are

        q = np.diag(np.asarray(q_weights, dtype=float))
        r = np.array([[float(r_weight)]])
        p = solve_continuous_are(self.a, self.b, q, r)
        return np.linalg.solve(r, self.b.T @ p)

    def acceleration_correction(self, state: Array, target_state: Array | None = None) -> float:
        target = self.reference_state if target_state is None else target_state
        error = state_error(state, target)
        return -float((self.gain @ error.reshape(-1, 1))[0, 0])


class TwoDegreeOfFreedomController:
    """Feedforward plus feedback architecture following the paper.

    For equilibrium maintenance the feedforward term is zero and the feedback is
    constant LQR. Later swing-up work can replace ConstantReference by a
    constrained trajectory reference without touching the plant or feedback API.
    """

    def __init__(
        self,
        model: TriplePendulumModel,
        reference: ConstantReference,
        feedback: LQRFeedback,
    ):
        self.model = model
        self.reference = reference
        self.feedback = feedback

    def acceleration(self, t: float, state: Array) -> float:
        target = self.reference.state(t)
        command = self.reference.acceleration(t) + self.feedback.acceleration_correction(
            state, target
        )
        limit = self.model.params.acceleration_limit
        if limit is not None:
            command = float(np.clip(command, -limit, limit))
        return float(command)

    def force(self, t: float, state: Array) -> float:
        return self.model.force_for_cart_acceleration(state, self.acceleration(t, state))


@dataclass
class TrajectoryReference:
    node_times: Array
    states: Array
    accelerations: Array

    def state(self, t: float) -> Array:
        t = float(np.clip(t, self.node_times[0], self.node_times[-1]))
        return np.array([np.interp(t, self.node_times, self.states[:, i]) for i in range(8)])

    def acceleration(self, t: float) -> float:
        t = float(np.clip(t, self.node_times[0], self.node_times[-1]))
        return float(np.interp(t, self.node_times, self.accelerations))


class TimeVaryingLQRController:
    """Finite-horizon LQR tracker around a sampled trajectory reference."""

    def __init__(
        self,
        model: TriplePendulumModel,
        reference: TrajectoryReference,
        q_weights: tuple[float, ...] = (250.0, 250.0, 250.0, 30.0, 35.0, 35.0, 35.0, 15.0),
        r_weight: float = 0.25,
        terminal_scale: float = 15.0,
    ):
        self.model = model
        self.reference = reference
        self.q = np.diag(np.asarray(q_weights, dtype=float))
        self.r = np.array([[float(r_weight)]])
        self.gains = self._solve_gains(terminal_scale)

    def _solve_gains(self, terminal_scale: float) -> Array:
        count = self.reference.node_times.size
        gains = np.zeros((count - 1, 1, self.model.state_size))
        p = terminal_scale * self.q
        for k in range(count - 2, -1, -1):
            dt = float(self.reference.node_times[k + 1] - self.reference.node_times[k])
            state = self.reference.states[k]
            control = float(self.reference.accelerations[k])
            a_cont, b_cont = linearize_acceleration_input(self.model, state, control=control)
            a = np.eye(self.model.state_size) + dt * a_cont
            b = dt * b_cont
            system = self.r + b.T @ p @ b
            gain = np.linalg.solve(system, b.T @ p @ a)
            gains[k] = gain
            p = self.q + a.T @ p @ (a - b @ gain)
        return gains

    def _gain_at(self, t: float) -> Array:
        index = int(np.searchsorted(self.reference.node_times, t, side="right") - 1)
        index = int(np.clip(index, 0, self.gains.shape[0] - 1))
        return self.gains[index]

    def acceleration(self, t: float, state: Array) -> float:
        target = self.reference.state(t)
        error = state_error(state, target)
        correction = float((self._gain_at(t) @ error.reshape(-1, 1))[0, 0])
        command = self.reference.acceleration(t) - correction
        limit = self.model.params.acceleration_limit
        if limit is not None:
            command = float(np.clip(command, -limit, limit))
        return float(command)


def make_equilibrium_controller(
    model: TriplePendulumModel,
    equilibrium: str,
    cart_position: float,
) -> TwoDegreeOfFreedomController:
    target = get_equilibrium(equilibrium).state(cart_position)
    reference = ConstantReference(target)
    feedback = LQRFeedback(model, target)
    return TwoDegreeOfFreedomController(model, reference, feedback)
