from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

Array = np.ndarray


def _triple(values: Iterable[float], label: str) -> Array:
    array = np.asarray(tuple(values), dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{label} must contain exactly three values")
    return array


@dataclass
class TriplePendulumParams:
    """Physical parameters.

    Defaults use the identified link parameters reported by Glueck, Eder, and
    Kugi (Automatica, 2013). The cart mass only matters when converting desired
    cart acceleration to equivalent cart force.
    """

    cart_mass: float = 1.0
    masses: Iterable[float] = field(default_factory=lambda: (0.876, 0.938, 0.553))
    lengths: Iterable[float] = field(default_factory=lambda: (0.323, 0.419, 0.484))
    com_lengths: Iterable[float] = field(default_factory=lambda: (0.215, 0.269, 0.226))
    inertias: Iterable[float] = field(default_factory=lambda: (0.013, 0.024, 0.018))
    joint_damping: Iterable[float] = field(default_factory=lambda: (0.215, 0.002, 0.002))
    cart_damping: float = 0.0
    gravity: float = 9.81
    force_limit: float | None = 250.0
    acceleration_limit: float | None = 80.0

    def __post_init__(self) -> None:
        self.masses = _triple(self.masses, "masses")
        self.lengths = _triple(self.lengths, "lengths")
        self.com_lengths = _triple(self.com_lengths, "com_lengths")
        self.inertias = _triple(self.inertias, "inertias")
        self.joint_damping = _triple(self.joint_damping, "joint_damping")


class TriplePendulumModel:
    """Triple pendulum on a cart in the paper's coordinates.

    Generalized coordinates are q = [phi1, phi2, phi3, s], where phi_i are
    absolute angles from the upward vertical and s is cart position. State is
    [q, q_dot]. The control architecture uses cart acceleration as in the paper,
    while force-input dynamics and acceleration-to-force conversion are also
    provided for cart-force simulation.
    """

    dof = 4
    state_size = 8

    def __init__(self, params: TriplePendulumParams | None = None):
        self.params = params or TriplePendulumParams()
        self._lever = self._lever_matrix()
        self._angle_mass_levers = self.params.masses @ self._lever
        self._angle_coupling = self._angle_coupling_matrix()

    def split_state(self, state: Array) -> tuple[Array, Array]:
        state = np.asarray(state, dtype=float)
        if state.shape != (self.state_size,):
            raise ValueError(f"state must have shape ({self.state_size},)")
        return state[: self.dof], state[self.dof :]

    def _lever_matrix(self) -> Array:
        lever = np.zeros((3, 3))
        for body in range(3):
            for angle in range(3):
                if angle < body:
                    lever[body, angle] = self.params.lengths[angle]
                elif angle == body:
                    lever[body, angle] = self.params.com_lengths[angle]
        return lever

    def _angle_coupling_matrix(self) -> Array:
        coupling = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                coupling[i, j] = np.sum(
                    self.params.masses * self._lever[:, i] * self._lever[:, j]
                )
        return coupling

    def mass_matrix(self, q: Array) -> Array:
        q = np.asarray(q, dtype=float)
        phi = q[:3]
        mass = np.zeros((4, 4))
        mass[3, 3] = self.params.cart_mass + np.sum(self.params.masses)

        for i in range(3):
            term = -self._angle_mass_levers[i] * np.cos(phi[i])
            mass[i, 3] = term
            mass[3, i] = term

        for i in range(3):
            for j in range(3):
                mass[i, j] = self._angle_coupling[i, j] * np.cos(phi[i] - phi[j])
            mass[i, i] += self.params.inertias[i]
        return mass

    def mass_matrix_derivatives(self, q: Array) -> Array:
        q = np.asarray(q, dtype=float)
        phi = q[:3]
        deriv = np.zeros((4, 4, 4))

        for r in range(3):
            value = self._angle_mass_levers[r] * np.sin(phi[r])
            deriv[r, r, 3] = value
            deriv[r, 3, r] = value

        for r in range(3):
            for i in range(3):
                for j in range(3):
                    diff = phi[i] - phi[j]
                    selector = (1.0 if r == i else 0.0) - (1.0 if r == j else 0.0)
                    deriv[r, i, j] = -self._angle_coupling[i, j] * np.sin(diff) * selector
        return deriv

    def gravity_gradient(self, q: Array) -> Array:
        q = np.asarray(q, dtype=float)
        grad = np.zeros(4)
        grad[:3] = -self.params.gravity * self._angle_mass_levers * np.sin(q[:3])
        return grad

    def damping_force(self, q_dot: Array) -> Array:
        d1, d2, d3 = self.params.joint_damping
        damping = np.zeros(4)
        damping[:3] = np.array(
            [
                (d1 + d2) * q_dot[0] - d2 * q_dot[1],
                -d2 * q_dot[0] + (d2 + d3) * q_dot[1] - d3 * q_dot[2],
                -d3 * q_dot[1] + d3 * q_dot[2],
            ]
        )
        damping[3] = self.params.cart_damping * q_dot[3]
        return damping

    def coriolis_centrifugal(self, q: Array, q_dot: Array) -> Array:
        d_mass = self.mass_matrix_derivatives(q)
        terms = np.zeros(4)
        for i in range(4):
            total = 0.0
            for j in range(4):
                for k in range(4):
                    christoffel = 0.5 * (
                        d_mass[k, i, j] + d_mass[j, i, k] - d_mass[i, j, k]
                    )
                    total += christoffel * q_dot[j] * q_dot[k]
            terms[i] = total
        return terms

    def potential_energy(self, q: Array) -> float:
        heights = self._lever @ np.cos(np.asarray(q, dtype=float)[:3])
        return float(np.sum(self.params.masses * self.params.gravity * heights))

    def force_input_acceleration(self, q: Array, q_dot: Array, force: float) -> Array:
        if self.params.force_limit is not None:
            force = float(np.clip(force, -self.params.force_limit, self.params.force_limit))
        generalized = np.array([0.0, 0.0, 0.0, force])
        rhs = (
            generalized
            - self.coriolis_centrifugal(q, q_dot)
            - self.damping_force(q_dot)
            - self.gravity_gradient(q)
        )
        return np.linalg.solve(self.mass_matrix(q), rhs)

    def acceleration_input_acceleration(
        self, q: Array, q_dot: Array, cart_acceleration: float
    ) -> Array:
        if self.params.acceleration_limit is not None:
            cart_acceleration = float(
                np.clip(
                    cart_acceleration,
                    -self.params.acceleration_limit,
                    self.params.acceleration_limit,
                )
            )
        mass = self.mass_matrix(q)
        bias = (
            self.coriolis_centrifugal(q, q_dot)
            + self.damping_force(q_dot)
            + self.gravity_gradient(q)
        )
        phi_ddot = np.linalg.solve(
            mass[:3, :3],
            -mass[:3, 3] * cart_acceleration - bias[:3],
        )
        return np.array([*phi_ddot, cart_acceleration], dtype=float)

    def force_for_cart_acceleration(self, state: Array, cart_acceleration: float) -> float:
        q, q_dot = self.split_state(state)
        q_ddot = self.acceleration_input_acceleration(q, q_dot, cart_acceleration)
        mass = self.mass_matrix(q)
        bias = (
            self.coriolis_centrifugal(q, q_dot)
            + self.damping_force(q_dot)
            + self.gravity_gradient(q)
        )
        force = float(mass[3, :] @ q_ddot + bias[3])
        if self.params.force_limit is not None:
            force = float(np.clip(force, -self.params.force_limit, self.params.force_limit))
        return force

    def derivative(self, t: float, state: Array, control: float, mode: str = "acceleration") -> Array:
        q, q_dot = self.split_state(state)
        if mode == "acceleration":
            q_ddot = self.acceleration_input_acceleration(q, q_dot, control)
        elif mode == "force":
            q_ddot = self.force_input_acceleration(q, q_dot, control)
        else:
            raise ValueError("mode must be 'acceleration' or 'force'")
        return np.concatenate((q_dot, q_ddot))

    def link_points(self, state_or_q: Array) -> Array:
        values = np.asarray(state_or_q, dtype=float)
        q = values[:4]
        phi = q[:3]
        cursor = np.array([q[3], 0.0], dtype=float)
        points = [cursor.copy()]
        for i, length in enumerate(self.params.lengths):
            cursor = cursor + np.array([-length * np.sin(phi[i]), length * np.cos(phi[i])])
            points.append(cursor.copy())
        return np.asarray(points)
