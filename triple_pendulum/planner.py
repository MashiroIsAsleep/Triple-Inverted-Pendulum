from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .control import (
    TimeVaryingLQRController,
    TrajectoryReference,
    make_equilibrium_controller,
    state_error,
)
from .equilibria import all_equilibria, get_equilibrium
from .model import TriplePendulumModel
from .simulation import SimulationResult, simulate

Array = np.ndarray


@dataclass(frozen=True)
class TransitionPlan:
    start: str
    target: str
    horizon: float
    accelerations: Array
    cart_position: float = 0.0
    reference_states: Array | None = None

    @property
    def node_times(self) -> Array:
        return np.linspace(0.0, self.horizon, self.accelerations.size)

    def acceleration(self, t: float) -> float:
        return float(np.interp(t, self.node_times, self.accelerations))


@dataclass(frozen=True)
class TransitionMetrics:
    max_angle_error_deg: float
    velocity_norm: float
    cart_error: float
    max_cart_travel: float
    max_acceleration: float

    @property
    def is_close(self) -> bool:
        return (
            self.max_angle_error_deg < 5.0
            and self.velocity_norm < 2.5
            and abs(self.cart_error) < 1.0
        )


@dataclass(frozen=True)
class TransitionSimulation:
    plan: TransitionPlan
    result: SimulationResult
    feedforward_metrics: TransitionMetrics
    final_metrics: TransitionMetrics


@dataclass(frozen=True)
class ChainStep:
    plan: TransitionPlan
    start_time: float
    handoff_time: float
    end_time: float
    feedforward_metrics: TransitionMetrics
    final_metrics: TransitionMetrics


@dataclass(frozen=True)
class ChainSimulation:
    route: tuple[str, ...]
    result: SimulationResult
    steps: tuple[ChainStep, ...]


STARTER_PLANS: dict[tuple[str, str], TransitionPlan] = {
    ("DDD", "UUU"): TransitionPlan(
        "DDD",
        "UUU",
        6.0,
        np.array(
            [
                0.0,
                5.751,
                0.07,
                4.379,
                2.966,
                -0.049,
                2.575,
                -8.044,
                2.161,
                -4.111,
                -10.662,
                1.177,
                -20.552,
                -11.142,
                17.03,
                18.501,
                0.0,
            ]
        ),
    ),
    ("DDD", "DDU"): TransitionPlan(
        "DDD",
        "DDU",
        2.5,
        np.array([0.0, 1.93, 0.577, -3.527, -2.507, 7.146, -6.243, 2.215, 0.0]),
    ),
    ("DDD", "DUD"): TransitionPlan(
        "DDD",
        "DUD",
        2.5,
        np.array([0.0, -0.757, -9.716, 3.481, 0.304, 2.789, -4.088, 27.548, 0.0]),
    ),
    ("DDD", "UDD"): TransitionPlan(
        "DDD",
        "UDD",
        2.5,
        np.array([0.0, -7.908, 5.276, 1.518, -0.175, 16.824, -10.175, -4.016, 0.0]),
    ),
    ("DUD", "DDD"): TransitionPlan(
        "DUD",
        "DDD",
        2.5,
        np.array([0.0, 3.513, 0.609, 0.337, -2.132, 0.588, -10.62, 4.174, 0.0]),
    ),
    ("DUU", "DDD"): TransitionPlan(
        "DUU",
        "DDD",
        2.5,
        np.array([0.0, -1.319, -3.712, -1.041, -3.996, 13.422, -5.831, 12.241, 0.0]),
    ),
    ("DDU", "DDD"): TransitionPlan(
        "DDU",
        "DDD",
        2.5,
        np.array([0.0, -4.62, 4.34, -2.82, 3.37, 3.62, 0.0]),
    ),
    ("DDU", "DUU"): TransitionPlan(
        "DDU",
        "DUU",
        2.5,
        np.array([0.0, 3.857, -8.624, 0.421, -5.182, 18.591, -16.608, 11.916, 0.0]),
    ),
    ("UDD", "DDD"): TransitionPlan(
        "UDD",
        "DDD",
        2.5,
        np.array([0.0, -3.12, 1.07, -2.88, 7.88, -1.46, 0.0]),
    ),
    ("UDU", "DDD"): TransitionPlan(
        "UDU",
        "DDD",
        4.5,
        np.array(
            [
                0.0,
                -1.448,
                3.833,
                -2.056,
                4.168,
                -1.181,
                -4.375,
                2.239,
                -7.376,
                5.743,
                -2.102,
                2.554,
                0.0,
            ]
        ),
    ),
    ("UUD", "DDD"): TransitionPlan(
        "UUD",
        "DDD",
        2.5,
        np.array([0.0, -4.897, -3.451, -4.748, 7.497, 3.874, 11.647, -0.526, 0.0]),
    ),
    ("UUU", "DDD"): TransitionPlan(
        "UUU",
        "DDD",
        2.5,
        np.array([0.0, -5.602, 3.578, -2.938, -0.862, 4.6, 9.537, -2.36, 0.0]),
    ),
}


def available_starter_transitions() -> tuple[tuple[str, str], ...]:
    return tuple(sorted(STARTER_PLANS))


def all_directed_transitions() -> tuple[tuple[str, str], ...]:
    names = tuple(sorted(all_equilibria()))
    return tuple((start, target) for start in names for target in names if start != target)


def missing_starter_transitions() -> tuple[tuple[str, str], ...]:
    available = set(available_starter_transitions())
    return tuple(pair for pair in all_directed_transitions() if pair not in available)


def transition_coverage() -> tuple[int, int]:
    return len(available_starter_transitions()), len(all_directed_transitions())


def find_starter_route(start: str, target: str) -> tuple[str, ...]:
    start = start.upper()
    target = target.upper()
    get_equilibrium(start)
    get_equilibrium(target)
    if start == target:
        return (start,)

    outgoing = {name: [] for name in all_equilibria()}
    for source, destination in available_starter_transitions():
        outgoing[source].append(destination)

    queue: list[tuple[str, ...]] = [(start,)]
    visited = {start}
    while queue:
        route = queue.pop(0)
        for next_name in sorted(outgoing[route[-1]]):
            if next_name in visited:
                continue
            next_route = (*route, next_name)
            if next_name == target:
                return next_route
            visited.add(next_name)
            queue.append(next_route)
    raise KeyError(f"no verified starter route from {start} to {target}")


def routable_transitions() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for start, target in all_directed_transitions():
        try:
            find_starter_route(start, target)
        except KeyError:
            continue
        pairs.append((start, target))
    return tuple(pairs)


def missing_routable_transitions() -> tuple[tuple[str, str], ...]:
    reachable = set(routable_transitions())
    return tuple(pair for pair in all_directed_transitions() if pair not in reachable)


def route_coverage() -> tuple[int, int]:
    return len(routable_transitions()), len(all_directed_transitions())


def has_starter_plan(start: str, target: str) -> bool:
    return (start.upper(), target.upper()) in STARTER_PLANS


def starter_plan(start: str, target: str) -> TransitionPlan:
    key = (start.upper(), target.upper())
    if key not in STARTER_PLANS:
        available = ", ".join(f"{a}->{b}" for a, b in STARTER_PLANS)
        raise KeyError(f"no starter plan for {start}->{target}; available: {available}")
    return STARTER_PLANS[key]


def simulate_plan(
    model: TriplePendulumModel,
    plan: TransitionPlan,
    dt: float = 0.005,
) -> SimulationResult:
    initial = get_equilibrium(plan.start).state(plan.cart_position)
    return simulate(
        model,
        initial,
        lambda t, state: plan.acceleration(t),
        duration=plan.horizon,
        dt=dt,
        mode="acceleration",
    )


def _simulate_feedforward(
    model: TriplePendulumModel,
    plan: TransitionPlan,
    initial_state: Array,
    dt: float,
    track_reference: bool,
) -> SimulationResult:
    if plan.reference_states is not None and track_reference:
        reference = TrajectoryReference(plan.node_times, plan.reference_states, plan.accelerations)
        tracker = TimeVaryingLQRController(model, reference)
        return simulate(
            model,
            initial_state,
            tracker.acceleration,
            duration=plan.horizon,
            dt=dt,
            mode="acceleration",
        )
    return simulate(
        model,
        initial_state,
        lambda t, state: plan.acceleration(t),
        duration=plan.horizon,
        dt=dt,
        mode="acceleration",
    )


def simulate_transition(
    model: TriplePendulumModel,
    plan: TransitionPlan,
    initial_state: Array | None = None,
    dt: float = 0.005,
    settle: float = 0.0,
    track_reference: bool = True,
) -> TransitionSimulation:
    if initial_state is None:
        initial_state = get_equilibrium(plan.start).state(plan.cart_position)
    feedforward = _simulate_feedforward(
        model,
        plan,
        np.asarray(initial_state, dtype=float),
        dt,
        track_reference,
    )
    feedforward_metrics = transition_metrics(feedforward, plan.target, plan.cart_position)
    result = feedforward
    final_metrics = feedforward_metrics
    if settle > 0.0:
        feedback = make_equilibrium_controller(model, plan.target, cart_position=plan.cart_position)
        settled = simulate(
            model,
            feedforward.states[-1],
            feedback.acceleration,
            duration=settle,
            dt=dt,
            mode="acceleration",
        )
        result = append_results(feedforward, settled)
        final_metrics = transition_metrics(result, plan.target, plan.cart_position)
    return TransitionSimulation(plan, result, feedforward_metrics, final_metrics)


def simulate_transition_chain(
    model: TriplePendulumModel,
    route: Sequence[str],
    dt: float = 0.005,
    settle: float = 2.0,
    track_reference: bool = True,
) -> ChainSimulation:
    route_names = tuple(item.upper() for item in route)
    if len(route_names) < 2:
        raise ValueError("route must contain at least two equilibria")

    current_state = get_equilibrium(route_names[0]).state(0.0)
    combined: SimulationResult | None = None
    steps: list[ChainStep] = []
    elapsed = 0.0
    for start, target in zip(route_names, route_names[1:]):
        plan = starter_plan(start, target)
        transition = simulate_transition(
            model,
            plan,
            initial_state=current_state,
            dt=dt,
            settle=settle,
            track_reference=track_reference,
        )
        handoff_time = elapsed + plan.horizon
        end_time = elapsed + transition.result.time[-1]
        steps.append(
            ChainStep(
                plan,
                elapsed,
                handoff_time,
                end_time,
                transition.feedforward_metrics,
                transition.final_metrics,
            )
        )
        combined = transition.result if combined is None else append_results(combined, transition.result)
        current_state = transition.result.states[-1]
        elapsed = end_time

    if combined is None:
        raise RuntimeError("chain simulation did not run any transitions")
    return ChainSimulation(route_names, combined, tuple(steps))


def transition_metrics(
    result: SimulationResult,
    target_name: str,
    cart_position: float = 0.0,
) -> TransitionMetrics:
    target = get_equilibrium(target_name).state(cart_position)
    error = state_error(result.states[-1], target)
    return TransitionMetrics(
        max_angle_error_deg=float(np.max(np.abs(np.rad2deg(error[:3])))),
        velocity_norm=float(np.linalg.norm(error[4:])),
        cart_error=float(error[3]),
        max_cart_travel=float(np.max(np.abs(result.states[:, 3] - cart_position))),
        max_acceleration=float(np.max(np.abs(result.controls))),
    )


def append_results(first: SimulationResult, second: SimulationResult) -> SimulationResult:
    if first.mode != second.mode:
        raise ValueError("can only append results with the same control mode")
    return SimulationResult(
        time=np.concatenate((first.time, first.time[-1] + second.time[1:])),
        states=np.vstack((first.states, second.states[1:])),
        controls=np.concatenate((first.controls, second.controls[1:])),
        mode=first.mode,
    )
