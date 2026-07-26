"""Numerical certificate for the co-invariant (Lukoyanov) expansion of the feature maps.

Certifies, on a genuine delay-differential-equation window, the hypotheses under which
the representation-agnostic verification theorem holds. The expansion tested is the one
of Lukoyanov (2000, equation (2.1)) and Gomoyunov--Lukoyanov (2024, Definition 7,
equation (3.6)), transported to the fixed-duration window: for every Lipschitz
continuation ``y`` of the window,

    Phi(w |> y_delta) - Phi(w)
        = horizontal_derivative * delta
        + instantaneous_state_derivative @ ( y(delta) - y(0) )
        + o(delta) ,

the ci-gradient being paired with the ENDPOINT DISPLACEMENT, not with a slope.

The augmented path is built with the TRUE time parametrisation
``(s, y(s), y(a)/(b-a) * s)``; for a uniformly sampled window this coincides with the
index parametrisation that ``SlidingSignatureJAX`` implements.

What each test establishes:

* ``test_expansion_holds_for_non_differentiable_continuation``
      the expansion holds along a Lipschitz continuation whose derivative at the origin
      does NOT exist -- the case a constant-slope definition cannot even state;
* ``test_slope_reading_fails_for_non_differentiable_continuation``
      pairing the ci-gradient with a slope instead fails, with a residual that does not
      converge; this is why the definition must quantify over all Lipschitz continuations;
* ``test_feature_path_is_lipschitz_in_time_across_the_knot``
      t -> Phi(x_t) is Lipschitz through the method-of-steps derivative jump, which is
      what makes the fundamental theorem of calculus applicable in the verification proof;
* ``test_signature_is_uniform_norm_lipschitz_only_on_velocity_bounded_sets``
      the signature feature map has no uniform-norm Lipschitz constant on the whole
      Lipschitz window space, but does have one on velocity-bounded subsets -- and the
      reachable windows are velocity-bounded;
* ``test_horizontal_derivative_can_fail_while_ci_gradient_survives``
      on a window with no one-sided derivative at its left endpoint the horizontal
      derivative does not exist, while the ci-gradient -- the only object the greedy
      control law contracts -- is unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest
from signax.module import SignatureTransform

# --------------------------------------------------------------------------------------
# A delay trajectory:  xdot(t) = A x(t) + A1 x(t - tau),  piecewise-C^1 (constant) history.
# --------------------------------------------------------------------------------------
DELAY = 1.0
STATE_DIMENSION = 2
SYSTEM_MATRIX = np.array([[0.0, 1.0], [-1.0, -0.5]])
DELAYED_STATE_MATRIX = -0.3 * np.eye(2)
INITIAL_HISTORY_VALUE = np.array([1.0, 0.0])
INTEGRATION_STEP = DELAY / 4000.0
FINAL_TIME = 3.0 * DELAY
SIGNATURE_DEPTH = 3
EVALUATION_TIME = 2.35 * DELAY

WINDOW_PAST_VERTICES = 400
WINDOW_APPENDED_VERTICES = 32
RAW_HISTORY_NODES = 8


def _integrate_delay_system() -> tuple[np.ndarray, np.ndarray]:
    history_steps = int(round(DELAY / INTEGRATION_STEP))
    forward_steps = int(round(FINAL_TIME / INTEGRATION_STEP))
    grid = np.arange(-history_steps, forward_steps + 1, dtype=float) * INTEGRATION_STEP
    states = np.zeros((grid.size, STATE_DIMENSION))
    states[: history_steps + 1] = INITIAL_HISTORY_VALUE

    def delayed_state(time_point: float) -> np.ndarray:
        position = (time_point + DELAY) / INTEGRATION_STEP
        lower = max(0, min(int(np.floor(position)), grid.size - 2))
        weight = position - lower
        return (1.0 - weight) * states[lower] + weight * states[lower + 1]

    def right_hand_side(time_point: float, state: np.ndarray) -> np.ndarray:
        return SYSTEM_MATRIX @ state + DELAYED_STATE_MATRIX @ delayed_state(time_point - DELAY)

    half = INTEGRATION_STEP / 2.0
    for index in range(history_steps, history_steps + forward_steps):
        time_point, state = grid[index], states[index]
        stage_1 = right_hand_side(time_point, state)
        stage_2 = right_hand_side(time_point + half, state + half * stage_1)
        stage_3 = right_hand_side(time_point + half, state + half * stage_2)
        stage_4 = right_hand_side(time_point + INTEGRATION_STEP, state + INTEGRATION_STEP * stage_3)
        states[index + 1] = state + INTEGRATION_STEP / 6.0 * (
            stage_1 + 2.0 * stage_2 + 2.0 * stage_3 + stage_4
        )
    return grid, states


_TIME_GRID, _STATE_GRID = _integrate_delay_system()
_signature_transform = SignatureTransform(depth=SIGNATURE_DEPTH)


def trajectory(time_points) -> np.ndarray:
    """Piecewise-linear interpolant of the solution: Lipschitz and piecewise C^1."""
    time_points = np.atleast_1d(np.asarray(time_points, dtype=float))
    return np.stack(
        [np.interp(time_points, _TIME_GRID, _STATE_GRID[:, j]) for j in range(STATE_DIMENSION)],
        axis=1,
    )


def augmented_signature(absolute_times: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Depth-3 signature of the Arribas-augmented polyline, true time parametrisation."""
    relative_times = (absolute_times - absolute_times[0]).reshape(-1, 1)
    basepoint_ramp = np.outer(relative_times[:, 0] / DELAY, states[0])
    augmented = np.concatenate([relative_times, states, basepoint_ramp], axis=1)
    return np.asarray(_signature_transform(augmented), dtype=float)


def window_vertices(
    window_end_time: float, extension_length: float, continuation
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices of the duration-DELAY window ending at ``window_end_time + extension_length``."""
    past_times = np.linspace(
        window_end_time + extension_length - DELAY, window_end_time, WINDOW_PAST_VERTICES
    )
    past_states = trajectory(past_times)
    if extension_length == 0.0:
        return (
            np.concatenate([past_times, np.full(WINDOW_APPENDED_VERTICES, window_end_time)]),
            np.vstack([past_states, np.repeat(past_states[-1:], WINDOW_APPENDED_VERTICES, axis=0)]),
        )
    elapsed = np.linspace(0.0, extension_length, WINDOW_APPENDED_VERTICES + 1)[1:]
    return (
        np.concatenate([past_times, window_end_time + elapsed]),
        np.vstack([past_states, np.asarray([continuation(step) for step in elapsed])]),
    )


def signature_feature(window_end_time, extension_length, continuation) -> np.ndarray:
    return augmented_signature(*window_vertices(window_end_time, extension_length, continuation))


def raw_history_feature(window_end_time, extension_length, continuation) -> np.ndarray:
    offsets = np.linspace(-DELAY, 0.0, RAW_HISTORY_NODES)
    end = window_end_time + extension_length
    samples = []
    for offset in offsets:
        sample_time = end + offset
        samples.append(
            continuation(sample_time - window_end_time)
            if sample_time > window_end_time
            else trajectory(sample_time)[0]
        )
    return np.concatenate(samples)


def markovian_feature(window_end_time, extension_length, continuation) -> np.ndarray:
    if extension_length > 0.0:
        return np.asarray(continuation(extension_length), dtype=float)
    return trajectory(window_end_time)[0]


FEATURE_MAPS = {
    "markovian": markovian_feature,
    "raw_history": raw_history_feature,
    "signature": signature_feature,
}


def affine_continuation(origin: np.ndarray, velocity: np.ndarray):
    return lambda elapsed: origin + velocity * elapsed


def oscillating_continuation(origin: np.ndarray, affine_part: np.ndarray, amplitude: np.ndarray):
    """y(s) = origin + a s + b s sin(log s): Lipschitz, with no derivative at s = 0."""

    def continuation(elapsed: float) -> np.ndarray:
        if elapsed <= 0.0:
            return origin.copy()
        return origin + affine_part * elapsed + amplitude * elapsed * np.sin(np.log(elapsed))

    return continuation


def coinvariant_pair(feature_map, window_end_time: float, reference_length: float):
    """Richardson-extrapolated (horizontal derivative, instantaneous-state derivative)."""
    origin = trajectory(window_end_time)[0]
    base = feature_map(window_end_time, 0.0, None)
    flat = affine_continuation(origin, np.zeros(STATE_DIMENSION))

    def horizontal_quotient(length: float) -> np.ndarray:
        return (feature_map(window_end_time, length, flat) - base) / length

    horizontal = 2.0 * horizontal_quotient(reference_length / 2) - horizontal_quotient(
        reference_length
    )

    columns = []
    for axis in range(STATE_DIMENSION):
        direction = np.zeros(STATE_DIMENSION)
        direction[axis] = 1.0

        def vertical_quotient(length: float, direction=direction) -> np.ndarray:
            sloped = feature_map(window_end_time, length, affine_continuation(origin, direction))
            return (sloped - feature_map(window_end_time, length, flat)) / length

        columns.append(2.0 * vertical_quotient(reference_length / 2) - vertical_quotient(
            reference_length
        ))
    return horizontal, np.stack(columns, axis=1)


AFFINE_PART = np.array([0.7, -0.4])
OSCILLATION_AMPLITUDE = np.array([0.5, 0.3])
EXTENSION_LENGTHS = np.array([2.0**-k for k in range(4, 13)]) * DELAY
REFERENCE_LENGTH = 2.0**-8


@pytest.mark.parametrize("feature_name", list(FEATURE_MAPS))
def test_expansion_holds_for_non_differentiable_continuation(feature_name):
    """The endpoint-displacement reading converges, at first order, for every continuation."""
    feature_map = FEATURE_MAPS[feature_name]
    origin = trajectory(EVALUATION_TIME)[0]
    horizontal, instantaneous = coinvariant_pair(feature_map, EVALUATION_TIME, REFERENCE_LENGTH)
    base = feature_map(EVALUATION_TIME, 0.0, None)
    continuation = oscillating_continuation(origin, AFFINE_PART, OSCILLATION_AMPLITUDE)

    normalised = []
    for length in EXTENSION_LENGTHS:
        increment = feature_map(EVALUATION_TIME, length, continuation) - base - horizontal * length
        displacement = continuation(length) - origin
        normalised.append(np.linalg.norm(increment - instantaneous @ displacement) / length)

    # The residual is o(delta): its normalised value falls by at least an order of
    # magnitude across the sweep, and the last value is small in absolute terms. For the
    # markovian map the remainder is identically zero, so the ratio test is vacuous there.
    assert normalised[-1] < 5e-3
    assert normalised[-1] < 1e-12 or normalised[-1] < normalised[0] / 10.0


@pytest.mark.parametrize("feature_name", list(FEATURE_MAPS))
def test_slope_reading_fails_for_non_differentiable_continuation(feature_name):
    """Pairing the ci-gradient with a slope does not converge: the definition needs the
    endpoint displacement, which is why it must quantify over all Lipschitz continuations."""
    feature_map = FEATURE_MAPS[feature_name]
    origin = trajectory(EVALUATION_TIME)[0]
    horizontal, instantaneous = coinvariant_pair(feature_map, EVALUATION_TIME, REFERENCE_LENGTH)
    base = feature_map(EVALUATION_TIME, 0.0, None)
    continuation = oscillating_continuation(origin, AFFINE_PART, OSCILLATION_AMPLITUDE)

    normalised = []
    for length in EXTENSION_LENGTHS:
        increment = feature_map(EVALUATION_TIME, length, continuation) - base - horizontal * length
        normalised.append(np.linalg.norm(increment - instantaneous @ (AFFINE_PART * length)) / length)

    # It oscillates instead of vanishing: the tail spread stays comparable to the head.
    tail = np.asarray(normalised[-6:])
    assert tail.max() - tail.min() > 0.1
    assert tail.max() > 0.1


@pytest.mark.parametrize("feature_name", list(FEATURE_MAPS))
def test_feature_path_is_lipschitz_in_time_across_the_knot(feature_name):
    """t -> Phi(x_t) is Lipschitz through the method-of-steps derivative jump at t = DELAY."""
    feature_map = FEATURE_MAPS[feature_name]
    quotients = []
    for length in (2.0**-6, 2.0**-9, 2.0**-11):
        for window_end_time in np.linspace(DELAY - 0.05, DELAY + 0.05, 11):
            here = feature_map(window_end_time, 0.0, None)
            there = feature_map(window_end_time + length, 0.0, None)
            quotients.append(np.linalg.norm(there - here) / length)
    quotients = np.asarray(quotients)
    assert np.isfinite(quotients).all()
    assert quotients.max() < 10.0  # a uniform bound; the measured maximum is near 2.7


def _area_enclosing_perturbation_ratio(oscillation_count: int, amplitude: float) -> float:
    vertices = max(1024, 64 * oscillation_count)
    times = np.linspace(EVALUATION_TIME - DELAY, EVALUATION_TIME, vertices)
    states = trajectory(times)
    phase = 2.0 * np.pi * oscillation_count * (times - times[0]) / DELAY
    perturbation = amplitude * np.stack([np.cos(phase) - 1.0, np.sin(phase)], axis=1)
    unperturbed = augmented_signature(times, states)
    perturbed = augmented_signature(times, states + perturbation)
    uniform_distance = np.max(np.linalg.norm(perturbation, axis=1))
    return float(np.linalg.norm(perturbed - unperturbed) / uniform_distance)


def test_signature_is_uniform_norm_lipschitz_only_on_velocity_bounded_sets():
    """No uniform-norm Lipschitz constant on the whole window space; one on velocity-bounded
    subsets, which is where the reachable windows live."""
    # Velocity unbounded (amplitude ~ k^{-1/2}): the ratio grows like sqrt(k).
    unbounded = [_area_enclosing_perturbation_ratio(k, k**-0.5) for k in (4, 64, 256)]
    assert unbounded[-1] > 5.0 * unbounded[0]

    # Velocity bounded (amplitude ~ c / k): the ratio settles.
    bounded = [_area_enclosing_perturbation_ratio(k, 0.2 / k) for k in (4, 64, 256)]
    assert max(bounded) / min(bounded) < 1.1


ROUGH_COEFFICIENT = 0.4


def _rough_window_states(times: np.ndarray, rough: bool) -> np.ndarray:
    states = np.stack([np.sin(1.3 * times + 0.4), np.cos(0.9 * times)], axis=1)
    if rough:
        radius = np.clip(times + DELAY, 1e-300, None)
        states[:, 0] += ROUGH_COEFFICIENT * radius * np.sin(np.log(radius))
    return states


def _rough_window_signature(extension_length: float, continuation, rough: bool) -> np.ndarray:
    past_times = np.linspace(extension_length - DELAY, 0.0, WINDOW_PAST_VERTICES)
    past_states = _rough_window_states(past_times, rough)
    if extension_length == 0.0:
        times = np.concatenate([past_times, np.zeros(WINDOW_APPENDED_VERTICES)])
        states = np.vstack(
            [past_states, np.repeat(past_states[-1:], WINDOW_APPENDED_VERTICES, axis=0)]
        )
    else:
        elapsed = np.linspace(0.0, extension_length, WINDOW_APPENDED_VERTICES + 1)[1:]
        times = np.concatenate([past_times, elapsed])
        states = np.vstack([past_states, np.asarray([continuation(step) for step in elapsed])])
    return augmented_signature(times, states)


def test_horizontal_derivative_can_fail_while_ci_gradient_survives():
    """On a window with no one-sided derivative at its LEFT endpoint the horizontal derivative
    does not converge, while the ci-gradient -- the only object the control law uses -- does."""
    lengths = [2.0**-k for k in range(4, 13)]
    results = {}
    for rough in (False, True):
        origin = _rough_window_states(np.array([0.0]), rough)[0]
        flat = affine_continuation(origin, np.zeros(STATE_DIMENSION))
        direction = np.zeros(STATE_DIMENSION)
        direction[0] = 1.0
        sloped = affine_continuation(origin, direction)
        base = _rough_window_signature(0.0, None, rough)

        horizontal_changes, vertical_changes = [], []
        previous_horizontal = previous_vertical = None
        for length in lengths:
            flat_value = _rough_window_signature(length, flat, rough)
            horizontal = (flat_value - base) / length
            vertical = (_rough_window_signature(length, sloped, rough) - flat_value) / length
            if previous_horizontal is not None:
                horizontal_changes.append(np.linalg.norm(horizontal - previous_horizontal))
                vertical_changes.append(np.linalg.norm(vertical - previous_vertical))
            previous_horizontal, previous_vertical = horizontal, vertical
        results[rough] = (np.asarray(horizontal_changes), np.asarray(vertical_changes))

    smooth_horizontal, smooth_vertical = results[False]
    rough_horizontal, rough_vertical = results[True]

    # Smooth left endpoint: both difference quotients are Cauchy.
    assert smooth_horizontal[-1] < smooth_horizontal[0] / 10.0
    assert smooth_vertical[-1] < smooth_vertical[0] / 10.0

    # Rough left endpoint: the horizontal quotient stops being Cauchy ...
    assert rough_horizontal[-3:].max() > 0.1
    # ... while the ci-gradient converges exactly as before.
    assert rough_vertical[-1] < rough_vertical[0] / 10.0
