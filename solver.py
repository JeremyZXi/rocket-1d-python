"""Adaptive vertical-flight solver with explicit liftoff, rail exit and apogee."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from models import Rocket, atmosphere


@dataclass
class FlightResult:
    time: np.ndarray
    altitude: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    mass: np.ndarray
    mach: np.ndarray
    dynamic_pressure: np.ndarray
    liftoff_time: float | None
    rail_exit_time: float | None
    rail_exit_speed: float | None
    apogee_time: float | None

    def export_csv(self, path: Path):
        names = ['time', 'altitude', 'velocity', 'acceleration', 'mass', 'mach', 'dynamic_pressure']
        np.savetxt(path, np.column_stack([getattr(self, name) for name in names]),
                   delimiter=',', header=','.join(names), comments='')


def simulate(rocket: Rocket, *, sample_interval=0.01, max_time=100.0, rail_length=15.0):
    if sample_interval <= 0 or max_time <= 0 or rail_length <= 0:
        raise ValueError('Sample interval, duration and rail length must be positive')

    def acceleration(t, h, v):
        rho, sound = atmosphere(h)
        drag = 0.5 * rho * v * np.abs(v) * rocket.reference_area * rocket.aerodynamics.cd(np.abs(v) / sound)
        gravity = 9.80665 * (6371000 / (6371000 + h)) ** 2
        return (rocket.motor.force(t) - drag) / rocket.mass(t) - gravity

    # On the pad, thrust minus weight is piecewise linear. Find the first
    # interval that permits liftoff, including a jump at the first sample.
    times = rocket.motor.time
    margins = rocket.motor.thrust - rocket.mass(times) * 9.80665
    liftoff = None
    for i, (t, margin) in enumerate(zip(times, margins)):
        if margin > 0:
            if i == 0 or margins[i - 1] > 0:
                liftoff = float(t)
            else:
                liftoff = float(times[i - 1] + (t - times[i - 1]) * (-margins[i - 1]) / (margin - margins[i - 1]))
            break

    apogee_time = rail_time = rail_speed = None
    segments = []
    if liftoff is not None and liftoff < max_time:
        def rhs(t, state):
            h, v = state
            return [v, acceleration(t, h, v)]

        def apogee(t, state):
            return state[1]
        apogee.direction = -1
        apogee.terminal = True

        def rail_exit(t, state):
            return state[0] - rail_length
        rail_exit.direction = 1
        rail_exit.terminal = False

        state = [0.0, 0.0]
        # Split at curve knots to resolve ignition and burnout discontinuities.
        boundaries = sorted({liftoff, max_time, *(float(t) for t in times if liftoff < t < max_time)})
        for start, end in zip(boundaries, boundaries[1:]):
            # Endpoint evaluations use the interior curve on each segment.
            def segment_rhs(t, y):
                return rhs(np.clip(t, np.nextafter(start, end), np.nextafter(end, start)), y)
            solution = solve_ivp(segment_rhs, (start, end), state, events=(apogee, rail_exit),
                                 dense_output=True, rtol=1e-8, atol=1e-9, max_step=0.1)
            if not solution.success:
                raise RuntimeError(solution.message)
            segments.append(solution)
            if solution.t_events[1].size and rail_time is None:
                rail_time = float(solution.t_events[1][0])
                rail_speed = float(solution.y_events[1][0, 1])
            if solution.t_events[0].size:
                apogee_time = float(solution.t_events[0][0])
                break
            state = solution.y[:, -1]

    finish = apogee_time if apogee_time is not None else max_time
    sample_time = np.unique(np.r_[np.arange(0, finish, sample_interval), finish,
                                [liftoff] if liftoff is not None and liftoff <= finish else []])
    h = np.zeros_like(sample_time)
    v = np.zeros_like(sample_time)
    for solution in segments:
        mask = (sample_time >= solution.t[0]) & (sample_time <= solution.t[-1])
        h[mask], v[mask] = solution.sol(sample_time[mask])
    a = acceleration(sample_time, h, v)
    if liftoff is None:
        a[:] = 0
    else:
        a[sample_time < liftoff] = 0
    rho, sound = atmosphere(h)
    return FlightResult(sample_time, h, v, a, rocket.mass(sample_time), v / sound,
                        0.5 * rho * v ** 2, liftoff, rail_time, rail_speed, apogee_time)
