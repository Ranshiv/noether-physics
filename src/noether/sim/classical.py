"""Classical simulations, solved with SciPy.

These are the systems people actually have intuition for -- a swing, a car's
suspension, a thrown ball, a planet. Each carries an ``everyday`` description
because a simulation you cannot connect to anything you have seen teaches very
little.

Every model states what it leaves out. A pendulum model that quietly assumes
small angles and then gets asked about a 170-degree swing should say which of
its answers is the approximation and which is the physics.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from .models import Model, Parameter, register
from .result import Convergence, Series, SimulationResult

#: Solver tolerances. Tight enough that the convergence check usually passes,
#: loose enough that a 2000-point run stays instant.
_RTOL = 1e-9
_ATOL = 1e-11


def _solve(rhs, y0, t_end, points, args=()):
    times = np.linspace(0.0, t_end, points)
    solution = solve_ivp(
        rhs, (0.0, t_end), y0, t_eval=times, args=args,
        rtol=_RTOL, atol=_ATOL, method="DOP853",
    )
    return times, solution


def _convergence(rhs, y0, t_end, points, args, coarse) -> Convergence:
    """Re-run at double resolution and compare the final state."""
    _, fine = _solve(rhs, y0, t_end, points * 2 - 1, args)
    if not fine.success:
        return Convergence(checked=False, note="refinement run failed")
    drift = float(np.max(np.abs(fine.y[:, -1] - coarse.y[:, -1])))
    return Convergence(checked=True, refined_points=points * 2 - 1, max_drift=drift)


# ---- pendulum ----------------------------------------------------------

def _run_pendulum(length: float, gravity: float, theta0: float, damping: float,
                  duration: float, points: int) -> SimulationResult:
    theta0_rad = math.radians(theta0)

    def rhs(_t, y, L, g, b):
        theta, omega = y
        return [omega, -(g / L) * math.sin(theta) - b * omega]

    args = (length, gravity, damping)
    times, solution = _solve(rhs, [theta0_rad, 0.0], duration, points, args)

    if not solution.success:
        return SimulationResult("pendulum", "Pendulum", error=solution.message)

    theta = np.degrees(solution.y[0])
    omega = solution.y[1]
    # Energy per unit mass: a conserved quantity when undamped, so watching it
    # drift is an independent check on the integrator.
    energy = 0.5 * (length * omega) ** 2 + gravity * length * (1 - np.cos(solution.y[0]))

    small_angle_period = 2 * math.pi * math.sqrt(length / gravity)
    # First-order correction: T = T0 (1 + theta0^2/16 + ...)
    corrected = small_angle_period * (1 + theta0_rad**2 / 16)

    result = SimulationResult(
        model_key="pendulum",
        model_name="Pendulum",
        parameters={"length": length, "gravity": gravity, "theta0": theta0,
                    "damping": damping, "duration": duration, "points": points},
        times=times.tolist(),
        series=[
            Series("angle", theta.tolist(), "degrees", "Displacement from vertical"),
            Series("angular_velocity", omega.tolist(), "rad/s", "Rate of change of angle"),
            Series("energy", energy.tolist(), "J/kg", "Energy per unit mass"),
        ],
        solver="DOP853 (explicit Runge-Kutta 8)",
        convergence=_convergence(rhs, [theta0_rad, 0.0], duration, points, args, solution),
        derived={
            "small_angle_period_s": small_angle_period,
            "amplitude_corrected_period_s": corrected,
            "energy_drift": float(abs(energy[-1] - energy[0])) if damping == 0 else None,
        },
        caveats=[
            "Rigid massless rod, point mass, no air resistance beyond the damping term.",
            f"The small-angle formula T = 2*pi*sqrt(L/g) gives {small_angle_period:.4f} s; at "
            f"{theta0:g} degrees the true period is nearer {corrected:.4f} s. The simulation "
            "uses the full nonlinear equation, so it does not rely on that approximation.",
        ],
    )
    return result


register(Model(
    key="pendulum",
    name="Pendulum",
    domain="classical",
    summary="A mass swinging on a rod under gravity, solved without the small-angle approximation.",
    everyday="A playground swing, or a grandfather clock. Push it further and each swing takes "
             "slightly longer -- which is why pendulum clocks keep a small, fixed amplitude.",
    equation=r"\frac{d^2\theta}{dt^2} = -\frac{g}{L}\sin\theta - b\frac{d\theta}{dt}",
    parameters=[
        Parameter("length", 1.0, "m", "Rod length", 0.001, 1000.0),
        Parameter("gravity", 9.81, "m/s^2", "Gravitational acceleration", 0.01, 100.0),
        Parameter("theta0", 20.0, "degrees", "Initial angle from vertical", -179.0, 179.0),
        Parameter("damping", 0.0, "1/s", "Linear damping coefficient", 0.0, 10.0),
        Parameter("duration", 10.0, "s", "How long to simulate", 0.01, 1000.0),
        Parameter("points", 800, "count", "Grid points", 50, 20000),
    ],
    observables=["angle", "angular_velocity", "energy"],
    runner=_run_pendulum,
))


# ---- damped driven oscillator -----------------------------------------

def _run_oscillator(mass: float, stiffness: float, damping: float, drive_amplitude: float,
                    drive_frequency: float, x0: float, duration: float,
                    points: int) -> SimulationResult:
    def rhs(t, y, m, k, c, F, w):
        x, v = y
        return [v, (F * math.cos(w * t) - c * v - k * x) / m]

    args = (mass, stiffness, damping, drive_amplitude, drive_frequency)
    times, solution = _solve(rhs, [x0, 0.0], duration, points, args)

    if not solution.success:
        return SimulationResult("oscillator", "Damped driven oscillator", error=solution.message)

    natural = math.sqrt(stiffness / mass)
    quality = (math.sqrt(mass * stiffness) / damping) if damping > 0 else float("inf")
    regime = (
        "undamped" if damping == 0
        else "underdamped" if damping < 2 * math.sqrt(mass * stiffness)
        else "critically damped" if abs(damping - 2 * math.sqrt(mass * stiffness)) < 1e-9
        else "overdamped"
    )

    return SimulationResult(
        model_key="oscillator",
        model_name="Damped driven oscillator",
        parameters={"mass": mass, "stiffness": stiffness, "damping": damping,
                    "drive_amplitude": drive_amplitude, "drive_frequency": drive_frequency,
                    "x0": x0, "duration": duration, "points": points},
        times=times.tolist(),
        series=[
            Series("position", solution.y[0].tolist(), "m", "Displacement from equilibrium"),
            Series("velocity", solution.y[1].tolist(), "m/s", "Rate of change of position"),
        ],
        solver="DOP853 (explicit Runge-Kutta 8)",
        convergence=_convergence(rhs, [x0, 0.0], duration, points, args, solution),
        derived={
            "natural_frequency_rad_s": natural,
            "natural_frequency_hz": natural / (2 * math.pi),
            "quality_factor": quality,
            "regime": regime,
        },
        caveats=[
            "Linear spring (Hooke's law) and linear damping. Real springs stiffen when stretched far.",
            f"This system is {regime}. Resonance is sharpest when the drive frequency approaches "
            f"{natural:.4g} rad/s.",
        ],
    )


register(Model(
    key="oscillator",
    name="Damped driven oscillator",
    domain="classical",
    summary="A mass on a spring with friction and an optional periodic push.",
    everyday="A car's suspension. Too little damping and you bounce down the road; too much and "
             "every bump jolts you. Manufacturers aim near critical damping.",
    equation=r"m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = F\cos(\omega t)",
    parameters=[
        Parameter("mass", 1.0, "kg", "Mass", 1e-6, 1e6),
        Parameter("stiffness", 10.0, "N/m", "Spring constant", 1e-6, 1e9),
        Parameter("damping", 0.5, "N s/m", "Damping coefficient", 0.0, 1e6),
        Parameter("drive_amplitude", 0.0, "N", "Driving force amplitude", 0.0, 1e6),
        Parameter("drive_frequency", 3.0, "rad/s", "Driving angular frequency", 0.0, 1e6),
        Parameter("x0", 1.0, "m", "Initial displacement", -1e3, 1e3),
        Parameter("duration", 20.0, "s", "How long to simulate", 0.01, 1e4),
        Parameter("points", 1000, "count", "Grid points", 50, 20000),
    ],
    observables=["position", "velocity"],
    runner=_run_oscillator,
))


# ---- projectile with drag ---------------------------------------------

def _run_projectile(speed: float, angle: float, mass: float, drag_coefficient: float,
                    gravity: float, duration: float, points: int) -> SimulationResult:
    angle_rad = math.radians(angle)

    def rhs(_t, y, m, k, g):
        _x, _z, vx, vz = y
        speed_now = math.hypot(vx, vz)
        return [vx, vz, -k * speed_now * vx / m, -g - k * speed_now * vz / m]

    args = (mass, drag_coefficient, gravity)
    y0 = [0.0, 0.0, speed * math.cos(angle_rad), speed * math.sin(angle_rad)]
    times, solution = _solve(rhs, y0, duration, points, args)

    if not solution.success:
        return SimulationResult("projectile", "Projectile with drag", error=solution.message)

    x, z = solution.y[0], solution.y[1]
    # Stop reporting once it is underground; the equations keep going, physics does not.
    landed = np.argmax(z < 0) if np.any(z < 0) else len(z)
    landed = max(landed, 2)

    vacuum_range = speed**2 * math.sin(2 * angle_rad) / gravity
    actual_range = float(x[landed - 1])

    return SimulationResult(
        model_key="projectile",
        model_name="Projectile with air resistance",
        parameters={"speed": speed, "angle": angle, "mass": mass,
                    "drag_coefficient": drag_coefficient, "gravity": gravity,
                    "duration": duration, "points": points},
        times=times[:landed].tolist(),
        series=[
            Series("x", x[:landed].tolist(), "m", "Horizontal distance"),
            Series("height", z[:landed].tolist(), "m", "Height above launch"),
            Series("speed", np.hypot(solution.y[2], solution.y[3])[:landed].tolist(),
                   "m/s", "Speed"),
        ],
        solver="DOP853 (explicit Runge-Kutta 8)",
        convergence=_convergence(rhs, y0, duration, points, args, solution),
        derived={
            "range_m": actual_range,
            "vacuum_range_m": vacuum_range,
            "drag_penalty_m": vacuum_range - actual_range,
            "max_height_m": float(np.max(z[:landed])),
            "flight_time_s": float(times[landed - 1]),
        },
        caveats=[
            "Quadratic drag with a constant coefficient; no lift, no spin, no wind.",
            f"In vacuum this would travel {vacuum_range:.3g} m. With drag it manages "
            f"{actual_range:.3g} m -- the gap is why textbook projectile answers overshoot reality.",
        ],
    )


register(Model(
    key="projectile",
    name="Projectile with air resistance",
    domain="classical",
    summary="A thrown object under gravity and quadratic drag.",
    everyday="A thrown ball or a batted cricket ball. The textbook parabola is what you get with "
             "the air removed; real trajectories fall short and come down steeper than they went up.",
    equation=r"m\frac{d\vec{v}}{dt} = m\vec{g} - k|\vec{v}|\vec{v}",
    parameters=[
        Parameter("speed", 30.0, "m/s", "Launch speed", 0.01, 10000.0),
        Parameter("angle", 45.0, "degrees", "Launch angle above horizontal", -89.0, 89.0),
        Parameter("mass", 0.145, "kg", "Mass (default: a baseball)", 1e-6, 1e4),
        Parameter("drag_coefficient", 0.0013, "kg/m", "Quadratic drag coefficient", 0.0, 10.0),
        Parameter("gravity", 9.81, "m/s^2", "Gravitational acceleration", 0.01, 100.0),
        Parameter("duration", 10.0, "s", "Maximum flight time to simulate", 0.01, 1000.0),
        Parameter("points", 1000, "count", "Grid points", 50, 20000),
    ],
    observables=["x", "height", "speed"],
    runner=_run_projectile,
))
