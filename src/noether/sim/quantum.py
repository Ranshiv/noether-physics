"""Quantum simulations, solved with QuTiP.

These are the systems the project was originally aimed at: a driven qubit, an
atom in a cavity, decoherence. Each one has a closed-form limit somewhere, and
each model states that limit and reports how far the numerics sit from it. A
quantum simulation that cannot be checked against anything is very hard to
trust, so where a check exists it is run automatically.

QuTiP 5 made ``c_ops`` and ``e_ops`` keyword arguments; the calls here reflect
that rather than the older positional form found in most tutorials.
"""

from __future__ import annotations

import math

import numpy as np

from .models import Model, Parameter, register
from .result import Convergence, Series, SimulationResult

#: QuTiP convention trap, verified empirically rather than assumed:
#:   sigmam() = [[0,0],[1,0]] maps |0> -> |1>
#:   destroy(2) = [[0,1],[0,0]] maps |1> -> |0>
#: They are adjoints of each other, so they disagree about which basis state is
#: excited. With sigmam() as the collapse operator, basis(2,0) is the EXCITED
#: state (sigma_z = +1) and basis(2,1) is the ground state. Getting this backwards
#: inverts every population label while leaving the dynamics -- and therefore the
#: analytic cross-checks -- looking perfectly correct.
EXCITED = 0
GROUND = 1


def _qutip():
    """Import QuTiP, with a message that says what to do if it is missing."""
    try:
        import qutip
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "quantum simulations need QuTiP: uv pip install -e '.[quantum]'"
        ) from exc
    return qutip


def _drift(coarse: list[np.ndarray], fine: list[np.ndarray]) -> float:
    """Largest disagreement between a run and its refinement, at shared times."""
    worst = 0.0
    for a, b in zip(coarse, fine):
        stride = (len(b) - 1) // (len(a) - 1) if len(a) > 1 else 1
        worst = max(worst, float(np.max(np.abs(np.asarray(a) - np.asarray(b)[::stride]))))
    return worst


# ---- Rabi oscillations -------------------------------------------------

def _run_rabi(rabi_frequency: float, detuning: float, decay: float, dephasing: float,
              duration: float, points: int) -> SimulationResult:
    qutip = _qutip()

    # H = (Delta/2) sigma_z + (Omega/2) sigma_x, in angular frequency units.
    hamiltonian = 0.5 * detuning * qutip.sigmaz() + 0.5 * rabi_frequency * qutip.sigmax()
    collapse = []
    if decay > 0:
        collapse.append(math.sqrt(decay) * qutip.sigmam())
    if dephasing > 0:
        collapse.append(math.sqrt(dephasing / 2) * qutip.sigmaz())

    excited = qutip.basis(2, EXCITED) * qutip.basis(2, EXCITED).dag()
    observables = [excited, qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]

    def solve(n: int):
        times = np.linspace(0.0, duration, n)
        result = qutip.mesolve(
            hamiltonian, qutip.basis(2, GROUND), times, c_ops=collapse, e_ops=observables
        )
        return times, result.expect

    times, expect = solve(points)
    _, refined = solve(points * 2 - 1)

    generalised = math.sqrt(rabi_frequency**2 + detuning**2)
    # On resonance and undamped, P_e = sin^2(Omega t / 2) exactly.
    analytic_note = ""
    max_analytic_error = None
    if decay == 0 and dephasing == 0:
        analytic = (rabi_frequency**2 / generalised**2) * np.sin(generalised * times / 2) ** 2
        max_analytic_error = float(np.max(np.abs(np.asarray(expect[0]) - analytic)))
        analytic_note = (
            f"Closed form available for this case; numerics agree to "
            f"{max_analytic_error:.2e}."
        )

    return SimulationResult(
        model_key="rabi",
        model_name="Driven qubit (Rabi oscillations)",
        parameters={"rabi_frequency": rabi_frequency, "detuning": detuning, "decay": decay,
                    "dephasing": dephasing, "duration": duration, "points": points},
        times=times.tolist(),
        series=[
            Series("excited_population", list(map(float, expect[0])), "", "Probability of the excited state"),
            Series("bloch_x", list(map(float, expect[1])), "", "<sigma_x>"),
            Series("bloch_y", list(map(float, expect[2])), "", "<sigma_y>"),
            Series("bloch_z", list(map(float, expect[3])), "", "<sigma_z>"),
        ],
        solver="QuTiP mesolve (Lindblad master equation)",
        convergence=Convergence(
            checked=True, refined_points=points * 2 - 1,
            max_drift=_drift([np.asarray(e) for e in expect], [np.asarray(e) for e in refined]),
        ),
        derived={
            "generalised_rabi_rad_s": generalised,
            "rabi_period_s": 2 * math.pi / generalised if generalised else None,
            "max_excited_population": float(np.max(expect[0])),
            "on_resonance": detuning == 0,
            "analytic_max_error": max_analytic_error,
        },
        caveats=[
            "Two-level approximation: real atoms and transmons have further levels that this ignores.",
            (
                "Rotating-wave approximation is built into this Hamiltonian; it fails when the drive "
                "strength approaches the transition frequency."
            ),
            (
                "Off resonance the qubit never fully inverts -- the ceiling is "
                f"{rabi_frequency**2 / generalised**2:.3f} here, not 1."
            ),
        ] + ([analytic_note] if analytic_note else []),
    )


register(Model(
    key="rabi",
    name="Driven qubit (Rabi oscillations)",
    domain="quantum",
    summary="A two-level system driven by a resonant field, with optional decay and dephasing.",
    everyday="Pushing a swing at exactly its natural rhythm makes it go higher and higher. Push "
             "slightly off-rhythm and it never reaches full height however long you push -- which "
             "is exactly what detuning does to a qubit.",
    equation=r"H = \frac{\Delta}{2}\sigma_z + \frac{\Omega}{2}\sigma_x",
    parameters=[
        Parameter("rabi_frequency", 6.283185, "rad/s", "Drive strength (Omega)", 0.0, 1e6),
        Parameter("detuning", 0.0, "rad/s", "Drive detuning from resonance (Delta)", -1e6, 1e6),
        Parameter("decay", 0.0, "1/s", "Energy relaxation rate (1/T1)", 0.0, 1e6),
        Parameter("dephasing", 0.0, "1/s", "Pure dephasing rate", 0.0, 1e6),
        Parameter("duration", 4.0, "s", "How long to simulate", 1e-9, 1e6),
        Parameter("points", 400, "count", "Grid points", 50, 20000),
    ],
    observables=["excited_population", "bloch_x", "bloch_y", "bloch_z"],
    runner=_run_rabi,
))


# ---- decoherence -------------------------------------------------------

def _run_decoherence(t1: float, t2: float, duration: float, points: int) -> SimulationResult:
    qutip = _qutip()

    if t2 > 2 * t1:
        return SimulationResult(
            "decoherence", "Qubit decoherence",
            error=(
                f"T2 = {t2:g} s exceeds 2*T1 = {2 * t1:g} s, which no physical qubit can do. "
                "T2 is bounded by 2*T1 because energy relaxation itself destroys phase coherence."
            ),
        )

    gamma1 = 1.0 / t1
    # 1/T2 = 1/(2 T1) + gamma_phi, so pure dephasing is what T2 has left over.
    gamma_phi = max(0.0, 1.0 / t2 - 1.0 / (2 * t1))

    collapse = [math.sqrt(gamma1) * qutip.sigmam()]
    if gamma_phi > 0:
        collapse.append(math.sqrt(gamma_phi / 2) * qutip.sigmaz())

    # Start on the equator: maximal coherence, so dephasing is visible.
    plus = (qutip.basis(2, EXCITED) + qutip.basis(2, GROUND)).unit()
    observables = [qutip.sigmax(), qutip.sigmaz(),
                   qutip.basis(2, EXCITED) * qutip.basis(2, EXCITED).dag()]

    def solve(n: int):
        times = np.linspace(0.0, duration, n)
        return times, qutip.mesolve(
            0 * qutip.sigmaz(), plus, times, c_ops=collapse, e_ops=observables
        ).expect

    times, expect = solve(points)
    _, refined = solve(points * 2 - 1)

    coherence = np.abs(np.asarray(expect[0]))
    analytic = np.exp(-times / t2)
    error = float(np.max(np.abs(coherence - analytic)))

    return SimulationResult(
        model_key="decoherence",
        model_name="Qubit decoherence",
        parameters={"t1": t1, "t2": t2, "duration": duration, "points": points},
        times=times.tolist(),
        series=[
            Series("coherence", coherence.tolist(), "", "|<sigma_x>|, the surviving superposition"),
            Series("analytic_envelope", analytic.tolist(), "", "exp(-t/T2), the expected decay"),
            Series("excited_population", list(map(float, expect[2])), "", "Population still excited"),
        ],
        solver="QuTiP mesolve (Lindblad master equation)",
        convergence=Convergence(
            checked=True, refined_points=points * 2 - 1,
            max_drift=_drift([np.asarray(e) for e in expect], [np.asarray(e) for e in refined]),
        ),
        derived={
            "t1_s": t1, "t2_s": t2, "pure_dephasing_rate_1_s": gamma_phi,
            "t2_limit_s": 2 * t1,
            "coherence_at_t2": float(np.interp(t2, times, coherence)) if duration >= t2 else None,
            "analytic_max_error": error,
        },
        caveats=[
            (
                "Markovian bath: the environment is assumed memoryless. Real 1/f noise in "
                "superconducting qubits is not, and produces non-exponential decay this cannot show."
            ),
            f"T2 is capped at 2*T1 = {2 * t1:g} s. Here T2 = {t2:g} s, leaving pure dephasing "
            f"at {gamma_phi:.4g} 1/s.",
            f"Coherence follows exp(-t/T2); numerics match that to {error:.2e}.",
        ],
    )


register(Model(
    key="decoherence",
    name="Qubit decoherence",
    domain="quantum",
    summary="A superposition decaying under energy relaxation (T1) and dephasing (T2).",
    everyday="Two identical pendulums started together drift out of step because each feels "
             "slightly different air currents. Decoherence is that, for a quantum superposition: "
             "the environment keeps 'measuring' it until the superposition is gone.",
    equation=r"\dot{\rho} = -\frac{i}{\hbar}[H,\rho] + \gamma_1 D[\sigma_-]\rho + \frac{\gamma_\phi}{2} D[\sigma_z]\rho",
    parameters=[
        Parameter("t1", 100e-6, "s", "Energy relaxation time", 1e-12, 1e3),
        Parameter("t2", 80e-6, "s", "Coherence time (must be <= 2*T1)", 1e-12, 1e3),
        Parameter("duration", 300e-6, "s", "How long to simulate", 1e-12, 1e3),
        Parameter("points", 400, "count", "Grid points", 50, 20000),
    ],
    observables=["coherence", "analytic_envelope", "excited_population"],
    runner=_run_decoherence,
))


# ---- Jaynes-Cummings ---------------------------------------------------

def _run_jaynes_cummings(coupling: float, detuning: float, photons: int, cavity_decay: float,
                         levels: int, duration: float, points: int) -> SimulationResult:
    qutip = _qutip()

    levels = int(levels)
    if photons >= levels:
        return SimulationResult(
            "jaynes_cummings", "Jaynes-Cummings model",
            error=(
                f"initial photon number {int(photons)} needs a Fock space larger than "
                f"{levels} levels, or the state is truncated away. Raise 'levels'."
            ),
        )

    a = qutip.tensor(qutip.destroy(levels), qutip.qeye(2))
    sm = qutip.tensor(qutip.qeye(levels), qutip.destroy(2))

    hamiltonian = detuning * sm.dag() * sm + coupling * (a.dag() * sm + a * sm.dag())
    collapse = [math.sqrt(cavity_decay) * a] if cavity_decay > 0 else []

    # Atom excited, cavity in a Fock state: the classic vacuum-Rabi setup.
    initial = qutip.tensor(qutip.basis(levels, int(photons)), qutip.basis(2, 1))
    observables = [sm.dag() * sm, a.dag() * a]

    def solve(n: int):
        times = np.linspace(0.0, duration, n)
        return times, qutip.mesolve(
            hamiltonian, initial, times, c_ops=collapse, e_ops=observables
        ).expect

    times, expect = solve(points)
    _, refined = solve(points * 2 - 1)

    vacuum_rabi = 2 * coupling * math.sqrt(int(photons) + 1)

    return SimulationResult(
        model_key="jaynes_cummings",
        model_name="Jaynes-Cummings model",
        parameters={"coupling": coupling, "detuning": detuning, "photons": photons,
                    "cavity_decay": cavity_decay, "levels": levels,
                    "duration": duration, "points": points},
        times=times.tolist(),
        series=[
            Series("atom_excited", list(map(float, expect[0])), "", "Probability the atom is excited"),
            Series("cavity_photons", list(map(float, expect[1])), "", "Mean photon number"),
        ],
        solver="QuTiP mesolve (Lindblad master equation)",
        convergence=Convergence(
            checked=True, refined_points=points * 2 - 1,
            max_drift=_drift([np.asarray(e) for e in expect], [np.asarray(e) for e in refined]),
        ),
        derived={
            "vacuum_rabi_frequency_rad_s": vacuum_rabi,
            "exchange_period_s": 2 * math.pi / vacuum_rabi if vacuum_rabi else None,
            "fock_levels": levels,
        },
        caveats=[
            (
                "Rotating-wave approximation; valid while the coupling is far below the transition "
                "frequency. Beyond that you need the quantum Rabi model."
            ),
            f"The Fock space is truncated at {levels} levels. If the mean photon number approaches "
            "that, the result is the truncation, not the physics.",
            f"Energy is exchanged at 2g*sqrt(n+1) = {vacuum_rabi:.4g} rad/s -- note that this is "
            "nonzero even with an empty cavity, which is the vacuum Rabi effect.",
        ],
    )


register(Model(
    key="jaynes_cummings",
    name="Jaynes-Cummings model",
    domain="quantum",
    summary="One atom exchanging a single excitation with one cavity mode.",
    everyday="Two pendulums joined by a weak spring: energy sloshes fully from one to the other "
             "and back. Here the 'spring' is the vacuum itself, which is why the exchange happens "
             "even when the cavity starts completely empty.",
    equation=r"H = \Delta\sigma^\dagger\sigma + g(a^\dagger\sigma + a\sigma^\dagger)",
    parameters=[
        Parameter("coupling", 1.0, "rad/s", "Atom-cavity coupling g", 0.0, 1e6),
        Parameter("detuning", 0.0, "rad/s", "Atom-cavity detuning", -1e6, 1e6),
        Parameter("photons", 0, "count", "Initial photon number", 0, 40),
        Parameter("cavity_decay", 0.0, "1/s", "Cavity loss rate kappa", 0.0, 1e6),
        # 20 levels is roughly where a laptop stops being instant; the ceiling is
        # about this machine, not about the physics.
        Parameter("levels", 12, "count", "Fock space truncation", 2, 60),
        Parameter("duration", 10.0, "s", "How long to simulate", 1e-9, 1e6),
        Parameter("points", 400, "count", "Grid points", 50, 20000),
    ],
    observables=["atom_excited", "cavity_photons"],
    runner=_run_jaynes_cummings,
))
