"""The concept library: curated physics, checked at read time.

Entries here are written by hand and deliberately so. Every equation is
dimension-checked and every worked example is recomputed when the entry is read,
so an error introduced here surfaces as a failing check rather than as confident
prose. Adding a concept means adding data, not code.

The analogies are chosen to be things a reader has physically felt -- a swing, a
car's suspension, a thrown ball -- because an analogy to something equally
abstract explains nothing.
"""

from __future__ import annotations

from .concepts import Concept, WorkedExample

CONCEPTS: list[Concept] = []


def add(concept: Concept) -> Concept:
    CONCEPTS.append(concept)
    return concept


# ---- classical ---------------------------------------------------------

add(Concept(
    key="simple-harmonic-motion",
    name="Simple harmonic motion",
    aliases=["shm", "harmonic oscillator", "spring", "oscillation"],
    domain="classical",
    definition="Motion where the restoring force is proportional to displacement and points back "
               "toward equilibrium, producing a sinusoid whose period does not depend on amplitude.",
    everyday="A child on a swing. Pull them back twice as far and they swing twice as wide — but "
             "each swing still takes the same time. That amplitude-independence is why pendulums "
             "were the world's clocks for three centuries.",
    detail=[
        "The defining property is F = -kx: push it further from rest and the force pulling it back "
        "grows in exact proportion. Nothing else in physics is quite so well behaved, which is why "
        "this one system stands behind so much else.",
        "Because the force is linear, the period is set by mass and stiffness alone. Amplitude "
        "cancels out. Double the swing and the mass moves twice as far, but it also feels twice "
        "the force, so it moves twice as fast and arrives at the same moment.",
        "Almost nothing is exactly harmonic — but almost everything is harmonic *near equilibrium*, "
        "because the bottom of any smooth potential well looks parabolic if you get close enough. "
        "That is why this model reappears in molecular vibrations, circuits, and quantum fields.",
    ],
    equations=[
        (r"F = -k x", "Restoring force is proportional to displacement, opposing it",
         {"F": "newton", "k": "newton/meter", "x": "meter"}),
        (r"\omega = \sqrt{\frac{k}{m}}", "Angular frequency from stiffness and mass",
         {"omega": "radian/second", "k": "newton/meter", "m": "kilogram"}),
        (r"T = 2 \pi \sqrt{\frac{m}{k}}", "Period — note that amplitude does not appear",
         {"T": "second", "m": "kilogram", "k": "newton/meter"}),
    ],
    examples=[
        WorkedExample(
            question="A 0.5 kg mass hangs from a spring with stiffness 20 N/m. What is its period?",
            given={"m": (0.5, "kilogram"), "k": (20.0, "newton/meter")},
            equation=r"T = 2 \pi \sqrt{\frac{m}{k}}",
            steps=[
                "The period depends only on mass and stiffness — not on how far you pull it.",
                "T = 2*pi*sqrt(m/k) = 2*pi*sqrt(0.5/20) = 2*pi*sqrt(0.025).",
                "sqrt(0.025) = 0.1581, so T = 2*pi*0.1581.",
            ],
            expected=0.9934588266,
            units="s",
            sanity="About one second — the same ballpark as a hanging mass you would bounce by "
                   "hand, which is a good sign the arithmetic is right.",
        ),
    ],
    simulation="oscillator",
    simulation_parameters={"mass": 0.5, "stiffness": 20.0, "damping": 0.0, "duration": 5.0},
    misconceptions=[
        "Amplitude does not change the period. Larger swings travel further but also move faster.",
        "A heavier mass oscillates *more slowly*, not faster — mass is in the numerator of the period.",
        "A real pendulum is only approximately harmonic. Past about 20 degrees the period lengthens "
        "measurably, because sin(theta) stops being close to theta.",
    ],
    reading_query="cat:physics.class-ph AND abs:oscillator",
    related=["resonance", "damping"],
))


add(Concept(
    key="resonance",
    name="Resonance",
    aliases=["resonant frequency", "driven oscillator"],
    domain="classical",
    definition="The large response a system gives when driven at its natural frequency, where each "
               "push arrives in step with the motion and energy accumulates.",
    everyday="Pushing a swing. Push at the rhythm the swing already has and it climbs higher with "
             "every push. Push at any other rhythm and you spend half your effort fighting the "
             "swing's own motion.",
    detail=[
        "A driven oscillator responds most strongly when the driving frequency matches its natural "
        "frequency. At that point the force is always in phase with the velocity, so every push "
        "adds energy instead of partly cancelling.",
        "What stops the response growing without limit is damping. The peak height is set by the "
        "quality factor Q: roughly, the amplitude at resonance is Q times the response you would "
        "get from a steady force of the same size.",
        "This is why soldiers break step on bridges, why a wine glass shatters at one particular "
        "note, and why an MRI machine works — nuclear spins absorb energy only at their own "
        "precession frequency.",
    ],
    equations=[
        (r"\omega_0 = \sqrt{\frac{k}{m}}", "Natural frequency of the undriven system",
         {"omega_0": "radian/second", "k": "newton/meter", "m": "kilogram"}),
        (r"Q = \frac{\sqrt{m k}}{c}", "Quality factor: how many oscillations before energy drains",
         {"Q": "dimensionless", "m": "kilogram", "k": "newton/meter", "c": "newton*second/meter"}),
    ],
    examples=[
        WorkedExample(
            question="A 2 kg mass on a 200 N/m spring has damping 4 N s/m. What is its Q?",
            given={"m": (2.0, "kilogram"), "k": (200.0, "newton/meter"),
                   "c": (4.0, "newton*second/meter")},
            equation=r"Q = \frac{\sqrt{m k}}{c}",
            steps=[
                "Q compares stored energy to energy lost per cycle.",
                "Q = sqrt(m*k)/c = sqrt(2*200)/4 = sqrt(400)/4 = 20/4.",
            ],
            expected=5.0,
            units="",
            sanity="Q = 5 is lightly damped: the amplitude at resonance is about five times the "
                   "static response, and it rings for a few cycles after you stop pushing.",
        ),
    ],
    simulation="oscillator",
    simulation_parameters={"mass": 2.0, "stiffness": 200.0, "damping": 4.0,
                           "drive_amplitude": 5.0, "drive_frequency": 10.0, "x0": 0.0,
                           "duration": 30.0},
    misconceptions=[
        "Resonance does not mean infinite amplitude. Damping always caps it, and real materials "
        "stop behaving linearly long before the mathematical peak.",
        "The peak of a driven, damped system sits slightly *below* the undamped natural frequency.",
    ],
    reading_query="cat:physics.class-ph AND abs:resonance",
    related=["simple-harmonic-motion", "damping"],
))


add(Concept(
    key="air-resistance",
    name="Air resistance",
    aliases=["drag", "projectile motion", "terminal velocity"],
    domain="classical",
    definition="The force opposing motion through a fluid, growing roughly with the square of "
               "speed, which makes real trajectories fall well short of textbook parabolas.",
    everyday="Put your hand out of a car window. At 30 km/h it is a gentle push; at 100 km/h it is "
             "hard to hold steady. The force grew by about eleven times for a threefold speed "
             "increase — that is the square law you can feel.",
    detail=[
        "At everyday speeds drag goes as v^2, so it matters enormously at the fast start of a "
        "flight and hardly at all near the top of an arc. That asymmetry is why a struck ball rises "
        "along a long shallow curve and comes down noticeably steeper.",
        "Because drag depends on speed and speed depends on drag, there is no closed-form "
        "trajectory. This is one of the first places in physics where you must integrate "
        "numerically — which is exactly what the linked simulation does.",
        "When drag grows to balance gravity, acceleration stops and the object falls at constant "
        "terminal velocity. A skydiver reaches about 55 m/s spread-eagled, roughly 80 m/s head-down.",
    ],
    equations=[
        (r"F = k v^2", "Quadratic drag: doubling speed quadruples the force",
         {"F": "newton", "k": "kilogram/meter", "v": "meter/second"}),
        (r"R = \frac{v^2 \sin(2 \theta)}{g}", "Range in vacuum — the answer drag makes wrong",
         {"R": "meter", "v": "meter/second", "g": "meter/second**2"}),
    ],
    examples=[
        WorkedExample(
            question="A ball is thrown at 30 m/s at 45 degrees. How far would it go in a vacuum?",
            given={"v": (30.0, "meter/second"), "theta": (0.7853981634, "radian"),
                   "g": (9.81, "meter/second**2")},
            equation=r"R = \frac{v^2 \sin(2 \theta)}{g}",
            steps=[
                "In vacuum, range = v^2*sin(2*theta)/g, and 45 degrees maximises sin(2*theta) at 1.",
                "R = 30^2 * 1 / 9.81 = 900/9.81.",
            ],
            expected=91.743119266,
            units="m",
            sanity="About 92 m. The simulation with real baseball drag gives roughly 58 m — so air "
                   "resistance costs a third of the distance, which is why this textbook formula "
                   "should never be trusted outdoors.",
        ),
    ],
    simulation="projectile",
    simulation_parameters={"speed": 30.0, "angle": 45.0},
    misconceptions=[
        "The vacuum parabola is not a small correction away from reality — for a thrown ball it "
        "overestimates range by tens of percent.",
        "Heavier objects do fall faster in air. Galileo's result holds only when drag is negligible; "
        "a feather and a hammer tie on the Moon, not on Earth.",
    ],
    reading_query="cat:physics.class-ph AND abs:drag",
    related=["simple-harmonic-motion"],
))


# ---- quantum -----------------------------------------------------------

add(Concept(
    key="superposition",
    name="Quantum superposition",
    aliases=["superposition principle", "qubit state"],
    domain="quantum",
    definition="A quantum system can be in a combination of states at once, described by complex "
               "amplitudes whose squared magnitudes give the probabilities of each measurement outcome.",
    everyday="A struck guitar string does not vibrate at one pure frequency — it carries the "
             "fundamental and several overtones simultaneously, and the sound is all of them at "
             "once. Superposition is that idea taken seriously for matter.",
    detail=[
        "A qubit's state is a|0> + b|1> with |a|^2 + |b|^2 = 1. Before measurement it is genuinely "
        "both, not secretly one of them. The difference is testable: an interference experiment "
        "gives results no 'it was really 0 all along' account can reproduce.",
        "The phase between a and b carries physical information even though it never shows up in a "
        "single measurement probability. Interference is what makes it visible, and it is where a "
        "quantum computer's advantage lives.",
        "Measurement in a given basis yields one outcome with probability |a|^2 or |b|^2. Two states "
        "differing only by an overall phase are the same state; only *relative* phase matters.",
    ],
    equations=[
        (r"E = \hbar \omega", "Energy of a quantum of angular frequency omega",
         {"E": "joule", "omega": "radian/second"}),
    ],
    examples=[
        WorkedExample(
            question="What is the energy of a photon from a 5 GHz microwave drive, as used on "
                     "superconducting qubits?",
            given={"omega": (3.14159265e10, "radian/second")},
            equation=r"E = \hbar \omega",
            steps=[
                "omega = 2*pi*f = 2*pi*5e9 = 3.1416e10 rad/s.",
                "E = hbar*omega, with hbar = 1.0546e-34 J s.",
            ],
            expected=3.313e-24,
            units="J",
            sanity="About 3.3e-24 J, or 20 micro-eV. Thermal energy at 1 K is 86 micro-eV, which is "
                   "precisely why these qubits must be run near 10 mK — otherwise heat alone would "
                   "excite them.",
        ),
    ],
    simulation="rabi",
    simulation_parameters={"rabi_frequency": 6.283185, "detuning": 0.0, "duration": 3.0},
    misconceptions=[
        "Superposition is not ignorance. 'It is really 0 or 1 and we do not know which' predicts "
        "different, experimentally excluded results.",
        "Measuring does not simply reveal a pre-existing value; it changes the state.",
        "A superposition of two places is not an object in two places at once in any everyday "
        "sense — it is a state whose interference pattern no single-location description reproduces.",
    ],
    reading_query="cat:quant-ph AND abs:superposition",
    related=["decoherence", "entanglement", "rabi-oscillations"],
))


add(Concept(
    key="decoherence",
    name="Decoherence",
    aliases=["dephasing", "t2", "coherence time", "quantum noise"],
    domain="quantum",
    definition="The loss of a superposition's phase relationships to the environment, which makes "
               "a quantum system behave classically without any measurement having been made.",
    everyday="Two identical pendulums started in perfect step gradually drift apart, because each "
             "feels slightly different air currents. Decoherence is that for quantum phase: the "
             "environment keeps nudging the system until the relative phase is scrambled beyond recovery.",
    detail=[
        "Two timescales matter. T1 is energy relaxation — the excited state decaying to the ground "
        "state, losing a quantum to the environment. T2 is coherence — how long the *phase* between "
        "components survives.",
        "T2 can never exceed 2*T1, because energy relaxation destroys phase as a side effect. When "
        "T2 is shorter than that ceiling, the extra loss is pure dephasing: the environment learning "
        "about the state without taking energy from it.",
        "Decoherence is why quantum computers need millikelvin refrigerators, magnetic shielding, "
        "and error correction. It is not a measurement problem to be philosophised away — it is a "
        "rate, in inverse seconds, that engineers fight.",
    ],
    equations=[
        (r"\frac{1}{T_2} = \frac{1}{2 T_1} + \gamma", "Coherence loss is relaxation plus pure dephasing",
         {"T_2": "second", "T_1": "second", "gamma": "1/second"}),
    ],
    examples=[
        WorkedExample(
            question="A transmon has T1 = 100 microseconds and T2 = 80 microseconds. What is its "
                     "pure dephasing rate?",
            given={"T_1": (100e-6, "second"), "T_2": (80e-6, "second")},
            equation=r"\gamma = \frac{1}{T_2} - \frac{1}{2 T_1}",
            steps=[
                "1/T2 = 1/80e-6 = 12500 per second.",
                "1/(2*T1) = 1/(200e-6) = 5000 per second.",
                "The difference is the dephasing the relaxation does not account for.",
            ],
            expected=7500.0,
            units="1/s",
            sanity="7500 per second, so pure dephasing acts on a 133 microsecond timescale — the "
                   "same order as T1, which is typical for a good transmon.",
        ),
    ],
    simulation="decoherence",
    simulation_parameters={"t1": 100e-6, "t2": 80e-6, "duration": 300e-6},
    misconceptions=[
        "Decoherence does not require a conscious observer or a deliberate measurement. A stray "
        "photon is enough.",
        "T2 > 2*T1 is impossible, not merely unusual.",
        "Decoherence does not 'collapse' the state so much as spread its phase information into the "
        "environment, where it is no longer recoverable in practice.",
    ],
    reading_query="cat:quant-ph AND abs:decoherence",
    related=["superposition", "entanglement"],
))


add(Concept(
    key="rabi-oscillations",
    name="Rabi oscillations",
    aliases=["rabi", "rabi flopping", "driven qubit"],
    domain="quantum",
    definition="The periodic exchange of population between two quantum levels driven by a "
               "resonant field, at a rate set by the drive strength.",
    everyday="Pushing a swing at exactly its own rhythm drives it steadily higher, then — if you "
             "kept perfect time and nothing lost energy — back down again. A driven qubit does "
             "precisely that, cycling between ground and excited state for as long as you drive it.",
    detail=[
        "On resonance, the excited-state probability is sin^2(Omega*t/2): a clean, complete "
        "oscillation. Drive for half a period and you have inverted the qubit — that is a pi pulse, "
        "the quantum equivalent of a NOT gate.",
        "Off resonance the oscillation gets faster but shallower. The rate becomes the generalised "
        "Rabi frequency sqrt(Omega^2 + Delta^2), while the maximum reachable population falls to "
        "Omega^2/(Omega^2 + Delta^2) — so a detuned drive can never fully invert the qubit, no "
        "matter how long you wait.",
        "Every gate on a superconducting or trapped-ion quantum computer is a carefully timed Rabi "
        "rotation. Calibrating a machine largely means measuring Omega precisely enough to stop the "
        "pulse at the right instant.",
    ],
    equations=[
        (r"\Omega_R = \sqrt{\Omega^2 + \Delta^2}", "Generalised Rabi frequency with detuning",
         {"Omega_R": "radian/second", "Omega": "radian/second", "Delta": "radian/second"}),
    ],
    examples=[
        WorkedExample(
            question="A qubit is driven at Omega = 2*pi*10 MHz with a detuning of 2*pi*10 MHz. How "
                     "fast does it oscillate?",
            given={"Omega": (6.2831853e7, "radian/second"), "Delta": (6.2831853e7, "radian/second")},
            equation=r"\Omega_R = \sqrt{\Omega^2 + \Delta^2}",
            steps=[
                "Omega_R = sqrt(Omega^2 + Delta^2) with both equal.",
                "= sqrt(2) * Omega = 1.41421 * 6.2832e7.",
            ],
            expected=8.8857659e7,
            units="rad/s",
            sanity="About sqrt(2) times faster than on resonance — but the population now only "
                   "reaches Omega^2/(Omega^2+Delta^2) = 0.5, so it never fully inverts.",
        ),
    ],
    simulation="rabi",
    simulation_parameters={"rabi_frequency": 6.283185, "detuning": 6.283185, "duration": 3.0},
    misconceptions=[
        "Detuning makes the oscillation faster, not slower — but shallower, which is the part that "
        "actually matters for gates.",
        "Rabi oscillations are coherent, not stochastic absorption and emission. Drive twice as "
        "long and you can end up back where you started.",
    ],
    reading_query="cat:quant-ph AND abs:Rabi",
    landmark_papers=["arXiv:2101.07336"],
    related=["superposition", "jaynes-cummings"],
))


add(Concept(
    key="jaynes-cummings",
    name="Jaynes-Cummings model",
    aliases=["jc model", "cavity qed", "vacuum rabi"],
    domain="quantum",
    definition="The exactly solvable model of one two-level atom coupled to one quantised mode of "
               "the electromagnetic field.",
    everyday="Two pendulums joined by a weak spring: energy sloshes entirely from one to the other "
             "and back again. Here the coupling is to the electromagnetic field itself — and the "
             "sloshing happens even when the cavity starts completely empty.",
    detail=[
        "The striking prediction is vacuum Rabi oscillation. An excited atom in an *empty* cavity "
        "still cycles its excitation into and out of the field, at rate 2g. The vacuum is not "
        "nothing; its fluctuations couple to the atom.",
        "With n photons already present the exchange runs at 2g*sqrt(n+1). That square root is "
        "distinctly quantum — a classical field would give a rate proportional to amplitude, with "
        "no offset at zero.",
        "This model underpins cavity and circuit QED, and is the reason superconducting qubits are "
        "read out by watching a resonator they are coupled to rather than being probed directly.",
    ],
    equations=[
        (r"\Omega_n = 2 g \sqrt{n + 1}", "Exchange rate with n photons — nonzero at n = 0",
         {"Omega_n": "radian/second", "g": "radian/second", "n": "dimensionless"}),
    ],
    examples=[
        WorkedExample(
            question="A cavity has coupling g = 2*pi*50 MHz and starts empty. How long does the "
                     "atom take to give up its excitation?",
            given={"g": (3.14159265e8, "radian/second"), "n": (0.0, "dimensionless")},
            equation=r"\Omega_n = 2 g \sqrt{n + 1}",
            steps=[
                "With n = 0, Omega_0 = 2*g*sqrt(1) = 2g = 6.2832e8 rad/s.",
                "A full exchange and return takes 2*pi/Omega_0; half of that transfers the excitation.",
            ],
            expected=6.2831853e8,
            units="rad/s",
            sanity="6.28e8 rad/s is a 10 ns round trip, so the excitation transfers in about 5 ns — "
                   "far faster than typical cavity loss, which is what makes strong coupling "
                   "observable at all.",
        ),
    ],
    simulation="jaynes_cummings",
    simulation_parameters={"coupling": 1.0, "photons": 0, "duration": 10.0},
    misconceptions=[
        "The vacuum Rabi frequency is not zero for an empty cavity — that is the whole point.",
        "The model assumes the rotating-wave approximation, which fails at ultrastrong coupling "
        "where the quantum Rabi model is needed instead.",
    ],
    reading_query="cat:quant-ph AND abs:Jaynes-Cummings",
    landmark_papers=["arXiv:2101.07336", "arXiv:2504.19943"],
    related=["rabi-oscillations", "superposition"],
))


add(Concept(
    key="uncertainty-principle",
    name="Heisenberg uncertainty principle",
    aliases=["uncertainty", "heisenberg"],
    domain="quantum",
    definition="A limit on how sharply two incompatible observables can simultaneously be defined: "
               "the product of their spreads has a floor set by hbar.",
    everyday="A very short musical note has no well-defined pitch — clap your hands and you cannot "
             "say what note it was. A long pure tone has precise pitch but no precise moment. "
             "Position and momentum trade off in exactly this way, and for the same mathematical reason.",
    detail=[
        "The principle is about the *states themselves*, not clumsy instruments. A state with sharp "
        "position simply does not possess a sharp momentum; there is nothing there to measure better.",
        "It follows from position and momentum being Fourier conjugates. Any wave narrow in one "
        "domain is broad in the other — the same theorem an audio engineer uses.",
        "It is why atoms do not collapse. Confining an electron closer to the nucleus raises its "
        "momentum spread and therefore its kinetic energy, and that cost balances the electrostatic "
        "attraction at a finite radius.",
    ],
    equations=[
        (r"\Delta x \Delta p \geq \frac{\hbar}{2}", "Position-momentum uncertainty relation",
         {"Delta": "dimensionless", "x": "meter", "p": "kilogram*meter/second"}),
    ],
    examples=[
        WorkedExample(
            question="An electron is confined to 0.1 nm, roughly an atom. What is the minimum "
                     "spread in its momentum?",
            given={"x": (1e-10, "meter")},
            equation=r"p = \frac{\hbar}{2 x}",
            steps=[
                "Minimum uncertainty gives Delta_p = hbar/(2*Delta_x).",
                "= 1.0546e-34 / (2 * 1e-10).",
            ],
            expected=5.2728e-25,
            units="kg m/s",
            sanity="5.3e-25 kg m/s corresponds to a speed spread near 580 km/s for an electron. "
                   "Atomic electrons really do move that fast — this is not a measurement artefact.",
        ),
    ],
    misconceptions=[
        "It is not about the observer disturbing the system. Even in principle, with perfect "
        "instruments, the limit holds.",
        "It does not say you cannot measure position precisely. You can — you then have no precise "
        "momentum to speak of.",
        "The bound is hbar/2, not hbar, for the standard-deviation form.",
    ],
    reading_query="cat:quant-ph AND abs:uncertainty relation",
    related=["superposition"],
))


def find(term: str) -> Concept | None:
    """Look up a concept by key, name or alias."""
    term = term.lower().strip()
    for concept in CONCEPTS:
        if concept.matches(term):
            return concept
    # Fall back to substring matching so "rabi oscillation" finds "rabi-oscillations".
    for concept in CONCEPTS:
        haystack = " ".join([concept.key, concept.name, *concept.aliases]).lower()
        if term in haystack:
            return concept
    return None


def search(term: str) -> list[Concept]:
    """Every concept whose text mentions the term."""
    term = term.lower().strip()
    return [
        c for c in CONCEPTS
        if term in " ".join([c.key, c.name, c.definition, *c.aliases]).lower()
    ]


def concept_keys() -> list[str]:
    return [c.key for c in CONCEPTS]
