"""Simulations and explanations.

The concept-library tests are the important ones. They recompute every worked
example and re-check every equation, so an error introduced into the library
data fails the suite instead of being taught to a user as fact.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from noether.explain import library
from noether.explain.compose import check_example, compose, render, suggestions
from noether.sim import engine
from noether.sim.models import Parameter


class TestModelRegistry:
    def test_every_model_is_runnable(self) -> None:
        for item in engine.available():
            assert item["parameters"], f"{item['key']} declares no parameters"
            assert item["observables"], f"{item['key']} reports nothing"

    def test_every_model_has_an_everyday_description(self) -> None:
        """A simulation you cannot connect to anything you have seen teaches little."""
        for item in engine.available():
            assert len(item["everyday"]) > 40, item["key"]

    def test_unknown_model_is_reported_not_raised(self) -> None:
        result = engine.run("does-not-exist")
        assert not result.ok
        assert "no model" in result.error

    def test_unknown_parameter_is_refused(self) -> None:
        result = engine.run("pendulum", {"nonsense": 1.0})
        assert not result.ok
        assert "no parameter" in result.error

    def test_out_of_range_parameter_is_refused_before_running(self) -> None:
        """A 500-level Fock space should fail in milliseconds, not after a hang."""
        result = engine.run("jaynes_cummings", {"levels": 500})
        assert not result.ok
        assert "maximum" in result.error


class TestParameterBounds:
    def test_below_minimum(self) -> None:
        assert Parameter("x", 1.0, "m", "", 0.5, 2.0).validate(0.1)

    def test_above_maximum(self) -> None:
        assert Parameter("x", 1.0, "m", "", 0.5, 2.0).validate(9.0)

    def test_in_range(self) -> None:
        assert Parameter("x", 1.0, "m", "", 0.5, 2.0).validate(1.0) is None


class TestClassical:
    def test_pendulum_small_angle_matches_the_formula(self) -> None:
        result = engine.run("pendulum", {"length": 1.0, "gravity": 9.81, "theta0": 2.0})
        assert result.ok
        assert result.derived["small_angle_period_s"] == pytest.approx(
            2 * math.pi * math.sqrt(1.0 / 9.81)
        )

    def test_undamped_pendulum_conserves_energy(self) -> None:
        """An independent check on the integrator, not on the model."""
        result = engine.run("pendulum", {"damping": 0.0, "duration": 20.0})
        assert result.derived["energy_drift"] < 1e-6

    def test_large_amplitude_period_exceeds_small_angle(self) -> None:
        result = engine.run("pendulum", {"theta0": 120.0})
        assert result.derived["amplitude_corrected_period_s"] > result.derived["small_angle_period_s"]

    def test_oscillator_natural_frequency(self) -> None:
        result = engine.run("oscillator", {"mass": 2.0, "stiffness": 200.0})
        assert result.derived["natural_frequency_rad_s"] == pytest.approx(10.0)

    def test_oscillator_damping_regime_is_named(self) -> None:
        assert engine.run("oscillator", {"damping": 0.0}).derived["regime"] == "undamped"
        heavy = engine.run("oscillator", {"mass": 1.0, "stiffness": 10.0, "damping": 100.0})
        assert heavy.derived["regime"] == "overdamped"

    def test_drag_shortens_the_range(self) -> None:
        """The whole point of the model: the vacuum parabola overestimates."""
        result = engine.run("projectile", {"speed": 30.0, "angle": 45.0})
        assert result.derived["range_m"] < result.derived["vacuum_range_m"]
        assert result.derived["drag_penalty_m"] > 0

    def test_vacuum_range_matches_the_closed_form(self) -> None:
        result = engine.run("projectile", {"speed": 30.0, "angle": 45.0, "gravity": 9.81})
        assert result.derived["vacuum_range_m"] == pytest.approx(30.0**2 / 9.81, rel=1e-6)

    def test_runs_report_convergence(self) -> None:
        for key in ("pendulum", "oscillator", "projectile"):
            assert engine.run(key, {}).convergence.converged is True, key


class TestQuantum:
    def test_resonant_rabi_matches_the_closed_form(self) -> None:
        result = engine.run("rabi", {"detuning": 0.0})
        assert result.ok
        assert result.derived["analytic_max_error"] < 1e-3

    def test_detuning_raises_rate_and_lowers_ceiling(self) -> None:
        """The physics that makes detuned gates fail, in two numbers."""
        omega = 6.283185
        on = engine.run("rabi", {"rabi_frequency": omega, "detuning": 0.0})
        off = engine.run("rabi", {"rabi_frequency": omega, "detuning": omega})

        assert off.derived["generalised_rabi_rad_s"] > on.derived["generalised_rabi_rad_s"]
        assert off.derived["max_excited_population"] < on.derived["max_excited_population"]
        assert off.derived["max_excited_population"] == pytest.approx(0.5, abs=0.02)

    def test_decoherence_follows_the_exponential(self) -> None:
        result = engine.run("decoherence", {"t1": 100e-6, "t2": 80e-6})
        assert result.derived["analytic_max_error"] < 1e-3

    def test_coherence_at_t2_is_one_over_e(self) -> None:
        result = engine.run("decoherence", {"t1": 100e-6, "t2": 80e-6, "duration": 300e-6})
        assert result.derived["coherence_at_t2"] == pytest.approx(1 / math.e, abs=0.01)

    def test_pure_dephasing_is_the_leftover_rate(self) -> None:
        result = engine.run("decoherence", {"t1": 100e-6, "t2": 80e-6})
        assert result.derived["pure_dephasing_rate_1_s"] == pytest.approx(7500.0)

    def test_t2_above_twice_t1_is_refused(self) -> None:
        """No physical qubit can do this, so the model must not pretend to."""
        result = engine.run("decoherence", {"t1": 100e-6, "t2": 300e-6})
        assert not result.ok
        assert "2*T1" in result.error

    def test_excited_population_decays_under_relaxation(self) -> None:
        """Caught in a live session, not by the analytic checks.

        QuTiP's sigmam() maps |0> -> |1>, the opposite of destroy(2), so
        basis(2,0) is the excited state. Labelling the wrong projector
        'excited_population' left every analytic cross-check passing while the
        reported curve rose under relaxation instead of falling.
        """
        result = engine.run("decoherence", {"t1": 100e-6, "t2": 80e-6, "duration": 300e-6})
        excited = result.get("excited_population")
        assert excited is not None
        assert excited.values[0] == pytest.approx(0.5, abs=0.02), "should start on the equator"
        assert excited.final < excited.values[0], "excited population must decay, not grow"
        assert excited.final < 0.1, "after 3*T1 almost nothing should remain excited"

    def test_rabi_starts_in_the_ground_state(self) -> None:
        """A driven qubit starts unexcited and is driven up, not the reverse."""
        result = engine.run("rabi", {"detuning": 0.0, "duration": 4.0})
        excited = result.get("excited_population")
        assert excited.values[0] == pytest.approx(0.0, abs=1e-6)
        assert max(excited.values) == pytest.approx(1.0, abs=0.01), "resonant drive fully inverts"

    def test_populations_stay_within_zero_and_one(self) -> None:
        """A probability outside [0, 1] means the model is wrong, whatever it matches."""
        for key in ("rabi", "decoherence", "jaynes_cummings"):
            result = engine.run(key, {})
            for series in result.series:
                if "population" in series.name or "excited" in series.name:
                    low, high = series.extremes
                    assert -1e-6 <= low and high <= 1 + 1e-6, f"{key}.{series.name}"

    def test_vacuum_rabi_is_nonzero_in_an_empty_cavity(self) -> None:
        """The defining prediction of the Jaynes-Cummings model."""
        result = engine.run("jaynes_cummings", {"coupling": 1.0, "photons": 0})
        assert result.derived["vacuum_rabi_frequency_rad_s"] == pytest.approx(2.0)

    def test_exchange_rate_scales_as_sqrt_n_plus_one(self) -> None:
        one = engine.run("jaynes_cummings", {"coupling": 1.0, "photons": 0})
        four = engine.run("jaynes_cummings", {"coupling": 1.0, "photons": 3})
        ratio = (
            four.derived["vacuum_rabi_frequency_rad_s"]
            / one.derived["vacuum_rabi_frequency_rad_s"]
        )
        assert ratio == pytest.approx(2.0)

    def test_photons_beyond_truncation_are_refused(self) -> None:
        result = engine.run("jaynes_cummings", {"photons": 20, "levels": 10})
        assert not result.ok
        assert "truncated" in result.error or "Fock" in result.error


class TestPlotting:
    def test_plot_writes_a_file(self, tmp_path: Path) -> None:
        result = engine.run("pendulum", {})
        path = engine.plot(result, tmp_path / "p.png")
        assert path and path.exists()

    def test_failed_run_draws_nothing(self, tmp_path: Path) -> None:
        result = engine.run("pendulum", {"nonsense": 1})
        assert engine.plot(result, tmp_path / "p.png") is None


class TestConceptLibrary:
    """The library checks itself. These are the tests that keep it honest."""

    def test_every_equation_parses_and_is_dimensionally_sound(self) -> None:
        for key in library.concept_keys():
            explanation = compose(key, run_simulation=False)
            assert explanation is not None
            for equation in explanation.equations:
                assert equation.parsed, f"{key}: {equation.latex} did not parse"
                assert equation.dimensions in ("consistent", "unknown"), (
                    f"{key}: {equation.latex} is {equation.dimensions}"
                )

    def test_declared_units_actually_resolve(self) -> None:
        """A concept that declares units must get a real verdict, not 'unknown'.

        Caught through the MCP server: the library writes T_1 while the parser
        emits T_{1}, so a fully-declared equation reported 'unknown' — which
        under this project's own rules means "we did not check". The alignment
        that worked examples already had was missing from equation checking.
        """
        for key in library.concept_keys():
            explanation = compose(key, run_simulation=False)
            for equation in explanation.equations:
                declared = next(
                    (u for latex, _says, u in explanation.concept.equations if latex == equation.latex),
                    {},
                )
                if declared:
                    assert equation.dimensions != "unknown", (
                        f"{key}: {equation.latex} declares units but reports unknown "
                        f"({equation.detail})"
                    )

    def test_every_worked_example_recomputes_to_its_stored_answer(self) -> None:
        """A typo in the library must fail here, not teach a user a wrong number."""
        for key in library.concept_keys():
            explanation = compose(key, run_simulation=False)
            for checked in explanation.examples:
                assert not checked.error, f"{key}: {checked.error}"
                assert checked.matches_expected is not False, (
                    f"{key}: {checked.discrepancy()}"
                )

    def test_every_linked_simulation_runs(self) -> None:
        for concept in library.CONCEPTS:
            if not concept.simulation:
                continue
            result = engine.run(concept.simulation, concept.simulation_parameters)
            assert result.ok, f"{concept.key}: {result.error}"

    def test_every_concept_has_an_analogy_and_misconceptions(self) -> None:
        for concept in library.CONCEPTS:
            assert len(concept.everyday) > 60, concept.key
            assert concept.definition, concept.key

    def test_related_concepts_are_not_dangling(self) -> None:
        known = set(library.concept_keys())
        for concept in library.CONCEPTS:
            for related in concept.related:
                # A forward reference to a concept not yet written is allowed,
                # but it must not be a typo of one that exists.
                if related not in known:
                    assert library.find(related) is None, (
                        f"{concept.key} points at {related!r}, which resolves elsewhere"
                    )


class TestLookup:
    def test_alias_lookup(self) -> None:
        assert library.find("t2").key == "decoherence"
        assert library.find("drag").key == "air-resistance"
        assert library.find("shm").key == "simple-harmonic-motion"

    def test_substring_fallback(self) -> None:
        assert library.find("rabi oscillation") is not None

    def test_unknown_returns_none(self) -> None:
        assert library.find("quantum gravity supersymmetry") is None

    def test_compose_returns_none_for_unknown(self) -> None:
        assert compose("not a real concept at all") is None

    def test_suggestions_offer_alternatives(self) -> None:
        assert suggestions("quantum")


class TestRendering:
    def test_brief_omits_the_mathematics(self) -> None:
        text = render(compose("resonance", run_simulation=False), depth="brief")
        assert "In everyday terms" in text
        assert "The mathematics" not in text

    def test_full_includes_misconceptions(self) -> None:
        text = render(compose("decoherence", run_simulation=False), depth="full")
        assert "Commonly got wrong" in text

    def test_no_corpus_says_so_rather_than_implying_papers(self) -> None:
        text = render(compose("resonance", run_simulation=False), depth="full")
        assert "Nothing ingested on this topic" in text

    def test_square_brackets_are_avoided_in_output(self) -> None:
        """Rich reads [...] as markup and would swallow the status line."""
        text = render(compose("decoherence", run_simulation=False), depth="full")
        assert "(parsed; dimensions:" in text


def test_check_example_detects_a_wrong_stored_answer() -> None:
    """The mechanism that keeps the library honest."""
    from noether.explain.concepts import WorkedExample

    wrong = WorkedExample(
        question="deliberately wrong",
        given={"m": (2.0, "kilogram"), "k": (200.0, "newton/meter"),
               "c": (4.0, "newton*second/meter")},
        equation=r"Q = \frac{\sqrt{m k}}{c}",
        expected=999.0,
    )
    checked = check_example(wrong)
    assert checked.ok
    assert checked.matches_expected is False
    assert "library entry is wrong" in checked.discrepancy()
