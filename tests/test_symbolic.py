"""Equation parsing and dimensional analysis -- guarantee 3.

``TestGrammarLimits`` pins what SymPy's LaTeX grammar can and cannot do. Those
tests exist so that a SymPy upgrade which fixes a limitation shows up as a
failure here, prompting us to delete the workaround rather than carry it forever.
"""

from __future__ import annotations

import sympy

from noether.compute.normalise import (
    collapse_double_braces,
    fix_leading_minus,
    insert_explicit_multiplication,
    strip_decorators,
)
from noether.compute.run import evaluate, solve_for, sweep
from noether.compute.symbolic import parse_equation, parse_rate
from noether.compute.units import Verdict, check_equation


class TestGrammarLimits:
    """What the underlying grammar accepts. Measured, not assumed."""

    def _parses(self, latex: str) -> bool:
        from sympy.parsing.latex import parse_latex

        try:
            return isinstance(parse_latex(latex, backend="lark"), sympy.Basic)
        except Exception:
            return False

    def test_lowercase_greek_is_accepted(self) -> None:
        assert self._parses(r"\omega")

    def test_uppercase_greek_is_rejected(self) -> None:
        """Why upper Greek needs a placeholder."""
        assert not self._parses(r"\Omega")

    def test_numeric_subscript_is_rejected(self) -> None:
        """Why placeholders are base-26 letters, not numbers."""
        assert not self._parses(r"X_{9001}")

    def test_letter_subscript_is_one_symbol(self) -> None:
        assert self._parses(r"X_{zzaa}")

    def test_most_lowercase_greek_is_accepted(self) -> None:
        assert self._parses(r"\alpha") and self._parses(r"\mu")

    def test_sigma_and_iota_are_rejected(self) -> None:
        """Not all lowercase Greek works, and sigma is everywhere in physics."""
        assert not self._parses(r"\sigma")
        assert not self._parses(r"\iota")

    def test_juxtaposed_parenthesis_is_ambiguous(self) -> None:
        """Why implicit multiplication is made explicit."""
        assert not self._parses(r"\omega (a + 1)")
        assert self._parses(r"\omega \cdot (a + 1)")


class TestNormalisation:
    def test_double_braces_collapse(self) -> None:
        assert collapse_double_braces("f_{{n}}") == "f_{n}"

    def test_decorators_are_dropped(self) -> None:
        assert strip_decorators(r"\vec{B} + \hat{H}") == "B + H"

    def test_leading_minus_gets_a_zero(self) -> None:
        assert fix_leading_minus("-x").startswith("0")
        assert fix_leading_minus("x") == "x"

    def test_multiplication_becomes_explicit(self) -> None:
        assert r"\cdot" in insert_explicit_multiplication("2 g x")

    def test_function_application_keeps_no_cdot(self) -> None:
        """sin(x) is application; omega(x) is multiplication."""
        assert r"\cdot" not in insert_explicit_multiplication(r"\sin (x)")

    def test_frac_arguments_stay_attached(self) -> None:
        """\\frac{1}{2} must not become one-half times two."""
        out = insert_explicit_multiplication(r"\frac{1}{2}")
        assert out.replace(" ", "") == r"\frac{1}{2}"

    def test_subscript_groups_are_not_split(self) -> None:
        """Recursing into {zzaa} would shatter a placeholder into z*z*a*a."""
        assert "X_{zzaa}" in insert_explicit_multiplication("X_{zzaa} + 1").replace(" ", "")


class TestParsing:
    def test_simple_equality(self) -> None:
        result = parse_equation(r"E = m c^2")
        assert result.ok and result.is_equality
        assert result.lhs == sympy.Symbol("E")

    def test_uppercase_greek_is_one_symbol_not_a_product(self) -> None:
        """The regression this module's placeholder machinery exists to prevent.

        Rewriting \\Omega to the text 'Omega' yields the product O*m*e*g*a, which
        parses cleanly and is silently wrong.
        """
        result = parse_equation(r"\Omega = 2 g")
        assert result.ok
        assert result.lhs == sympy.Symbol("Omega")
        assert not result.lhs.is_Mul

    def test_hbar_survives_as_one_symbol(self) -> None:
        result = parse_equation(r"E = \hbar \omega")
        assert result.ok
        assert sympy.Symbol("hbar") in result.rhs.free_symbols

    def test_pi_is_a_constant_not_a_symbol(self) -> None:
        """Otherwise every sweep demands a value for pi."""
        result = parse_equation(r"T = 2 \pi r")
        assert result.ok
        assert "pi" not in {str(s) for s in result.rhs.free_symbols}

    def test_creation_operator(self) -> None:
        result = parse_equation(r"H = \hbar \omega (a^\dagger a + \frac{1}{2})")
        assert result.ok
        assert sympy.Symbol("adag") in result.rhs.free_symbols

    def test_fraction_with_compound_numerator(self) -> None:
        result = parse_equation(r"E_n = \frac{n^2 \pi^2 \hbar^2}{2 m L^2}")
        assert result.ok
        assert sympy.Symbol("hbar") in result.rhs.free_symbols

    def test_derivative_is_carried_and_reported(self) -> None:
        result = parse_equation(r"\frac{d^2 x}{d t^2} = -\omega^2 x")
        assert result.ok
        assert result.lossy
        assert any("derivative" in n for n in result.notes)

    def test_relation_is_not_silently_an_equality(self) -> None:
        """Dropping >= would turn an inequality into a false equation."""
        result = parse_equation(r"\Delta x \Delta p \geq \frac{\hbar}{2}")
        assert result.ok
        assert result.relation == ">="
        assert any("relation" in n for n in result.notes)

    def test_sigma_survives_as_one_symbol(self) -> None:
        r"""The grammar rejects \sigma outright, so it takes the placeholder path."""
        result = parse_equation(r"\sigma = \frac{\hbar}{2}")
        assert result.ok
        assert result.lhs == sympy.Symbol("sigma")

    def test_subscript_folds_into_the_placeholder(self) -> None:
        r"""A placeholder is itself subscripted; \sigma_x must not double up."""
        result = parse_equation(r"\sigma_x = 1")
        assert result.ok
        assert result.lhs == sympy.Symbol("sigma_x")

    def test_subscripted_uppercase_greek(self) -> None:
        result = parse_equation(r"\Omega_R = 2 g")
        assert result.ok
        assert result.lhs == sympy.Symbol("Omega_R")

    def test_ambiguous_function_resolves_to_the_written_command(self) -> None:
        r"""tanh(x) beats tan(h*x): the author wrote one command, \tanh."""
        result = parse_equation(r"y = \tanh(x)")
        assert result.ok
        assert "tanh" in str(result.rhs)

    def test_matrix_environment_is_refused_not_mangled(self) -> None:
        r"""Stripping \begin leaves {bmatrix}, read as b*m*a*t*r*i*x."""
        result = parse_equation(r"\begin{bmatrix} 1 & 2 \end{bmatrix}")
        assert not result.ok
        assert "bmatrix" in result.error

    def test_failure_is_reported_not_raised(self) -> None:
        result = parse_equation(r"\begin{bmatrix} 1 & 2 \end{bmatrix} \otimes")
        assert not result.ok
        assert result.error

    def test_parse_rate_over_a_batch(self) -> None:
        batch = [
            r"E = m c^2", r"\lambda = \frac{h}{p}", r"F = G \frac{m_1 m_2}{r^2}",
            r"T = 2 \pi \sqrt{\frac{L}{g}}", r"\Omega = 2 g \sqrt{n+1}",
        ]
        assert parse_rate([parse_equation(e) for e in batch]) == 1.0


class TestDimensions:
    UNITS = {
        "E": "joule", "m": "kilogram", "v": "meter/second",
        "g": "meter/second**2", "h": "meter", "c": "meter/second",
    }

    def test_consistent_equation(self) -> None:
        check = check_equation(parse_equation(r"E = m c^2"), {"E": "joule", "m": "kilogram"})
        assert check.verdict is Verdict.CONSISTENT
        assert check.ok

    def test_inconsistent_equation_is_caught(self) -> None:
        check = check_equation(parse_equation(r"E = m c"), {"E": "joule", "m": "kilogram"})
        assert check.verdict is Verdict.INCONSISTENT

    def test_mixed_sum_is_inconsistent_not_unknown(self) -> None:
        """A DimensionalityError proves inconsistency; it is not a failure to check."""
        check = check_equation(
            parse_equation(r"E = m v^2 + m g"), self.UNITS
        )
        assert check.verdict is Verdict.INCONSISTENT

    def test_correct_physics_is_not_flagged(self) -> None:
        """Kinetic plus potential energy. An early version called this wrong."""
        check = check_equation(parse_equation(r"E = \frac{1}{2} m v^2 + m g h"), self.UNITS)
        assert check.verdict is Verdict.CONSISTENT

    def test_ambiguous_symbol_is_unknown_not_guessed(self) -> None:
        """h is Planck's constant or a height; guessing produces false verdicts."""
        check = check_equation(parse_equation(r"E = \frac{1}{2} m v^2 + m g h"))
        assert check.verdict is Verdict.UNKNOWN
        assert "h" in check.unknown_symbols
        assert "Planck" in check.hint()

    def test_unknown_is_not_a_pass(self) -> None:
        check = check_equation(parse_equation(r"\Psi = \Xi \zeta"))
        assert check.verdict is Verdict.UNKNOWN
        assert not check.ok

    def test_unambiguous_constants_need_no_declaration(self) -> None:
        check = check_equation(
            parse_equation(r"E = \hbar \omega"), {"E": "joule", "omega": "radian/second"}
        )
        assert check.verdict is Verdict.CONSISTENT

    def test_declaration_overrides_the_default(self) -> None:
        """In a paper's own context the author knows what a symbol means."""
        check = check_equation(parse_equation(r"E = m c^2"), {"E": "joule", "m": "kilogram", "c": "meter/second"})
        assert check.verdict is Verdict.CONSISTENT


class TestEvaluation:
    def test_pendulum_period(self) -> None:
        import math

        parsed = parse_equation(r"T = 2 \pi \sqrt{\frac{L}{g}}")
        result = evaluate(parsed, {"L": 1.0, "g": 9.81})
        assert result.ok
        assert abs(result.value - 2 * math.pi * math.sqrt(1 / 9.81)) < 1e-9

    def test_constants_are_supplied(self) -> None:
        result = evaluate(parse_equation(r"E = \hbar \omega"), {"omega": 1e15})
        assert result.ok
        assert 1.0e-19 < result.value < 1.1e-19

    def test_explicit_value_overrides_a_constant(self) -> None:
        """Natural units must be obeyed, not overruled."""
        result = evaluate(parse_equation(r"E = \hbar \omega"), {"omega": 2.0, "hbar": 1.0})
        assert result.value == 2.0

    def test_missing_value_is_reported(self) -> None:
        result = evaluate(parse_equation(r"y = m x + b"), {"x": 1.0})
        assert not result.ok
        assert "no value supplied" in result.error

    def test_inconsistent_equation_is_refused(self) -> None:
        """Producing a number from an equation shown to be wrong is the worst case."""
        parsed = parse_equation(r"E = m c")
        result = evaluate(parsed, {"m": 1.0}, {"E": "joule", "m": "kilogram"})
        assert not result.ok
        assert "refusing" in result.error

    def test_sweep_produces_points(self) -> None:
        result = sweep(parse_equation(r"y = x^2"), "x", 0.0, 4.0, points=50)
        assert result.coverage == 1.0
        assert len(result.finite) == 50

    def test_solve_for(self) -> None:
        roots = solve_for(parse_equation(r"E = m c^2"), "m")
        assert roots and str(roots[0]) == "E/c**2"
