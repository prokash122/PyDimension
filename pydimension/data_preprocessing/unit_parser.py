"""
Unit string parsing: convert human-readable unit strings into fundamental
dimension vectors [Mass, Length, Time, Temperature, Current, Amount, Luminous].
"""

import re
from typing import Dict, List


UNIT_RECOMMENDATIONS: Dict[str, str] = {
    "etaP": "W",
    "Vs": "m/s",
    "r0": "m",
    "alpha": "m²/s",
    "rho": "kg/m³",
    "cp": "J/(kg·K)",
    "Tv-T0": "K",
    "Lv": "J/kg",
    "Tl-T0": "K",
    "Lm": "J/kg",
    "e": "dimensionless",
    "Ke": "dimensionless",
    "e*": "dimensionless",
    "p*": "dimensionless",
}


def infer_units(variables: List[str]) -> Dict[str, str]:
    """Best-effort unit inference from variable names using known patterns."""
    units: Dict[str, str] = {}
    for var in variables:
        if var in UNIT_RECOMMENDATIONS:
            units[var] = UNIT_RECOMMENDATIONS[var]
        elif var.startswith("p") and var[1:].isdigit():
            units[var] = "dimensionless"
        elif var.endswith("*"):
            units[var] = "dimensionless"
        else:
            units[var] = "dimensionless"
    return units


# ── Compositional unit-expression parser ────────────────────────────────────
# Grammar (after normalisation):
#   expr   := term ("/" term)*            each "/" divides by the whole term
#   term   := factor ("·"? factor)*       adjacency = multiplication
#   factor := "(" expr ")" exp? | symbols exp?
#   exp    := "^"? signed integer         (superscripts are pre-translated)
# Alphabetic runs are split greedily into known unit symbols, so "kgm" parses
# as kg·m and an exponent binds to the last symbol of the run ("ms-1" = m·s⁻¹).

_UNIT_VECTORS: Dict[str, tuple] = {
    # base units             M  L  T  Θ  I  N  J
    "kg":  (1, 0, 0, 0, 0, 0, 0),
    "g":   (1, 0, 0, 0, 0, 0, 0),
    "m":   (0, 1, 0, 0, 0, 0, 0),
    "s":   (0, 0, 1, 0, 0, 0, 0),
    "k":   (0, 0, 0, 1, 0, 0, 0),
    "a":   (0, 0, 0, 0, 1, 0, 0),
    "mol": (0, 0, 0, 0, 0, 1, 0),
    "cd":  (0, 0, 0, 0, 0, 0, 1),
    # derived units
    "n":   (1, 1, -2, 0, 0, 0, 0),   # newton
    "w":   (1, 2, -3, 0, 0, 0, 0),   # watt
    "j":   (1, 2, -2, 0, 0, 0, 0),   # joule
    "pa":  (1, -1, -2, 0, 0, 0, 0),  # pascal
    "hz":  (0, 0, -1, 0, 0, 0, 0),   # hertz
}
_SYMBOLS_BY_LENGTH = sorted(_UNIT_VECTORS, key=len, reverse=True)

_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
})

_TOKEN_RE = re.compile(r"\^[+-]?\d+|[+-]?\d+|[a-z]+|[()·/]")

_DIMENSIONLESS_STRINGS = {"", "-", "1", "dimensionless", "none", "unitless"}


class _UnitParseError(ValueError):
    """Internal: the compositional parser could not handle the string."""


def _vec_add(a, b, scale=1):
    return [x + scale * y for x, y in zip(a, b)]


def _split_symbols(run: str) -> List[str]:
    """Greedily split an alphabetic run into known unit symbols."""
    symbols = []
    rest = run
    while rest:
        for sym in _SYMBOLS_BY_LENGTH:
            if rest.startswith(sym):
                symbols.append(sym)
                rest = rest[len(sym):]
                break
        else:
            raise _UnitParseError(f"unknown unit symbol in {run!r}")
    return symbols


def _maybe_exponent(tokens: List[str], i: int):
    if i < len(tokens) and re.fullmatch(r"\^[+-]?\d+|[+-]?\d+", tokens[i]):
        return int(tokens[i].lstrip("^")), i + 1
    return 1, i


def _parse_factor(tokens: List[str], i: int):
    if i >= len(tokens):
        raise _UnitParseError("unexpected end of unit string")
    tok = tokens[i]
    if tok == "(":
        vec, i = _parse_expr(tokens, i + 1)
        if i >= len(tokens) or tokens[i] != ")":
            raise _UnitParseError("unbalanced parentheses")
        exp, i = _maybe_exponent(tokens, i + 1)
        return [v * exp for v in vec], i
    if re.fullmatch(r"[a-z]+", tok):
        symbols = _split_symbols(tok)
        vec = [0] * 7
        for sym in symbols[:-1]:
            vec = _vec_add(vec, _UNIT_VECTORS[sym])
        exp, i = _maybe_exponent(tokens, i + 1)
        vec = _vec_add(vec, _UNIT_VECTORS[symbols[-1]], scale=exp)
        return vec, i
    raise _UnitParseError(f"unexpected token {tok!r}")


def _parse_term(tokens: List[str], i: int):
    vec, i = _parse_factor(tokens, i)
    while i < len(tokens) and (tokens[i] == "·" or tokens[i] == "("
                               or re.fullmatch(r"[a-z]+", tokens[i])):
        if tokens[i] == "·":
            i += 1
        rhs, i = _parse_factor(tokens, i)
        vec = _vec_add(vec, rhs)
    return vec, i


def _parse_expr(tokens: List[str], i: int):
    vec, i = _parse_term(tokens, i)
    while i < len(tokens) and tokens[i] == "/":
        rhs, i = _parse_term(tokens, i + 1)
        vec = _vec_add(vec, rhs, scale=-1)
    return vec, i


def _parse_dimensions_compositional(unit: str) -> List[int]:
    normalized = (unit.strip().translate(_SUPERSCRIPT_MAP).lower()
                  .replace(" ", "").replace("⋅", "·").replace("*", "·")
                  .replace("×", "·"))
    if normalized in _DIMENSIONLESS_STRINGS:
        return [0] * 7
    tokens = []
    pos = 0
    while pos < len(normalized):
        match = _TOKEN_RE.match(normalized, pos)
        if match is None:
            raise _UnitParseError(f"cannot tokenize {unit!r} at {normalized[pos:]!r}")
        tokens.append(match.group())
        pos = match.end()
    vec, i = _parse_expr(tokens, 0)
    if i != len(tokens):
        raise _UnitParseError(f"trailing tokens in {unit!r}: {tokens[i:]}")
    return vec


def parse_dimensions(unit: str) -> List[int]:
    """Parse a unit string into a 7-element dimension vector.

    Order: [Mass, Length, Time, Temperature, Current, Amount, Luminous].

    Uses a compositional parser (products, quotients, parentheses, and
    integer exponents over SI base and common derived units), falling back
    to the legacy keyword heuristics for strings it cannot interpret.
    """
    try:
        return _parse_dimensions_compositional(unit)
    except _UnitParseError:
        return _parse_dimensions_legacy(unit)


def _parse_dimensions_legacy(unit: str) -> List[int]:
    """Legacy keyword-heuristic parser (kept as a fallback)."""
    dims = [0, 0, 0, 0, 0, 0, 0]
    unit_lower = unit.lower().replace(" ", "").replace("·", "").replace("⋅", "").replace("*", "")

    if "dimensionless" in unit_lower or unit == "1":
        return dims

    cp_patterns = ["j/(kgk)", "j/kg/k", "jkg^-1k^-1", "jkg-1k-1", "j/(kg·k)", "j/(kg*k)"]
    if any(p in unit_lower for p in cp_patterns):
        return [0, 2, -2, -1, 0, 0, 0]

    if "kg" in unit_lower:
        dims[0] = -1 if "/kg" in unit_lower else 1

    if "kg/m³" in unit_lower or "kg/m^3" in unit_lower:
        dims[1] = -3
    elif "m²/s" in unit_lower or "m^2/s" in unit_lower:
        dims[1] = 2
    elif "m³" in unit_lower or "m^3" in unit_lower:
        dims[1] = 3
    elif "m²" in unit_lower or "m^2" in unit_lower:
        dims[1] = 2
    elif "m/s" in unit_lower:
        dims[1] = 1
    elif unit_lower == "m":
        dims[1] = 1

    if "/s²" in unit_lower or "/s^2" in unit_lower:
        dims[2] = -2
    elif "/s³" in unit_lower or "/s^3" in unit_lower:
        dims[2] = -3
    elif "/s" in unit_lower:
        dims[2] = -1

    if "(kg·k)" in unit_lower or "/(kg·k)" in unit_lower:
        dims[3] = -1
    elif unit_lower.endswith("k") or "k)" in unit_lower or unit_lower == "k":
        dims[3] = 1

    if "w" in unit_lower and "j" not in unit_lower:
        dims[0], dims[1], dims[2] = 1, 2, -3
    elif any(p in unit_lower for p in ["j/(kg·k)", "j/(kg*k)", "j/kg/k", "j/(kgk)"]):
        dims[0], dims[1], dims[2], dims[3] = 0, 2, -2, -1
    elif "j/kg" in unit_lower:
        dims[0], dims[1], dims[2] = 0, 2, -2
    elif "j" in unit_lower:
        dims[0], dims[1], dims[2] = 1, 2, -2

    return dims
