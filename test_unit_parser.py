"""Tests for pydimension.data_preprocessing.unit_parser.parse_dimensions.

Runnable standalone (python test_unit_parser.py) or via pytest.
Dimension vector order: [Mass, Length, Time, Temperature, Current, Amount, Luminous].
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pydimension.data_preprocessing.unit_parser import parse_dimensions

CASES = {
    # simple / base units
    "W":             [1, 2, -3, 0, 0, 0, 0],
    "m":             [0, 1, 0, 0, 0, 0, 0],
    "m/s":           [0, 1, -1, 0, 0, 0, 0],
    "m/s²":          [0, 1, -2, 0, 0, 0, 0],
    "m²/s":          [0, 2, -1, 0, 0, 0, 0],
    "K":             [0, 0, 0, 1, 0, 0, 0],
    "mol":           [0, 0, 0, 0, 0, 1, 0],
    "kg/m³":         [1, -3, 0, 0, 0, 0, 0],
    # dimensionless spellings
    "-":             [0, 0, 0, 0, 0, 0, 0],
    "1":             [0, 0, 0, 0, 0, 0, 0],
    "dimensionless": [0, 0, 0, 0, 0, 0, 0],
    # compound units the legacy heuristics mishandled
    "W/(m·K)":       [1, 1, -3, -1, 0, 0, 0],
    "W/m/K":         [1, 1, -3, -1, 0, 0, 0],
    "N/m":           [1, 0, -2, 0, 0, 0, 0],
    "N·m":           [1, 2, -2, 0, 0, 0, 0],
    "Pa·s":          [1, -1, -1, 0, 0, 0, 0],
    # energy forms
    "J/kg":          [0, 2, -2, 0, 0, 0, 0],
    "J/(kg·K)":      [0, 2, -2, -1, 0, 0, 0],
    "J/kg/K":        [0, 2, -2, -1, 0, 0, 0],
    # explicit base-unit products, superscripts and carets
    "kg·m²·s⁻³":     [1, 2, -3, 0, 0, 0, 0],
    "kg m^2 s^-3":   [1, 2, -3, 0, 0, 0, 0],
}


def test_parse_dimensions():
    for unit, expected in CASES.items():
        got = parse_dimensions(unit)
        assert got == expected, f"{unit!r}: got {got}, expected {expected}"


if __name__ == "__main__":
    failures = 0
    for unit, expected in CASES.items():
        got = parse_dimensions(unit)
        ok = got == expected
        failures += not ok
        print(f"{'OK ' if ok else 'FAIL'} {unit!r:18s} -> {got}"
              + ("" if ok else f"  expected {expected}"))
    print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
