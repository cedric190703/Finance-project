"""Smoke tests: the package imports and the compiled core is wired up."""

from __future__ import annotations

import aegis


def test_package_exposes_version() -> None:
    assert aegis.__version__


def test_compiled_core_is_importable_and_matches_package_version() -> None:
    assert aegis.core_version() == aegis.__version__
