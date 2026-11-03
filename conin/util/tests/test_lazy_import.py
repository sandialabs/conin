"""conin.util.lazy_import."""

import sys

import pytest

from conin.util import lazy_import


def test_imports_on_first_use(monkeypatch):
    monkeypatch.delitem(sys.modules, "colorsys", raising=False)
    colorsys = lazy_import("colorsys")
    assert "colorsys" not in sys.modules
    assert colorsys.rgb_to_hsv(0, 0, 0) == (0, 0, 0)
    assert "colorsys" in sys.modules
    assert bool(colorsys)


def test_bool_imports():
    json = lazy_import("json.decoder")
    assert json
    assert json.JSONDecodeError is sys.modules["json.decoder"].JSONDecodeError


def test_unavailable_module():
    missing = lazy_import("conin_no_such_module")
    assert not missing
    with pytest.raises(
        ImportError, match="conin_no_such_module module is not available"
    ):
        missing.anything
    assert repr(missing) == "lazy_import('conin_no_such_module')"


def test_dunder_lookups_do_not_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "colorsys", raising=False)
    colorsys = lazy_import("colorsys")
    assert not hasattr(colorsys, "__wrapped__")
    assert "colorsys" not in sys.modules
