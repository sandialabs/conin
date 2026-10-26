"""Tests for conin.util.constraint_data.merge_constraint_data."""

import types

import munch

from conin.util.constraint_data import merge_constraint_data


def test_no_app_data_returns_the_fields():
    fields = munch.Munch(T=[0, 1])
    assert merge_constraint_data(None, fields) is fields


def test_no_app_data_and_no_fields():
    assert merge_constraint_data(None, None) is None


def test_munch_app_data_gets_the_fields_on_a_copy():
    app = munch.Munch(limit=3, T="app")
    data = merge_constraint_data(app, munch.Munch(T=[0, 1], evidence=None))
    assert data == munch.Munch(limit=3, T=[0, 1], evidence=None)
    assert isinstance(data, munch.Munch)
    # The application data is never modified.
    assert app == munch.Munch(limit=3, T="app")
    assert data is not app


def test_dict_app_data():
    app = {"limit": 3}
    data = merge_constraint_data(app, munch.Munch(T=[0]))
    assert data == {"limit": 3, "T": [0]}
    assert app == {"limit": 3}


def test_no_fields_gives_a_copy_of_the_app_data():
    app = munch.Munch(limit=3)
    data = merge_constraint_data(app, None)
    assert data == app
    assert data is not app


def test_object_app_data_gets_attributes():
    app = types.SimpleNamespace(limit=3)
    data = merge_constraint_data(app, munch.Munch(T=[0, 1]))
    assert (data.limit, data.T) == (3, [0, 1])
    assert not hasattr(app, "T")


def test_the_copy_is_shallow():
    shared = [1, 2]
    app = munch.Munch(values=shared)
    data = merge_constraint_data(app, munch.Munch(T=[0]))
    assert data["values"] is shared
