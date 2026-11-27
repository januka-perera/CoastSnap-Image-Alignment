from types import SimpleNamespace

import pytest

import coastsnap_align.backends as backends


def test_plugin_factory_receives_configuration(monkeypatch):
    config = object()
    plugin = SimpleNamespace(name="other", load=lambda: lambda cfg: ("backend", cfg))
    monkeypatch.setattr(backends, "entry_points", lambda **kwargs: [plugin])
    assert backends.create_backend("other", config) == ("backend", config)
    assert "other" in backends.backend_names()


def test_plugins_cannot_shadow_builtins(monkeypatch):
    monkeypatch.setattr(
        backends, "entry_points", lambda **kwargs: [SimpleNamespace(name="lightglue")]
    )
    with pytest.raises(ValueError, match="reserved"):
        backends.backend_names()
