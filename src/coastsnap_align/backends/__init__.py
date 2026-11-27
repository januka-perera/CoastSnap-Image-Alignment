"""Built-in and third-party matcher discovery without importing PyTorch."""

from importlib.metadata import entry_points

from .base import Backend, Matches

__all__ = ["Backend", "Matches", "create_backend", "backend_names"]


def _plugins():
    plugins = {}
    for ep in entry_points(group="coastsnap_align.backends"):
        if ep.name in {"lightglue", "flexglue"} or ep.name in plugins:
            raise ValueError(f"Duplicate/reserved backend name: {ep.name}")
        plugins[ep.name] = ep
    return plugins


def backend_names():
    return ["lightglue", "flexglue", *sorted(_plugins())]


def create_backend(name, config):
    if name in {"lightglue", "flexglue"}:
        from .glue import GlueBackend

        return GlueBackend(name, config)
    plugins = _plugins()
    if name not in plugins:
        raise ValueError(f"Unknown backend {name!r}; available: {', '.join(backend_names())}")
    return plugins[name].load()(config)
