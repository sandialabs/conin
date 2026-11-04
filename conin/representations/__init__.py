"""Model representations used by conin inference backends."""

from .base import (
    Representation,
    register_representation,
    unregister_representation,
    supported_representations,
    get_representation_builder,
    build_representation,
)
