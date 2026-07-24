"""
SpladeX: A custom semantic search plugin and server for Sphinx.
"""

from __future__ import annotations

from typing import Any

from .extension import setup as extension_setup

__version__ = "0.1.2"


def setup(app: Any) -> dict[str, Any]:
    """
    Sphinx extension entrypoint.
    Delegates setup to the extension module.
    """
    return extension_setup(app)
