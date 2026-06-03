# -*- coding: utf-8 -*-
"""Console helpers shared by local CLI scripts."""

from __future__ import annotations

import sys
from typing import TextIO


def _reconfigure_stream(stream: TextIO) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, TypeError, ValueError):
        return


def configure_utf8_stdio(
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> None:
    """Prefer UTF-8 console output so Korean logs remain readable on Windows."""
    _reconfigure_stream(sys.stdout if stdout is None else stdout)
    _reconfigure_stream(sys.stderr if stderr is None else stderr)
