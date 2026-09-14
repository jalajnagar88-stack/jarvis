"""Allows ``python -m jarvis``."""

from __future__ import annotations

import sys

from jarvis.cli import main

if __name__ == "__main__":
    sys.exit(main())
