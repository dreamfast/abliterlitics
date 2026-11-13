"""Allow running src.db as a module: python3 -m src.db <command>."""

from __future__ import annotations

from src.db.cli import main

if __name__ == "__main__":
    main()
