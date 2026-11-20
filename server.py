# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp>=1.29,<2"]
# ///
"""Source entry point for a relocatable Data Engineering plugin."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai_data.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
