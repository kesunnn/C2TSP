"""Compatibility entry point for the historical two-round training program."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tsp_onetree.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
