"""Deprecated location: the implementation moved into the `chb` package.

Use `from chb import compute_fingerprint` for the Python API or the `chb`
console command (equivalently `python -m chb ...`) for the CLI. This shim
keeps old imports and `python chb_metrics.py ...` invocations working.
"""
import warnings as _warnings

from chb.metrics import *  # noqa: F401,F403
from chb.metrics import (  # noqa: F401  (non-star helpers used by older scripts)
    _combined_cli_main,
    _json_safe,
    _has_new_density,
)

_warnings.warn(
    "chb_metrics.py is deprecated; use `from chb import compute_fingerprint` "
    "or the `chb` console command.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    _combined_cli_main()
