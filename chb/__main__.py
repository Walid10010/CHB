"""Allow `python -m chb ...` as an alias for the `chb` console script."""
from .metrics import _combined_cli_main

if __name__ == "__main__":
    _combined_cli_main()
