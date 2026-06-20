"""Permite ejecutar `python -m skfold_kge`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
