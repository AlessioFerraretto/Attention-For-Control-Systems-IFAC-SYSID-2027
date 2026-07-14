"""Project directory and data path constants.

This module centralizes filesystem paths used across the project so tests
and training scripts can import canonical references to datasets and
preprocessed artifacts.

Notes:
        - Paths are created relative to the current working directory via
                `pathlib.Path()` and may change depending on where scripts are executed.
        - The module intentionally prints the resolved root directory during
                import to aid interactive debugging; consumers in production can
                import constants without relying on the printed message.
"""

from pathlib import Path

ROOT_DIR = Path()
print(f"Executing from '{ROOT_DIR.absolute()}'")

DATA_DIR = ROOT_DIR / "data"
