"""Private crude-oil dataset package.

The CSV files in this directory are the raw inputs for the E1-a experiments.
Dataset-specific parsing and unit conversion are kept in :mod:`adapter` so a
new dataset can be added without changing the shared training code.
"""

from .adapter import load_dataset

__all__ = ["load_dataset"]
