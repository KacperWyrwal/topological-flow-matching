"""Topological Flow Matching (topofm)

Modular package factoring the original everything.py into submodules.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("topofm")
except PackageNotFoundError:  # local editable or src layout without install
    __version__ = "0.0.0"
