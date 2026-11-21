"""Public interface for the SeqTrust package."""

from .core import SeqTrust
from .io import read_sequence_file

__all__ = ["SeqTrust", "read_sequence_file"]
