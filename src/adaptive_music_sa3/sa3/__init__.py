"""Vendored Stable Audio 3 edit helpers (from sa3-inversion).

Re-exports FlowEdit, RF-Inversion invert/sample, and encode/decode helpers
so the package does not depend on path hacks into the sibling repo.
"""

from .audio import decode_latent, drift, load_latent, load_latent_from_wav, write_wav
from .flow_edit import flow_edit
from .rf_inversion import invert, invert_and_edit, make_cond, sample

__all__ = [
    "decode_latent",
    "drift",
    "flow_edit",
    "invert",
    "invert_and_edit",
    "load_latent",
    "load_latent_from_wav",
    "make_cond",
    "sample",
    "write_wav",
]
