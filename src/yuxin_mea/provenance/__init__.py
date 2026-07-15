"""Reproducibility provenance: fingerprint raw inputs + config, stamp results.

See ``doc/caching.md`` (Provenance & reproducibility) for the model. The core
guarantee: every pipeline output records a fingerprint of the exact raw data
(``data.raw.h5`` + ``mxassay.metadata``) and resolved config that produced it,
so drift between an output and the current on-disk raw can always be detected.
"""

from .fingerprint import (
    file_hash,
    h5_fingerprint,
    params_hash,
    raw_fingerprint,
    stat_sig,
)
from .sidecar import PROVENANCE_FILENAME, read_sidecar, write_sidecar
from .verify import VerifyReport, classify_task, verify_provenance

__all__ = [
    "file_hash",
    "h5_fingerprint",
    "params_hash",
    "raw_fingerprint",
    "stat_sig",
    "PROVENANCE_FILENAME",
    "read_sidecar",
    "write_sidecar",
    "VerifyReport",
    "classify_task",
    "verify_provenance",
]
