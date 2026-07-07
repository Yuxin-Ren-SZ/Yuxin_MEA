"""Groupname handling for group-scoped aggregate tasks.

S1 does NOT resolve groupname collisions — it only WARNS. ``groupname`` values
in the experiment cache collide across samples (e.g. 'Control' in every sample)
and have spelling variants ('10uM_H2O2' vs '10 uM H2O2'). Grouping raw
groupnames is therefore only unambiguous within a ``sample_id`` (why the group
presets include ``sample_id``).

``canonicalize_groupname`` is the named hook the future S2 rewrite replaces with
a real resolver (e.g. the ``canonical_group`` template CSV in
``scripts/compare_treatment_groups.py``). In S1 it is the identity.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .well_dims import WellDims


def canonicalize_groupname(raw: str) -> str:
    """S1: identity. S2 swaps this for a real resolver/map."""
    return raw


def _norm(raw: str) -> str:
    """Normalized form for variant detection only (strips space/underscore/case)."""
    return re.sub(r"[\s_]+", "", str(raw).strip().lower())


def detect_groupname_issues(dims_list: "list[WellDims]") -> list[str]:
    """Return human-readable WARNING lines for groupname collisions/variants.

    Two classes of issue:
      * cross-sample collision — the same raw groupname under >1 sample_id
        (so a ``(groupname,)``-only scope would pool unrelated cultures);
      * spelling variants — distinct raw groupnames that normalize equal
        (so ``(sample_id, groupname)`` still splits what is one treatment).
    """
    warnings: list[str] = []

    samples_by_group: dict[str, set[str]] = defaultdict(set)
    raws_by_norm: dict[str, set[str]] = defaultdict(set)
    for d in dims_list:
        g = d.groupname
        if g == "?" or not g:
            continue
        samples_by_group[g].add(d.sample_id)
        raws_by_norm[_norm(g)].add(g)

    for group, samples in sorted(samples_by_group.items()):
        if len(samples) > 1:
            warnings.append(
                f"groupname {group!r} spans {len(samples)} samples "
                f"{sorted(samples)} — group scopes without sample_id would pool them."
            )
    for _norm_key, raws in sorted(raws_by_norm.items()):
        if len(raws) > 1:
            warnings.append(
                f"groupname spelling variants collapse to the same treatment: "
                f"{sorted(raws)} — these are NOT merged in S1 (resolve manually)."
            )
    return warnings
