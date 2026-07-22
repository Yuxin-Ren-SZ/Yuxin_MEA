"""Cohort-scale panels — burst archetypes (F6c) and the pooled-UMAP grid (F6b).

These two figures differ from the rest in that their panels are not independent:
F6c's four views all read one UMAP fit over every pooled burst in the cohort, and
F6b's grid cells all read one pooled per-well embedding. Rendering each panel
from scratch would refit a minutes-long embedding per tile.

:meth:`RenderContext.dataset` exists for exactly this. The expensive work runs
once per context and every panel that asks for it gets the same result, so a
composition can hold all four F6c views, or a row of F6b timepoints, at the cost
of one fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as S
from .panels import (
    Bool, Choice, Int, PanelDataMissing, PanelSpec, Text, register,
    simple_panel, group_color, ordered_groups,
)

_ARCH_PALETTE = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#aa3377", "#66ccee"]


# --------------------------------------------------------------------------- #
# F6c — shared burst-archetype dataset
# --------------------------------------------------------------------------- #
def _build_archetypes(ctx, max_rows_per_group: int = 40, max_bursts: int = 9000):
    """Pool bursts across wells, embed once, cluster globally.

    Mirrors ``fig6c_bursttypes.build_f6c``'s preparation step so the composer and
    the batch figure cluster identically.
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    from .fig6c_bursttypes import FEATURES, LOG_FEATURES, _cluster, collect_bursts

    bursts = collect_bursts(ctx.tidy, ctx.analysis_root, seed=ctx.seed,
                            groups=ctx.groups,
                            max_rows_per_group=max_rows_per_group,
                            max_bursts=max_bursts)
    if bursts is None or bursts.empty:
        raise PanelDataMissing("no bursts collected for the selected groups")

    X = bursts[FEATURES].to_numpy(float)
    for j, f in enumerate(FEATURES):
        if f in LOG_FEATURES:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Z = StandardScaler().fit_transform(X)
    emb = np.asarray(umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                               random_state=ctx.seed).fit_transform(Z))
    labels, k = _cluster(Z, seed=ctx.seed)
    return {"bursts": bursts.assign(arch=labels), "emb": emb, "Z": Z,
            "labels": labels, "k": int(k)}


def _archetypes(ctx, max_rows_per_group=40, max_bursts=9000):
    return ctx.dataset("f6c.archetypes", _build_archetypes,
                       max_rows_per_group=int(max_rows_per_group),
                       max_bursts=int(max_bursts))


@simple_panel
def _panel_burst_space(ax, ctx, color_by="group", max_rows_per_group=40,
                       max_bursts=9000, title=""):
    d = _archetypes(ctx, max_rows_per_group, max_bursts)
    emb, bursts, k = d["emb"], d["bursts"], d["k"]

    if color_by == "group":
        for g in ordered_groups(bursts.group.unique()):
            m = bursts.group.to_numpy() == g
            ax.scatter(emb[m, 0], emb[m, 1], s=3, c=group_color(g), lw=0,
                       alpha=0.5, label=g, rasterized=True)
        ax.legend(fontsize=6, markerscale=2, loc="best")
        default_title = "Burst feature space by group"
    else:
        for a in range(k):
            m = d["labels"] == a
            ax.scatter(emb[m, 0], emb[m, 1], s=3, lw=0, alpha=0.6,
                       c=_ARCH_PALETTE[a % len(_ARCH_PALETTE)],
                       label=f"type {a}", rasterized=True)
        ax.legend(fontsize=6, markerscale=2, loc="best",
                  title=f"archetype (k={k})", title_fontsize=6)
        default_title = "Global burst archetypes"

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(title or default_title, loc="left", fontweight="bold")


@simple_panel
def _panel_composition(ax, ctx, max_rows_per_group=40, max_bursts=9000, title=""):
    """Per-well archetype fractions by arm, with the MWU-vs-Control result computed.

    Per-well fractions rather than pooled burst counts: pooling bursts treats
    every burst as independent and makes groups look separated when they are not.
    """
    from .fig6c_bursttypes import _composition_stats, _stars

    d = _archetypes(ctx, max_rows_per_group, max_bursts)
    bursts, k = d["bursts"], d["k"]
    groups = [g for g in ordered_groups(bursts.group.unique()) if g in ctx.groups]

    wc = (bursts.groupby(["well_uid", "group", "arch"]).size()
          .unstack("arch", fill_value=0))
    wc = wc.div(wc.sum(axis=1), axis=0).reset_index()
    cstats = _composition_stats(wc, k, groups)
    qlook = ({(int(r.arch), r.arm): r.q for r in cstats.itertuples()}
             if not cstats.empty else {})
    n_sig = int((cstats.q < 0.05).sum()) if not cstats.empty else 0

    off = np.linspace(-0.3, 0.3, max(len(groups), 1))
    for a in range(k):
        for gi, g in enumerate(groups):
            v = (wc.loc[wc.group == g, a].dropna().to_numpy()
                 if a in wc.columns else np.array([]))
            if not len(v):
                continue
            pos = a + off[gi]
            bp = ax.boxplot(v, positions=[pos], widths=0.14, patch_artist=True,
                            showfliers=False)
            bp["boxes"][0].set(facecolor=group_color(g), alpha=0.5, lw=0.5)
            bp["medians"][0].set(color="k", lw=0.8)
            q = qlook.get((a, g), np.nan)
            if not np.isnan(q) and q < 0.05:
                ax.text(pos, min(v.max() + 0.04, 0.98), _stars(q), ha="center",
                        va="bottom", fontsize=7, color=group_color(g))

    ax.set_xticks(range(k))
    ax.set_xticklabels([f"type {a}" for a in range(k)])
    ax.set_ylabel("per-well fraction", fontsize=7)
    # fractions only reach 1.0; the headroom is for the legend, which would
    # otherwise sit on top of the boxes since several span the full range
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("well = unit; MWU vs Control, BH-FDR"
                  + ("  (all n.s.)" if n_sig == 0 else f"  ({n_sig} q<0.05)"),
                  fontsize=5.5)
    handles = [ax.plot([], [], marker="s", ls="", color=group_color(g), label=g)[0]
               for g in groups]
    ax.legend(handles=handles, fontsize=5.5, loc="upper center", ncol=len(groups),
              frameon=False, columnspacing=1.0, handletextpad=0.4)
    ax.set_title(title or ("Composition by well "
                           + ("(n.s.)" if n_sig == 0 else f"({n_sig} sig)")),
                 loc="left", fontweight="bold")


def _panel_archetype_profiles(fig, subspec, ctx, max_rows_per_group=40,
                              max_bursts=9000, title=""):
    """Composite: the profile heatmap plus its own colourbar inside one cell."""
    from .fig6c_bursttypes import FEAT_LABELS, FEATURES

    d = _archetypes(ctx, max_rows_per_group, max_bursts)
    Z, labels, k = d["Z"], d["labels"], d["k"]
    inner = subspec.subgridspec(1, 2, width_ratios=[1, 0.045], wspace=0.06)
    ax = fig.add_subplot(inner[0, 0])
    cax = fig.add_subplot(inner[0, 1])

    prof = np.vstack([Z[labels == a].mean(0) for a in range(k)])
    im = ax.imshow(prof, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEAT_LABELS, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"type {a}" for a in range(k)], fontsize=7)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("z-score", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(title or "Archetype feature profiles", loc="left",
                 fontweight="bold")


_F6C_PARAMS = {
    "max_rows_per_group": Int(40, "Well-recordings / group", min=2, max=400),
    "max_bursts": Int(9000, "Pooled bursts cap", min=200, max=100000),
    "title": Text("", "Panel title"),
}
_F6C_NOTE = ("Shares one cohort-wide UMAP + KMeans fit with the other archetype "
             "panels; changing the sampling parameters refits it.")

register(PanelSpec(
    id="archetypes.space", title="Burst feature space", section="Burst types",
    fn=_panel_burst_space, default_w=6, default_h=5, families=("ml",),
    description="Pooled per-burst UMAP, coloured by group or by archetype. "
                + _F6C_NOTE,
    params={"color_by": Choice("group", ["group", "archetype"], "Colour by"),
            **_F6C_PARAMS}))

register(PanelSpec(
    id="archetypes.composition", title="Archetype composition by well",
    section="Burst types", fn=_panel_composition, default_w=6, default_h=5,
    families=("ml",),
    description="Per-well archetype fractions per arm; stars are Mann-Whitney vs "
                "Control with BH-FDR, computed here rather than asserted. "
                + _F6C_NOTE,
    params=dict(_F6C_PARAMS)))

register(PanelSpec(
    id="archetypes.profiles", title="Archetype feature profiles",
    section="Burst types", fn=_panel_archetype_profiles, default_w=6, default_h=5,
    families=("ml",),
    description="Standardised feature means per archetype, for reading off what "
                "each type is. " + _F6C_NOTE,
    params=dict(_F6C_PARAMS)))


# --------------------------------------------------------------------------- #
# F6b — pooled per-well UMAP at one developmental timepoint
# --------------------------------------------------------------------------- #
def _build_pooled_well(ctx, well_uid: str):
    from .fig_umap_migration import _pool_well

    d = _pool_well(ctx.tidy, ctx.analysis_root, well_uid)
    if d is None:
        raise PanelDataMissing(f"no debug traces for {well_uid}")
    return d


def _pooled(ctx, well_uid: str):
    return ctx.dataset("f6b.pooled", _build_pooled_well, well_uid=well_uid)


@simple_panel
def _panel_state_manifold(ax, ctx, well="", arm="IVH_Late", stage=0, title=""):
    """One cell of the F6b grid: a well's pooled embedding at one timepoint.

    The embedding is fit once over *all* of the well's recordings, so the axes
    mean the same thing at every stage — which is the whole point of the figure
    and why the fit is shared through the context rather than done per panel.
    """
    from .fig_umap_migration import (
        SELECTED_WELLS, _GREY_SAMPLE, _auto_days, _stride_keep, _BURST_C, _REST_C,
    )

    wuid = well or SELECTED_WELLS.get(arm)
    if not wuid:
        raise PanelDataMissing(f"no well selected for {arm}")
    if wuid not in set(ctx.tidy.well_uid):
        raise PanelDataMissing(f"{wuid} is not in the selected groups")

    d = _pooled(ctx, wuid)
    days = _auto_days(d["tau"])
    stage = int(stage)
    day = days[stage] if stage < len(days) else None

    grey = d["emb"][_stride_keep(len(d["emb"]), _GREY_SAMPLE)]
    ax.scatter(grey[:, 0], grey[:, 1], s=1.0, c="0.88", lw=0, zorder=1,
               rasterized=True)
    if day is not None:
        sel = d["tau"] == day
        rest, brst = sel & ~d["is_burst"], sel & d["is_burst"]
        ax.scatter(d["emb"][rest, 0], d["emb"][rest, 1], s=2, c=_REST_C, lw=0,
                   alpha=0.5, zorder=2, rasterized=True)
        ax.scatter(d["emb"][brst, 0], d["emb"][brst, 1], s=4, c=_BURST_C, lw=0,
                   alpha=0.75, zorder=3, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.03, 0.96, f"Day {day}" if day is not None else "n/a",
            transform=ax.transAxes, fontsize=6.5, va="top", color="0.25")
    ax.set_title(title or f"{arm} · {wuid.split('|')[0]}", loc="left",
                 fontweight="bold", fontsize=7.5)


register(PanelSpec(
    id="manifold.state", title="Network-state manifold", section="Burst types",
    fn=_panel_state_manifold, default_w=3, default_h=4, families=("ml",),
    description=("One well's pooled resting/bursting UMAP at one post-treatment "
                 "stage. Axes are fit over all of the well's recordings at once, "
                 "so positions are comparable across stages; place several side "
                 "by side to show the migration."),
    params={
        "arm": Choice("IVH_Late", ["Control", "IVH_Early", "IVH_Late"], "Arm"),
        "well": Text("", "Well uid (blank = curated pick)"),
        "stage": Int(0, "Stage (0 = treatment day)", min=0, max=8),
        "title": Text("", "Panel title"),
    }))
