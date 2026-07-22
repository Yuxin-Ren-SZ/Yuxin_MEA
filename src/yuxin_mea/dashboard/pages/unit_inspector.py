"""Single-unit inspector page.

Browse each sorted neuron from a well: waveform template, waveform overlay
(random-subset snippets behind the template), ISI histogram, autocorrelogram,
amplitude-over-time, probe location, and a per-unit metrics card — plus manual
curation (good/MUA/noise labels, notes, merge, remove, amplitude-threshold
split).

Navigation: Recording → Well → Unit. Per-well data comes from the
``SortingAnalyzer`` binary-folder written by ``AnalyzerTask``, read via
``yuxin_mea.analysis.unit_inspector`` (plain ``np.load`` — no SpikeInterface),
which memoizes each well's bundle by source-file signature.

Manual curation is authored here and persisted to a durable, pipeline-independent
manifest (``yuxin_mea.analysis.manual_curation_store``). The dashboard **never
runs compute** — applying the curation is the ``manual_curation`` CLI task, whose
exact ``yuxin-mea-run`` invocation is surfaced at the bottom of the curation card.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import dash
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from flask import current_app

from yuxin_mea.dashboard.components import (
    build_filter_bar,
    filter_id,
    filter_kwargs,
    iso_to_yymmdd,
    yymmdd_to_iso,
)


dash.register_page(__name__, path="/unit-inspector", name="Unit inspector", order=6)

# Filter bar fields — status is dropped (every listed recording already has a
# completed analyzer task, so a status filter would be a no-op here).
_FILTER_FIELDS = ("sample", "scan-type", "date", "group")


_EMPTY = go.Figure().update_layout(
    annotations=[{"text": "(no unit selected)", "xref": "paper", "yref": "paper",
                  "x": 0.5, "y": 0.5, "showarrow": False,
                  "font": {"size": 13, "color": "#84807a"}}],
    xaxis={"visible": False}, yaxis={"visible": False},
    margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=300,
)
_LABELS = ["good", "MUA", "noise"]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _graph(gid: str) -> dcc.Graph:
    return dcc.Graph(id=gid, figure=_EMPTY, config={"displayModeBar": False})


def _curation_card() -> html.Div:
    return html.Div([
        html.Div(html.Span("curation", className="h-title"), className="card-head"),
        html.Div([
            html.Div([
                html.Span("label", className="section-label",
                          style={"marginRight": "8px"}),
                *[html.Button(lab, id=f"unit-insp-label-{lab.lower()}", n_clicks=0,
                              className="btn chip") for lab in _LABELS],
                html.Span(style={"width": "14px", "display": "inline-block"}),
                html.Button("remove", id="unit-insp-remove-btn", n_clicks=0,
                            className="btn chip"),
            ], style={"display": "flex", "alignItems": "center",
                      "flexWrap": "wrap", "gap": "6px", "marginBottom": "10px"}),

            dcc.Textarea(id="unit-insp-note", placeholder="notes for this unit…",
                         style={"width": "100%", "minHeight": "44px",
                                "fontFamily": "var(--font-mono)", "fontSize": "11px",
                                "marginBottom": "10px"}),

            html.Div([
                html.Span("merge units", className="section-label",
                          style={"marginRight": "8px"}),
                html.Div(dcc.Dropdown(id="unit-insp-merge-select", options=[],
                                      value=[], multi=True,
                                      placeholder="pick ≥2 units"),
                         style={"flex": "1 1 240px"}),
                html.Button("group", id="unit-insp-merge-add", n_clicks=0,
                            className="btn"),
            ], style={"display": "flex", "alignItems": "center", "gap": "6px",
                      "marginBottom": "6px"}),
            html.Div(id="unit-insp-merge-groups", style={"marginBottom": "10px"}),

            html.Div([
                html.Span("split by amplitude ≤", className="section-label",
                          style={"marginRight": "8px"}),
                dcc.Input(id="unit-insp-split-thr", type="number", debounce=True,
                          placeholder="µV", style={"width": "90px"}),
                html.Button("set", id="unit-insp-split-set", n_clicks=0,
                            className="btn"),
                html.Button("clear", id="unit-insp-split-clear", n_clicks=0,
                            className="btn"),
                html.Span(id="unit-insp-split-state",
                          style={"marginLeft": "8px", "fontSize": "11px",
                                 "color": "var(--ink-3)",
                                 "fontFamily": "var(--font-mono)"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "6px",
                      "marginBottom": "12px", "flexWrap": "wrap"}),

            html.Div([
                html.Button("Save curation", id="unit-insp-save", n_clicks=0,
                            className="btn primary"),
                html.Span(id="unit-insp-status-msg",
                          style={"marginLeft": "12px", "fontSize": "11px",
                                 "fontFamily": "var(--font-mono)",
                                 "color": "var(--ink-3)"}),
                html.Span(id="unit-insp-stale-badge", style={"marginLeft": "10px"}),
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "10px"}),

            html.Div("Apply (in a terminal):", className="section-label"),
            html.Pre(id="unit-insp-apply-cmd", className="code terminal",
                     style={"whiteSpace": "pre-wrap", "fontSize": "11px"}),
        ], className="card-body"),
    ], className="card", style={"marginTop": "12px"})


layout = html.Div([
    html.Div(
        html.Div([
            html.Div([
                html.Span("workspace"),
                html.Span("spike_sorting"),
                html.Span("single-unit"),
            ], className="breadcrumb"),
            html.H1("Unit inspector"),
            html.Div(
                "Review each sorted neuron: template, waveform overlay, ISI / "
                "autocorrelogram, amplitude drift, probe location, metrics — and "
                "curate by hand.",
                className="subtitle"),
        ]),
        className="view-head",
    ),

    html.Div(id="unit-insp-banner"),

    # Recording filter bar (Sample · Scan · Date · Group) narrows the dropdown.
    build_filter_bar("unit", show=_FILTER_FIELDS),

    html.Div([
        html.Label("Recording", className="section-label",
                   style={"marginRight": "6px"}),
        html.Div(dcc.Dropdown(id="unit-insp-rec", options=[], value=None,
                              clearable=False), style={"width": "340px"}),
        html.Span(style={"display": "inline-block", "width": "14px"}),
        html.Label("Well", className="section-label", style={"marginRight": "6px"}),
        html.Div(dcc.Dropdown(id="unit-insp-well", options=[], value=None,
                              clearable=False), style={"width": "160px"}),
        html.Button("Rescan", id="unit-insp-rescan", n_clicks=0,
                    className="btn", style={"marginLeft": "12px"}),
        html.Span(id="unit-insp-status", style={
            "marginLeft": "12px", "fontFamily": "var(--font-mono)",
            "fontSize": "11px", "color": "var(--ink-3)"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "6px",
              "marginBottom": "16px", "flexWrap": "wrap"}),

    dcc.Store(id="unit-insp-analyzer-dir", data=None),
    dcc.Store(id="unit-insp-curation-dir", data=None),
    dcc.Store(id="unit-insp-selected-unit", data=None),
    dcc.Store(id="unit-insp-manifest", data=None),

    html.Div([
        html.Div([
            html.Div([
                html.Span("units", className="h-title"),
                html.Span(id="unit-insp-count", className="h-actions",
                          style={"color": "var(--ink-3)", "fontSize": "10px",
                                 "fontFamily": "var(--font-mono)",
                                 "textTransform": "none", "letterSpacing": "0"}),
            ], className="card-head"),
            # Per-well unit filters.
            html.Div([
                dcc.Dropdown(
                    id="unit-insp-fltr-label",
                    options=[{"label": x, "value": x} for x in
                             ("good", "MUA", "noise", "unlabeled")],
                    value=[], multi=True, placeholder="label",
                    style={"flex": "1 1 130px", "fontSize": "11px"}),
                dcc.Dropdown(
                    id="unit-insp-fltr-curated",
                    options=[{"label": "auto: all", "value": "all"},
                             {"label": "auto: pass", "value": "pass"},
                             {"label": "auto: reject", "value": "reject"}],
                    value="all", clearable=False,
                    style={"flex": "0 0 110px", "fontSize": "11px"}),
                dcc.Input(id="unit-insp-fltr-fr", type="number", debounce=True,
                          placeholder="min FR", style={"width": "72px"}),
                dcc.Input(id="unit-insp-fltr-n", type="number", debounce=True,
                          placeholder="min n", style={"width": "64px"}),
            ], style={"display": "flex", "gap": "6px", "flexWrap": "wrap",
                      "padding": "8px 10px", "alignItems": "center",
                      "borderBottom": "1px solid var(--line)"}),
            html.Div(id="unit-insp-list",
                     style={"maxHeight": "62vh", "overflowY": "auto",
                            "padding": "0"}),
        ], className="card", style={"flex": "0 0 340px"}),

        html.Div([
            html.Div(id="unit-insp-metrics-card", style={"marginBottom": "12px"}),
            html.Div([
                html.Div(_graph("unit-insp-fig-template"), style={"flex": "1 1 420px"}),
                html.Div(_graph("unit-insp-fig-overlay"), style={"flex": "1 1 420px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
            html.Div([
                html.Div(_graph("unit-insp-fig-isi"), style={"flex": "1 1 300px"}),
                html.Div(_graph("unit-insp-fig-acg"), style={"flex": "1 1 300px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
            html.Div([
                html.Div(_graph("unit-insp-fig-amp"), style={"flex": "1 1 300px"}),
                html.Div(_graph("unit-insp-fig-probe"), style={"flex": "1 1 300px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
            _curation_card(),
        ], style={"flex": "1 1 640px"}),
    ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start",
              "flexWrap": "wrap"}),
], className="page")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> dict[str, Any]:
    return current_app.config.get("YUXIN_MEA", {})


def _curation_dir_for(analyzer_dir: str, analysis_root: str) -> str | None:
    try:
        rel = Path(analyzer_dir).parent.relative_to(Path(analysis_root) / "analyzer_data")
    except ValueError:
        return None
    cdir = Path(analysis_root) / "curation_data" / rel / "auto_curation"
    return str(cdir) if cdir.exists() else None


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def _load_or_default_manifest(analyzer_dir: str, curation_dir: str | None) -> dict:
    """Load the well's manual-curation manifest, or build a fresh default."""
    from yuxin_mea.analysis import manual_curation_store as mc
    from yuxin_mea.analysis import unit_inspector as ui

    analysis_root = _ctx().get("analysis_root")
    loc = mc.locate_from_analyzer_dir(analysis_root, analyzer_dir)
    bundle = ui.load_unit_bundle_cached(analyzer_dir, curation_dir=curation_dir)
    fp = mc.sorting_fingerprint(analyzer_dir, bundle.unit_ids)
    if loc is not None:
        rk, rec, well = loc
        path = mc.manifest_path(analysis_root, rk, rec, well)
        existing = mc.load_manifest(path)
        if existing is not None:
            return existing
        return mc.default_manifest(rk, rec, well, bundle.unit_ids, fp)
    return mc.default_manifest("", "", "", bundle.unit_ids, fp)


# ---------------------------------------------------------------------------
# Callbacks — navigation
# ---------------------------------------------------------------------------


@callback(
    Output("unit-insp-banner", "children"),
    Output("unit-insp-rec", "options"),
    Output("unit-insp-rec", "value"),
    Output("unit-insp-status", "children"),
    Output(filter_id("unit", "sample"), "options"),
    Output(filter_id("unit", "scan-type"), "options"),
    Output(filter_id("unit", "date"), "min_date_allowed"),
    Output(filter_id("unit", "date"), "max_date_allowed"),
    Output(filter_id("unit", "group"), "options"),
    Input("unit-insp-rescan", "n_clicks"),
    Input(filter_id("unit", "sample"), "value"),
    Input(filter_id("unit", "scan-type"), "value"),
    Input(filter_id("unit", "date"), "start_date"),
    Input(filter_id("unit", "date"), "end_date"),
    Input(filter_id("unit", "group"), "value"),
    State("unit-insp-rec", "value"),
)
def _populate_recordings(_n, f_sample, f_scan, f_from, f_to, f_group, current_rec):
    from pathlib import Path

    from yuxin_mea.analysis import unit_inspector as ui
    from yuxin_mea.dashboard.components import no_config_banner
    from yuxin_mea.dashboard.data import filter_recordings, load_recordings_detail

    ctx_app = _ctx()
    banner = None if ctx_app.get("config_exists") else no_config_banner()
    analysis_root = ctx_app.get("analysis_root")
    empty: list[dict] = []
    if analysis_root is None:
        return banner, [], None, "analysis_root not set.", empty, empty, None, None, empty

    if ctx.triggered_id == "unit-insp-rescan":
        ui.clear_cache()

    analyzer_set = set(ui.list_analyzer_recordings(analysis_root))
    recordings, well_status = load_recordings_detail(Path(analysis_root))
    # Only offer filter values that lead to an analyzer-backed recording.
    recs_with_analyzer = [r for r in recordings if r["cache_key"] in analyzer_set]
    sample_opts = sorted({r["sample_id"] for r in recs_with_analyzer})
    scan_opts = sorted({r["scan_type"] for r in recs_with_analyzer})
    date_opts = sorted({r["date"] for r in recs_with_analyzer})
    date_min = yymmdd_to_iso(date_opts[0]) if date_opts else None
    date_max = yymmdd_to_iso(date_opts[-1]) if date_opts else None
    group_opts = sorted({g for r in recs_with_analyzer for g in r.get("groups", [])})

    kwargs = filter_kwargs({
        "sample": f_sample, "scan-type": f_scan,
        "date-from": iso_to_yymmdd(f_from), "date-to": iso_to_yymmdd(f_to),
        "group": f_group,
    })
    filtered = filter_recordings(recordings, well_status, **kwargs)
    rec_keys = sorted(r["cache_key"] for r in filtered
                      if r["cache_key"] in analyzer_set)

    rec_options = [{"label": k, "value": k} for k in rec_keys]
    value = current_rec if current_rec in rec_keys else (rec_keys[0] if rec_keys else None)
    if not analyzer_set:
        status = "No analyzer output found in pipeline_cache."
    else:
        status = f"{len(rec_keys)} of {len(analyzer_set)} analyzer recording(s) match."
    return (
        banner, rec_options, value, status,
        [{"label": s, "value": s} for s in sample_opts],
        [{"label": s, "value": s} for s in scan_opts],
        date_min, date_max,
        [{"label": g, "value": g} for g in group_opts],
    )


@callback(
    Output("unit-insp-well", "options"),
    Output("unit-insp-well", "value"),
    Input("unit-insp-rec", "value"),
)
def _populate_wells(rec_key: str | None):
    from yuxin_mea.analysis import unit_inspector as ui

    analysis_root = _ctx().get("analysis_root")
    if not (rec_key and analysis_root):
        return [], None
    dirs = ui.analyzer_dirs_for_recording(analysis_root, rec_key)
    if not dirs:
        return [], None
    opts = [{"label": w, "value": w} for w in sorted(dirs)]
    return opts, opts[0]["value"]


@callback(
    Output("unit-insp-analyzer-dir", "data"),
    Output("unit-insp-curation-dir", "data"),
    Input("unit-insp-rec", "value"),
    Input("unit-insp-well", "value"),
)
def _resolve_dirs(rec_key: str | None, well_id: str | None):
    from yuxin_mea.analysis import unit_inspector as ui

    analysis_root = _ctx().get("analysis_root")
    if not (rec_key and well_id and analysis_root):
        return None, None
    dirs = ui.analyzer_dirs_for_recording(analysis_root, rec_key)
    adir = dirs.get(well_id)
    if adir is None:
        return None, None
    cdir = _curation_dir_for(str(adir), str(analysis_root))
    return str(adir), cdir


@callback(
    Output("unit-insp-merge-select", "options"),
    Input("unit-insp-analyzer-dir", "data"),
    State("unit-insp-curation-dir", "data"),
)
def _merge_options(analyzer_dir, curation_dir):
    from yuxin_mea.analysis import unit_inspector as ui

    if not analyzer_dir:
        return []
    bundle = ui.load_unit_bundle_cached(analyzer_dir, curation_dir=curation_dir)
    return [{"label": f"unit {u}", "value": u} for u in bundle.unit_ids]


# ---------------------------------------------------------------------------
# Callbacks — master list + detail figures
# ---------------------------------------------------------------------------


def _filter_units(tbl, labels, fltr_label, fltr_curated, fltr_fr, fltr_n):
    """Apply the per-well unit filters to the unit table (returns a view)."""
    if fltr_label:
        want = set(fltr_label)

        def _keep(uid: int) -> bool:
            lab = labels.get(str(uid))
            return (lab in want) or (lab is None and "unlabeled" in want)

        tbl = tbl[[_keep(int(u)) for u in tbl["unit_id"]]]
    if fltr_curated in ("pass", "reject") and "auto_curated" in tbl.columns:
        tbl = tbl[tbl["auto_curated"] == (fltr_curated == "pass")]
    if fltr_fr is not None and "firing_rate" in tbl.columns:
        tbl = tbl[tbl["firing_rate"] >= float(fltr_fr)]
    if fltr_n is not None and "n_spikes" in tbl.columns:
        tbl = tbl[tbl["n_spikes"] >= int(fltr_n)]
    return tbl


@callback(
    Output("unit-insp-list", "children"),
    Output("unit-insp-selected-unit", "data"),
    Output("unit-insp-count", "children"),
    Input("unit-insp-analyzer-dir", "data"),
    Input("unit-insp-curation-dir", "data"),
    Input("unit-insp-manifest", "data"),
    Input({"type": "unit-insp-row", "index": ALL}, "n_clicks"),
    Input("unit-insp-fltr-label", "value"),
    Input("unit-insp-fltr-curated", "value"),
    Input("unit-insp-fltr-fr", "value"),
    Input("unit-insp-fltr-n", "value"),
    State("unit-insp-selected-unit", "data"),
)
def _render_list(analyzer_dir, curation_dir, manifest, _clicks,
                 fltr_label, fltr_curated, fltr_fr, fltr_n, current_unit):
    from yuxin_mea.analysis import unit_inspector as ui

    if not analyzer_dir:
        return (html.Div("Select a recording and well.",
                         style={"padding": "12px", "color": "var(--ink-3)"}),
                None, "")

    trig = ctx.triggered_id
    bundle = ui.load_unit_bundle_cached(analyzer_dir, curation_dir=curation_dir)
    labels = (manifest or {}).get("labels", {})
    removed = set((manifest or {}).get("removed", []))

    full = ui.unit_table(bundle)
    tbl = _filter_units(full, labels, fltr_label, fltr_curated, fltr_fr, fltr_n)
    visible = [int(u) for u in tbl["unit_id"]]

    if isinstance(trig, dict) and trig.get("type") == "unit-insp-row":
        selected = trig["index"]
    elif current_unit in visible:
        selected = current_unit
    else:
        selected = visible[0] if visible else None

    count = f"{len(visible)}/{bundle.n_units()} units"
    if not visible:
        return (html.Div("No units match the filters.",
                         style={"padding": "12px", "color": "var(--ink-3)"}),
                selected, count)

    header = html.Div([
        html.Span("unit", style={"flex": "0 0 40px"}),
        html.Span("label", style={"flex": "0 0 52px"}),
        html.Span("FR", style={"flex": "1 1 38px", "textAlign": "right"}),
        html.Span("amp", style={"flex": "1 1 42px", "textAlign": "right"}),
        html.Span("n", style={"flex": "1 1 42px", "textAlign": "right"}),
    ], style={"display": "flex", "gap": "6px", "padding": "6px 10px",
              "position": "sticky", "top": "0", "background": "var(--bg-deep)",
              "fontFamily": "var(--font-mono)", "fontSize": "10px",
              "color": "var(--ink-3)", "borderBottom": "1px solid var(--line)"})

    rows = [header]
    for _, r in tbl.iterrows():
        uid = int(r["unit_id"])
        is_sel = uid == selected
        manual = labels.get(str(uid))
        lab = manual or str(r.get("ks_label", ""))
        if uid in removed and manual != "noise":
            lab += " ✕"
        rows.append(html.Div([
            html.Span(str(uid), style={"flex": "0 0 40px"}),
            html.Span(lab, style={"flex": "0 0 52px", "color": "var(--ink-2)",
                                  "fontWeight": "600" if manual else "400"}),
            html.Span(_fmt(float(r.get("firing_rate", float("nan")))),
                      style={"flex": "1 1 38px", "textAlign": "right"}),
            html.Span(_fmt(float(r.get("amplitude_median", float("nan")))),
                      style={"flex": "1 1 42px", "textAlign": "right"}),
            html.Span(str(int(r.get("n_spikes", 0))),
                      style={"flex": "1 1 42px", "textAlign": "right"}),
        ],
            id={"type": "unit-insp-row", "index": uid},
            n_clicks=0,
            className="unit-row" + (" selected" if is_sel else ""),
            style={"display": "flex", "gap": "6px", "padding": "6px 10px",
                   "cursor": "pointer", "fontFamily": "var(--font-mono)",
                   "fontSize": "11px", "borderBottom": "1px solid var(--line)"}))
    return rows, selected, count


@callback(
    Output("unit-insp-metrics-card", "children"),
    Output("unit-insp-fig-template", "figure"),
    Output("unit-insp-fig-overlay", "figure"),
    Output("unit-insp-fig-isi", "figure"),
    Output("unit-insp-fig-acg", "figure"),
    Output("unit-insp-fig-amp", "figure"),
    Output("unit-insp-fig-probe", "figure"),
    Input("unit-insp-selected-unit", "data"),
    # dirs are Inputs (not State) so switching well/recording refreshes the detail
    # even when the newly-selected unit id equals the previous one (unit 0 exists
    # in every well → a same-value selected-unit write would otherwise dedupe and
    # leave the previous well's figures on screen).
    Input("unit-insp-analyzer-dir", "data"),
    Input("unit-insp-curation-dir", "data"),
)
def _render_detail(unit_id, analyzer_dir, curation_dir):
    from yuxin_mea.analysis import unit_inspector as ui

    if unit_id is None or not analyzer_dir:
        return ("", _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY)
    bundle = ui.load_unit_bundle_cached(analyzer_dir, curation_dir=curation_dir)
    if unit_id not in bundle.unit_ids:
        return ("", _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY)

    kv = ui.unit_metrics_kv(bundle, unit_id)
    card = html.Div(
        [html.Span(f"unit {unit_id}", style={"fontWeight": "600",
                                             "marginRight": "16px"})]
        + [html.Span([html.Span(f"{k}: ", style={"color": "var(--ink-3)"}),
                      html.Span(v)],
                     style={"marginRight": "16px",
                            "fontFamily": "var(--font-mono)", "fontSize": "11px"})
           for k, v in kv.items()],
        className="card", style={"padding": "10px 14px",
                                 "display": "flex", "flexWrap": "wrap",
                                 "alignItems": "center"})
    return (
        card,
        ui.fig_unit_template(bundle, unit_id),
        ui.fig_unit_overlay(bundle, unit_id),
        ui.fig_isi_hist(bundle, unit_id),
        ui.fig_autocorrelogram(bundle, unit_id),
        ui.fig_amplitude_time(bundle, unit_id),
        ui.fig_probe_location(bundle, unit_id),
    )


# ---------------------------------------------------------------------------
# Callbacks — curation authoring
# ---------------------------------------------------------------------------


@callback(
    Output("unit-insp-manifest", "data"),
    Output("unit-insp-status-msg", "children"),
    Input("unit-insp-analyzer-dir", "data"),
    Input("unit-insp-label-good", "n_clicks"),
    Input("unit-insp-label-mua", "n_clicks"),
    Input("unit-insp-label-noise", "n_clicks"),
    Input("unit-insp-remove-btn", "n_clicks"),
    Input("unit-insp-note", "n_blur"),
    Input("unit-insp-merge-add", "n_clicks"),
    Input("unit-insp-split-set", "n_clicks"),
    Input("unit-insp-split-clear", "n_clicks"),
    State("unit-insp-manifest", "data"),
    State("unit-insp-selected-unit", "data"),
    State("unit-insp-curation-dir", "data"),
    State("unit-insp-note", "value"),
    State("unit-insp-merge-select", "value"),
    State("unit-insp-split-thr", "value"),
)
def _mutate_manifest(analyzer_dir, _g, _m, _no, _rm, _nb, _mg, _ss, _sc,
                     manifest, selected, curation_dir, note, merge_sel, split_thr):
    from yuxin_mea.analysis import manual_curation_store as mc

    trig = ctx.triggered_id
    if not analyzer_dir:
        return None, ""
    # (Re)load on well change or first render.
    if trig == "unit-insp-analyzer-dir" or manifest is None:
        return _load_or_default_manifest(analyzer_dir, curation_dir), ""

    if selected is None:
        return manifest, ""

    # Fold the currently-open note first so switching action (e.g. clicking a
    # label) never drops text the user typed but hasn't blurred/saved.
    if note is not None:
        manifest = mc.set_note(manifest, selected, note)

    if trig == "unit-insp-label-good":
        manifest = mc.set_label(manifest, selected, "good")
    elif trig == "unit-insp-label-mua":
        manifest = mc.set_label(manifest, selected, "MUA")
    elif trig == "unit-insp-label-noise":
        manifest = mc.set_label(manifest, selected, "noise")
    elif trig == "unit-insp-remove-btn":
        manifest = mc.toggle_removed(manifest, selected)
    elif trig == "unit-insp-note":
        manifest = mc.set_note(manifest, selected, note or "")
    elif trig == "unit-insp-merge-add":
        if merge_sel and len(merge_sel) >= 2:
            manifest = mc.add_merge_group(manifest, merge_sel)
    elif trig == "unit-insp-split-set":
        if split_thr is not None:
            manifest = mc.set_split(manifest, selected, "amplitude", float(split_thr))
    elif trig == "unit-insp-split-clear":
        manifest = mc.clear_split(manifest, selected)

    return manifest, "● unsaved changes"


@callback(
    Output("unit-insp-status-msg", "children", allow_duplicate=True),
    Input("unit-insp-save", "n_clicks"),
    State("unit-insp-manifest", "data"),
    State("unit-insp-analyzer-dir", "data"),
    State("unit-insp-note", "value"),
    State("unit-insp-selected-unit", "data"),
    prevent_initial_call=True,
)
def _save(_n, manifest, analyzer_dir, note, selected):
    from yuxin_mea.analysis import manual_curation_store as mc

    if not manifest or not analyzer_dir:
        return "nothing to save"
    analysis_root = _ctx().get("analysis_root")
    loc = mc.locate_from_analyzer_dir(analysis_root, analyzer_dir)
    if loc is None:
        return "could not resolve manifest path"
    # Fold the currently-open note before writing.
    if selected is not None:
        manifest = mc.set_note(manifest, selected, note or "")
    rk, rec, well = loc
    path = mc.manifest_path(analysis_root, rk, rec, well)
    mc.save_manifest(path, manifest)
    return f"saved ✓  {path}"


@callback(
    Output("unit-insp-label-good", "className"),
    Output("unit-insp-label-mua", "className"),
    Output("unit-insp-label-noise", "className"),
    Output("unit-insp-remove-btn", "className"),
    Output("unit-insp-note", "value"),
    Output("unit-insp-merge-groups", "children"),
    Output("unit-insp-split-state", "children"),
    Output("unit-insp-stale-badge", "children"),
    Output("unit-insp-apply-cmd", "children"),
    Input("unit-insp-manifest", "data"),
    Input("unit-insp-selected-unit", "data"),
    State("unit-insp-analyzer-dir", "data"),
    State("unit-insp-curation-dir", "data"),
)
def _render_curation(manifest, selected, analyzer_dir, curation_dir):
    base = "btn chip"
    if not manifest or selected is None:
        return (base, base, base, base, "", "", "", "", "")

    labels = manifest.get("labels", {})
    cur = labels.get(str(selected))
    cls = [base + (" active" if cur == lab else "") for lab in _LABELS]
    removed = set(manifest.get("removed", []))
    rm_cls = "btn chip" + (" active" if selected in removed else "")

    note = manifest.get("notes", {}).get(str(selected), "")

    groups = manifest.get("merges", [])
    group_chips = [
        html.Span("+".join(f"u{u}" for u in g),
                  className="merge-chip",
                  style={"display": "inline-block", "padding": "2px 8px",
                         "marginRight": "6px", "borderRadius": "10px",
                         "background": "var(--bg-deep)", "fontSize": "11px",
                         "fontFamily": "var(--font-mono)"})
        for g in groups] or [html.Span("no merge groups",
                                       style={"color": "var(--ink-3)",
                                              "fontSize": "11px"})]

    split = next((s for s in manifest.get("splits", [])
                  if int(s["unit_id"]) == int(selected)), None)
    split_state = (f"split at {split['threshold']:.1f} µV" if split
                   else "no split")

    # Stale badge (sorting changed since labels were authored).
    from yuxin_mea.analysis import manual_curation_store as mc
    from yuxin_mea.analysis import unit_inspector as ui
    stale_badge: Any = ""
    if analyzer_dir:
        bundle = ui.load_unit_bundle_cached(analyzer_dir, curation_dir=curation_dir)
        fp = mc.sorting_fingerprint(analyzer_dir, bundle.unit_ids)
        if mc.is_stale(manifest, fp):
            stale_badge = html.Span(
                "⚠ stale — sorting changed",
                style={"padding": "2px 10px", "borderRadius": "10px",
                       "background": "#fdecec", "color": "#c62828",
                       "fontSize": "11px", "fontFamily": "var(--font-mono)"})

    cfg = _ctx().get("config_path")
    rk = manifest.get("recording_key", "")
    cmd = (f"yuxin-mea-run --config {cfg} "
           f"--tasks manual_curation --recordings {rk}") if rk else \
        "yuxin-mea-run --tasks manual_curation --recordings <recording_key>"

    return (*cls, rm_cls, note, group_chips, split_state, stale_badge, cmd)
