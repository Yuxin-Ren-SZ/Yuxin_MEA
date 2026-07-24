"""One module per report subplot.

Each module renders exactly one panel to a caption-free image and writes its
caption to a sibling ``.txt``, so any panel can be dropped into a slide, poster
or a different figure layout without carrying the report's framing with it::

    python -m scripts.report.panels.f4c_firing_rate
    python -m scripts.report.render_panels --figures f4      # all of F4, one process

Modules define ``NAME``, ``CAPTION``, ``FIGSIZE`` and either ``build(ax, ctx)`` or
``build_fig(fig, ctx)``; :mod:`scripts.report.panel_base` does the rest, and
:mod:`scripts.report.panel_data` gives them a shared, disk-cached data context so
running them separately does not re-fit the expensive embeddings.
"""
