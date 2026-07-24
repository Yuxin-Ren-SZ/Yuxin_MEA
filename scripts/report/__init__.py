"""Journal-report figure generation for the HD-MEA network-scan study.

Self-contained package that turns the pipeline's rolled-up metric tables and
per-well artifacts into publication-grade, vector (PDF/SVG) + raster (PNG)
figures with a single consistent style/palette.

The deliverable is **one image per subplot**. Main figures F2-F8 are panels under
``panels/``::

    python -m scripts.report.render_panels --all   # render every panel
    python -m scripts.report.assemble_figure       # arrange them into sheets
    python -m scripts.report.panel_pptx            # editable Letter deck

The supplement S1-S5 is still composed, one figure per builder::

    python -m scripts.report.make_report --qpcr-dir <qPCR root>
    python -m scripts.report.build_pptx            # the supplement deck

Both paths take their cohort from ``panel_data.PanelContext``, so the main
figures and the supplement always describe the same recordings. See
``README.md`` for the figure catalog and the statistics.
"""
