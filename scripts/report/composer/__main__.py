"""``python -m scripts.report.composer`` — serve the figure composer."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="pipeline config JSON (default: pipeline_config_local.json)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8060)
    ap.add_argument("--cache-dir", default=None,
                    help="panel-preview cache (default: ~/.cache/yuxin_mea/composer)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    from .server import create_app

    app = create_app(args.config,
                     Path(args.cache_dir) if args.cache_dir else None)
    info = app.extensions["composer"]
    print(f"\nFigure composer  ->  http://{args.host}:{args.port}")
    print(f"  analysis_root : {info['analysis_root']}")
    print(f"  tidy_long     : {info['fingerprint']['tidy_rows']} rows, "
          f"sha {info['fingerprint']['tidy_sha256'][:12]}")
    print(f"  specs         : {info['specs_dir']}\n")
    # threaded so a slow /api/preview does not stall the page; the actual
    # rendering happens in the cache's process pool
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True,
            use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
