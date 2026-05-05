from __future__ import annotations

import json
import tempfile
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .burst_detector import BurstResults


class BurstOutputWriter(ABC):
    """ABC for persisting and reloading BurstResults.

    Concrete implementations choose the serialization format (pickle, JSON, etc.)
    without coupling the detector algorithm or the pipeline task to a specific format.

    Implementations must guarantee that write() + read() is lossless for all
    fields of BurstResults.
    """

    @abstractmethod
    def write(self, results: BurstResults, output_dir: Path) -> None:
        """Persist results into output_dir. Creates the directory if needed."""

    @abstractmethod
    def read(self, output_dir: Path) -> BurstResults:
        """Reconstruct BurstResults from a previously written output_dir."""


class PickleBurstOutputWriter(BurstOutputWriter):
    """Writes burst events as pickled DataFrames plus JSON/npy for non-tabular data.

    Output layout inside output_dir::

        burstlets.pkl
        network_bursts.pkl
        superbursts.pkl
        metrics.json
        diagnostics.json
        plot_signals.npy

    Pickle preserves the DataFrame index, so callers that want to ignore it on
    reload can simply call ``df.reset_index(drop=True)``.
    """

    _EVENT_FILES = {
        "burstlets": "burstlets.pkl",
        "network_bursts": "network_bursts.pkl",
        "superbursts": "superbursts.pkl",
    }

    def write(self, results: BurstResults, output_dir: Path) -> None:
        import pandas as pd

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for attr, filename in self._EVENT_FILES.items():
            df = getattr(results, attr)
            dest = output_dir / filename
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_pickle(dest)
            else:
                pd.DataFrame().to_pickle(dest)

        self._atomic_json_write(results.metrics, output_dir / "metrics.json")
        self._atomic_json_write(results.diagnostics, output_dir / "diagnostics.json")

        # plot_data contains numpy arrays — saved as a dict via allow_pickle
        np.save(output_dir / "plot_signals.npy", results.plot_data)  # type: ignore[arg-type]

    def read(self, output_dir: Path) -> BurstResults:
        import pandas as pd

        output_dir = Path(output_dir)

        dataframes = {}
        for attr, filename in self._EVENT_FILES.items():
            path = output_dir / filename
            dataframes[attr] = pd.read_pickle(path) if path.exists() else pd.DataFrame()

        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)

        with open(output_dir / "diagnostics.json") as f:
            diagnostics = json.load(f)

        plot_data = np.load(  # type: ignore[call-overload]
            output_dir / "plot_signals.npy", allow_pickle=True
        ).item()

        return BurstResults(
            burstlets=dataframes["burstlets"],
            network_bursts=dataframes["network_bursts"],
            superbursts=dataframes["superbursts"],
            metrics=metrics,
            diagnostics=diagnostics,
            plot_data=plot_data,
        )

    @staticmethod
    def _atomic_json_write(data: dict, dest: Path) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# Backwards-compatible import name for callers that used the previous writer API.
# The current on-disk format is pickle/JSON/NPY, not parquet.
ParquetBurstOutputWriter = PickleBurstOutputWriter
