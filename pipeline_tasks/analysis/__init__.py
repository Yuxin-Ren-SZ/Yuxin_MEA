from .burst_detector import (
    BurstDetectorConfig,
    BurstDetectorError,
    BurstResults,
    compute_network_bursts,
)
from .burst_output import (
    BurstOutputWriter,
    ParquetBurstOutputWriter,
    PickleBurstOutputWriter,
)

__all__ = [
    "BurstDetectorConfig",
    "BurstDetectorError",
    "BurstResults",
    "compute_network_bursts",
    "BurstOutputWriter",
    "PickleBurstOutputWriter",
    "ParquetBurstOutputWriter",
    "PlateViewerConfig",
    "WellRecord",
    "build_plate_figure",
]


def __getattr__(name: str):
    if name in {"PlateViewerConfig", "WellRecord", "build_plate_figure"}:
        from .plate_raster_synchrony import (
            PlateViewerConfig,
            WellRecord,
            build_plate_figure,
        )

        values = {
            "PlateViewerConfig": PlateViewerConfig,
            "WellRecord": WellRecord,
            "build_plate_figure": build_plate_figure,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
