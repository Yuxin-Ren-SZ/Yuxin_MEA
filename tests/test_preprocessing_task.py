from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class _FakeDType:
    def __init__(self, kind: str) -> None:
        self.kind = kind


class _FakeRecording:
    def __init__(self, calls: list, dtype_kind: str = "u") -> None:
        self._calls = calls
        self._dtype_kind = dtype_kind

    def get_dtype(self) -> _FakeDType:
        return _FakeDType(self._dtype_kind)

    def annotate(self, **kwargs) -> None:
        self._calls.append(("annotate", kwargs))

    def save(self, **kwargs) -> None:
        self._calls.append(("save", kwargs))


def _install_fake_spikeinterface(
    fail_local_reference: bool = False,
    fail_global_reference: bool = False,
):
    calls: list = []

    spikeinterface = types.ModuleType("spikeinterface")
    full = types.ModuleType("spikeinterface.full")
    preprocessing = types.ModuleType("spikeinterface.preprocessing")

    def read_maxwell(path, *, stream_id, rec_name):
        calls.append(("read_maxwell", path, stream_id, rec_name))
        return _FakeRecording(calls, dtype_kind="u")

    def unsigned_to_signed(recording):
        calls.append(("unsigned_to_signed",))
        recording._dtype_kind = "i"
        return recording

    def bandpass_filter(recording, *, freq_min, freq_max):
        calls.append(("bandpass_filter", freq_min, freq_max))
        return recording

    def common_reference(recording, *, reference, operator, local_radius=None):
        calls.append(("common_reference", reference, operator, local_radius))
        if reference == "local" and fail_local_reference:
            raise RuntimeError("missing locations")
        if reference == "global" and fail_global_reference:
            raise RuntimeError("global reference failed")
        return recording

    def astype(recording, dtype):
        calls.append(("astype", dtype))
        recording._dtype_kind = "f"
        return recording

    full.read_maxwell = read_maxwell
    preprocessing.unsigned_to_signed = unsigned_to_signed
    preprocessing.bandpass_filter = bandpass_filter
    preprocessing.common_reference = common_reference
    preprocessing.astype = astype

    sys.modules["spikeinterface"] = spikeinterface
    sys.modules["spikeinterface.full"] = full
    sys.modules["spikeinterface.preprocessing"] = preprocessing
    sys.modules.pop("pipeline_tasks.preprocessing", None)

    return calls


def _import_task():
    module = importlib.import_module("pipeline_tasks.preprocessing")
    return module.PreprocessingTask


class PreprocessingTaskTests(unittest.TestCase):
    def tearDown(self) -> None:
        for name in [
            "pipeline_tasks.preprocessing",
            "pipeline_tasks",
            "spikeinterface",
            "spikeinterface.full",
            "spikeinterface.preprocessing",
        ]:
            sys.modules.pop(name, None)

    def test_splits_compound_well_id(self):
        _install_fake_spikeinterface()
        task_cls = _import_task()

        self.assertEqual(
            task_cls.split_compound_well_id("rec0000/well000"),
            ("rec0000", "well000"),
        )

    def test_invalid_compound_well_id_raises_clear_error(self):
        _install_fake_spikeinterface()
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "compound well_id"):
                task.run(
                    "SampleA/240415/PlateX/Network/001",
                    "well000",
                    tmp_path / "data.raw.h5",
                    {},
                )

    def test_builds_zarr_output_path(self):
        _install_fake_spikeinterface()
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_path = task.build_output_path(
                output_root=tmp_path,
                recording_key="SampleA/240415/PlateX/Network/001",
                rec_name="rec0000",
                well_id="well000",
            )

            self.assertEqual(
                output_path,
                tmp_path
                / "SampleA"
                / "240415"
                / "PlateX"
                / "Network"
                / "001"
                / "rec0000"
                / "well000"
                / "preprocessed.zarr",
            )

    def test_run_applies_preprocessing_in_order(self):
        calls = _install_fake_spikeinterface()
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.raw.h5"
            output_path = task.run(
                "SampleA/240415/PlateX/Network/001",
                "rec0000/well000",
                data_path,
                {
                    "output_root": str(tmp_path / "preprocessed"),
                    "n_jobs": 2,
                    "chunk_duration": "500ms",
                    "progress_bar": False,
                    "overwrite": True,
                },
            )

            expected_output = (
                tmp_path
                / "preprocessed"
                / "SampleA"
                / "240415"
                / "PlateX"
                / "Network"
                / "001"
                / "rec0000"
                / "well000"
                / "preprocessed.zarr"
            )
            self.assertEqual(output_path, expected_output)
            self.assertEqual(
                calls,
                [
                    ("read_maxwell", str(data_path), "well000", "rec0000"),
                    ("unsigned_to_signed",),
                    ("bandpass_filter", 300, 3000),
                    ("common_reference", "local", "median", (0, 250)),
                    ("astype", "float32"),
                    (
                        "save",
                        {
                            "folder": expected_output,
                            "format": "zarr",
                            "overwrite": True,
                            "n_jobs": 2,
                            "chunk_duration": "500ms",
                            "progress_bar": False,
                        },
                    ),
                ],
            )

    def test_local_reference_falls_back_to_global(self):
        calls = _install_fake_spikeinterface(fail_local_reference=True)
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task.run(
                "SampleA/240415/PlateX/Network/001",
                "rec0000/well000",
                tmp_path / "data.raw.h5",
                {"output_root": str(tmp_path / "preprocessed")},
            )

        self.assertIn(("common_reference", "local", "median", (0, 250)), calls)
        self.assertIn(("common_reference", "global", "median", None), calls)

    def test_global_reference_does_not_try_local_reference(self):
        calls = _install_fake_spikeinterface()
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task.run(
                "SampleA/240415/PlateX/Network/001",
                "rec0000/well000",
                tmp_path / "data.raw.h5",
                {
                    "output_root": str(tmp_path / "preprocessed"),
                    "reference": "global",
                },
            )

        self.assertIn(("common_reference", "global", "median", None), calls)
        self.assertNotIn(("common_reference", "local", "median", (0, 250)), calls)

    def test_local_and_global_reference_failures_raise_runtime_error(self):
        _install_fake_spikeinterface(
            fail_local_reference=True,
            fail_global_reference=True,
        )
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaisesRegex(
                RuntimeError,
                "Common reference failed",
            ):
                task.run(
                    "SampleA/240415/PlateX/Network/001",
                    "rec0000/well000",
                    tmp_path / "data.raw.h5",
                    {"output_root": str(tmp_path / "preprocessed")},
                )

    def test_global_reference_failure_raises_runtime_error(self):
        _install_fake_spikeinterface(fail_global_reference=True)
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaisesRegex(
                RuntimeError,
                "Global common reference failed",
            ):
                task.run(
                    "SampleA/240415/PlateX/Network/001",
                    "rec0000/well000",
                    tmp_path / "data.raw.h5",
                    {
                        "output_root": str(tmp_path / "preprocessed"),
                        "reference": "global",
                    },
                )

    def test_invalid_reference_raises_value_error(self):
        _install_fake_spikeinterface()
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Invalid reference type"):
                task.run(
                    "SampleA/240415/PlateX/Network/001",
                    "rec0000/well000",
                    tmp_path / "data.raw.h5",
                    {
                        "output_root": str(tmp_path / "preprocessed"),
                        "reference": "not-a-reference",
                    },
                )


if __name__ == "__main__":
    unittest.main()
