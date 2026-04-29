from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline_manager import PipelineManager
from pipeline_manager.task_record import TaskStatus
from pipeline_tasks import PreprocessingTask


class _FakeRecording:
    def __init__(self, sampling_frequency: float = 20_000.0) -> None:
        self._sampling_frequency = sampling_frequency

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency


class _FakeSorting:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    def remove_empty_units(self):
        self._calls.append(("remove_empty_units",))
        return self

    def save(self, *, folder, overwrite):
        self._calls.append(("save_cleaned_sorting", folder, overwrite))


def _install_fake_runtime(
    *,
    cuda_available: bool = True,
    total_vram_gb: float = 16.0,
    sampling_frequency: float = 20_000.0,
):
    calls: list = []

    spikeinterface = types.ModuleType("spikeinterface")
    full = types.ModuleType("spikeinterface.full")
    torch = types.ModuleType("torch")

    recording = _FakeRecording(sampling_frequency=sampling_frequency)
    sorting = _FakeSorting(calls)

    def load(folder):
        calls.append(("load", folder))
        return recording

    def run_sorter(**kwargs):
        calls.append(("run_sorter", kwargs))
        return sorting

    def remove_excess_spikes(sorting_obj, recording_obj):
        calls.append(("remove_excess_spikes", sorting_obj, recording_obj))
        return sorting_obj

    class _FakeCuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def get_device_properties(_device_index):
            return types.SimpleNamespace(
                total_memory=int(total_vram_gb * (1024**3)),
            )

        @staticmethod
        def empty_cache():
            calls.append(("empty_cache",))

    full.load = load
    full.run_sorter = run_sorter
    full.remove_excess_spikes = remove_excess_spikes
    full.BaseRecording = _FakeRecording
    full.BaseSorting = _FakeSorting
    torch.cuda = _FakeCuda

    sys.modules["spikeinterface"] = spikeinterface
    sys.modules["spikeinterface.full"] = full
    sys.modules["torch"] = torch
    sys.modules.pop("pipeline_tasks.sorting", None)

    return calls, recording, sorting


def _import_task():
    module = importlib.import_module("pipeline_tasks.sorting")
    return module.SortingTask


class SortingTaskTests(unittest.TestCase):
    def tearDown(self) -> None:
        for name in [
            "pipeline_tasks.sorting",
            "spikeinterface",
            "spikeinterface.full",
            "torch",
        ]:
            sys.modules.pop(name, None)

    def test_task_exports_sorting_dependency(self):
        _install_fake_runtime()
        from pipeline_tasks import SortingTask

        self.assertEqual(SortingTask.task_name, "sorting")
        self.assertEqual(SortingTask.dependencies, ["preprocessing"])

    def test_default_params_expose_high_and_low_vram_presets(self):
        _install_fake_runtime()
        task_cls = _import_task()

        defaults = task_cls.default_params()

        self.assertEqual(defaults["min_high_vram_gb"], 14)
        self.assertEqual(defaults["high_vram_sorter_kwargs"]["batch_size_seconds"], 2.0)
        self.assertEqual(defaults["high_vram_sorter_kwargs"]["cluster_downsampling"], 20)
        self.assertIsNone(defaults["high_vram_sorter_kwargs"]["max_cluster_subset"])
        self.assertEqual(defaults["high_vram_sorter_kwargs"]["dmin"], 17)
        self.assertEqual(defaults["low_vram_sorter_kwargs"]["batch_size_seconds"], 0.5)
        self.assertEqual(defaults["low_vram_sorter_kwargs"]["cluster_downsampling"], 30)
        self.assertEqual(defaults["low_vram_sorter_kwargs"]["max_cluster_subset"], 50_000)

    def test_builds_sorter_and_cleaned_output_paths(self):
        _install_fake_runtime()
        task_cls = _import_task()

        sorter_output, cleaned_output = task_cls.build_output_paths(
            output_root="/tmp/sorted",
            recording_key="SampleA/240415/PlateX/Network/001",
            rec_name="rec0000",
            well_id="well000",
        )

        expected_base = (
            Path("/tmp/sorted")
            / "SampleA"
            / "240415"
            / "PlateX"
            / "Network"
            / "001"
            / "rec0000"
            / "well000"
        )
        self.assertEqual(sorter_output, expected_base / "sorter_output")
        self.assertEqual(cleaned_output, sorter_output)

    def test_run_loads_preprocessed_recording_and_uses_high_vram_config(self):
        calls, recording, sorting = _install_fake_runtime(
            cuda_available=True,
            total_vram_gb=16,
            sampling_frequency=20_000,
        )
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_path = task.run(
                "SampleA/240415/PlateX/Network/001",
                "rec0000/well000",
                tmp_path / "data.raw.h5",
                {
                    "preprocessing_output_root": str(tmp_path / "preprocessed"),
                    "output_root": str(tmp_path / "sorted"),
                    "docker_image": "mea-spikesorter",
                    "verbose": False,
                    "overwrite": True,
                    "high_vram_sorter_kwargs": {
                        "batch_size_seconds": 3.0,
                        "clear_cache": False,
                        "invert_sign": False,
                        "cluster_downsampling": 11,
                        "max_cluster_subset": 12_345,
                        "nblocks": 2,
                        "dmin": 21,
                        "do_correction": True,
                    },
                },
            )

        preprocessed_path = (
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
        sorter_output = (
            tmp_path
            / "sorted"
            / "SampleA"
            / "240415"
            / "PlateX"
            / "Network"
            / "001"
            / "rec0000"
            / "well000"
            / "sorter_output"
        )
        cleaned_output = sorter_output

        run_sorter_call = next(call for call in calls if call[0] == "run_sorter")
        kwargs = run_sorter_call[1]

        self.assertEqual(output_path, cleaned_output)
        self.assertIn(("load", preprocessed_path), calls)
        self.assertIn(("empty_cache",), calls)
        self.assertEqual(kwargs["sorter_name"], "kilosort4")
        self.assertIs(kwargs["recording"], recording)
        self.assertEqual(kwargs["folder"], str(sorter_output))
        self.assertEqual(kwargs["docker_image"], "mea-spikesorter")
        self.assertFalse(kwargs["verbose"])
        self.assertFalse(kwargs["delete_output_folder"])
        self.assertTrue(kwargs["remove_existing_folder"])
        self.assertEqual(kwargs["batch_size"], 60_000)
        self.assertNotIn("batch_size_seconds", kwargs)
        self.assertFalse(kwargs["clear_cache"])
        self.assertFalse(kwargs["invert_sign"])
        self.assertEqual(kwargs["cluster_downsampling"], 11)
        self.assertEqual(kwargs["max_cluster_subset"], 12_345)
        self.assertEqual(kwargs["nblocks"], 2)
        self.assertEqual(kwargs["dmin"], 21)
        self.assertTrue(kwargs["do_correction"])
        self.assertEqual(
            calls[-3:],
            [
                ("remove_excess_spikes", sorting, recording),
                ("remove_empty_units",),
                ("save_cleaned_sorting", cleaned_output, True),
            ],
        )

    def test_run_uses_low_vram_config_when_cuda_is_unavailable(self):
        calls, _recording, _sorting = _install_fake_runtime(
            cuda_available=False,
            sampling_frequency=20_000,
        )
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            task.run(
                "SampleA/240415/PlateX/Network/001",
                "rec0000/well000",
                tmp_path := Path(tmp) / "data.raw.h5",
                {
                    "preprocessing_output_root": str(Path(tmp) / "preprocessed"),
                    "output_root": str(Path(tmp) / "sorted"),
                    "low_vram_sorter_kwargs": {
                        "batch_size_seconds": 0.25,
                        "clear_cache": False,
                        "invert_sign": False,
                        "cluster_downsampling": 17,
                        "max_cluster_subset": 111,
                        "nblocks": 3,
                        "do_correction": True,
                    },
                },
            )

        self.assertEqual(tmp_path.name, "data.raw.h5")
        run_sorter_call = next(call for call in calls if call[0] == "run_sorter")
        kwargs = run_sorter_call[1]
        self.assertEqual(kwargs["batch_size"], 5_000)
        self.assertNotIn("batch_size_seconds", kwargs)
        self.assertFalse(kwargs["clear_cache"])
        self.assertFalse(kwargs["invert_sign"])
        self.assertEqual(kwargs["cluster_downsampling"], 17)
        self.assertEqual(kwargs["max_cluster_subset"], 111)
        self.assertEqual(kwargs["nblocks"], 3)
        self.assertTrue(kwargs["do_correction"])
        self.assertNotIn("dmin", kwargs)

    def test_explicit_preset_batch_size_overrides_batch_size_seconds(self):
        calls, _recording, _sorting = _install_fake_runtime(
            cuda_available=True,
            total_vram_gb=16,
            sampling_frequency=20_000,
        )
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task.run(
                "SampleA/240415/PlateX/Network/001",
                "rec0000/well000",
                tmp_path / "data.raw.h5",
                {
                    "preprocessing_output_root": str(tmp_path / "preprocessed"),
                    "output_root": str(tmp_path / "sorted"),
                    "high_vram_sorter_kwargs": {
                        "batch_size": 777,
                        "batch_size_seconds": 9.0,
                        "cluster_downsampling": 13,
                    },
                },
            )

        run_sorter_call = next(call for call in calls if call[0] == "run_sorter")
        kwargs = run_sorter_call[1]
        self.assertEqual(kwargs["batch_size"], 777)
        self.assertNotIn("batch_size_seconds", kwargs)
        self.assertEqual(kwargs["cluster_downsampling"], 13)

    def test_sorter_kwargs_override_vram_preset(self):
        calls, _recording, _sorting = _install_fake_runtime(
            cuda_available=True,
            total_vram_gb=16,
        )
        task_cls = _import_task()
        task = task_cls()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task.run(
                "SampleA/240415/PlateX/Network/001",
                "rec0000/well000",
                tmp_path / "data.raw.h5",
                {
                    "preprocessing_output_root": str(tmp_path / "preprocessed"),
                    "output_root": str(tmp_path / "sorted"),
                    "sorter_kwargs": {
                        "batch_size": 123,
                        "cluster_downsampling": 7,
                    },
                },
            )

        run_sorter_call = next(call for call in calls if call[0] == "run_sorter")
        kwargs = run_sorter_call[1]
        self.assertEqual(kwargs["batch_size"], 123)
        self.assertEqual(kwargs["cluster_downsampling"], 7)

    def test_pipeline_manager_schedules_sorting_after_preprocessing(self):
        _install_fake_runtime()
        task_cls = _import_task()

        with TemporaryDirectory() as tmp:
            manager = PipelineManager(Path(tmp))
            manager.register_task(PreprocessingTask)
            manager.register_task(task_cls)
            manager.add_well("SampleA/240415/PlateX/Network/001", "rec0000/well000")

            first = manager.get_next_task(n=1)
            self.assertEqual(first[0].task_name, "preprocessing")

            manager.update_status(
                first[0],
                TaskStatus.COMPLETE,
                output_path=Path(tmp) / "preprocessed.zarr",
            )

            second = manager.get_next_task(n=1)
            self.assertEqual(second[0].task_name, "sorting")
