# Yuxin MEA

SadeghLab 的 MEA 记录、流水线与分析库。

[English](README.md) | **中文**

---

## 项目简介

面向 MaxWell MaxTwo HD-MEA 板的端到端分析栈。流水线流程：

```
预处理（带通滤波 + 共参考）
   → Kilosort4 spike sorting
   → （可选）SpikeInterface auto-merge
   → analyzer（波形、单元位置、质量指标）
   → 基于指标阈值的自动 curation
   → 网络 burst 检测
```

项目附带两个 burst 检测器：经典阈值检测器（`burst_detection`），以及一个**迭代式 Fisher-LDA 检测器**（`iterative_burst_detection`）—— 后者在多轮迭代中精修复合发放率信号，并可选地通过 GMM 对事件聚类以识别 super-burst 结构。检测结果可输出到交互式 Plotly **plate viewer**（每条记录一份 HTML），也可在多页 **Dash dashboard** 中查看数据集、流水线状态与 burst 诊断。

## 仓库结构

```
src/yuxin_mea/          可安装的命名空间（`pip install -e .`）
  config/               JSON 配置加载器 + ParamSpec 模式
  dataset/              原始 MEA 扫描、mxassay 元数据解析、recording/well 缓存
  pipeline/             逐 well 的任务 DAG + JSON 状态缓存
  tasks/                preprocessing、sorting、auto_merge、analyzer、
                        auto_curation、burst_detection、
                        iterative_burst_detection、plate_viewer
  analysis/             算法代码（burst 检测器、burst_diagnostic 库、
                        curation_summary、synthetic_validation）
  dashboard/            多页 Dash 应用（Home / Recordings / Pipeline /
                        Burst Diagnostic / Settings）
config/                 示例流水线配置 JSON
notebooks/v2/           重构后的标准流水线 notebook
notebooks/              原始 notebook（保留作参考）
tests/                  pytest 测试套件（约 200 个测试）
scripts/                辅助脚本（如 strip_notebook_outputs.py）
doc/architecture.md     早期架构说明（命名部分已过时；以 AGENTS.md 为准）
AGENTS.md               详尽的模块/符号索引 —— 权威参考
```

## 安装

科学依赖（torch+CUDA、Kilosort、SpikeInterface、Dash、Plotly、scikit-learn、h5py、zarr 等）由 conda 管理 —— `pyproject.toml` 故意保持 `dependencies = []`，避免 `pip install -e .` 重新从 PyPI 解析 conda 已管理的包。

```bash
conda env create -f environment.yml
conda activate yuxin_mea
pip install -e .
```

预设的 torch 版本针对 **CUDA 12.8**。如需仅 CPU 或其它 CUDA 版本，请在创建环境前修改 `environment.yml` 中的 `pip:` 段。

## 快速开始

三种入口，按使用者类型选择。

### 1. Dashboard（面向非技术用户）

```bash
yuxin-mea-dashboard --config config/pipeline_config.example.json
```

随后打开 `http://127.0.0.1:8050`。页面：

- **Home** —— 配置路径、数据根目录、缓存条目数
- **Recordings** —— 来自 `experiment_cache.json` 的可排序/筛选表格
- **Pipeline** —— `(recording × well) × task` 状态矩阵
- **Burst Diagnostic** —— 批量运行迭代式 burst 检测器并浏览诊断图
- **Settings** —— 基于模式（schema）的配置编辑器，按每个任务的 `ParamSpec` 校验

若配置文件尚不存在，dashboard 仍会启动并显示提示横幅 —— 可通过 Settings 页面初始化配置。

### 2. Notebook（v2 版）

```
notebooks/v2/00_full_pipeline.ipynb   —— 端到端运行
notebooks/v2/01_si_preprocessing.ipynb
notebooks/v2/01_plate_viewer.ipynb
notebooks/v2/03_auto_merge.ipynb
notebooks/v2/04_analyzer.ipynb
notebooks/v2/05_auto_curation.ipynb
notebooks/v2/06_iterative_burst_detector_synthetic_validation.ipynb
```

在 `yuxin_mea` 这个 conda 环境下用 JupyterLab 打开。原始 `notebooks/` 仅作参考，不再随重构更新。

### 3. 库调用（面向开发者）

```python
from yuxin_mea.dataset import DatasetManager
from yuxin_mea.pipeline import PipelineManager
from yuxin_mea.config import ConfigManager
from yuxin_mea.analysis.iterative_burst_detector import compute_iterative_bursts
```

完整的公开 API（按模块逐符号列出）见 `AGENTS.md`。

## 配置

所有配置集中在一份 JSON 文件中：

```json
{
  "global": {
    "data_root":     "/path/to/raw/recordings",
    "analysis_root": "./data/analysis",
    "figure_root":   "./output/figures"
  },
  "tasks": {
    "preprocessing":  { "bandpass_freq_min": 300, "bandpass_freq_max": 3000, ... },
    "sorting":        { "sorter": "kilosort4", ... },
    "auto_merge":     { "enabled": false, ... },
    "analyzer":       { ... },
    "auto_curation":  { "presence_ratio_min": 0.75, ... },
    "burst_detection":           { ... },
    "iterative_burst_detection": { ... },
    "plate_viewer":              { ... }
  }
}
```

可通过 dashboard 的 **Settings** 标签页编辑（每个字段都按对应任务的 `ParamSpec` 校验），也可参照 `config/pipeline_config.example.json` 手工编辑。每个任务的 `params_schema()` 与 `default_params()` 必须键完全一致，由 `tests/test_params_schema.py` 强制保证。

## 流水线流程

```
raw → preprocessing → sorting → (auto_merge) → analyzer → auto_curation → burst_detection
                                                                       ↘ iterative_burst_detection → plate_viewer
```

设计要点：

- `DatasetManager` 与 `PipelineManager` **彼此独立**，互不 import。调用方通过 `recording_key + "/" + well_id` 这一复合 `pipeline_key` 把两者关联起来。
- 每个任务进入 `running` 状态时会快照当前配置。`is_task_complete()` 仅当状态为 `complete` **且** 快照与当前配置相等时才为真 —— 因此修改任一任务的配置会自动让该任务及其所有下游失效。
- 所有缓存写入（`experiment_cache.json`、`pipeline_cache.json`、任务输出 JSON）均为原子操作（tempfile + `os.replace`），可在 NAS 上承受中断写入。

## 测试

```bash
conda run -n yuxin_mea pytest
```

约 200 个测试，覆盖：dataset 缓存与扫描、每个任务的 params schema 与输出路径、burst 检测器的精度与参考实现等价性、配置构建器的表单渲染与嵌套字典重建、curation summary、合成尖峰序列验证，以及 `notebooks/v2` 的执行。

## 延伸阅读

- **`AGENTS.md`** —— 详尽的模块/符号索引，权威参考。读代码请从这里开始。
- **`doc/architecture.md`** —— 早期架构说明。命名部分已过时（重构前的 `stages`/`stage_record` 对应当前的 `tasks`/`task_record`），但依赖传递性与配置失效机制仍然适用。
- **`TODO.md`** —— 当前进行中的工作。

## 联系方式

SadeghLab 内部项目 —— 如需访问原始数据和基础设施，请联系维护者。
