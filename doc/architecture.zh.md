# Yuxin_MEA 项目架构

> 本文档对应 4 阶段重构（commit `8631fde`）之后的代码结构。
> 英文版：[`architecture.md`](architecture.md)。

## 项目定位

这是一个用于 **多电极阵列（MEA, Multi-Electrode Array）神经记录数据分析** 的实验室软件。完整流程：从 MaxWell 设备产生的原始 `.h5` 文件 → 预处理 → spike sorting → 单元筛选 → burst 检测 → 可视化。

---

## 顶层结构

```
src/yuxin_mea/          ← 可安装的 Python 包（pip install -e .）
├── dataset/            ← "我有哪些录音?"
├── pipeline/           ← "怎么调度任务?"
├── tasks/              ← "做什么分析步骤?"
├── config/             ← "用什么参数?"
├── analysis/           ← "底层算法和图表函数"
├── dashboard/          ← "浏览器 UI"
└── cli/                ← "命令行入口"

notebooks/v2/           ← 用新 API 写的可执行教程
tests/                  ← 236 个测试（pytest）
config/                 ← 示例 JSON 配置
AGENTS.md               ← 给未来 AI 助手看的"代码地图"
pyproject.toml          ← 包定义
```

---

## 核心架构理念：三个独立但可组合的层

### 第 1 层：`dataset/` — 数据发现层

**职责**：扫描磁盘上的 MaxWell 录音文件，缓存元数据。

```
dataset/
├── manager.py        DatasetManager  ← 主入口
├── entries.py        RecordingEntry, WellEntry  ← 数据结构
├── metadata.py       解析 mxassay.metadata 文件
├── cache.py          JsonCacheStore  ← experiment_cache.json 读写
└── _mxassay_decoder.py  Qt 序列化格式的解码器（实现细节）
```

**关键不变量**：
- `RecordingEntry` 是 **冻结的 dataclass**（`frozen=True`）—— 一旦从磁盘扫描出来就不能改字段
- `cache_key` = `"sample_id/date/plate_id/scan_type/run_id"` —— 整个系统的"主键"
- 复合 well_id = `"rec0000/well000"` —— h5 文件内部的双层结构

**设计要点**：
- **为什么用冻结的 dataclass？** 录音元数据（采样率、通道映射）一旦确定就不该改变。`frozen=True` 在 Python 层面把这条约束写死，让任何"不小心修改了字段"的 bug 在运行时立即崩溃，而不是默默污染下游分析结果。
- **`cache_key` 这条字符串是整个系统的连接组织**。Dataset、Pipeline、Dashboard 三层之间不互相导入对方的对象 —— 它们都只交换 `cache_key`。这是教科书式的"通过 ID 解耦"模式。

---

### 第 2 层：`pipeline/` — 任务调度层

**职责**：跟踪 "哪个 well 跑完了哪个任务"，提供任务依赖图调度。

```
pipeline/
├── manager.py            PipelineManager  ← 调度器
├── base_task.py          BaseAnalysisTask  ← 任务基类（每个 Task 继承）
├── task_record.py        TaskStatus（not_run/running/complete/failed）+ TaskRecord
├── pipeline_entry.py     PipelineEntry：每个 (recording, well) 对的状态容器
├── work_item.py          WorkItem：可调度的"待办事项"
├── cache.py              JsonPipelineCacheStore  ← pipeline_cache.json
├── config_provider.py    BaseConfigProvider 接口
└── well_metadata.py      未来扩展用的 well 元数据提供者接口
```

**关键不变量**（来自 `AGENTS.md`）：
- `PipelineManager` 与 `DatasetManager` **完全独立** —— 互不 import
- `TaskRecord.config` 在状态转为 `running` 时 **快照保存** ← "可复现性"的关键
- `is_task_complete()` 只有在"状态 complete + 当前 config 与快照一致"时才返回 True
- 启动时所有非 complete 的任务会被重置为 `not_run`

**设计要点**：
- **Config snapshot 模式**：当任务开始运行时，当时的参数被冻结写入 `TaskRecord.config`。下次用户改了配置文件后再问"这个 well 跑完了吗?"，系统会比对当前配置 vs 快照配置；如果不一致，即使状态是 complete 也算"过期"。这让"重跑还是跳过"的决定变成自动的，不需要人记得自己改了什么。
- **任务依赖在类属性里声明**（`dependencies: list[str]`），不在调度器里。这意味着 `PipelineManager` 完全不知道"具体业务" —— 它只看类属性。新增一个 task 不需要改 `PipelineManager` 一行代码。

---

### 第 3 层：`tasks/` — 业务任务层

每个 task 就是 `BaseAnalysisTask` 的一个子类，包含 4 个核心方法：

```python
class SortingTask(BaseAnalysisTask):
    task_name = "sorting"                    # ← 唯一名字
    dependencies = ["preprocessing"]         # ← 依赖 DAG

    @classmethod
    def default_params(cls) -> dict:         # ← 默认参数
        return {"sorter": "kilosort4", ...}

    @classmethod
    def params_schema(cls) -> dict[str, ParamSpec]:   # ← Phase 3 新增！表单 schema
        return {"sorter": ParamSpec("str", "kilosort4",
                                    choices=["kilosort4", ...]), ...}

    def run(self, recording_key, well_id, data_path, params) -> Path:
        ...                                  # ← 实际工作
```

当前的 7 个 task（按 DAG 顺序）：

```
preprocessing → sorting → auto_merge → analyzer → auto_curation → burst_detection
                                                                ↘
                                                                 ml_burst_detection
```

（Phase 5 从 pipeline 中移除了 `plate_viewer`：可视化不是 processing。
它现在是 Dashboard 的一个页面 —— 见第 6 层。）

---

### 第 4 层：`config/` — 配置层

```
config/
├── manager.py    ConfigManager  ← JSON 配置的读写
├── schema.py     ParamSpec dataclass + validate_value()  ← Phase 3 新增
└── globals.py    GLOBALS_SCHEMA：data_root / analysis_root / figure_root
```

**配置文件结构**（`pipeline_config.json`）：

```json
{
  "global": {
    "data_root": "/path/to/raw",
    "analysis_root": "/path/to/analysis",
    "figure_root": "/path/to/figures"
  },
  "tasks": {
    "preprocessing": {"bandpass_freq_min": 300, ...},
    "sorting": {"sorter": "kilosort4", ...}
  }
}
```

**设计要点**：
- **`default_params()` 与 `params_schema()` 必须一一对应**（key 集合相等）—— 这条不变量由 `tests/test_params_schema.py` 强制。这是"单一真理源"的反例的反例：我们故意有两个来源，但用测试保证它们同步。为什么？因为 `default_params()` 返回值，`params_schema()` 返回**类型 + 校验规则 + UI 提示**。把它们分开让 task 类不必依赖 `ParamSpec`，但又能被 Dashboard 自动渲染成表单。
- **Dashboard 不直接 `import DatasetManager`** —— 它用 `JsonCacheStore` 直接读 JSON。为什么？因为 `DatasetManager.__init__` 会 **扫描磁盘并 mutate cache**。Dashboard 是只读的，一旦构造 Manager 就破坏了"只读"承诺。

---

### 第 5 层：`analysis/` — 纯算法层

```
analysis/
├── burst_detector.py              compute_network_bursts（传统检测）
├── ml_burst_detector.py           compute_ml_bursts（HMM + HDBSCAN 检测）
├── ml_burst_hmm.py / ml_burst_features.py / ml_burst_cluster.py  ML 检测器构件
├── burst_common.py                共享底层工具（脉冲矩阵、多尺度 FF、合并）
├── burst_output.py                把 BurstResults 序列化为 pickle/parquet
├── plate_raster_synchrony.py      多孔板可视化的 Plotly 图表
├── burst_diagnostic.py            ← Phase 2b：诊断仪表板用的图函数 + run_batch + 缓存
├── curation_summary.py            ← Phase 4：质控结果聚合
├── synthetic_validation.py        ← Phase 4：合成 spike train + GT 评分
└── （load_plate_data 是 plate_raster_synchrony.py 里的公共函数 — Phase 5）
```

**关键约束**：`analysis/` 中的代码 **不能 import Dash、Pipeline、Dataset** —— 只能用 numpy / scipy / sklearn / plotly。这让算法可以单独在 Jupyter / 测试中使用。

---

### 第 6 层：`dashboard/` — 浏览器 UI

```
dashboard/
├── cli.py                  ← yuxin-mea-dashboard 命令行入口
├── app.py                  ← build_app(config_path) -> Dash
├── data.py                 ← 只读地加载缓存文件
├── components/
│   ├── layout.py           ← 顶部导航条 + page_container
│   └── form_builder.py     ← 把 ParamSpec 渲染成 Dash widget
└── pages/                  ← Dash 4.x 多页面（pages_folder 自动发现）
    ├── home.py             /                  (order=0)
    ├── recordings.py       /recordings        (order=1)
    ├── pipeline.py         /pipeline          (order=2)
    ├── plate_viewer.py     /plate-viewer      (order=3) ← Phase 5：原来是 task，现在是页面
    ├── burst_diagnostic.py /burst-diagnostic  (order=4)
    └── settings.py         /settings          (order=10) ← 配置编辑器
```

**使用方法**：

```bash
yuxin-mea-dashboard --config pipeline_config.json
# 打开 http://127.0.0.1:8050
```

如果配置文件不存在，仪表板会进入 **"config-only mode"** —— 数据页面显示"还没有配置"横幅，但 Settings 页面照常可用，第一次点 Save 时会创建文件。

**设计要点**：
- **Pattern-matched IDs `{"form": "...", "field": "..."}`** 是 Dashboard 配置编辑器能用一组 callback 处理 8 个不同 task 表单的关键。Dash 的 `ALL` 通配符在 callback 输入中匹配所有相同 pattern 的组件，比写 8 个几乎相同的 callback 干净得多。
- **`app.server.config["YUXIN_MEA"]`** 是页面之间共享状态的方式。Flask 的 config 字典在所有请求间共享，而 Dash 的 `dcc.Store` 是 per-session 的。需要"整个 app 共用一份配置"时，stash 到 server config 是惯用做法。

---

## 数据流：一次完整分析的全链路

```
原始 .h5 文件
   │
   ↓  (DatasetManager 扫描)
experiment_cache.json
   │
   ↓  (用户在 Dashboard Settings 填配置)
pipeline_config.json
   │
   ↓  (PipelineManager.add_well + register_task)
pipeline_cache.json   ← 每个 (recording, well) 的任务状态
   │
   ↓  (调用 task.run(...) 循环)
<analysis_root>/preprocessed_data/<rec>/<well>/preprocessed.zarr
<analysis_root>/spikesorted_data/<rec>/<well>/...
<analysis_root>/auto_merge_data/<rec>/<well>/...
<analysis_root>/analyzer_data/<rec>/<well>/...
<analysis_root>/curation_data/<rec>/<well>/quality_metrics.pkl + curated_spike_times.npy
<analysis_root>/burst_detection_data/<rec>/<well>/...
   │
   ↓  (Dashboard 只读地展示这些)
浏览器中的 Recordings / Pipeline / Burst diagnostic 页面
```

---

## 一般用户怎么用

### 路径 A：完全用浏览器（推荐给非技术用户）

```bash
# 第一次：创建配置文件
yuxin-mea-dashboard --config pipeline_config.json
# → 浏览器打开 → Settings → 填 data_root/analysis_root/figure_root + 各 task 参数 → Save

# 之后：监控管线
yuxin-mea-dashboard --config pipeline_config.json
# → Recordings 页面看有哪些录音
# → Pipeline 页面看每个 task 跑到哪了（颜色编码：绿/蓝/红/灰）
# → Burst diagnostic 页面交互式探索 burst 检测结果
```

### 路径 B：用 Notebook（推荐给做研究的）

`notebooks/v2/` 下的 8 个 notebook 是新的"教程兼工作流"：

```
00_full_pipeline.ipynb       ← 跑完整流水线（所有 task）
01_si_preprocessing.ipynb    ← 单步：预处理
03_auto_merge.ipynb          ← 单步：自动合并
04_analyzer.ipynb            ← 单步：分析器
05_auto_curation.ipynb       ← 单步：自动筛选 + 用 curation_summary 做汇总
06_..._synthetic_validation  ← 用 synthetic_validation 验证 burst 检测器
01_plate_viewer / 02_generate_default_params  ← 重定向到 Dashboard
```

每个 notebook 的结构都遵循同一个模板（`03_auto_merge.ipynb` 是标准模板）：

1. Imports（统一从 `yuxin_mea.*` 导入）
2. 加载 `ConfigManager`（如果配置不存在，写模板并停止）
3. 扫描录音
4. 注册任务 + 添加 well
5. 状态总览表
6. 主循环：`get_next_task` → `task.run()` → `update_status`
7. 最终状态报告

---

## 怎么扩展：4 个常见场景

### 场景 1：添加一个新的 Task

最常见的扩展。完整步骤：

```python
# 1. 在 src/yuxin_mea/tasks/my_new_task.py 中：
from yuxin_mea.config import ParamSpec
from yuxin_mea.pipeline import BaseAnalysisTask

class MyNewTask(BaseAnalysisTask):
    task_name = "my_new_task"
    dependencies = ["analyzer"]               # ← 上游 task 名

    @classmethod
    def default_params(cls):
        return {"my_param": 42, "output_root": "./my_data"}

    @classmethod
    def params_schema(cls):                   # ← Dashboard 会自动渲染
        return {
            "my_param": ParamSpec("int", 42, "什么作用", min=0),
            "output_root": ParamSpec("path", "./my_data", "输出位置"),
        }

    def run(self, recording_key, well_id, data_path, params):
        p = self.resolve_params(params)       # ← 合并默认值
        # ... 实际逻辑 ...
        return output_path                    # ← 必须返回输出路径
```

```python
# 2. 在 src/yuxin_mea/tasks/__init__.py 加一行：
from .my_new_task import MyNewTask

# 3. 在 Dashboard 的 settings.py 中把它加入 _TASK_CLASSES 列表
```

**测试自动覆盖**：`test_params_schema.py` 是参数化的，新 task 会自动被"schema vs default_params key 一致性"检查覆盖。

**设计要点**：
- **几乎不需要碰 `pipeline/manager.py`**。这就是基类 + 类属性声明依赖的好处 —— 调度器看你的 `task_name` 和 `dependencies` 就够了。所有"新增 task"需要做的事都是局部的（一个文件 + 两个 import 行）。
- **`resolve_params()` 在 run() 开头调用** 是约定俗成的做法。它把 JSON 中的用户值与 `default_params()` 合并，JSON 值胜出。这让用户可以只在配置文件里写"我想改的"参数，其他用默认值。

### 场景 2：添加一个新的算法函数

放进 `src/yuxin_mea/analysis/`：

```python
# src/yuxin_mea/analysis/my_algorithm.py
import numpy as np

def detect_my_thing(spike_times: dict[str, np.ndarray]) -> dict:
    """纯算法，没有 Dash / Pipeline 依赖。"""
    ...
```

然后：
- 在 `notebooks/v2/` 中 import 它演示使用
- 如果有合理的"输入输出契约"，写 `tests/test_my_algorithm.py`
- 如果它要进 Pipeline，再包一个 Task 类（见场景 1）

**关键纪律**：`analysis/` 模块 **绝不 import Dash / Pipeline / Dataset / Config**。这条约束让算法可以在脱离整个仪表板栈的环境中使用（比如纯 Jupyter 实验）。

### 场景 3：添加一个新的 Dashboard 页面

```python
# src/yuxin_mea/dashboard/pages/my_page.py
import dash
from dash import callback, Input, Output, html

dash.register_page(__name__, path="/my-page", name="My Page", order=5)

layout = html.Div([html.H2("My Page"), html.Div(id="my-content")])

@callback(Output("my-content", "children"), Input("my-content", "id"))
def _render(_id):
    return "Hello!"
```

Dash 4.x 的 `pages_folder="pages"` 自动发现机制会捡起来 —— 顶部导航条自动多一个链接。

### 场景 4：扩展配置 schema（增加一个 global）

```python
# src/yuxin_mea/config/globals.py 加一行：
GLOBALS_SCHEMA = {
    "data_root": ParamSpec(...),
    "analysis_root": ParamSpec(...),
    "figure_root": ParamSpec(...),
    "my_new_global": ParamSpec("path", "", "什么用"),   # ← 这里
}
```

Dashboard Settings 页面的 Globals 标签会自动多一个字段。其他代码 `cm.get_global("my_new_global")` 立刻可用。

---

## 测试纪律

236 个测试。分四类：

1. **业务测试**（旧的） —— 验证每个 manager / task 的行为
2. **架构不变量测试**（重构期间新增） —— 守护跨阶段约束：
   - `test_params_schema.py`：schema/default 一致性
   - `test_notebooks_v2.py`：notebook 不含旧包名
   - `test_config_builder.py`：嵌套 dict 表单不会丢字段
3. **算法测试** —— `synthetic_validation.py` 提供合成数据 + 评分函数，让 burst 检测器有 ground-truth 测试
4. **Dashboard smoke 测试** —— `build_app(missing_path)` 不崩、所有页面能被注册

**重要约束**：**没有测试调用 `app.run()`**。`app.run()` 会阻塞整个测试进程。所有测试只构造 app，断言属性，然后退出。

---

## 找文档的地方

- **`AGENTS.md`**（473 行）：给未来的 AI 助手看的代码地图。每个模块、每个公开符号一行说明。这是项目的"单一真理源"对外参考。
- **每个 task 类的 docstring** 简要说明算法目的。
- **`config/pipeline_config.example.json`**：可直接复制使用的配置示例，跟当前 schema 完全对应。
- **`~/.claude/plans/please-review-the-codebase-goofy-shannon.md`**：4 阶段重构的完整设计文档（含每个 commit 的决策依据）。

---

## 总结：架构是怎么框出来的

整个系统的 **核心抽象** 是三组**通过字符串 ID 连接的独立子系统**：

```
DatasetManager  ←(cache_key)→  PipelineManager  ←(task_name)→  TaskRegistry
       ↑                              ↑                              ↑
       │                              │                              │
       └──── ConfigManager ───────────┴───── Dashboard (read-only) ──┘
```

- **没有 God object**：没有"主控制器"一类的东西在管理一切
- **没有循环依赖**：依赖关系是单向 DAG（`analysis` → 谁都不依赖；`tasks` → `pipeline + config + analysis`；`dashboard` → 所有；`cli` → `dashboard`）
- **不变量由测试守护**：架构层面的承诺（"schema 与 default 一致"、"notebook 不含旧导入"、"dashboard 只读"）都有对应的测试
- **扩展点都在边缘**：新 task = 加一个文件；新页面 = 加一个文件；新 global = 改一行。核心调度器和 manager 几乎永远不需要改

这种结构对实验室软件特别合适 —— 业务（新算法、新分析步骤）经常变，而骨架（"怎么知道哪些录音存在"、"怎么知道什么跑完了"）需要稳定。把易变的东西放在边缘的 task 类和 page 模块里，把稳定的东西放在 manager 和 schema 里。
