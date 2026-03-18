# trace_gen

高性能 trace 生成器：内置分布数据并快速采样生成 `(q_len, kv_len, output_len)` 请求三元组。

支持两种使用方式：

- **静态生成**（CLI）：预生成一批带/不带时间戳的请求样本，支持 HPP / NHPP / NB 三种到达过程
- **动态调用**（Python API）：外部系统按需逐条获取请求，到达时间由调用方控制

## 安装

### 依赖

- Python >= 3.8
- numpy

### 方式一：pip / uv 安装 wheel

```bash
pip install trace_gen-<version>-cp38-abi3-<platform>.whl
```

或使用 uv：

```bash
uv pip install trace_gen-<version>-cp38-abi3-<platform>.whl
```

### 方式二：直接使用 .so 文件（无需安装）

解压 `trace_gen-<version>-<platform>.tar.gz`，将 `trace_gen_bundle/trace_gen/` 目录放到项目中或加入 `PYTHONPATH`：

```bash
tar xzf trace_gen-0.1.0-linux_x86_64.tar.gz
export PYTHONPATH="$PWD/trace_gen_bundle:$PYTHONPATH"

# 直接使用
python -c "from trace_gen import core; print(core.load_pack_default())"

# 或运行 CLI
python -m trace_gen.cli --scenes all --n 10000 --out-dir ./trace_output
```

bundle 目录结构：

```
trace_gen_bundle/
├── trace_gen/
│   ├── __init__.py
│   ├── cli.py
│   └── core.abi3.so
├── example/
│   └── dynamic_call.py
└── README.md
```

### 支持平台

| 平台 | 架构 | Python |
|------|------|--------|
| Linux | x86_64 | 3.8 – 3.14 |
| Linux | aarch64 | 3.8 – 3.14 |

## 快速使用

安装后即可直接运行：

```bash
python -m trace_gen.cli --scenes all --n 10000 --out-dir ./trace_output
```

## CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--scenes` | `all` 或逗号分隔的 scene_id | `all` |
| `--n` | 每个场景生成样本数 | `10000` |
| `--seed` | 随机种子 | `1` |
| `--out-dir` | 输出目录 | `./trace_output` |
| `--sample-mode` | `all` / `qkv` / `out` | `all` |
| `--with-time` | 是否输出到达时间列 | 否 |
| `--qps-mode` | `hpp` / `nhpp` / `nb` | `hpp` |
| `--duration-sec` | 生成时间长度（秒） | `10.0` |
| `--qps` | 常量 QPS（用于 `hpp` / `nb`） | `1000.0` |
| `--nb-alpha` | 负二项过分散参数 | `1.0` |
| `--nhpp-bucket-sec` | NHPP 分段长度（秒） | `1.0` |
| `--nhpp-rates` | 逗号分隔的分段 QPS（不足时循环） | - |
| `--allow-zero-q` | 保留 `q_len=0` 的样本（少数场景 cache 完全命中） | 默认过滤 |
| `--allow-zero-output` | 保留 `output_len=0` 的样本（少数场景系统无输出） | 默认过滤 |

> **关于零值过滤**：少数场景下采样可能产生 `q_len=0`（cache 完全命中）或 `output_len=0`（系统无输出）的请求。默认自动过滤这些样本，以避免下游系统收到无效请求。如需保留，分别传入 `--allow-zero-q` 或 `--allow-zero-output`。

## 静态生成（CLI）

通过 CLI 一次性预生成请求样本至 CSV 文件。

### 仅生成长度样本

```bash
python -m trace_gen.cli --scenes all --n 10000 --out-dir ./trace_output
```

输出 CSV：`q_len, kv_len, output_len`

### HPP（恒定 QPS）+ 时间序列

```bash
python -m trace_gen.cli --scenes all --n 10000 --out-dir ./trace_output \
    --with-time --qps-mode hpp --duration-sec 10 --qps 1000
```

请求以恒定速率到达，适合模拟平稳负载。

### NHPP（分段变化 QPS）+ 时间序列

```bash
python -m trace_gen.cli --scenes all --n 10000 --out-dir ./trace_output \
    --with-time --qps-mode nhpp --duration-sec 10 --nhpp-bucket-sec 1 \
    --nhpp-rates 800,1200,900,1500
```

每个 bucket 内的 QPS 可以不同，`--nhpp-rates` 指定各段 QPS，长度不足时自动循环。可模拟：

- **流量高峰**：`--nhpp-rates 500,500,2000,2000,500`
- **周期波动**：`--nhpp-rates 300,800,1500,800,300`
- **渐进增长**：`--nhpp-rates 100,200,400,800,1600`

### NB（过分散 QPS）+ 时间序列

```bash
python -m trace_gen.cli --scenes all --n 10000 --out-dir ./trace_output \
    --with-time --qps-mode nb --duration-sec 10 --qps 1000 --nb-alpha 1.0
```

每秒请求数服从负二项分布，`alpha` 越大方差越大，适合模拟突发流量。

> 带 `--with-time` 时输出 CSV：`time, q_len, kv_len, output_len`

> 如需对静态生成的数据自定义到达时间，可直接读取 CSV 后自行重排 time 列。

## 动态调用（Python API）

当外部系统需要按需获取请求时（如负载生成器、仿真框架），每次调用采样器获取一条或一小批请求，到达时间由调用方自行控制。

采样默认过滤 `q_len=0` 和 `output_len=0` 的样本，可通过 `allow_zero_q` / `allow_zero_output` 参数控制：

```python
import numpy as np
from trace_gen import core

# 初始化
pack = core.load_pack_default()
scene = pack.scenes[sorted(pack.scenes.keys())[0]]
rng = np.random.default_rng(42)

def next_request():
    """按需获取一条请求（默认过滤零值）"""
    q, kv, out = core.sample_from_scene(
        rng=rng,
        case_type=scene.case_type,
        alias_accept=scene.alias_accept,
        alias_index=scene.alias_index,
        spikes=scene.spikes,
        spikes_count=scene.spikes_count,
        bins=scene.bins,
        bins_count=scene.bins_count,
        out_bins=scene.out_bins,
        out_alias_accept=scene.out_alias_accept,
        out_alias_index=scene.out_alias_index,
        n=1,
        out_quantiles=scene.out_quantiles,
        # allow_zero_q=False,      # 默认过滤 q_len=0
        # allow_zero_output=False,  # 默认过滤 output_len=0
    )
    return int(q[0]), int(kv[0]), int(out[0])

# 外部系统在需要时调用
q_len, kv_len, output_len = next_request()
```

完整可运行示例见 [`example/dynamic_call.py`](example/dynamic_call.py)。

## API 参考

### `trace_gen.core`

```python
from trace_gen import core
```

**数据加载**

| 函数 | 说明 |
|------|------|
| `load_pack_default()` | 加载内置数据，返回包含所有场景的对象 |

**采样**

| 函数 | 说明 |
|------|------|
| `sample_from_scene(rng, ..., n, allow_zero_q=False, allow_zero_output=False)` | 从指定场景采样 `n` 条请求，返回 `(q, kv, out)` 三个 `ndarray[int32]`；默认过滤零值 |

**QPS 生成器（用于静态生成）**

| 类 | 说明 |
|----|------|
| `HPP(qps)` | 齐次泊松过程（恒定 QPS） |
| `NHPPPiecewise(rates, bucket_sec)` | 非齐次泊松过程（分段变化 QPS） |
| `NBCounts(mean_qps, bucket_sec, alpha)` | 负二项分布（过分散 QPS） |

所有 QPS 生成器提供 `.generate(duration_sec, rng)` 方法，返回排序后的到达时间数组 `ndarray[float64]`。

## Citation

If you use this software in academic research that results in a publication, presentation, or publicly released manuscript, please cite:

```bibtex
@inproceedings{cai2026characterizing,
  title={Characterizing Cloud-Native LLM Inference at Bytedance and Exposing Optimization Challenges and Opportunities for Future AI Accelerators},
  author={Cai, Jingwei and Kong, Dehao and Huang, Hantao and Jiang, Zishan and Ma, Zixuan and Guo, Qingyu and Zhang, Zhenxing and Shi, Guiming and Gao, Mingyu and Ma, Kaisheng and others},
  booktitle={2026 IEEE International Symposium on High Performance Computer Architecture (HPCA)},
  pages={1--19},
  year={2026},
  organization={IEEE}
}
```

