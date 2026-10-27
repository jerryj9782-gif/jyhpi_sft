# NAVSIM v1 输入、推理与官方 PDM

本项目输出 8 个未来自车相对位姿，间隔 0.5s、时域 4s。标签由当前位姿和未来位姿做 SE(2) 相对变换，x 向前、y 向左，heading 为弧度并 wrap 到 `[-π,π)`。输出是相对当前帧的绝对未来位置，不是逐步 delta。

相机固定 F0/L0/R0，仅当前帧 RGB uint8；三路图像通过同一 letterbox 预处理。ego state 使用当前速度和加速度，command 使用当前日志中的 recorded command。未来位姿只用于训练标签和评测误差，不能用于生成输入命令。归一化仅由训练集拟合。

## 官方 navtest 清单

训练用自定义 recording-disjoint train/val。测试另使用所安装 NAVSIM v1 仓库的官方 `navtest.yaml`：读取指定 log_names/tokens，必须完整覆盖、不抽样，同时检查与 train、val 的 token 和 recording 不重叠。测试时间戳异常记录进 audit，保留官方 token 覆盖，不因训练过滤规则悄悄缩小测试集。不会用 test 重算统计。

在本项目训练环境、项目根目录执行（读取原始数据，不加载 GPU）：

```bash
python scripts/prepare_navtest.py --root /your/navsim \
  --navsim-root /path/to/navsim --train-manifest assets/navsim/train.jsonl \
  --val-manifest assets/navsim/val.jsonl --output assets/navtest
```

该脚本适配 v1 的 4 history / 10 future / frame_interval=1 / has_route=true 筛选配置，遇到不同版本会拒绝。数据需包含 `navsim_logs/test`、`sensor_blobs/test`。检查所有三路图像是否存在，但不全量解码。

## 批量预测再用 CPU 评分

推荐将 GPU 预测和官方 CPU scorer 分开，避免官方多 worker 各自重复加载完整 VLM。先根据验证集选定唯一 checkpoint，再测试。

```bash
# 本项目环境；会用 GPU，等空闲后执行
python scripts/evaluate.py --checkpoint /path/to/selected_checkpoint \
  --manifest assets/navtest/navtest.jsonl --batch-size 4 --steps 10 \
  --output runs/navtest_predictions
```

修改配置 `data.root` 为你的数据根目录。输出 `predictions.jsonl` 每行 `{token, trajectory}`，`trajectory` 是 8×3；模型和归一化来自该训练 checkpoint。seed 按 token 派生，因此不受 batch 划分影响。`metrics.json` 的 ADE/FDE/heading 和恒速基线只是位移指标。

使用已有的 NAVSIM v1 环境、地图和对应 metric cache，运行本仓库评分脚本。脚本只依赖评测环境，不 import navpi，也不需要训练环境的 Transformers/JAX：

```bash
/path/to/navsim_env/bin/python /path/to/jyhpi_sft/scripts/score_navsim.py \
  --navsim-root /path/to/navsim \
  --cache /path/to/pdm_cache_navtest \
  --manifest /path/to/jyhpi_sft/assets/navtest/navtest.jsonl \
  --predictions /path/to/jyhpi_sft/runs/navtest_predictions/predictions.jsonl \
  --output /path/to/jyhpi_sft/runs/navtest_pdm --workers 4
```

metric cache 由对应 NAVSIM checkout 的官方 caching 流程生成，需匹配场景/地图版本；其中 pickle 只应来自可信本地数据。路径元数据中的 cache 路径应在当前机器有效。本脚本调用该环境的 `navsim.evaluate.pdm_score.pdm_score`、`PDMSimulator` 和 `PDMScorer`；模拟 40 poses×0.1s，预测仍是 8 poses×0.5s。结果包括官方各分项、0～1 `score` 和乘 100 的 `pdm_score_percent`。

预测 token 必须与完整清单顺序、数量一致。缺失 cache、非法轨迹或 scorer 错误会导致失败，不生成假“全部成功”的平均分；中途失败可能留有部分 `scores.jsonl`，完整成功才写 `metrics.json`。这实现的是 **v1 PDM，不是 v2 EPDMS**。本次未执行真实官方评分。

## 直接 Agent 接口

`navpi.navsim_agent.NavPiAgent` 实现 NAVSIM AbstractAgent，默认 `requires_scene=False`，只读取 AgentInput 当前状态与三路图像，不读取未来 Scene。SensorConfig 的 F0/L0/R0 索引为 `[3]`，其他传感器空，与默认 4 帧历史配置对应。

```python
from navpi.navsim_agent import NavPiAgent
agent = NavPiAgent('/path/to/selected_checkpoint', device='cuda', steps=10)
agent.initialize()
trajectory = agent.compute_trajectory(agent_input)
assert trajectory.poses.shape == (8, 3)
```

需要该评测环境同时能 import `navpi` 及其固定 PyTorch/Transformers 依赖，避免直接升级现有 NAVSIM 环境依赖。`configs/navsim_agent.yaml` 提供 Hydra agent 配置，可复制到对应 NAVSIM 配置目录；多 worker 直接 agent 评分需自己控制 GPU 进程数，因此更推荐上面的批量导出方案。当前 adapter 同 seed 42，每次生成同起始噪声；批量评测则按 token 派生 seed，比较结果时应固定使用同一入口。

独立推理 API：

```python
from navpi.policy import NavPiPolicy
policy = NavPiPolicy('/path/to/selected_checkpoint', device='cuda', steps=10)
poses = policy.predict(
    images={'cam_front_center': front_rgb, 'cam_front_left': left_rgb,
            'cam_front_right': right_rgb},
    state=[speed, vx, vy, ax, ay],
    prompt='Drive safely. Recorded navigation command: 0 1 0 0', seed=42)
```

接口依据 [NAVSIM 官方 agents 文档](https://github.com/autonomousvision/navsim/blob/main/docs/agents.md) 和本机 v1 源码核对。CPU adapter 测试验证传感器选择、状态、格式与最新帧读取；不能替代实际地图环境的 PDM 运行。
