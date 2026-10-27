# OpenPI → NAVSIM SFT

独立 PyTorch 项目，将 π0.5 的预训练动作专家适配到 NAVSIM。只输入 **F0 / L0 / R0 三路当前相机**，直接预测 **`[B, 8, 3]`**：未来 0.5～4.0 秒、相对当前自车坐标系的 `(x 米, y 米, heading 弧度)`。不使用 32 维机器人动作填充，不依赖同级 `openpi_old`。

默认训练策略是 **ViT 冻结 1 epoch 后小学习率解冻，projector、VLM、动作专家与新动作头全程训练**。支持从第一步全参训练和冻结 ViT 对照。最多 8 epochs，每 epoch 验证、保存、按验证 ADE 选模型并早停。训练使用 BF16 autocast、FP32 参数和 AdamW 状态，支持单卡、多卡和多节点 DDP。

本次交付只准备代码、资产和 CPU 检查，没有启动真实训练、GPU benchmark 或官方 PDM 评测。80GB 最优 batch/速度需在目标空闲 GPU 上测量。

## 模型与训练选择

| 部分 | 实现 |
|---|---|
| ViT | 预训练 SigLIP，三路共享，224×224 等比 letterbox；三个视角合批编码 |
| projector | 保留预训练 `1152 → 2048`，从开始训练，峰值 LR `1e-5` |
| VLM | Gemma 2B prefix，峰值 LR `1e-5` |
| 动作专家 | 预训练 Gemma 300M、时间 MLP、AdaRMSNorm，峰值 LR `1e-5` |
| 新动作头 | 新建 `3 → 1024` 输入投影、`1024 → 3` 输出投影，峰值 LR `1e-4` |
| ViT 日程 | 第 1 epoch 冻结，随后峰值 LR `1e-6`，独立 50 steps warmup |
| 状态 / 命令 | 当前 `[speed, vx, vy, ax, ay]` 与 recorded command；训练集归一化后编码为状态文本 |
| 输入长度 | 768 个图像 tokens + 最长 64 个文本 tokens；超长报错 |

**ViT 和 projector 的结构都不需要因为相机数和动作维度而改动。** projector 始终训练以适应驾驶场景；ViT 默认先稳定新动作头再解冻。研究中两种视觉训练策略都存在，不能从某篇文章的 RFT 冻结设置推导出本项目 SFT 必须冻结。依据、取舍、消融命令见 [80GB 训练方案](docs/80GB训练方案.md)。

模型有 3,353,374,451 个参数，基础 checkpoint 严格复用 807 个张量。去掉无用词表输出头，仅重建动作输入/输出投影；兼容原权重的 tied embedding 别名。最后一层 prefix 的部分参数不参与动作损失，DDP 使用 `find_unused_parameters=True`，详见 [实现说明](docs/使用说明.md)。

## 新机器从 GitHub 开始

```bash
git clone https://github.com/jerryj9782-gif/jyhpi_sft.git
cd jyhpi_sft
PYTHON=python3.11 bash scripts/install.sh
source .venv/bin/activate

# 首次使用：从官方公开 GCS 下载 π0.5 JAX 权重，并在 CPU 转换
python -m pip install -e '.[checkpoint]'
python scripts/prepare_checkpoint.py

# 使用自己的合法 NAVSIM 数据，检查三路图片是否存在
python scripts/prepare_data.py --root /your/navsim \
  --source-split trainval --stride 4 --output assets/navsim --check-images

bash scripts/check_cpu.sh
bash scripts/check_ddp_cpu.sh
```

安装在新的项目 `.venv`，不会覆盖旧环境。固定 PyTorch 2.8.0、Transformers 4.53.2；包内附带 OpenPI 修改的 Gemma/SigLIP，不修改 site-packages。CUDA wheel 默认 cu128，目标驱动须支持。JAX/Orbax 仅首次官方权重转换需要，训练本身不需要。

下载使用官方公共地址，无需 GitHub/Hugging Face 登录；文件逐个校验、复用已下载的完整文件，转换只用 CPU。预留约 30GB 磁盘和至少 40GiB 可用主机内存，建议 64GiB；不要在主机内存紧张时转换。也可直接导入已有 OpenPI PyTorch 权重：

```bash
python scripts/prepare_checkpoint.py --source /path/to/pi05_base_pytorch \
  --tokenizer-source /path/to/paligemma_tokenizer.model
```

下载细节、哈希与转换差异见 [checkpoint 说明](checkpoints/README.md)。数据需包含 `navsim_logs/trainval`、`navsim_logs/test` 和 `sensor_blobs/trainval`；训练准备只读取 test 文件名排除 recording。输入格式见 [数据说明](assets/navsim/README.md)。

## 本机已准备的资产

目录：`/home/jyh/project_jyh/jyhpi/openpi_navsim_sft`。

- 独立复制的基础 checkpoint：`checkpoints/pi05_base/model.safetensors`，7,233,650,408 字节，非软链接。
- `assets/tokenizer/tokenizer.model`。
- `assets/navsim/{train.jsonl,val.jsonl,norm_stats.json,audit.json}`：30,266 train / 7,569 val，沿用旧项目 recording-disjoint 自定义划分，非完整官方 navtrain。
- `reports/`：CPU 测试、DDP、权重映射、数据与公开下载地址检查记录。

原始图片留在 `/data/open_datasets/navsim`，通过配置 `data.root` 访问。GitHub 不包含数 GB 权重、私有数据清单、缓存或训练产物；下载与构建脚本均包含在仓库中。

## 之后训练

先修改 `configs/sft_80gb.yaml` 的数据根目录和输出目录。以下命令供 GPU 空闲后执行。

```bash
# 单卡 80GB，micro batch 2 × 累积 32 = 有效 batch 64
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash scripts/train.sh \
  --set train.grad_accum=32 --set train.output=runs/navsim_1gpu

# 8 卡 80GB，8 × 2 × 4 = 有效 batch 64
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
  bash scripts/train.sh --set train.output=runs/navsim_8gpu

# 从第一步全参训练（含 ViT）；不使用冻结阶段
NPROC_PER_NODE=8 bash scripts/train.sh \
  --set train.vision_mode=full --set train.output=runs/full_from_start
```

配置默认 micro batch 2、累积 4、逐层重计算，是待实测的保守起点；如果直接单卡用默认值，有效 batch 是 8。DDP 每卡复制完整模型与优化器，不合并多卡显存。

默认最多 8 epochs，至少到第 4 epoch 才允许早停，验证 ADE 连续 3 次未改善超过 0.001m 时停止。`best.json` 指向选中的 checkpoint，`validation.jsonl` 记录验证曲线。细节与步数计算见 [80GB 训练方案](docs/80GB训练方案.md)。

```bash
# 精确续训：相同配置 / world size，目录替换为实际保存的 checkpoint
NPROC_PER_NODE=8 bash scripts/train.sh --resume runs/navsim_8gpu/step_00000473 \
  --set train.output=runs/navsim_8gpu

# 改训练规模：保留已训 NAVSIM 头，重建优化器及日程，使用新输出目录
NPROC_PER_NODE=1 bash scripts/train.sh --warm-start runs/navsim_8gpu/step_00000473 \
  --set train.grad_accum=32 --set train.output=runs/new_experiment
```

每 epoch 保存完整模型、优化器、日程、归一化、tokenizer 和各 rank RNG；完整 checkpoint 预算约 38GiB，8 份建议预留 320GiB 以上。不自动删除旧检查点。

## 80GB 显存与速度

```bash
# CPU 预处理：缓存像素，不缓存视觉特征；建议本地 SSD
python scripts/cache_data.py --split train --output data_cache/train --workers 4
python scripts/cache_data.py --split val --output data_cache/val --workers 4

# 会运行临时 GPU 更新，仅在目标 GPU 空闲后执行
python scripts/tune_batch.py --gpus 8 --batches 1 2 4 8 \
  --checkpointing full vision none --target-batch 64 --memory-budget-gib 74 \
  --set data.train_cache=data_cache/train --set data.val_cache=data_cache/val \
  --output runs/tuning_80gb

NPROC_PER_NODE=8 bash scripts/train.sh --config runs/tuning_80gb/recommended.yaml
```

调优默认按 **ViT 已解冻** 的完整模型测量前向、反向、优化器状态和 DDP，排除前两步 warmup，按显存预算内最高 samples/s 生成配置。不是按显存占用越高越好来选。默认 `compile=false`；显存与吞吐估算、冻结对照、失败处理见 [80GB 训练方案](docs/80GB训练方案.md)。

## 验证与 NAVSIM

```bash
python scripts/evaluate.py --checkpoint runs/navsim_8gpu/step_00000473 \
  --output runs/val_predictions
```

导出固定 token seed 的轨迹和 ADE/FDE/heading、恒速基线。这些误差不等于 PDM。仓库还提供 `NavPiAgent`、官方 v1 navtest 清单准备和独立 CPU PDM 评分脚本；完整可执行流程见 [NAVSIM 适配](docs/NAVSIM适配.md)。NAVSIM/nuPlan/地图与 metric cache 使用独立评测环境。

## 检查与交接

检查范围见 [检查记录](docs/检查记录.md)，模型及数据细节见 [使用说明](docs/使用说明.md)。已验证的 CPU 逻辑不能代替完整模型 GPU 吞吐或官方驾驶评估。

```bash
python scripts/export_release.py --output dist/handoff_code.tar.gz
# 另行交接本机 checkpoint / tokenizer / 清单，不含 NAVSIM 原图
python scripts/export_release.py --with-assets --output dist/handoff_with_assets.tar
```

代码保留 Apache-2.0 许可与来源记录；模型另遵循 [Gemma 条款](LICENSE_GEMMA.txt)。上游：[OpenPI](https://github.com/Physical-Intelligence/openpi)、[NAVSIM](https://github.com/autonomousvision/navsim)。
