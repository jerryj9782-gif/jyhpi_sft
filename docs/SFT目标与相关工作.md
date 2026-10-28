# SFT、ADE/FDE 与 PDM：当前方案为什么这样训练

核对日期：2026-10-28（Asia/Shanghai）。本次只核对论文、官方代码和本地实现，没有启动 GPU 训练或真实 PDM 评测。

## 当前方案是否合适

作为将预训练 π0.5 适配到 NAVSIM 的第一阶段，**保留 flow matching、用真实驾驶轨迹监督，是合理的 SFT 基线**。模型已经有预训练的视觉、语言和动作专家，新动作投影改为原生 8×3 后，继续训练条件轨迹分布比直接替换成无训练基础的 PDM 优化更自然。实际驾驶质量仍需验证，不能仅凭 loss 下降判定成功。

本项目 `src/navpi/model.py::NavPi.forward` 使用归一化后的真实轨迹 a 和高斯噪声 ε：

```text
x_t = t ε + (1-t) a
target velocity = ε - a
loss = mean((predicted_velocity - target_velocity)²)
```

这是 **flow-matching 速度预测 MSE**。它不是米制轨迹 ADE、FDE，也不是 PDM reward。速度场积分得到最终轨迹后，验证脚本才计算 ADE/FDE：ADE 是全部未来位置的平均欧氏距离，FDE 是最后一个位置的欧氏距离。heading 单独报告弧度误差；训练 loss 则包含全部三个归一化动作维度。

建议继续使用当前 flow-matching SFT，先完整训练 20 epochs。ViT 默认冻结 1 轮、随后低学习率解冻；projector/VLM/动作专家从开始训练。这个冻结日程是工程选择，不是论文证明的 NAVSIM 最优策略，可以用全程冻结/全程解冻对照验证。三路当前图像适合先做高吞吐基线，但没有多帧历史，动态目标运动信息可能不足，应作为后续独立消融。

## 相关工作的实际目标

| 工作及阶段 | 实际训练信号 | PDM 的角色 |
|---|---|---|
| ReCogDrive Stage 1 | 驾驶理解/轨迹等输出的 VLM 监督学习 | 不是直接优化 PDM 的阶段 |
| ReCogDrive Stage 2 IL | 论文使用 DDPM 噪声预测 MSE；官方代码另外支持 flow 速度预测 MSE | 用于评估生成轨迹 |
| ReCogDrive Stage 3 DiffGRPO | 采样轨迹，按 reward 计算组内 advantage 更新策略 | NAVSIM PDM 作为 reward |
| AutoVLA SFT | 输出 token 的负对数似然，额外加强 action-token 监督 | 不是 PDM loss |
| AutoVLA RFT | GRPO 策略优化 | NAVSIM 使用 PDM driving reward，另含推理长度惩罚 |
| NAVSIM TransFuser | 轨迹 L1，加感知辅助损失 | 用于最终驾驶评估 |
| Hydra-MDP | 模仿监督 + 仿真各项分数的监督蒸馏 | 可以成为评分头的监督标签 |

依据：[ReCogDrive 论文方法与阶段说明](https://arxiv.org/html/2506.08052)、[ReCogDrive 官方 planner 实现](https://github.com/xiaomi-research/recogdrive/blob/main/navsim/agents/recogdrive/recogdrive_diffusion_planner.py)、[AutoVLA §3.3–3.4](https://arxiv.org/html/2506.13757v2#S3.SS3)、[TransFuser 官方损失](https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/transfuser_loss.py)、[Hydra-MDP 论文](https://arxiv.org/abs/2406.06978)。

ReCogDrive 本地 `recogdrive_agent.py::compute_loss` 在训练且非 GRPO 时返回 planner 的 `predictions.loss`；它的非训练分支才计算生成轨迹的 L1。因此只看到 agent 文件里的 L1 就认为整个 IL 在训练 ADE/FDE，会误读调用路径。

不能概括为“所有 SFT 都训练 ADE/FDE”，也不能说“PDM 只能用于 RL”。主要有三种情况：轨迹模仿损失、PDM 监督评分/蒸馏、PDM reward 的策略优化。当前项目实现第一种，并提供 PDM 评估入口；没有实现 Hydra 评分头或 PDM-RL。

## 为什么不能直接把 loss 换成负 PDM

标准 NAVSIM PDM scorer 包含模拟、碰撞/道路规则判定和 NumPy 计算，不是连接到当前 PyTorch 模型的可微损失。将输出 detach 后送进 scorer，再把 score 包成 tensor，不会产生所需的策略梯度。若要优化 PDM，需要实现有效的策略优化方法，或者额外训练可微评分/排序模型；这都是单独的方法改动。

PDM 同时衡量碰撞、道路合规、进度、TTC、舒适度等，位移误差没有覆盖这些目标。一个模型可以更接近记录轨迹，却在安全分项上更差。因此当前自动按验证 ADE 记录 `best.json` 只是一种低成本选择，不能称为“PDM 最佳模型”。

## 本项目的模型选择建议

训练仍用 flow matching；每 epoch 记录验证 ADE/FDE/heading。建议预先固定在第 5、10、15、20 epoch 对同一验证集补充 PDM 评估，并包含 ADE 最佳的候选 checkpoint；以验证 PDM 和安全分项判断最终候选，记录所选路径，之后才运行 navtest。不能用 navtest 反复选 epoch、LR、采样 seed 或去噪步数。

以当前 30,266 样本、8 卡、micro batch 2、累积 4 为例，每 epoch 473 次更新，对应候选 step 为 2365、4730、7095、9460。实际数据重建后应从训练日志确定 epoch 对应的 step。固定 seed=42、采样 10 步，所有候选使用相同清单和 scorer 版本。

下面是单个候选的验证 PDM 流程；预测会用 GPU，需之后主动执行。本项目当前不会在训练循环中自动启动官方 scorer：

```bash
# 训练环境中导出验证集预测
python scripts/evaluate.py --checkpoint runs/navsim_8gpu/step_00002365 \
  --manifest assets/navsim/val.jsonl --steps 10 --seed 42 \
  --output runs/val_epoch5_predictions

# NAVSIM v1 环境中使用覆盖这些验证 token 的 metric cache
/path/to/navsim_env/bin/python scripts/score_navsim.py \
  --navsim-root /path/to/navsim --cache /path/to/pdm_cache_trainval \
  --manifest assets/navsim/val.jsonl \
  --predictions runs/val_epoch5_predictions/predictions.jsonl \
  --output runs/val_epoch5_pdm --workers 4
```

当前自定义 val 不保证与官方 navtrain 的完整筛选一致，metric cache 必须覆盖全部 val tokens；缺失时先按相同场景构建 cache，不能静默跳过失败样本。该流程只是验证模型，不会把 PDM 分数反向传播进 SFT。

## 为什么改成 20 epochs

此前默认 8 epochs 是试跑预算，没有当前数据上的收敛实验支持它足够。现设置 `epochs=20`、`early_stop_patience=0`、`min_epochs=20`、`max_steps=null`：正常执行会完整训练 20 轮，早期最佳 checkpoint 仍可保存。

[ReCogDrive 论文](https://arxiv.org/html/2506.08052) 的第二阶段训练 200 epochs，但冻结 VLM，只训练 diffusion planner；第一阶段 VLM 训练预算和第二阶段不同。[AutoVLA 附录 C](https://arxiv.org/html/2506.13757v2#A3) 给出 5 epochs 的 SFT 设置。不同数据量、全参/冻结范围、batch、模型初始化和损失不能只用 epoch 数比较。本项目当前有效 batch 64，20 轮约 9,460 次更新；是否增加至 30 轮，应结合验证曲线和计算预算，而非直接照搬某篇论文。

具体配置、30 轮可选设置、磁盘预算及续训限制见 [80GB 训练方案](80GB训练方案.md)。
