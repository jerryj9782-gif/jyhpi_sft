# 基础权重

本机已准备 `pi05_base/model.safetensors`（6.74 GiB / 7.23 GB），来源为已经下载到本机的官方 OpenPI `gs://openpi-assets/checkpoints/pi05_base` 的 PyTorch 转换缓存。没有再次联网重复下载，也没有使用旧 NAVSIM SFT 的已训练动作头。

SHA256：

```text
3fca799bc5ba147e0085df1dc4189d2ad01ec447c553881e62e3d9cd955066cd
```

PaliGemma tokenizer SHA256：

```text
8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6
```

同一份本地权重的参数文件中记录了 `action_horizon=10`，这不是 NAVSIM 推理时域。基础模型没有固定 horizon 的可学习位置表；本项目配置明确使用 8 步和 3 维动作。不是通过修改旧 config 文件来误认为已有新的训练好动作头。

收到完整资产包后可直接运行：

```bash
python scripts/prepare_checkpoint.py \
  --expected-sha256 3fca799bc5ba147e0085df1dc4189d2ad01ec447c553881e62e3d9cd955066cd
```

或从你自己的本地 OpenPI PyTorch 缓存准备：

```bash
python scripts/prepare_checkpoint.py --source /path/to/pi05_base_pytorch \
  --tokenizer-source /path/to/paligemma_tokenizer.model
```

工具还支持 `--hf-repo <明确的OpenPI格式镜像> --revision <固定commit>`，用于你已经选定并核实的镜像。没有默认下载第三方不明权重。只要包内 tokenizer 缺失，会使用官方 `https://storage.googleapis.com/big_vision/paligemma_tokenizer.model` 下载。

不同格式转换、不同 dtype 的权重哈希可能不同，但必须通过所有复用参数的名称与形状校验。LeRobot 的参数名称/配置可能不同，不能直接互换。本项目提供官方 JAX 权重的下载与 CPU 转换入口，映射改编自 [OpenPI 官方转换流程](https://github.com/Physical-Intelligence/openpi#pytorch-support)。

`prepare_checkpoint.py` 只在 CPU/meta device 检查完整架构，不在 GPU 上加载模型。它记录本地来源、文件字节数和哈希；哈希证明文件完整性，不能单独证明远程来源身份。代码和模型使用须遵守根目录 `LICENSE`、`LICENSE_GEMMA.txt`。

## 仅有 GitHub 代码时

```bash
python -m pip install -e '.[checkpoint]'
python scripts/prepare_checkpoint.py
```

默认从 `gs://openpi-assets/checkpoints/pi05_base/params` 的公开 HTTPS 接口下载，不需要账户登录。原始 JAX 文件约 12.44 GB，缓存在 `.checkpoint_download/pi05_base`；逐个对象校验大小及 MD5/CRC32C，通过后原子改名。重跑复用已校验完整文件，未完成对象重新下载，不宣称字节级断点续传。

CPU Orbax 恢复后按官方 tensor layout 映射，保存 BF16 safetensors；不分配 CUDA，不实例化完整 FP32 PyTorch 模型。转换前检查所有参数键和复用形状，转换后 meta device 再严格校验。预留约 30GB 磁盘、至少 40GiB 可用主机内存，建议 64GiB；转换完可手动移走原始下载缓存。不要在主机内存紧张、其他任务需要内存时执行转换。

已有官方 JAX 缓存可跳过网络：

```bash
python scripts/prepare_checkpoint.py --jax-source /path/to/pi05_base
```

新转换输出不含随机初始化、未使用的 expert 词表输出头，因此文件比本机旧转换缓存更小、SHA256 不同；807 个实际复用参数仍按同样布局加载。上面 `3fca...` 哈希只用于核对本机已交接的那一份文件，不适用于所有不同序列化版本。每次准备记录实际 SHA256、来源和映射报告。tokenizer 固定官方哈希，错误文件会拒绝。

测试已覆盖真实 GCS 清单、小文件下载校验、真实 Orbax time MLP 子树的 CPU 恢复及权重精确比对、完整预转换权重检查及缩小后的全部 JAX tensor layout。为了避免占用当前训练任务的主机资源，没有重新执行全尺寸 Orbax 恢复/转换；新机器应先运行准备命令确认环境兼容。训练不 import JAX/Orbax；这些可选依赖只用于首次转换。
