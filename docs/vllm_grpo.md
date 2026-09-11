# GRPO vLLM 隔离环境与 Benchmark

## 分支与工作区

- 工作区：`/mingli01/project/xiyongkai/qwen3vl-seg-vllm`
- 分支：`codex/grpo-vllm`
- 主线工作区：`/mingli01/project/xiyongkai/qwen3vl-seg`，保持不动。

## 环境

- Prefix：`/mingli01/data/xyk/conda/vllm-qwen3vl`
- swift：`4.5.2`
- vLLM：`0.20.2`
- transformers：`5.8.1`
- torch：`2.11.0+cu130`
- trl：`0.28.0`
- peft：`0.18.1`

构建要点：

1. 以已有 `dirl_grpo` vLLM 环境为底座克隆，避免重新下载 CUDA/vLLM 栈。
2. 在新 prefix 中禁用 DIRLPlanning 的 `ms-swift 4.0.0.dev0` editable 链接。
3. 从 `ms-swift-grpo` 复制已验证的 `ms-swift 4.5.2`。
4. 安装 `qwen_vl_utils>=0.0.14`。
5. 将 `transformers` 升级到 `5.8.1`，与 ms-swift 4.5.2 兼容。

## 缓存目录（重要）

ms-swift 的 dataset cache 使用 ModelScope 的 `get_cache_dir()`，不是 `HF_DATASETS_CACHE`。必须把
`MODELSCOPE_CACHE` 一起重定向，否则会写满受限 NFS 并触发 `Errno 122 Disk quota exceeded`。

`vllm_grpo_248k_smoke.sh` 已把以下目录放到 GPU 节点 `/dev/shm` 的 job 专属目录：

```text
HF_HOME
HF_DATASETS_CACHE
TRANSFORMERS_CACHE
TMPDIR
MODELSCOPE_CACHE
MS_CACHE_HOME
```

## 启动 4 卡 vLLM GRPO Smoke

```bash
cd /mingli01/project/xiyongkai/qwen3vl-seg-vllm
sbatch --parsable \
  --export=ALL,GEN_BATCH=32,PER_DEVICE_BATCH=8,MAX_STEPS=20,OUT_DIR=/mingli01/data/xyk/grpo/train248k_vllm_bench_g32 \
  scripts/slurm/vllm_grpo_248k_smoke.sh
```

关键参数：

```text
--use_vllm true
--vllm_mode colocate
--vllm_tensor_parallel_size 1
--vllm_enable_lora true
--lora_rank 8
--vllm_gpu_memory_utilization 0.45
--sleep_level 1
```

## Benchmark 结果

### gen_batch=8, 5 steps

- Job：`101405`
- `5/5` 完成，`checkpoint-5`
- 显存：约 `51.1 GiB/GPU`
- 第一步：`86.4s`
- 5 步总耗时：`129.9s`

### gen_batch=32, 20 steps

- Job：`101406`
- `20/20` 完成，`checkpoint-20`
- 显存：约 `58.3 GiB/GPU`
- 稳定约 `21.94s/step`
- 每步 4 个 prompt，约 `5.5s/prompt`
- 20 步总耗时：`438.9s`

对比非 vLLM 4 卡路径约 `9s/prompt`，当前 vLLM + 大批次约 `1.6x` 提升。

## batch reward 修复结果

commit `7d61116` 已修复 vLLM 路径下的 batch mask reward 回退：

- 没有图像 token 或没有有效 `image_grid_thw` 的样本直接跳过，mask reward = 0；
- 单个样本 multimodal 编码失败只跳过该样本，不再拖垮整个 batch；
- 不再出现 `batch mask decode failed` 警告。

验证：job `101542`，`gen_batch=32`，10 步完成，保存 `checkpoint-10`，显存约
`58.25 GiB/GPU`，稳定约 `21.6s/step`。修复前同样配置约 `21.6–21.9s/step`，
说明 reward 回退不是当前 vLLM 路径的主要瓶颈；当前瓶颈更可能在 vLLM rollout、
LoRA 权重同步和 colocate sleep/wake。

## 70k + G=4 + generation_batch_size=64

推荐正式训练配置：

```text
dataset: /mingli01/data/xyk/grpo/rl_hard_70k.jsonl
num_generations: 4
generation_batch_size: 64
per_device_train_batch_size: 8
world_size: 4
steps_per_generation: 2
```

验证结果：

- Job：`101685`
- `10/10` 完成，保存 `checkpoint-10`
- 显存：约 `59.38 GiB/GPU`
- 稳定约 `19.08s/step`
- 10 步总耗时：`221.4s`
- 无 batch mask reward fallback

训练时间估算：

```text
70,090 prompts / 8 prompts per step ≈ 8,762 steps/epoch
8,762 × 19.08s ≈ 167,200s ≈ 46.4h
```

已经低于 72h，比原来的 `G=8 + gen_batch=32` 配置明显更快。

## per_device_train_batch_size=16 结果

在 `70k + G=4 + gen_batch=64` 基础上，把 `per_device_train_batch_size` 从 8 调到 16：

- Job：`101697`
- `10/10` 完成，保存 `checkpoint-10`
- 显存：`68.13 GiB/GPU`
- 稳定 step_time：约 `27–31s`
- 10 步总耗时：`652s`（包含首次 TorchInductor 编译约 6–7 分钟）
- 无 batch mask reward fallback

训练时间估算：

```text
每个 step 16 prompts
70,090 / 16 ≈ 4,381 steps/epoch
4,381 × 29s ≈ 127,000s ≈ 35.3h
```

按 `31s/step` 估算约为 `37.7h`。首步编译时间在完整 epoch 训练中可忽略。

结论：`per_device_train_batch_size=16` 比 8 的约 46h 更快，推荐作为正式训练默认配置。
