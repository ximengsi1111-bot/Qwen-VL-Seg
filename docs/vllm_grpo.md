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

## 已知问题

vLLM colocate 下偶发：

```text
[SegIoUReward] batch mask decode failed, falling back: index 1 is out of bounds for dimension 0 with size 1
```

原因是同一 batch 中个别样本的图像 token 在输入中被截断，decoder 批量前向返回的行数少于 batch
item 数。当前会自动回退到单样本 mask reward，数值正确，但 batch reward 的加速没有完全生效。
后续可把这类样本单独分组处理。

## 与 72h 一个 epoch 的关系

在 `gen_batch=32` 下，每步 4 个 prompt，105135 个 prompt 约需 26284 步：

```text
26284 × 21.94s ≈ 576,000s ≈ 160h
```

仍超过 72h。若目标是 72h 内跑完一个 epoch，还需要：

- 继续增大 generation batch（8/16 prompts per step）；
- 减少 `num_generations`；
- 使用数据子集；
- 或增加 GPU 数。
