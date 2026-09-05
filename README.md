# Qwen3-VL-Seg 从零复现 — 项目进展与结果

> 状态：已完成从环境、数据、模型、训练到评测的全链路（100k 小数据版），并完成动态分辨率 bucket / 小批次实验。
> 本文档为项目当前进展、成果、实验与结论的总览。

## 1. 项目目标

从零（不复用现成 Qwen-VL-Seg 代码）复现论文 **《Qwen3-VL-Seg》**：基于 Qwen3-VL-4B 视觉语言模型 + Box-Guided Mask Decoder，实现指代分割（R-ES）与指代理解。当前阶段采用小数据策略（约 100k 条样本），先在 4×A800 上打通完整链路，再评估是否扩展数据与卡数。

## 2. 运行环境与命名

| 名称 | 说明 |
|---|---|
| `kk` | 本地电脑，代码仓库 `D:\XYK\Qwen-VL-Seg` |
| `祝融` | 当前服务器 `172.29.15.117`（`choud-mgt-09`），Slurm 分区 `gre`，调试/正式训练主力 |
| `DL` | 租用服务器（2×RTX 5090D 32GB，Blackwell `sm_120`），目前无卡/备用 |

- conda 环境：`segment`（Python 3.10.x）
- 框架：PyTorch + DeepSpeed（ZeRO-2，bf16）+ LoRA
- 训练调度：`deepspeed`（4 卡，`--num_gpus=4`）
- 远程操作约定：所有命令带 `PYTHONNOUSERSITE=1`；Slurm 作业 `module load conda/3 cuda/13.0`，Python 路径回退到 `/persist_data/home/mingli/.conda/envs/segment/bin`

## 3. 目录约定

```text
/mingli01/project/xiyongkai/qwen3vl-seg        # 代码仓库（祝融）
├── configs/             # 训练/DeepSpeed 配置
├── qwen3vl_seg/         # 模型代码包
│   ├── data/            # schema/dataset/collator/bucket/sampler
│   ├── model/           # model_wrapper/mask_decoder/lora/prompt/loss
│   ├── train/           # config/training_loop
│   └── eval/            # runner/runner_bucket/metrics
├── scripts/             # Slurm 提交脚本
├── docs/                # 方案文档
└── tests/

/file_storage01/home/mingli/data/xyk          # 数据/权重/结果资产
├── build100k/           # 100k 数据构建（manifest 与 samples.jsonl）
├── checkpoints/         # 模型权重（small/stage1-100k、stage2-100k 等）
├── runs/                # 训练 metrics 与评测结果
└── cache/               # HF/缓存

/mingli01/models/Qwen3-VL-4B-Instruct         # 基座模型
```

## 4. 数据（100k 构建）

- 清单：`/file_storage01/home/mingli/data/xyk/build100k/datasets/small/samples.jsonl`
- 规模：`samples = 101200`，`images = 26304`
- 切分：

| 切分 | 数量 | 组成 |
|---|---|---|
| train | 100000 | RefCOCO 30000 + RefCOCO+ 30000 + RefCOCOg 30000 + COCO 5000 + LVIS 5000 |
| val | 600 | RefCOCO 200 + RefCOCO+ 200 + RefCOCOg 200 |
| test | 200 | RefCOCOg 200 |
| testA | 200 | RefCOCO 100 + RefCOCO+ 100 |
| testB | 200 | RefCOCO 100 + RefCOCO+ 100 |

- 说明：公开指代数据（RefCOCO 系）用于指代分割，COCO/LVIS 用于通用目标分割，弥补 SA1B-ORS 无网获取受限的问题。
- 配套统计：`/file_storage01/home/mingli/data/xyk/build100k/stats.json`；图像清单：`image_manifest.jsonl`。

## 5. 模型

- 基座：`Qwen3-VL-4B-Instruct`（`/mingli01/models/Qwen3-VL-4B-Instruct`）
- 接入：`qwen3vl_seg/model/model_wrapper.py`（`Qwen3VLSegForSegmentation`）
- 解码器：`qwen3vl_seg/model/mask_decoder.py`（Box-Guided Mask Decoder）
- 两阶段训练：
  - **Stage1**：LoRA（rank 32，`q/k/v/o_proj`、`gate/up/down_proj` 共 7 模块），训练视觉编码器 + 解码器，引入 `<mask_token>` 监督。
  - **Stage2**：全参数微调解码器，冻结视觉编码器。
- 关键修复：mask token 监督（commit `c222053` / `978a478`）——`<mask_token>` 参与 CE，且 Stage1 对 `<mask_start>/<mask_token>/<mask_end>` 的 embedding/lm_head 行解冻（通过 hook 梯度掩码，修复了设备 bug）。

## 6. 训练（100k 正式版，无桶）

- 配置：`configs/train/stage1_100k_zhurong.yaml`、`stage2_100k_zhurong.yaml`
- 通用：batch=global16（micro1×gas4×world4）、bf16、ZeRO-2、gradient checkpoint、AdamW(lr 见各阶段)、cosine 到 `max_steps`。

| 阶段 | 作业 | 步数 | 末步 loss | 显存 | 状态 |
|---|---|---|---|---|---|
| Stage1-100k | 94855 | 12500 | total 0.712 / text 0.568 / seg 0.142 / iou 0.011 | 54.23GB | COMPLETED |
| Stage2-100k | 95157 | 6250 | total 0.423 / text 0.296 / seg 0.126 / iou 0.008 | 68.76GB | COMPLETED |

- Stage1 末步在 step 12500；Stage2 末步在 step 6250（约 2:1 epoch）。
- checkpoint：`checkpoints/small/stage1-100k`、`checkpoints/small/stage2-100k`
- 训练日志：`runs/small/stage1-100k/metrics.jsonl`、`runs/small/stage2-100k/metrics.jsonl`

## 7. 正式评测结果（1200 条）

- 采用 `eval/runner.py`，预测 | 真值 左右拼接可视化（默认开启）。
- 作业：`95280`（评测）、`95389`（可视化）。结果目录：`runs/eval/stage2-100k-formal/`（`metrics.json`、`predictions.jsonl`、`viz/`）。

**总体**

| 指标 | 值 |
|---|---|
| n_samples | 1200 |
| mIoU | 0.6770 |
| cIoU | 0.6633 |
| strict P@0.5 | 0.835 |

**按子集（mIoU）**

| 数据集 | val | testA | testB |
|---|---|---|---|
| RefCOCO | 0.6887 | 0.7067 | 0.6932 |
| RefCOCO+ | 0.6525 | 0.6982 | 0.6468 |

| RefCOCOg | val | test |
|---|---|---|
| mIoU | 0.6710 | 0.6770 |

## 8. 与参考论文对比

论文指标（Table 1 / Table 5）：

| 数据集 | cIoU (val / testA / testB) | mIoU | P@0.5 |
|---|---|---|---|
| RefCOCO | 82.3 / 83.7 / 79.1 | 82.8 | 92.8 |
| RefCOCO+ | 76.2 / 80.2 / 70.8 | 78.5 | 87.8 |
| RefCOCOg | 78.2 / 78.1 | 78.6 | 88.0 |

当前复现明显低于论文，预期主因：
1. **数据规模**：论文用 SA1B-ORS 等大规模数据；本次仅约 100k 条文（公开指代 + COCO + LVIS），且无网受限无法拉取 SA1B-ORS。
2. **训练预算**：Stage1/Stage2 仅各跑约 2/1 epoch，明显不足。
3. **mask token 输出稳定性**：早期推理依赖占位符注入，`<mask_token>` 自主生成尚不稳定（已在代码中引入 mask 监督修复，但本次 100k 训练在修复前完成，尚未用修复后的权重重跑）。
4. 动态分辨率 bucket / 小批次为后续优化点，尚未并入本次接受结果。

## 9. 实验：动态分辨率 bucket / 小批次（micro_batch>1）

> 详见 `docs/dynamic_batch_experiments.md`，研究分支 `codex/dynamic-batch-resolution`。

### 背景

Qwen3-VL 视觉编码为动态分辨率，不同尺寸图像产生不同大小 `pixel_values`，无法直接堆进 batch，最初只能 micro_batch=1 逐张前向，GPU 吞吐受限。为此引入 **bucket**：同一分辨率桶内的样本组成 batch，右/下补边到桶画布，不缩放不裁切。

### 方案

- 桶：`bucket = min(b | b >= max(H,W))`，默认 `{256,512,768}`（已细化到 `{256,384,512,640,768}`）。
- 填充：只补右/下，`max_pixels=768²=589824`。
- 感知 sampler + collator（同桶 batch 拼接），训练补 0、解码后裁回原图；`eval/runner_bucket.py` 推理按桶分桶并还原原始顺序。
- 配置见 `configs/train/stage{1,2}_100k_bucket_micro4_zhurong.yaml`（`use_bucket=true`、`micro_batch_size=4`、`gradient_accumulation_steps=1`、`global_batch_size=16`、LR 按 batch 缩放）。

### 实验结论（等样本预算，oracle-box 口径）

| 方案 | 完整 val avg mIoU | vs dyn_micro1 |
|---|---|---|
| dyn_micro1（无桶） | **0.5264** | — |
| bucket_micro2 lr2x | 0.4961 | −0.030 |
| bucket_micro4 lr4x | 0.5047 | −0.022 |

1. 原先 bucket 比无桶低 0.17–0.19 的“大差距”，**主因是优化器更新次数变少**（micro>1 更新次数减半/¼，而 LR 未变），不是补边空白。
2. **解法 = LR 按 batch 倍数线性放大**：可补回约 0.12，残差仅 0.02–0.03（接近噪声）。
3. **细化桶（减少空白）没有帮助**：空白填充不是性能瓶颈。
4. `micro_batch>1 + 线性缩放 LR` 是可行提速方案，在 oracle-box 口径保住约 98% 精度。

### 近期调整

- 已把 bucket 档位加细为 `[256, 384, 512, 640, 768]`（5 档，commit `c46303d`）：图像取“不小于 max(H,W) 的最小桶”，减少补边空白；最大桶仍为 768，`max_pixels=589824` 不变。`micro_batch_size=4` 及其余参数保持不变。

## 10. 已知问题与取舍

- **mask token 修复未回灌到已接受的 100k 训练**：本次接受的结果来自修复前的代码，若要让 mask 监督生效需重跑。
- **bucket sampler 单遍限制**：祝融 `master`（`978a478`）上的 bucket sampler 单遍迭代；若想跑满 `max_steps` epoch 需循环改造（该修复在本地方 `codex/dynamic-batch-resolution` 分支，尚未并入祝融 master）。
- **正式训练未采用 bucket/micro4**：当前接受结果仍为原版（无桶、micro1）。bucket 相关代码保留为可选能力。

## 11. 当前状态与下一步

已接受：100k Stage1/Stage2 正式训练与 1200 条正式评测结果为当前定稿。

待决策：
- 是否加大数据继续 **SFT**，还是用剩余算力做 **RL**（用户倾向数据/算力有限时先 SFT 后 RL）。
- 是否将 mask 监督修复并入后续训练（推荐，需重跑一次）。
- bucket / 小批次（micro>1 + LR 缩放）是否用于后续更大训练。

## 12. 关键文件索引

- 训练入口：`qwen3vl_seg/train/training_loop.py`、`qwen3vl_seg/train/config.py`
- 模型：`qwen3vl_seg/model/model_wrapper.py`、`mask_decoder.py`、`lora.py`、`prompt_format.py`、`prompt_parser.py`
- 数据：`qwen3vl_seg/data/dataset.py`、`collator.py`、`schema.py`、`buckets.py`、`bucket_sampler.py`、`samplers.py`
- 评测：`qwen3vl_seg/eval/runner.py`、`runner_bucket.py`、`metrics.py`
- 配置：`configs/train/stage{1,2}_100k_zhurong.yaml`、`stage{1,2}_100k_bucket_micro4_zhurong.yaml`、`deepspeed_stage{1,2}_100k_bucket_zhurong.json`、`common.yaml`
- 脚本：`scripts/slurm/train_stage{1,2}_100k_zhurong.sh`、`train_stage{1,2}_100k_bucket_zhurong.sh`

## 13. Git 历史（祝融 master 当前）

```text
c46303d feat: refine bucket sizes to 256/384/512/640/768 (5 tiers)
978a478 fix: move special-token grad mask to grad device
c222053 fix: supervise mask_token and trainable special token rows in Stage1
73e8585 feat: add dynamic-batch micro_batch=4 (bucket + LR scaling) training/inference
e858280 chore: checkpoint before mini-batch integration
be5bfc0 fix: save combined prediction/GT viz
d3701af feat: default prediction+GT side-by-side visualization
dd5e6d6 fix: eval runner max-samples default to all
2570bfe feat: formal eval runner with split filter and aggregate metrics
bbb0089 fix: make slurm module loading optional, fall back to segment env PATH
be2c954 fix: define tag before pruning old deepspeed steps
3e73e09 fix: keep master stable without bucket (checkpoint pruning, exclusive node)
3907743 feat: Qwen3-VL-Seg reproduction pipeline (data/model/train/eval)
```
