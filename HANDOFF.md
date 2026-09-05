# Qwen3-VL-Seg — 会话交接（Handoff）

> 给新会话的开场：项目现状、环境、已完成成果、待决策项。连接祝融后建议先 `cd /mingli01/project/xiyongkai/qwen3vl-seg && git log --oneline -5` 确认分支与 HEAD（当前 `master`，HEAD=`3d5b1be`）。

## 1. 项目

从零（不复用现成 Qwen-VL-Seg 代码）复现《Qwen3-VL-Seg》：Qwen3-VL-4B + Box-Guided Mask Decoder。已完成环境、数据、模型、训练、评测全链路（100k 小数据版），目前处于正式训练后的收尾/决策阶段。

## 2. 机器与连接

| 名称 | 说明 |
|---|---|
| `kk` | 本地电脑，git 根 `D:\XYK\Qwen-VL-Seg`，代码在 `D:\XYK\Qwen-VL-Seg\qwen3vl-seg` |
| `祝融` | 主运行机 `172.29.15.117`（`choud-mgt-09`），SSH 私钥 `C:\Users\kk\.ssh\01.txt`，用户 `mingli`；`ssh -i C:\Users\kk\.ssh\01.txt mingli@172.29.15.117`；Slurm 分区 `gre`，调试固定 4 卡 |
| `DL` | AutoDL 备用 `connect.westc.seetacloud.com:16874`，密码 `3Zia47hcTjVS`，当前无卡 |

- 远程约定：命令带 `PYTHONNOUSERSITE=1`；用 `segment` conda 环境；DeepSpeed 4 卡 / ZeRO-2 / bf16。

## 3. 仓库（祝融 = 事实源，约定直接改祝融并 commit）

- 祝融代码：`/mingli01/project/xiyongkai/qwen3vl-seg`，分支 `master`。
- 当前 HEAD=`3d5b1be`（README）；其上是 `c46303d`（bucket 五档细化）；再上是 `978a478`/`c222053`（mask token 监督修复）。
- 本地 kk 分支：`codex/dynamic-batch-resolution`，含 bucket 研究；`eab013a` 的 bucket_sampler `while True` 循环修复**未并入祝融**（祝融上仍是单遍）。

## 4. 资产根：`/file_storage01/home/mingli/data/xyk`

- 数据：`build100k/datasets/small/samples.jsonl`，101200 样本 / 26304 图。
  - train 100000：RefCOCO / RefCOCO+ / RefCOCOg 各 30000 + COCO / LVIS 各 5000。
  - val 600；test / testA / testB 各 200。
- 权重：`checkpoints/small/stage1-100k`、`stage2-100k`；基座 `/mingli01/models/Qwen3-VL-4B-Instruct`。
- 结果：`runs/small/stage{1,2}-100k/metrics.jsonl`、`runs/eval/stage2-100k-formal/`。
- 统计：`build100k/stats.json`；图像清单 `image_manifest.jsonl`。

## 5. 已完成成果

- 100k 正式训练（无桶，micro_batch=1）：
  - Stage1 作业 `94855`：12500 步，末步 total 0.712 / text 0.568 / seg 0.142 / 显存 54.23GB。
  - Stage2 作业 `95157`：6250 步，末步 total 0.423 / text 0.296 / seg 0.126 / 显存 68.76GB。
- 正式评测 `95280`（1200 条）：mIoU 0.677、cIoU 0.663、strict P@0.5 0.835。
  - 按子集 mIoU：RefCOCO val 0.689 / testA 0.707 / testB 0.693；RefCOCO+ 0.653 / 0.698 / 0.647；RefCOCOg 0.671 / test 0.677。
- 可视化 `95389`：预测 | 真值 左右拼接，1200 张，默认开启。
- 明显低于论文（RefCOCO cIoU 82.3 / 83.7 / 79.1），主因：数据规模（100k vs SA1B-ORS）、训练仅约 2/1 epoch、mask token 输出不稳定。

## 6. 实验结论（dynamic-batch / bucket）

- 动态分辨率下 micro_batch>1 需 LR 按 batch 倍数线性放大（可补回约 0.12，残差 0.02–0.03）；细化桶无助于精度；正式训练倾向精度优先 = micro1，bucket 代码保留为可选能力。
- 当前 bucket 档位已细化 `[256, 384, 512, 640, 768]`（5 档，commit `c46303d`），`max_pixels=589824` 不变。
- bucket_micro4 配置：`configs/train/stage{1,2}_100k_bucket_micro4_zhurong.yaml`（`use_bucket=true`、`micro_batch_size=4`、`gradient_accumulation_steps=1`、`global_batch_size=16`、LR 已按 4x 缩放）。

## 7. 文档与清理

- 根 `README.md`：`/mingli01/project/xiyongkai/qwen3vl-seg/README.md`（提交 `3d5b1be`，13 节，含成果/实验/结论）。
- `logs/` 已清理，仅保留 `94855`、`95157`、`95280`、`95389` 四个最终运行的 `.out/.err`。

## 8. 待决策 / 下一步

1. 加大数据继续 SFT，还是用剩余算力做 RL（倾向数据/算力有限时先 SFT 后 RL）。
2. 是否将 mask-token 监督修复（`c222053`/`978a478`）纳入重训（当前 100k 是在修复前完成的，需重跑才会生效）。
3. 后续更大训练是否使用 bucket / micro>1 + LR 缩放。

## 9. 关键入口

- 训练：`qwen3vl_seg/train/training_loop.py`、`train/config.py`
- 模型：`qwen3vl_seg/model/model_wrapper.py`、`mask_decoder.py`、`lora.py`
- 数据：`qwen3vl_seg/data/dataset.py`、`collator.py`、`buckets.py`、`bucket_sampler.py`
- 评测：`qwen3vl_seg/eval/runner.py`、`runner_bucket.py`、`metrics.py`
- 配置：`configs/train/common.yaml`、`stage{1,2}_100k_zhurong.yaml`、`stage{1,2}_100k_bucket_micro4_zhurong.yaml`
