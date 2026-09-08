# GRPO 研究现状（Qwen3-VL-Seg）

> 目标：在 Qwen3-VL-4B + Box-Guided Mask Decoder 上，用 ms-swift 做 GRPO（box IoU + mask IoU 双奖励）。

## 1. 已完成的资产（都在 `/mingli01/data/xyk`）
| 项 | 位置 |
|---|---|
| 更大 SFT 数据（248,108 样本）| `xyk/datasets/small/` |
| GRPO 60k prompt 池 | `xyk/grpo/prompts.jsonl` |
| GRPO RLHF 数据（ms-swift 格式）| `xyk/grpo/rl_dataset.jsonl`、`rl_smoke.jsonl` |
| ms-swift policy（base+LoRA 合并+seg token）| `xyk/model/grpo-policy` |
| GRPO 环境 | `xyk/conda/ms-swift-grpo`（py3.10 / torch 2.6.0 / transformers 5.8.1 / ms-swift 4.5.2 / trl 0.28 / peft 0.19）|

## 2. GRPO 关键代码（`codex/grpo` 分支）
- reward：`qwen3vl_seg/grpo/reward.py`（`SegIoUReward`：box IoU + mask IoU，mask 用冻结 decoder）
- 运行入口：`qwen3vl_seg/grpo/run_rlhf.py`（进程内注册 `seg_iou` → 调 `swift.pipelines.train.rlhf.rlhf_main`，**不改 site-packages**）
- 数据格式化：`scripts/data/build_grpo_rl_dataset.py`
- policy 转换：`scripts/model/convert_stage2_policy.py`
- 冒烟启动：`scripts/slurm/rlhf_grpo_smoke.sh`（非 vLLM，GRPO）
- 评测/测试：`scripts/eval/{grpo_eval_iou,test_mask_iou}.py`

## 3. 非 vLLM GRPO 冒烟：已通过 ✅
`rlhf_grpo_smoke.sh`（1 GPU / G=4 / 1 epoch）：
```
reward 0.7916 | reward_std 0.0098 | frac_reward_zero_std 0 | memory 17.6 GiB
global_step 64/192 | Saving checkpoint → .../v22-.../checkpoint-64
```
- reward 非 0（box+mask 双奖励），advantage 正常，训练完成并保存 checkpoint。

## 4. 关键根因（踩坑记录）
- **GRPO 训练报 `StopIteration`（`model.device` 拿不到参数）**，最初误判为 vllm/torch 2.11 问题。
- **真正根因**：启动脚本里 vLLM 实验加的 `export SWIFT_SINGLE_DEVICE_MODE=1`（强制 `is_mp()=False`）+ `--exclusive`（占整节点 128 CPU），导致 ms-swift 的 GRPO loss 前向走错分支。
- **修复**：回退脚本——去掉 `SWIFT_SINGLE_DEVICE_MODE` / `NPROC_PER_NODE` / `--exclusive`（只占 1 GPU+4 CPU），保留 `--report_to none`。

## 5. vLLM 探索结论（暂搁置）
- 目标版本 **vllm 0.23.0**（官方支持 Qwen3-VL，且兼容 transformers 5.8.1 `!=5.0-5.5`）。
- 但 **PyPI 的 vllm 0.9/0.10 未打包 `qwen3_vl.py`；vllm 0.23.0 需 `transformers<5` / torch 2.11**，与本栈有冲突；且祝融/kk 对大文件下载反复卡顿。
- 因此 **vLLM 加速在"Qwen3-VL + 现有栈"下暂不可行**，已回退到非 vLLM（HF 生成）GRPO。vLLM 留待网络/环境条件成熟或单独建专门推理环境。

## 6. 下一步
- **上大规模 GRPO**：用 60k prompt 池 / 调大 `num_generations`（如 8）/ 更多步数，验证 mIoU 提升。
- 可先把 `use_vllm` 相关尝试记录在分支，不再追。