# PhraseCut GRPO 数据实验

## 目标

用 PhraseCut 的自由形式指代表达替换原先由 35k 基础样本重复增强得到的 70k GRPO 数据，
验证 GRPO 效果不佳是否主要来自数据重复与表达分布过窄。

## 现有 70k 数据审计

- 70,090 条记录，只有 35,045 个基础图像样本。
- `hflip/color/noise` 使每条基础样本平均出现约 2 次。
- COCO/LVIS 类别提示约占 82.5%，RefCOCO 系列 phrasal 约占 17.4%。
- 因此原先的 70k 主要是同语义样本的增强变体，而非 70k 个独立场景或表达。

## PhraseCut 数据

- 官方仓库：https://github.com/ChenyunWu/PhraseCutDataset
- 训练集：71,746 张图像、310,816 个 phrase-region 对。
- 每个 phrase 自带 Visual Genome polygon 级区域标注。
- 数据镜像：`https://hf-mirror.com/datasets/junbeomHong/phrasecut/resolve/main/phrasecut.tar.gz`
- 数据根目录：`/mingli01/data/xyk/grpo/newdata/phrasecut`
- 解压后数据根：`.../extracted/LISA/PhraseCutDataset/data/VGPhraseCut_v0`

## Pilot 构造规则

- 从官方训练集中均匀选择图像。
- 每张图像最多取一个 phrase，避免用增强扩大样本数。
- 优先选择结构完整、mask 非空且面积不过极端的任务。
- 不做难度筛选，不优先保留难例。
- 输出格式与当前 GRPO JSONL 保持一致。

输出：

- JSONL：`processed/phrasecut_5k_pilot.jsonl`
- Masks：`processed/masks/train/`
- Stats：`processed/phrasecut_5k_pilot_stats.json`

## 命令

```bash
cd /mingli01/project/xiyongkai/qwen3vl-seg-phrasecut

python scripts/data/build_phrasecut_jsonl.py \
  --root /mingli01/data/xyk/grpo/newdata/phrasecut/extracted/LISA/PhraseCutDataset/data/VGPhraseCut_v0 \
  --output /mingli01/data/xyk/grpo/newdata/phrasecut/processed/phrasecut_5k_pilot.jsonl \
  --stats /mingli01/data/xyk/grpo/newdata/phrasecut/processed/phrasecut_5k_pilot_stats.json \
  --mask-dir /mingli01/data/xyk/grpo/newdata/phrasecut/processed/masks \
  --num-samples 5000 --workers 16 --seed 42

python scripts/data/validate_phrasecut_jsonl.py \
  /mingli01/data/xyk/grpo/newdata/phrasecut/processed/phrasecut_5k_pilot.jsonl \
  --check-images --check-masks

sbatch scripts/slurm/vllm_grpo_phrasecut_5k_200.sh
```

## 训练对照

- 使用当前 `codex/grpo-on-policy-reward` 的 live-policy mask reward 路径。
- 6 卡，`NUM_GENERATIONS=4`，`GEN_BATCH=96`，`PER_DEVICE_BATCH=16`。
- Pilot 训练 `200` 步，结果写入 `runs/phrasecut_5k_200`。
- 正式 70k 只有在本 pilot 的 reward、显存、吞吐和日志均正常后启动。
