# 仓库布局与同步规范

## 数据流水线（先看这里）

- **248k 指的是 SFT 数据量**（248,108 条），不是 GRPO 数据量。
- SFT 先于 GRPO：SFT 产出 `stage1-248k` / `stage2-248k`，合并为 `model/grpo-policy-248k`。
- **所有 GRPO 分支都从 248k-SFT 的 policy 出发**，分支之间的差别是 **GRPO 数据**，不是 SFT 数据。
- `/mingli01/data/xyk/grpo/` 下的 `train248k_vllm_70k_*` 读作「248k-SFT 基座 + 70k GRPO 数据」。

| 分支 | SFT 基座 | 该分支的 GRPO 数据 |
|---|---|---|
| `sft-248k` | 248k SFT 本体 | 无（只做 SFT） |
| `grpo-hard-aug` | 248k SFT | `grpo/rl_hard.jsonl`：从 248k SFT train 采 miou 0.2–0.5 的难例 + 增强；非 vLLM，4 卡 |
| `grpo-70k` | 248k SFT | 70k 子集；vLLM rollout |
| `grpo-onpolicy-70k` | 248k SFT | 70k 子集；on-policy，checkpoint-200 评测 |
| `grpo-phrasecut-50k` | 248k SFT | 混合 50k：PhraseCut 30k + RefCOCO/+/g 12k + COCO/LVIS 8k |

## 分支（GitHub: ximengsi1111-bot/Qwen-VL-Seg）

| 分支 | 内容 | 历史名 |
|---|---|---|
| `sft-248k` | 248k SFT 主线（Qwen3-VL-4B + Box-Guided Mask Decoder） | `master` |
| `grpo-hard-aug` | 248k-SFT 基座 + rl_hard.jsonl 的 GRPO（4 卡、rank-local reward、冻结 mask decoder 批处理） | `codex/grpo` → `grpo-248k` |
| `grpo-70k` | 70k GRPO（vLLM rollout） | `codex/grpo-vllm` / 旧 GitHub `grpo` |
| `grpo-onpolicy-70k` | 70k on-policy + checkpoint-200 评测 | `codex/grpo-on-policy-reward` |
| `grpo-phrasecut-50k` | 混合 50k 数据 + Hungarian F1 reward + 评测口径修正（最新） | `codex/grpo-phrasecut-data` |

祖先关系：`grpo-70k` → `grpo-onpolicy-70k`；`grpo-hard-aug` → `grpo-phrasecut-50k`；
`grpo-phrasecut-50k` 同时包含 `grpo-hard-aug` 与 `grpo-70k` 两条线。

> `master` 目前在 GitHub 上仍作为 `sft-248k` 的别名保留（二者指向同一 commit）。
> 待仓库默认分支切换为 `sft-248k` 后即可删除 `master`。

## 祝融上的 checkout

| 路径 | 分支 |
|---|---|
| `/mingli01/project/xiyongkai/qwen3vl-seg` | `grpo-hard-aug`（主库，`.git` 在此） |
| `/mingli01/project/xiyongkai/qwen3vl-seg-phrasecut` | `grpo-phrasecut-50k`（worktree） |
| `/mingli01/project/xiyongkai/qwen3vl-seg-vllm` | `grpo-onpolicy-70k`（worktree） |

## 同步流程

祝融是唯一事实源，所有改动只在祝融 commit。

祝融没有 GitHub 凭据（`git config --global credential.helper` 为空），推送走本地中转：

```bash
# 本地镜像仓 D:\XYK\Qwen-VL-Seg-backup-20260915\repo
git fetch zhurong '+refs/heads/*:refs/remotes/zhurong/*'
git push origin <branch>
```

本地工作副本：`D:\XYK\Qwen-VL-Seg\repo`（GitHub 克隆，含全部 5 条分支）。

## 资源

- 数据/权重/评测结果：`/mingli01/data/xyk`，索引见该目录 `README.md`
- 评测结果清单：`/mingli01/data/xyk/runs/eval/INDEX.md`
- 本地归档（2026-10-18 整理）：`D:\XYK\_archive_20261018`

## 约定

- 大文件（数据、权重、predictions*.jsonl）不进 git
- 新脚本一律放 `scripts/` 下，在祝融 commit 后推送