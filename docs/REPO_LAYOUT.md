# 仓库布局与同步规范

## 分支（GitHub: ximengsi1111-bot/Qwen-VL-Seg）

| 分支 | 内容 | 说明 |
|---|---|---|
| `master` | 248k SFT 主线（Qwen3-VL-4B + Box-Guided Mask Decoder） | 稳定分支，不跑 GRPO |
| `grpo-70k` | 70k GRPO（vLLM rollout） | 原 `codex/grpo-vllm`，即旧 GitHub `grpo` |
| `grpo-248k` | 248k GRPO（4 卡、rank-local reward、冻结 mask decoder 批处理） | 原 `codex/grpo` |
| `grpo-onpolicy-70k` | 70k on-policy + checkpoint-200 评测 | 原 `codex/grpo-on-policy-reward` |
| `grpo-phrasecut-50k` | 混合 50k 数据 + Hungarian F1 reward + 评测口径修正 | 原 `codex/grpo-phrasecut-data`，**当前最新** |

祖先关系：`grpo-70k` → `grpo-onpolicy-70k`；`grpo-248k` → `grpo-phrasecut-50k`；
`grpo-phrasecut-50k` 同时包含 `grpo-248k` 与 `grpo-70k` 两条线。

## 祝融上的 checkout

| 路径 | 分支 |
|---|---|
| `/mingli01/project/xiyongkai/qwen3vl-seg` | `grpo-248k`（主库，`.git` 在此） |
| `/mingli01/project/xiyongkai/qwen3vl-seg-phrasecut` | `grpo-phrasecut-50k`（worktree） |
| `/mingli01/project/xiyongkai/qwen3vl-seg-vllm` | `grpo-onpolicy-70k`（worktree） |

## 同步流程

祝融是唯一事实源，所有改动只在祝融 commit。

祝融没有 GitHub 凭据（`git config --global credential.helper` 为空），因此推送走本地中转：

```bash
# 本地镜像仓 D:\XYK\Qwen-VL-Seg-backup-20260915\repo
git fetch zhurong '+refs/heads/*:refs/remotes/zhurong/*'
git push origin <branch>
```

祝融侧可直接 `git fetch origin` 读取 GitHub（已验证连通）。

## 资源

- 数据/权重/评测结果：`/mingli01/data/xyk`，索引见该目录下 `README.md`
- 评测结果清单：`/mingli01/data/xyk/runs/eval/INDEX.md`
- 本地归档（2026-10-18 整理）：`D:\XYK\_archive_20261018`

## 约定

- 大文件（数据、权重、predictions*.jsonl）不进 git
- 新脚本一律放 `scripts/` 下并在祝融 commit 后推送