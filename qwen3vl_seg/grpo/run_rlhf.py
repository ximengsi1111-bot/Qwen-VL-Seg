"""Run ms-swift RLHF (GRPO) in-process with the project's SegIoU reward.

Registers the custom reward into ``swift.rewards.orms`` at runtime so we do
NOT modify the installed site-packages. CLI args are read from ``sys.argv``.
"""
import os
import sys
from pathlib import Path

# Ensure this worktree's project is importable, not the main worktree.
_PROJ = str(Path(__file__).resolve().parents[2])
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


def _register_seg_iou_reward() -> None:
    """Register SegIoUReward into ms-swift's reward registry (orms)."""
    try:
        from qwen3vl_seg.grpo.reward import SegIoUReward as _BaseSegIoU
        import swift.rewards.orm as _orm
        class _SegIoU(_BaseSegIoU):
            def __init__(self, args=None, **kw):
                super().__init__(args, box_weight=0.3, mask_weight=0.7, **kw)
        _orm.orms.setdefault("seg_iou", _SegIoU)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not register seg_iou reward: {exc}")


def _register_on_policy_trainer() -> None:
    """Select the live-policy GRPO trainer when explicitly enabled."""
    enabled = os.environ.get("GRPO_ON_POLICY_REWARD", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    from swift.trainers.trainer_factory import TrainerFactory
    TrainerFactory.TRAINER_MAPPING["grpo"] = "qwen3vl_seg.grpo.trainer.OnPolicySegGRPOTrainer"
    print("[run_rlhf] using OnPolicySegGRPOTrainer", flush=True)


if __name__ == "__main__":
    _register_seg_iou_reward()
    _register_on_policy_trainer()
    from swift.pipelines.train.rlhf import rlhf_main
    rlhf_main()
