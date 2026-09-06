"""Run ms-swift RLHF (GRPO) in-process with the project's SegIoU reward.

Registers the custom reward into ``swift.rewards.orms`` at runtime so we do
NOT modify the installed site-packages. CLI args are read from ``sys.argv``.
"""
import sys

# Ensure the project is importable.
_PROJ = "/mingli01/project/xiyongkai/qwen3vl-seg"
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


def _register_seg_iou_reward() -> None:
    """Register SegIoUReward into ms-swift's reward registry (orms)."""
    try:
        from qwen3vl_seg.grpo.reward import SegIoUReward
        import swift.rewards.orm as _orm
        _orm.orms.setdefault("seg_iou", SegIoUReward)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not register seg_iou reward: {exc}")


if __name__ == "__main__":
    _register_seg_iou_reward()
    from swift.pipelines.train.rlhf import rlhf_main
    rlhf_main()