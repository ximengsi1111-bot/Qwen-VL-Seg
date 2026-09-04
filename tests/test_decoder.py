import torch

from qwen3vl_seg.model.mask_decoder import BoxGuidedMaskDecoder, MaskDecoderConfig


def test_decoder_single_and_multi_instance_shapes() -> None:
    config = MaskDecoderConfig(
        visual_hidden_dim=32,
        text_hidden_dim=32,
        hidden_dim=32,
        num_heads=4,
        feedforward_dim=128,
        num_decoder_layers=2,
        mask_passes=2,
    )
    decoder = BoxGuidedMaskDecoder(config)
    visual_features = [torch.randn(1, 32, 2, 2) for _ in range(4)]
    mm_features = torch.randn(1, 32, 2, 2)
    image = torch.randn(1, 3, 64, 64)

    for count in (1, 3):
        seg_features = torch.randn(1, count, 32)
        boxes = torch.tensor(
            [[[0.2 + i * 0.1, 0.2, 0.6 + i * 0.05, 0.7] for i in range(count)]],
            dtype=torch.float32,
        )
        output = decoder(visual_features, mm_features, seg_features, boxes, image)
        assert output["mask_logits"].shape == (1, count, 8, 8)
        assert output["iou_scores"].shape == (1, count)


def test_decoder_is_trainable_and_loss_finite() -> None:
    config = MaskDecoderConfig(
        visual_hidden_dim=16,
        text_hidden_dim=16,
        hidden_dim=16,
        num_heads=4,
        feedforward_dim=64,
        num_decoder_layers=1,
        mask_passes=1,
    )
    decoder = BoxGuidedMaskDecoder(config)
    output = decoder(
        [torch.randn(1, 16, 4, 4) for _ in range(4)],
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 2, 16),
        torch.rand(1, 2, 4),
        torch.randn(1, 3, 128, 128),
    )
    target = (torch.rand(1, 2, 16, 16) > 0.5).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["mask_logits"], target
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert next(decoder.parameters()).grad is not None
