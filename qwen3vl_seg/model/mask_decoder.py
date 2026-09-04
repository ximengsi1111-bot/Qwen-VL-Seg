from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MaskDecoderConfig:
    visual_hidden_dim: int = 2560
    text_hidden_dim: int = 2560
    hidden_dim: int = 256
    num_heads: int = 8
    feedforward_dim: int = 1024
    num_decoder_layers: int = 2
    mask_passes: int = 2
    box_expand: float = 0.15
    gate_alpha: float = 20.0
    eps: float = 1e-6
    stride: int = 8


class SpatialFeatureInjector(nn.Module):
    """Near-identity spatial adapter for one multi-scale visual feature."""

    def __init__(self, in_channels: int, hidden_dim: int, s: float = 1e-3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.s = nn.Parameter(torch.full((), s))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        detail = F.gelu(self.dwconv(self.group_norm(x)))
        return x + self.s * detail


class MultiScaleVisualFusion(nn.Module):
    def __init__(self, visual_hidden_dim: int, hidden_dim: int):
        super().__init__()
        self.injectors = nn.ModuleList(
            [
                SpatialFeatureInjector(visual_hidden_dim, hidden_dim)
                for _ in range(4)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

    def forward(self, visual_features: list[torch.Tensor]) -> torch.Tensor:
        aligned = []
        for feature, injector in zip(visual_features, self.injectors):
            if len(aligned):
                feature = F.interpolate(
                    feature, size=aligned[0].shape[-2:], mode="bilinear", align_corners=False
                )
            aligned.append(injector(feature))
        return self.fuse(torch.cat(aligned, dim=1))


class FourierPositionalEncoding(nn.Module):
    """Fourier features for one scalar coordinate."""

    def __init__(self, num_frequencies: int = 32):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_frequencies, dtype=torch.float)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)
        freqs = self.freqs.to(x.device)
        angles = x.unsqueeze(-1) * freqs  # (..., 1, F)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(
            start_dim=-2
        )


class BoxEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, num_frequencies: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies
        self.pe = FourierPositionalEncoding(num_frequencies)
        in_dim = 4 * num_frequencies * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        # boxes: B,N,4 in normalized [0,1] as x1,y1,x2,y2
        x1 = boxes[..., 0:1]
        y1 = boxes[..., 1:2]
        width = (boxes[..., 2:3] - boxes[..., 0:1]).clamp_min(1e-6)
        height = (boxes[..., 3:4] - boxes[..., 1:2]).clamp_min(1e-6)
        values = torch.cat(
            [
                x1,
                y1,
                0.2 * width.log() + 0.5,
                0.2 * height.log() + 0.5,
            ],
            dim=-1,
        )  # B,N,4
        encoded = self.pe(values.unsqueeze(-1))  # B,N,4,2F -> flatten via mlp
        encoded = encoded.view(*values.shape[:-1], 4 * self.num_frequencies * 2)
        return self.mlp(encoded)


class ObjectQueryInitializer(nn.Module):
    def __init__(
        self,
        text_hidden_dim: int,
        hidden_dim: int,
        num_frequencies: int = 32,
    ):
        super().__init__()
        self.box_embed = BoxEmbedding(hidden_dim, num_frequencies)
        self.text_proj = nn.Linear(text_hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        seg_features: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(self.box_embed(boxes) + self.text_proj(seg_features))


class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, feedforward_dim: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        x = query
        x = x + self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False
        )[0]
        x = x + self.cross_attn(
            self.norm2(x), memory, memory, need_weights=False
        )[0]
        return x + self.ffn(self.norm3(x))


class PixelShuffleStage(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv(x))
        x = F.pixel_shuffle(x, upscale_factor=2)
        return self.norm(x)


class ShallowStem(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, 64)
        self.norm2 = nn.GroupNorm(8, 128)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.conv1(image)))
        x = F.gelu(self.norm2(self.conv2(x)))
        return F.gelu(self.conv3(x))


class SoftBoxGate(nn.Module):
    """Differentiable soft spatial gate from expanded normalized boxes."""

    def __init__(self, alpha: float = 20.0, expand: float = 0.15):
        super().__init__()
        self.alpha = alpha
        self.expand = expand

    @staticmethod
    def _line_gate(coord: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float):
        # a,b normalized, coord normalized in [0,1]
        return torch.sigmoid(alpha * (coord - a)) * torch.sigmoid(alpha * (b - coord))

    def forward(
        self,
        boxes: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        device = boxes.device
        if boxes is None or boxes.numel() == 0:
            return torch.ones(1, 1, height, width, device=device)
        yy = (torch.arange(height, device=device, dtype=boxes.dtype) + 0.5) / height
        xx = (torch.arange(width, device=device, dtype=boxes.dtype) + 0.5) / width
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]
        w = (x2 - x1).clamp_min(1e-6)
        h = (y2 - y1).clamp_min(1e-6)
        x1p = (x1 - self.expand * w).clamp(0, 1)
        x2p = (x2 + self.expand * w).clamp(0, 1)
        y1p = (y1 - self.expand * h).clamp(0, 1)
        y2p = (y2 + self.expand * h).clamp(0, 1)
        gates = []
        for b in range(boxes.shape[0]):
            per_box = []
            for n in range(boxes.shape[1]):
                gx = self._line_gate(
                    xx[None, :], x1p[b, n], x2p[b, n], self.alpha
                )
                gy = self._line_gate(
                    yy[:, None], y1p[b, n], y2p[b, n], self.alpha
                )
                per_box.append(gy * gx)
            gates.append(torch.stack(per_box, dim=0).amax(dim=0))
        return torch.stack(gates, dim=0).unsqueeze(1)


class DynamicMaskHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, query: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        kernels = self.proj(query)  # B,N,D
        return torch.einsum("bnd,bdhw->bnhw", kernels, fused)


class MaskRefinement(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        mask_logits: torch.Tensor,
        fused: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        soft = torch.sigmoid(mask_logits)
        numerator = torch.einsum("bnhw,bdhw->bnd", soft, fused)
        denominator = soft.sum(dim=(-1, -2)).unsqueeze(-1) + eps
        pooled = numerator / denominator
        return self.norm(query + self.proj(pooled))


class BoxGuidedMaskDecoder(nn.Module):
    """Box-guided mask decoder from the paper with two mask passes."""

    def __init__(self, config: MaskDecoderConfig | None = None):
        super().__init__()
        config = config or MaskDecoderConfig()
        self.config = config
        self.visual_fusion = MultiScaleVisualFusion(
            config.visual_hidden_dim, config.hidden_dim
        )
        self.visual_memory_proj = nn.Conv2d(
            config.text_hidden_dim, config.hidden_dim, kernel_size=1
        )
        self.memory_pos = nn.Parameter(
            torch.zeros(1, config.hidden_dim, 1, 1)
        )
        self.query_init = ObjectQueryInitializer(
            config.text_hidden_dim, config.hidden_dim
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.feedforward_dim,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.stem = ShallowStem(config.hidden_dim)
        self.upsample = nn.Sequential(
            PixelShuffleStage(config.hidden_dim),
            PixelShuffleStage(config.hidden_dim),
        )
        self.gate = SoftBoxGate(config.gate_alpha, config.box_expand)
        self.mask_head = DynamicMaskHead(config.hidden_dim)
        self.refine = MaskRefinement(config.hidden_dim)
        self.iou_head = nn.Linear(config.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keep the decoder numerically tame at initialization.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        visual_features: list[torch.Tensor],
        mm_features: torch.Tensor,
        seg_features: torch.Tensor,
        boxes: torch.Tensor,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # visual_features: list of B,C,H,W at merged visual resolution
        # mm_features: B,D_text,H,W at merged visual resolution
        # seg_features: B,num_instances,D_text
        # boxes: B,num_instances,4 normalized
        fused = self.visual_fusion(visual_features)
        b, _, fh, fw = fused.shape
        mem = fused + self.visual_memory_proj(mm_features) + self.memory_pos
        memory = mem.flatten(start_dim=2).permute(0, 2, 1)

        query = self.query_init(seg_features, boxes)
        for block in self.blocks:
            query = block(query, memory)

        fused_up = fused
        for stage in self.upsample:
            fused_up = stage(fused_up)
        image_feature = self.stem(image)
        image_feature = F.interpolate(
            image_feature,
            size=fused_up.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate = self.gate(boxes, fused_up.shape[-2], fused_up.shape[-1])
        pixel_feature = fused_up + gate * image_feature

        logits = self.mask_head(query, pixel_feature)
        for _ in range(max(1, self.config.mask_passes) - 1):
            query = self.refine(
                query, logits, pixel_feature, self.config.eps
            )
            logits = self.mask_head(query, pixel_feature)
        iou = self.iou_head(query).squeeze(-1)
        return {"mask_logits": logits, "iou_scores": iou}


def num_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
