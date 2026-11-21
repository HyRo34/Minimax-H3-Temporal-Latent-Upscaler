"""3D backbone adapted from LBH-123-AI. See LICENSES/LBH-MIT.txt."""
import re

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file


LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD  = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std

def normalization(channels):
    return nn.GroupNorm(32, channels)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h

class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for block_index in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and block_index % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for block_index in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and block_index % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)


def _extract_upscaler_sd(sd):
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd

def _detect_arch(sd):
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m: temporal_out_indices.add(int(m.group(1)))

    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    return cfg


def load_resizer(checkpoint_path, device):
    state = _extract_upscaler_sd(load_file(str(checkpoint_path), device="cpu"))
    if "t_embed.0.weight" not in state:
        raise ValueError("Select a George0667 v2 temporal checkpoint (the filename contains '_temporal_').")
    dtype = state["conv_in.weight"].dtype
    with torch.device("meta"):
        model = LatentResizer3D(**_detect_arch(state))
        model.t_embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
    model.load_state_dict(state, assign=True, strict=True)
    del state
    model.to(device).eval()
    return model, dtype


def expand_video(video, target_tokens, ratio, device, motion, spatial_scale, target_hw, loaded):
    model, dtype = loaded
    mean, std = _make_norm_tensors(device, dtype)
    features = (video.to(device=device, dtype=dtype) - mean) / std
    spatial = model.embed(torch.full((1, 1), spatial_scale - 1.0, dtype=dtype, device=device))
    temporal = model.t_embed(torch.full((1, 1), ratio - 1.0, dtype=dtype, device=device))
    embedding = (spatial + temporal).expand(features.shape[0], -1)
    features = model.conv_in(features)
    for block in model.in_blocks:
        features = block(features, embedding) if isinstance(block, ResBlockEmb3D) else block(features)

    if ratio != 1:
        centers = torch.tensor(motion._token_centers(video.shape[2]), dtype=torch.float64)
        targets = torch.tensor(motion._token_centers(target_tokens), dtype=torch.float64) / ratio
        targets = targets.clamp(centers[0], centers[-1])
        hi = torch.searchsorted(centers, targets, right=True).clamp(1, len(centers) - 1)
        lo = hi - 1
        weight = (targets - centers[lo]) / (centers[hi] - centers[lo])
        weight = weight.to(device=device, dtype=dtype).view(1, 1, -1, 1, 1)
        features = features.index_select(2, lo.to(device)).lerp(features.index_select(2, hi.to(device)), weight)
    if target_hw is not None and tuple(features.shape[3:]) != target_hw:
        features = F.interpolate(features, size=(features.shape[2], *target_hw), mode="trilinear", align_corners=False)
    for block in model.out_blocks:
        features = block(features, embedding) if isinstance(block, ResBlockEmb3D) else block(features)
    features = model.conv_out(F.silu(model.norm_out(features)))
    return (features * std + mean).to(video)
