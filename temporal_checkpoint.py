"""Run the H3 temporal/spatial 3D latent resizer on ComfyUI core's optimized channels-last slabbed kernel path."""
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.storage
import comfy.utils
from comfy.ldm.minimax.latent_upscaler import (
    MiniMaxH3LatentUpscaler,
    clip_stats,
    conv3d_cl,
    norm_silu,
    slabs,
)
from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD


_RESIZER_CACHE = {}


def _strip_prefix(sd):
    for key in ("state_dict", "model", "upscaler"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    for prefix in ("model.", "upscaler.", "module."):
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    return sd


def _resample_temporal(features, source_times, target_times):
    device = features.device
    src_t = torch.as_tensor(source_times, dtype=torch.float64)
    dst_t = torch.as_tensor(target_times, dtype=torch.float64).clamp(src_t[0], src_t[-1])
    hi = torch.searchsorted(src_t, dst_t, right=True).clamp(1, src_t.numel() - 1)
    lo = hi - 1
    fraction = ((dst_t - src_t[lo]) / (src_t[hi] - src_t[lo])).to(device=device, dtype=features.dtype).view(1, 1, -1, 1, 1)
    lo_dev = lo.to(device)
    hi_dev = hi.to(device)
    # Resample per-slab if large, preserving channels_last_3d for downstream convs.
    out = features.index_select(2, lo_dev).lerp(features.index_select(2, hi_dev), fraction)
    return out.contiguous(memory_format=torch.channels_last_3d)


class MiniMaxH3TemporalLatentUpscaler(MiniMaxH3LatentUpscaler):
    """Core MiniMaxH3LatentUpscaler extended with the optional temporal scale embedding (`t_embed`)."""

    def __init__(self, in_channels=24, channels=512, in_blocks=12, out_blocks=12, temporal_kernel=5,
                 has_temporal=True, dtype=None, device=None, operations=None):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            in_blocks=in_blocks,
            out_blocks=out_blocks,
            temporal_kernel=temporal_kernel,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.has_temporal = has_temporal
        if has_temporal:
            self.t_embed = nn.Sequential(
                operations.Linear(1, 64, dtype=dtype, device=device),
                nn.SiLU(),
                operations.Linear(64, 64, dtype=dtype, device=device),
            )
        else:
            self.t_embed = None

    def memory_required(self, shape, size):
        n, _, t_in, h_in, w_in = shape
        t_out, h_out, w_out = size
        t_max = max(t_in, t_out)
        hw_max = max(h_in * w_in, h_out * w_out)
        return n * self.conv_in.out_channels * (2 * t_max + 96) * hw_max * self.dtype.itemsize

    def forward_temporal(self, x, target_hw=None, spatial_scale=1.0, ratio=1.0,
                         needs_temporal=False, source_times=None, target_times=None,
                         uniform_first=False, uniform_times=None, target_source_times=None):
        if target_hw is not None:
            eff_spatial_scale = (target_hw[0] / x.shape[3] + target_hw[1] / x.shape[4]) / 2.0
        else:
            eff_spatial_scale = float(spatial_scale)

        mean = torch.tensor(LATENTS_MEAN, device=x.device, dtype=x.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENTS_STD, device=x.device, dtype=x.dtype).view(1, -1, 1, 1, 1)
        # Trained on VAE latents normalized a second time with the same stats.
        x = (x - mean) / std

        spatial = self.embed(torch.full((1, 1), eff_spatial_scale - 1.0, device=x.device, dtype=x.dtype))
        if self.t_embed is not None:
            temporal = self.t_embed(torch.full((1, 1), float(ratio) - 1.0, device=x.device, dtype=x.dtype))
            emb = spatial + temporal
        else:
            emb = spatial

        x = self.conv_in(x).contiguous(memory_format=torch.channels_last_3d)
        x = self.run_blocks(self.in_blocks, x, emb)
        if needs_temporal:
            x = _resample_temporal(x, source_times, target_times)
        if target_hw is not None and tuple(x.shape[3:]) != target_hw:
            x = F.interpolate(
                x, size=(x.shape[2], *target_hw), mode="trilinear", align_corners=False,
            ).contiguous(memory_format=torch.channels_last_3d)
        x = self.run_blocks(self.out_blocks, x, emb)

        out = torch.empty((x.shape[0], self.conv_out.out_channels) + tuple(x.shape[2:]), device=x.device, dtype=x.dtype)
        stats = clip_stats(x, self.norm_out)
        for t0, t1, a, b in slabs(x.shape[2], 1):
            out[:, :, t0:t1] = conv3d_cl(
                self.conv_out, norm_silu(x[:, :, a:b], self.norm_out, stats)
            )[:, :, t0 - a:t1 - a]
        del x
        if uniform_first:
            out = _resample_temporal(out, uniform_times, target_source_times).contiguous()
        return torch.addcmul(mean, out, std)


def load_resizer(checkpoint_path, device=None):
    path_str = str(Path(checkpoint_path).resolve())
    mtime = os.path.getmtime(path_str)
    cache_key = (path_str, mtime)
    cached = _RESIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    sd = _strip_prefix(comfy.utils.load_torch_file(path_str, safe_load=True))
    fast_disk = comfy.storage.state_dict_fast_disk(sd)
    dtype = sd["conv_in.weight"].dtype
    has_temporal = "t_embed.0.weight" in sd
    model = MiniMaxH3TemporalLatentUpscaler(
        in_channels=sd["conv_in.weight"].shape[1],
        channels=sd["conv_in.weight"].shape[0],
        in_blocks=len({k.split(".")[1] for k in sd if k.startswith("in_blocks.") and ".in_layers." in k}),
        out_blocks=len({k.split(".")[1] for k in sd if k.startswith("out_blocks.") and ".in_layers." in k}),
        temporal_kernel=sd["in_blocks.1.dwconv.weight"].shape[2],
        has_temporal=has_temporal,
        dtype=dtype,
        operations=comfy.ops.disable_weight_init,
    )
    model.requires_grad_(False).eval()
    comfy.model_management.archive_model_dtypes(model)
    load_device = device if device is not None else comfy.model_management.get_torch_device()
    patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
        fast_disk=fast_disk,
    )
    model.load_state_dict(sd, assign=patcher.is_dynamic(), strict=True)
    del sd

    _RESIZER_CACHE.clear()
    result = (patcher, None, dtype)
    _RESIZER_CACHE[cache_key] = result
    return result


@torch.inference_mode()
def expand_video(video, target_tokens, ratio, checkpoint_path, device, motion,
                 target_source_times=None, uniform_first=False,
                 spatial_scale=1.0, target_hw=None, loaded=None):
    if loaded is None:
        patcher, _, dtype = load_resizer(checkpoint_path, device)
    else:
        patcher, _, dtype = loaded

    source_times = motion._token_centers(video.shape[2])
    needs_temporal = (
        uniform_first
        or target_source_times is not None
        or float(ratio) != 1.0
        or video.shape[2] != target_tokens
    )
    uniform_times = None
    if uniform_first:
        frames = motion._token_frame_spans(video.shape[2])[-1][1] + 1
        _, _, _, uniform_tokens, _ = motion.temporal_insert_map([ratio] * frames)
        uniform_times = [center / ratio for center in motion._token_centers(uniform_tokens)]
        target_times = uniform_times
    elif target_source_times is None:
        target_times = [center / ratio for center in motion._token_centers(target_tokens)]
    else:
        target_times = target_source_times

    out_t = len(target_times) if needs_temporal else video.shape[2]
    out_hw = target_hw if target_hw is not None else (video.shape[3], video.shape[4])
    mem_req = patcher.model.memory_required(video.shape, (out_t, out_hw[0], out_hw[1]))
    comfy.model_management.load_models_gpu([patcher], memory_required=mem_req)

    x = video.to(device=patcher.load_device, dtype=dtype)
    out = patcher.model.forward_temporal(
        x,
        target_hw=target_hw,
        spatial_scale=spatial_scale,
        ratio=ratio,
        needs_temporal=needs_temporal,
        source_times=source_times,
        target_times=target_times,
        uniform_first=uniform_first,
        uniform_times=uniform_times,
        target_source_times=target_source_times,
    )
    return out.to(video)
