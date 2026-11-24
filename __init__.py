import gc
from itertools import accumulate
import json
import logging
import math
from pathlib import Path
import sys

import folder_paths
import torch
from comfy import model_management
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import ComfyExtension, io

TEMPORAL_MODELS = "h3_temporal_models"
SPATIAL_MODELS = "latent_upscale_models"
folder_paths.folder_names_and_paths.setdefault(TEMPORAL_MODELS, ([], {".safetensors"}))
folder_paths.add_model_folder_path(TEMPORAL_MODELS, str(Path(folder_paths.models_dir) / TEMPORAL_MODELS))

PREFERRED_CHECKPOINT = "minimax_h3_latent_upscaler_3d_conv_v2_temporal_bf16.safetensors"


def _is_h3_spatial_upscaler(path):
    """Inspect only the safetensors JSON header (<1ms) to confirm a pure H3 24-ch spatial upscaler."""
    try:
        with open(path, "rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            if header_len <= 0 or header_len > 16 * 1024 * 1024:
                return False
            header = json.loads(handle.read(header_len).decode("utf-8"))
    except Exception:
        return False

    def _strip(key):
        for prefix in ("model.", "upscaler.", "module."):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    meta = {_strip(k): v for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)}
    conv_in = meta.get("conv_in.weight")
    conv_out = meta.get("conv_out.weight")
    if not conv_in or not conv_out or "in_blocks.1.dwconv.weight" not in meta:
        return False
    if "t_embed.0.weight" in meta:
        return False
    in_shape = conv_in.get("shape") or ()
    out_shape = conv_out.get("shape") or ()
    return len(in_shape) == 5 and in_shape[1] == 24 and len(out_shape) == 5 and out_shape[0] == 24


def _spatial_checkpoint_options():
    names = folder_paths.get_filename_list(SPATIAL_MODELS) if SPATIAL_MODELS in folder_paths.folder_names_and_paths else []
    models = [name for name in names if name.lower().endswith(".safetensors")]
    return ["auto", *models, "same_as_temporal"]


def _resolve_spatial_checkpoint(selection):
    """Resolve a dedicated H3 spatial upscaler path, or (None, None) to use the temporal checkpoint."""
    choice = (selection or "auto").strip()
    if choice == "same_as_temporal":
        return None, None
    if choice != "auto":
        return folder_paths.get_full_path_or_raise(SPATIAL_MODELS, choice), choice

    def _rank(name):
        low = name.lower()
        score = (4 if ("minimax" in low and "h3" in low) else (2 if ("h3" in low or "minimax" in low) else 0))
        if "upscal" in low or "spatial" in low:
            score += 1
        if "temporal" in low:
            score -= 5
        return (-score, low)

    spatial_names = folder_paths.get_filename_list(SPATIAL_MODELS) if SPATIAL_MODELS in folder_paths.folder_names_and_paths else []
    temporal_names = folder_paths.get_filename_list(TEMPORAL_MODELS) if TEMPORAL_MODELS in folder_paths.folder_names_and_paths else []
    candidates = [(SPATIAL_MODELS, name) for name in sorted(spatial_names, key=_rank) if name.lower().endswith(".safetensors")]
    candidates.extend((TEMPORAL_MODELS, name) for name in sorted(temporal_names, key=_rank) if name.lower().endswith(".safetensors"))
    for folder, name in candidates:
        full_path = folder_paths.get_full_path(folder, name)
        if full_path and _is_h3_spatial_upscaler(full_path):
            return full_path, name
    return None, None


def _mainodes():
    import nodes

    insert = nodes.NODE_CLASS_MAPPINGS.get("H3TemporalInsert")
    if insert is None:
        raise RuntimeError("Install ComfyUI-MAINodes with H3TemporalInsert, then restart ComfyUI.")
    motion = sys.modules[insert.__module__]
    required = ("_token_centers", "_token_frame_spans", "temporal_insert_map")
    if not all(hasattr(motion, name) for name in required):
        raise RuntimeError("Update ComfyUI-MAINodes to a version with the H3 latent timeline helpers.")
    return insert, motion


def _video(samples):
    video = samples["samples"]
    if video.is_nested:
        video = video.unbind()[0]
    if video.ndim != 5 or video.shape[1] != 24:
        raise ValueError("Expected an H3 video latent shaped [batch, 24, time, height, width].")
    if video.shape[2] < 2 or (video.shape[2] - 2) % 5:
        raise ValueError("The H3 source latent must have 5k+2 time tokens.")
    return video


def _audio(samples):
    raw = samples["samples"]
    if getattr(raw, "is_nested", False) and hasattr(raw, "unbind"):
        parts = raw.unbind()
        if len(parts) > 1:
            return parts[1]
    return None


def _estimate_pass2_tokens(video, audio):
    vid_tokens = int(video.shape[2]) * ((int(video.shape[3]) + 1) // 2) * ((int(video.shape[4]) + 1) // 2)
    aud_tokens = int(audio.shape[-1]) if audio is not None else 0
    return vid_tokens + aud_tokens


def _disarm_dyrope():
    """Clear H3DyRoPE / H3TrueClock module globals when Pass 2 auto-skips in 0 steps."""
    for mod in tuple(sys.modules.values()):
        if mod is None:
            continue
        dyrope = getattr(mod, "_DYROPE", None)
        clock = getattr(mod, "_TRUE_CLOCK", None)
        if isinstance(dyrope, dict) and isinstance(clock, dict) and "active" in dyrope and "spans" in clock:
            clock["spans"] = None
            dyrope["active"] = False
            dyrope["mode"] = None
            dyrope["alt_angles"] = None
            dyrope["alt_table"] = None
            dyrope["sigma"] = None
            dyrope["sigma_max"] = None


def _stretch_audio_latent(src_audio, holds, target_len):
    """Map Pass-1 audio latent tokens onto the dilated video timeline via the hold map."""
    world_len = len(holds)
    dilated_len = sum(holds)
    src_len = src_audio.shape[-1]
    if src_len == target_len and all(hold == 1 for hold in holds):
        return src_audio.clone()
    holds_t = torch.tensor(holds, dtype=torch.float64)
    cum = torch.zeros(world_len + 1, dtype=torch.float64)
    cum[1:] = torch.cumsum(holds_t, dim=0)
    dil_frames = (torch.arange(target_len, dtype=torch.float64) + 0.5) * (dilated_len / float(target_len))
    idx = torch.searchsorted(cum[1:].contiguous(), dil_frames, right=True).clamp(0, world_len - 1)
    frac = (dil_frames - cum[idx]) / holds_t[idx].clamp_min(1.0)
    src_frames = idx.to(torch.float64) + frac
    src_pos = (src_frames * (src_len / float(world_len)) - 0.5).clamp(0.0, float(src_len - 1))
    lo = torch.floor(src_pos).to(torch.long)
    hi = (lo + 1).clamp(max=src_len - 1)
    weight = (src_pos - lo.to(torch.float64)).to(device=src_audio.device, dtype=src_audio.dtype)
    lo_dev = lo.to(src_audio.device)
    hi_dev = hi.to(src_audio.device)
    return src_audio.index_select(-1, lo_dev).lerp(src_audio.index_select(-1, hi_dev), weight)


def _auto_block_metrics(video):
    """Measure per-block sustained foreground motion, peak burst, and global mean velocity."""
    vid = video.detach().float()
    token_count = vid.shape[2]
    world_len = (token_count - 2) // 5 * 17 + 5
    block_count = (token_count - 2) // 5 + 1
    scores = []
    peak_steps = []
    global_means = []
    for block_idx in range(block_count):
        base = 5 * block_idx
        if block_idx < block_count - 1:
            diff_4f = (vid[:, :, base + 2:base + 5] - vid[:, :, base + 1:base + 4]).pow(2).mean(dim=1).sqrt()
            excess = sorted((
                float(torch.quantile(diff_4f[:, step].flatten(), 0.98) - torch.quantile(diff_4f[:, step].flatten(), 0.50))
                for step in range(3)
            ), reverse=True)
            peak_steps.append(excess[0])
            scores.append(0.60 * excess[0] + 0.40 * excess[1])
            global_means.append(float(diff_4f.mean()))
        else:
            if base == 0:
                if token_count >= 2:
                    diff_tail = (vid[:, :, 1:2] - vid[:, :, 0:1]).pow(2).mean(dim=1).sqrt() * (4.0 / 2.5)
                else:
                    scores.append(0.0)
                    peak_steps.append(0.0)
                    global_means.append(0.0)
                    continue
            else:
                diff_tail = (vid[:, :, base + 1:base + 2] - vid[:, :, base - 1:base]).pow(2).mean(dim=1).sqrt() * (4.0 / 5.0)
            flat = diff_tail.flatten()
            tail_score = float(torch.quantile(flat, 0.98) - torch.quantile(flat, 0.50))
            peak_steps.append(tail_score)
            scores.append(tail_score)
            global_means.append(float(diff_tail.mean()))
    return world_len, scores, peak_steps, global_means


def _auto_block_scores(video):
    """Score each 17-frame H3 block's foreground latent velocity without VAE decode."""
    world_len, scores, _, global_means = _auto_block_metrics(video)
    return world_len, scores, global_means


def _auto_motion_windows(video, abs_thresh=0.45, rel_ratio=0.55):
    """Detect high-motion 17-frame H3 blocks directly in latent space without VAE decode."""
    world_len, scores, peak_steps, global_means = _auto_block_metrics(video)
    clip_peak = max(peak_steps) if peak_steps else 0.0
    gate = max(abs_thresh, clip_peak * rel_ratio)
    burst_gate = max(0.75, gate + 0.10)
    windows = []
    last_block = len(scores) - 1
    for block_idx, (score, peak_step, gmean) in enumerate(zip(scores, peak_steps, global_means)):
        is_short_tail = (block_idx == last_block and (world_len - block_idx * 17) <= 5)
        score_gate = max(gate, 0.55) if is_short_tail else gate
        peak_gate = max(burst_gate, 0.85) if is_short_tail else burst_gate
        gmean_hit = gmean >= (1.05 if is_short_tail else 0.90)
        if score >= score_gate or peak_step >= peak_gate or gmean_hit:
            start = block_idx * 17
            end = min(world_len, (block_idx + 1) * 17)
            if windows and windows[-1][1] == start:
                windows[-1] = (windows[-1][0], end)
            else:
                windows.append((start, end))
    return windows, [round(score, 3) for score in scores], round(gate, 3)


def _auto_adaptive_block_dilations(video, low_thresh=0.45, high_thresh=0.75, rel_ratio=0.55):
    """Assign 1x (calm), 2x (moderate motion), or 3x (extreme motion) per 17-frame block."""
    world_len, scores, peak_steps, global_means = _auto_block_metrics(video)
    clip_peak = max(peak_steps) if peak_steps else 0.0
    low_gate = max(low_thresh, clip_peak * rel_ratio)
    high_gate = max(high_thresh, low_gate + 0.10)
    block_dilations = []
    windows = []
    last_block = len(scores) - 1
    for block_idx, (score, peak_step, gmean) in enumerate(zip(scores, peak_steps, global_means)):
        is_short_tail = (block_idx == last_block and (world_len - block_idx * 17) <= 5)
        score_low = max(low_gate, 0.55) if is_short_tail else low_gate
        peak_high = max(high_gate, 0.85) if is_short_tail else high_gate
        gmean_high = 1.25 if is_short_tail else 1.10
        gmean_low = 1.05 if is_short_tail else 0.90
        if peak_step >= peak_high or gmean >= gmean_high:
            dil = 3
        elif score >= score_low or gmean >= gmean_low:
            dil = 2
        else:
            dil = 1
        block_dilations.append(dil)
        if dil > 1:
            start = block_idx * 17
            end = min(world_len, (block_idx + 1) * 17)
            if windows and windows[-1][1] == start:
                windows[-1] = (windows[-1][0], end)
            else:
                windows.append((start, end))
    return block_dilations, windows, [round(score, 3) for score in scores], round(low_gate, 3), round(high_gate, 3)


def _repair_ranges(value, length):
    text = (value or "").strip()
    if text.lower() in ("all", "full", "*"):
        return [(0, length)]
    try:
        windows = sorted(tuple(int(part) for part in span.split("-")) for span in text.split(","))
    except ValueError as error:
        raise ValueError("Enter comma-separated source frame ranges like 5-17,18-22 or 'all'.") from error
    previous_end = 0
    for window in windows:
        if len(window) != 2:
            raise ValueError("Each repair range needs a start and an exclusive end.")
        start, end = window
        if not previous_end <= start < end <= length:
            raise ValueError(f"Repair ranges must be non-overlapping and inside the {length}-frame source.")
        previous_end = end
    return windows


def _oracle_windows(oracle_hold_map, length):
    data = json.loads(oracle_hold_map)
    holds = data.get("holds") if isinstance(data, dict) else None
    if not isinstance(holds, list) or len(holds) != length or any(type(hold) is not int or hold < 1 for hold in holds):
        raise ValueError(f"Oracle hold map must have {length} positive integer holds, one per source frame.")
    windows = []
    start = None
    for index in range(min(len(holds), length)):
        if holds[index] > 1:
            if start is None:
                start = index
        elif start is not None:
            windows.append((start, index))
            start = None
    if start is not None:
        windows.append((start, min(len(holds), length)))
    return windows


def _target_latent_hw(video, spatial_scale):
    if spatial_scale <= 1.0:
        return None
    px_height = video.shape[3] * 16
    px_width = video.shape[4] * 16
    target_px_h = max(32, int(round(px_height * spatial_scale / 32.0) * 32))
    target_px_w = max(32, int(round(px_width * spatial_scale / 32.0) * 32))
    return (target_px_h // 16, target_px_w // 16)


def _block_holds(length, windows, dilation):
    holds = [1] * length
    for start, end in windows:
        for frame in range(start // 17 * 17, min(length, math.ceil(end / 17) * 17)):
            holds[frame] = dilation
    return holds


def _block_mask(motion, base_video, latent, holds, spatial_scale=1.0, untouched_spatial_denoise=1.0):
    _, audio_mask = latent["noise_mask"].unbind()
    video, audio = latent["samples"].unbind()
    video_mask = torch.zeros_like(video[:, :1])
    frame_map = [frame for frame, hold in enumerate(holds) for _ in range(hold)]
    if spatial_scale > 1.0:
        untouched = float(untouched_spatial_denoise)
        video_mask.fill_(untouched)
        if untouched < 1.0:
            for token, (first, last) in enumerate(motion._token_frame_spans(video.shape[2])):
                support = frame_map[first:last + 1]
                if any(holds[frame] > 1 for frame in support):
                    video_mask[:, :, token] = 1.0
    else:
        for token, (first, last) in enumerate(motion._token_frame_spans(video.shape[2])):
            support = frame_map[first:last + 1]
            if any(holds[frame] > 1 for frame in support):
                video_mask[:, :, token] = 1.0
            if first == last and support[0] % 17 == 0:
                native = support[0] // 17 * 5
                video[:, :, token] = base_video[:, :, native]
                if (support[0] == 0
                        or (support[0] > 0 and holds[support[0] - 1] == 1)
                        or holds[support[0]] == 1):
                    video_mask[:, :, token] = 0.0
    latent["samples"] = NestedTensor((video, audio))
    latent["noise_mask"] = NestedTensor((video_mask, audio_mask))


class H3AdaptiveLatentRepair(io.ComfyNode):
    """Production node for H3 latent auto-motion repair and joint spatial+temporal upscaling."""

    MOTION_MODES = [
        "auto_adaptive (1x/2x/3x)",
        "auto_2x",
        "auto_3x",
        "full_2x",
        "full_3x",
        "manual",
    ]

    @classmethod
    def define_schema(cls):
        checkpoints = folder_paths.get_filename_list(TEMPORAL_MODELS) or [PREFERRED_CHECKPOINT]
        default_ckpt = PREFERRED_CHECKPOINT if PREFERRED_CHECKPOINT in checkpoints else checkpoints[0]
        spatial_options = _spatial_checkpoint_options()
        return io.Schema(
            node_id="H3AdaptiveLatentRepair",
            display_name="H3 Latent Auto Repair & Upscale",
            category="latent/minimax/motion",
            description="Auto-detects fast-motion 17-frame blocks directly in latent space, applies adaptive 1x/2x/3x temporal dilation and optional 1.5x-2.0x decoupled spatial upscaling (official H3 spatial upscaler when present), preserves first-pass audio, and aligns FL2V/I2V keyframes.",
            inputs=[
                io.Latent.Input("samples"),
                io.Conditioning.Input("conditioning"),
                io.Combo.Input("checkpoint", options=checkpoints, default=default_ckpt,
                               tooltip="Place the temporal checkpoint in ComfyUI/models/h3_temporal_models/ (George0667 v2 temporal BF16 or FP16)."),
                io.Combo.Input("motion_mode", options=cls.MOTION_MODES, default="auto_adaptive (1x/2x/3x)",
                               tooltip="auto_adaptive: 1x calm / 2x moderate / 3x extreme bursts. auto_2x/auto_3x: uniform dilation on hot blocks only. full_2x/full_3x: whole clip. manual: use manual_ranges."),
                io.Float.Input("spatial_scale", default=1.0, min=1.0, max=2.0, step=0.25,
                               tooltip="1.0 = same-res repair; 1.5 = 640x384->960x576; 2.0 = 640x384->1280x768."),
                io.Float.Input("untouched_spatial_denoise", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Denoise mask on untouched 1x blocks when spatial_scale > 1.0 (1.0 is sharpest and ~15% faster; lower values blend more raw resizer output)."),
                io.String.Input("manual_ranges", default="", optional=True,
                                tooltip="Used when motion_mode='manual' (e.g. '17-50') or optional threshold override (e.g. '0.45,0.75' for adaptive, '0.40' for auto_2x/3x)."),
                io.String.Input("oracle_hold_map", force_input=True, optional=True,
                                tooltip="Optional external hold_map (e.g. from H3 Jerk Oracle) to override motion_mode."),
                io.Combo.Input(
                    "spatial_checkpoint", options=spatial_options, default="auto", optional=True,
                    tooltip="Spatial upscaler used when spatial_scale > 1.0. 'auto' picks the official MiniMax H3 spatial upscaler from models/latent_upscale_models/ (clean 1.25x-2.0x grids, crisp keyframes, faster prep) and falls back to checkpoint if absent.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="samples"),
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="hold_map"),
                io.Int.Output(display_name="expanded_frames"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, samples, conditioning, checkpoint, motion_mode="auto_adaptive (1x/2x/3x)",
                spatial_scale=1.0, untouched_spatial_denoise=1.0,
                manual_ranges="", oracle_hold_map=None, spatial_checkpoint="auto"):
        insert, motion = _mainodes()
        video = _video(samples)
        source_audio = _audio(samples)
        world_len = (video.shape[2] - 2) // 5 * 17 + 5
        extra = (manual_ranges or "").strip()
        dilation = 3 if motion_mode in ("auto_3x", "full_3x") else 2
        factors = None
        if oracle_hold_map and oracle_hold_map.strip():
            ranges = _oracle_windows(oracle_hold_map, world_len)
        elif motion_mode == "auto_adaptive (1x/2x/3x)":
            thresholds = [float(value) for value in extra.split(",")] if extra else [0.45, 0.75]
            if len(thresholds) not in (1, 2) or any(not math.isfinite(value) or value < 0 for value in thresholds):
                raise ValueError("Enter one or two non-negative motion thresholds, such as 0.45,0.75.")
            low = thresholds[0]
            high = thresholds[-1] if len(thresholds) == 2 else 0.75
            factors, ranges, _, _, _ = _auto_adaptive_block_dilations(video, low, high)
        elif motion_mode in ("auto_2x", "auto_3x"):
            threshold = float(extra) if extra else 0.45
            if not math.isfinite(threshold) or threshold < 0:
                raise ValueError("Enter one non-negative motion threshold, such as 0.45.")
            ranges, _, _ = _auto_motion_windows(video, threshold)
        elif motion_mode == "manual":
            ranges = _repair_ranges(extra, world_len)
        else:
            ranges = [(0, world_len)]

        if factors is not None:
            holds = [factors[frame // 17] for frame in range(world_len)]
        else:
            holds = _block_holds(world_len, ranges, dilation)
            factors = holds[::17]

        target_hw = _target_latent_hw(video, spatial_scale)
        if target_hw == tuple(video.shape[-2:]):
            target_hw = None
        needs_spatial = target_hw is not None
        needs_temporal = any(factor > 1 for factor in factors)
        unpadded = sum(holds)
        used = {"world_len": world_len, "holds": holds}
        if needs_temporal:
            output, _, snapped, _ = insert().insert(
                {"samples": video}, json.dumps(used), expand_to_end=False,
            )
            used = json.loads(snapped)
            holds = used["holds"]
            expanded_video, audio = output["samples"].unbind()
            if source_audio is not None:
                audio = _stretch_audio_latent(source_audio, holds, audio.shape[-1]).to(audio)
        else:
            from comfy_extras.nodes_minimax_h3 import temporal_shape

            expanded_video = video
            audio = source_audio
            if audio is None:
                audio = video.new_zeros((video.shape[0], 32, 2, temporal_shape(world_len)[2]))
            output = {}
        expanded = sum(holds)
        output["noise_mask"] = NestedTensor((torch.zeros_like(expanded_video[:, :1]), torch.zeros_like(audio)))

        base_video = video
        loaded = None
        path = None
        spatial_path, spatial_name = _resolve_spatial_checkpoint(spatial_checkpoint) if needs_spatial else (None, None)
        temporal_path = None
        if needs_temporal or (needs_spatial and spatial_path is None):
            if not checkpoint.endswith(".safetensors"):
                raise ValueError("Download a .safetensors temporal checkpoint into ComfyUI/models/h3_temporal_models/.")
            temporal_path = folder_paths.get_full_path_or_raise(TEMPORAL_MODELS, checkpoint)
        use_decoupled_spatial = (
            needs_spatial
            and spatial_path is not None
            and (temporal_path is None or Path(spatial_path).resolve() != Path(temporal_path).resolve())
        )
        spatial_note = None
        device = model_management.get_torch_device()
        if needs_temporal or needs_spatial:
            from .temporal_checkpoint import expand_video, load_resizer

            if use_decoupled_spatial:
                if needs_temporal:
                    loaded_temp = load_resizer(temporal_path, device)
                    try:
                        uniform = {}
                        for factor in sorted(set(factors)):
                            if factor == 1:
                                uniform[factor] = video
                                continue
                            _, _, _, target_tokens, _ = motion.temporal_insert_map([factor] * world_len)
                            uniform[factor] = expand_video(
                                video, target_tokens, factor, temporal_path, device, motion,
                                spatial_scale=1.0, target_hw=None, loaded=loaded_temp,
                            )
                        pieces = []
                        for block_index, factor in enumerate(factors):
                            first = block_index * 5 * factor
                            end = first + 5 * factor if block_index + 1 < len(factors) else None
                            pieces.append(uniform[factor][:, :, first:end])
                        repaired_1x = torch.cat(pieces, dim=2)
                        if repaired_1x.shape[2] != expanded_video.shape[2]:
                            raise RuntimeError("H3 working timeline and checkpoint token count differ.")
                    finally:
                        del loaded_temp
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    repaired_1x = video

                path = spatial_path
                loaded = load_resizer(spatial_path, device)
                repaired = expand_video(
                    repaired_1x, repaired_1x.shape[2], 1.0, spatial_path, device, motion,
                    spatial_scale=spatial_scale, target_hw=target_hw, loaded=loaded,
                )
                base_video = repaired
                spatial_note = f"Spatial upscale: {spatial_scale:g}x ({target_hw[1]*16}x{target_hw[0]*16}) via {spatial_name} (decoupled)."
            else:
                path = temporal_path
                loaded = load_resizer(path, device)
                uniform = {}
                for factor in sorted(set(factors)):
                    if factor == 1 and not needs_spatial:
                        uniform[factor] = video
                        continue
                    if factor == 1:
                        target_tokens = video.shape[2]
                    else:
                        _, _, _, target_tokens, _ = motion.temporal_insert_map([factor] * world_len)
                    uniform[factor] = expand_video(
                        video, target_tokens, factor, path, device, motion,
                        spatial_scale=spatial_scale, target_hw=target_hw, loaded=loaded,
                    )
                base_video = uniform.get(1, video)
                pieces = []
                for block_index, factor in enumerate(factors):
                    first = block_index * 5 * factor
                    end = first + 5 * factor if block_index + 1 < len(factors) else None
                    pieces.append(uniform[factor][:, :, first:end])
                repaired = torch.cat(pieces, dim=2)
                if repaired.shape[2] != expanded_video.shape[2]:
                    raise RuntimeError("H3 working timeline and checkpoint token count differ.")
                if needs_spatial:
                    if (spatial_checkpoint or "auto").strip() == "auto":
                        spatial_note = (
                            f"Spatial upscale: {spatial_scale:g}x ({target_hw[1]*16}x{target_hw[0]*16}) via {checkpoint} (joint fallback; "
                            "place minimax_h3_latent_upscaler_3d_fp16.safetensors in models/latent_upscale_models/ for cleaner fractional grids & keyframes)."
                        )
                    else:
                        spatial_note = f"Spatial upscale: {spatial_scale:g}x ({target_hw[1]*16}x{target_hw[0]*16}) via {checkpoint} (joint)."
        else:
            repaired = video.clone()

        output["samples"] = NestedTensor((repaired, audio))
        mask_scale = spatial_scale if needs_spatial else 1.0
        _block_mask(motion, base_video, output, holds, mask_scale, untouched_spatial_denoise)

        starts = list(accumulate([0, *holds]))
        resized_guides = {}
        adjusted = []
        kf_has_temporal = bool(getattr(loaded[0].model, "has_temporal", True)) if loaded is not None else True
        try:
            for embedding, metadata in conditioning:
                remapped = []
                for guide in metadata.get("minimax_keyframes") or ():
                    frame = guide["resolved_frame_index"]
                    index = math.floor(frame)
                    if not 0 <= index < world_len:
                        raise ValueError(f"H3 keyframe {frame} is outside the source clip.")
                    shifted = {**guide, "resolved_frame_index": starts[index] + (frame - index) * holds[index]}
                    keyframe = guide.get("latent")
                    if needs_spatial and keyframe is not None and tuple(keyframe.shape[-2:]) != target_hw:
                        cache_key = id(keyframe)
                        if cache_key not in resized_guides:
                            if keyframe.shape[2] == 1 and kf_has_temporal:
                                guide_input = keyframe.repeat(1, 1, 5, 1, 1)
                                resized = expand_video(
                                    guide_input, guide_input.shape[2], 1, path, device, motion,
                                    spatial_scale=spatial_scale, target_hw=target_hw, loaded=loaded,
                                )
                                resized_guides[cache_key] = resized[:, :, 2:3].clone()
                            else:
                                resized = expand_video(
                                    keyframe, keyframe.shape[2], 1, path, device, motion,
                                    spatial_scale=spatial_scale, target_hw=target_hw, loaded=loaded,
                                )
                                resized_guides[cache_key] = resized[:, :, :keyframe.shape[2]].clone()
                        shifted["latent"] = resized_guides[cache_key]
                    remapped.append(shifted)
                if remapped:
                    metadata = {**metadata, "minimax_keyframes": sorted(remapped, key=lambda guide: guide["resolved_frame_index"])}
                adjusted.append([embedding, metadata])
        finally:
            if loaded is not None:
                del loaded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        used.update(repair_ranges=ranges, block_dilations=factors, unpadded_expanded_frames=unpadded)
        if needs_spatial:
            used["spatial_checkpoint"] = spatial_name if use_decoupled_spatial else checkpoint
        est_tokens = _estimate_pass2_tokens(repaired, audio)
        report = (
            f"H3 block dilation: {factors}. Working timeline: {expanded} frames (~{est_tokens:,} AV tokens). "
            f"Final timeline: {world_len} frames."
        )
        if spatial_note:
            report += f" {spatial_note}"
        if est_tokens > 250000:
            warn = f"Very high Pass-2 token count (~{est_tokens:,}) near attention int32 limits; lower spatial_scale or clip length."
            report += f" WARNING: {warn}"
            logging.warning("[H3LatentRepair] %s", warn)
        elif est_tokens > 55000:
            note = f"High Pass-2 token count (~{est_tokens:,}); for faster runs on long clips use spatial_scale=1.5 + RTX VSR 2x or auto_2x."
            report += f" Note: {note}"
            logging.info("[H3LatentRepair] %s", note)
        logging.info("[H3LatentRepair] %s", report)
        return io.NodeOutput(output, adjusted, json.dumps(used), expanded, report)


class H3RepairSchedule(io.ComfyNode):
    """Build the Pass-2 inject sigma tail, returning an empty 0-step schedule when nothing is masked."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RepairSchedule",
            display_name="H3 Repair Inject Schedule (Auto-Skip)",
            category="latent/minimax/motion",
            description="Generates the Pass-2 denoising tail schedule (matching H3InjectSchedule) and automatically returns an empty 0-step schedule when the repair noise_mask is all zero (calm clips at 1x), skipping Pass-2 sampling completely.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("samples"),
                io.Combo.Input("scheduler", options=["beta", "simple", "normal", "sgm_uniform", "karras", "exponential"], default="beta"),
                io.Int.Input("total_steps", default=12, min=4, max=100,
                             tooltip="Total schedule steps before slicing the tail (12/0.42 = 5 steps @ sigma 0.883 fast smudge-free default across 1.0x & 1.5x; 16/0.35 = 6 steps @ sigma 0.855 or 20/0.30 = 6 steps @ sigma 0.787 for tighter Pass-1 pose lock)."),
                io.Float.Input("inject", default=0.42, min=0.05, max=1.0, step=0.01,
                               tooltip="Fraction of total_steps to run in Pass 2 (default 0.42 of 12 = 5 steps @ sigma 0.883 with a dense final detail step at 0.271; pair with deterministic res_multistep)."),
                io.Boolean.Input("skip_if_unmasked", default=True,
                                 tooltip="When true and noise_mask is all zero (no dilated blocks at 1x), return an empty schedule so SamplerCustomAdvanced skips Pass 2 in 0 steps."),
            ],
            outputs=[
                io.Sigmas.Output(display_name="sigmas"),
            ],
        )

    @classmethod
    def execute(cls, model, samples, scheduler="beta", total_steps=12, inject=0.42, skip_if_unmasked=True):
        import comfy.samplers as comfy_samplers

        if skip_if_unmasked and isinstance(samples, dict) and "noise_mask" in samples:
            mask = samples["noise_mask"]
            if mask is not None:
                if getattr(mask, "is_nested", False):
                    active = any(float(t.max().item()) > 1e-6 for t in mask.unbind() if t.numel() > 0)
                else:
                    active = (mask.numel() > 0 and float(mask.max().item()) > 1e-6)
                if not active:
                    _disarm_dyrope()
                    logging.info("[H3RepairSchedule] noise_mask is all zero -> skipping Pass 2 (0 steps).")
                    return io.NodeOutput(torch.empty((0,), dtype=torch.float32))

        ms = model.get_model_object("model_sampling")
        n_steps = int(total_steps)
        full = comfy_samplers.calculate_sigmas(ms, scheduler, n_steps).cpu()
        run = max(1, int(round(n_steps * float(inject))))
        tail = full[n_steps - run:]
        logging.info(
            "[H3RepairSchedule] scheduler=%s total=%d inject=%.2f -> %d steps, start_sigma=%.3f tail=%s",
            scheduler, n_steps, float(inject), run, float(tail[0]), [round(float(x), 3) for x in tail],
        )
        return io.NodeOutput(tail)


class H3ExactVAEDecode(io.ComfyNode):
    """Decode an H3 video latent and recover exact real-time frames, trimming trailing snap-pad chunks."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ExactVAEDecode",
            display_name="H3 Exact VAE Decode & Recover",
            category="latent/minimax/motion",
            description="Decodes an H3 video latent and applies exact hold-map frame recovery in one step. Skips trailing VAE chunks beyond the last kept frame and falls back to normal decode when hold_map is empty (e.g. when Pass 2 is bypassed).",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
                io.String.Input("hold_map", default="", optional=True,
                                tooltip="Optional hold_map JSON from H3 Latent Auto Repair. Leave blank or unlinked when Pass 2 is bypassed."),
                io.Boolean.Input("trim_tail_padding", default=True, optional=True,
                                 tooltip="Skip trailing VAE temporal chunks that fall strictly after the last recovered frame."),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def _decode_trimmed_temporal(cls, vae, video, starts):
        fsm = vae.first_stage_model
        vae.throw_exception_if_invalid()
        memory_used = vae.memory_used_decode(video.shape, vae.vae_dtype)
        model_management.load_models_gpu([vae.patcher], memory_required=memory_used, force_full_load=vae.disable_offload)

        with model_management.cuda_device_context(vae.device):
            z = video.to(vae.vae_dtype).to(vae.device)
            latents_mean = fsm.latents_mean.view(1, -1, 1, 1, 1).to(z)
            latents_std = fsm.latents_std.view(1, -1, 1, 1, 1).to(z)
            z = z * latents_std + latents_mean

            chunk_dec = fsm.tokens_chunk_size * fsm.vae_ratio_t
            split_count = int(fsm.token_drop > 0) + 1
            full_shape = fsm.decode_output_shape(z.shape)
            num_keep = len(starts)
            keep_len = min(int(starts[-1]) + 1, full_shape[2])
            output_buffer = torch.empty(
                (full_shape[0], full_shape[1], num_keep, full_shape[3], full_shape[4]),
                dtype=torch.float32,
                device=model_management.intermediate_device(),
            )

            pad_tokens, num_chunks = fsm._decode_temporal_chunks(z.shape[2])
            if pad_tokens > 0:
                pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
                z = torch.cat([z, pad_z], dim=2)

            dec_overlap = None
            write_pos = 0
            keep_cursor = 0

            def write_part(part):
                nonlocal write_pos, keep_cursor
                part_frames = part.shape[2]
                if part_frames <= 0 or write_pos >= keep_len or keep_cursor >= num_keep:
                    return
                copy_frames = min(part_frames, keep_len - write_pos)
                chunk_end = write_pos + copy_frames
                start_keep = keep_cursor
                local_idx = []
                while keep_cursor < num_keep and starts[keep_cursor] < chunk_end:
                    local_idx.append(starts[keep_cursor] - write_pos)
                    keep_cursor += 1
                if local_idx:
                    selected = part[:, :, local_idx, :, :]
                    output_buffer[:, :, start_keep:keep_cursor, :, :].copy_(fsm._finalize_pixels(selected))
                write_pos = chunk_end

            for i in range(num_chunks):
                if write_pos >= keep_len or keep_cursor >= num_keep:
                    break
                t_start_idx = i * fsm.tokens_chunk_size
                t_end_idx = t_start_idx + fsm.tokens_chunk_size + fsm.token_overlap
                clip_z = z[:, :, t_start_idx:t_end_idx, :, :]
                clip_dec = fsm._adaptive_decode(clip_z)

                for j in range(split_count):
                    f_start_idx = j * chunk_dec
                    f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
                    clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
                    clip_dec_chunk = clip_dec_chunk[:, :, fsm.frame_pre_padding:, :, :]
                    if j == 0:
                        if dec_overlap is not None:
                            clip_dec_chunk = fsm.blend(dec_overlap, clip_dec_chunk, fsm.frame_overlap, dim=-3)
                            dec_overlap = None
                        write_part(clip_dec_chunk)
                    else:
                        dec_overlap = clip_dec_chunk.contiguous()

                if i == num_chunks - 1 and dec_overlap is not None and write_pos < keep_len:
                    write_part(dec_overlap)
                    dec_overlap = None

                del clip_dec, clip_z, clip_dec_chunk

        if keep_cursor != num_keep:
            raise RuntimeError(f"H3ExactVAEDecode recovered {keep_cursor} of {num_keep} frames.")
        images = output_buffer.movedim(1, -1)
        if images.ndim == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    @classmethod
    @torch.no_grad()
    def execute(cls, samples, vae, hold_map="", trim_tail_padding=True):
        gc.collect()
        model_management.soft_empty_cache()
        video = _video(samples)
        holds = None
        if isinstance(hold_map, str) and hold_map.strip().startswith("{"):
            try:
                data = json.loads(hold_map)
                if isinstance(data, dict) and data.get("holds"):
                    holds = data["holds"]
            except (json.JSONDecodeError, TypeError):
                holds = None

        if not holds:
            images = vae.decode(video)
            if images.ndim == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            result = images.cpu()
            del images
            model_management.unload_model_and_clones(vae.patcher)
            model_management.soft_empty_cache()
            return io.NodeOutput(result)

        starts, cur = [], 0
        for h in holds:
            starts.append(cur)
            cur += h

        fsm = getattr(vae, "first_stage_model", None)
        can_trim = (
            trim_tail_padding
            and fsm is not None
            and not getattr(fsm, "noncausal", True)
            and hasattr(fsm, "_decode_temporal_chunks")
            and hasattr(fsm, "_adaptive_decode")
            and video.shape[2] > 1
        )
        if can_trim:
            expected_total = fsm.decode_output_shape(video.shape)[2]
            if cur != expected_total:
                raise RuntimeError(f"Hold map sums to {cur} frames, expected {expected_total} from latent.")
            images = cls._decode_trimmed_temporal(vae, video, starts)
            result = images.cpu()
            del images
            model_management.unload_model_and_clones(vae.patcher)
            model_management.soft_empty_cache()
            return io.NodeOutput(result)

        images = vae.decode(video)
        if images.ndim == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        if cur != images.shape[0]:
            raise RuntimeError(f"Hold map sums to {cur} frames, decoded {images.shape[0]}.")
        result = images[torch.tensor(starts)].cpu()
        del images
        model_management.unload_model_and_clones(vae.patcher)
        model_management.soft_empty_cache()
        return io.NodeOutput(result)


class H3RepairDecodeAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RepairDecodeAudio",
            display_name="H3 Repair · Decode Original Audio",
            category="audio/minimax",
            description="Decode first-pass H3 audio, replacing invalid samples before normalization.",
            inputs=[io.Latent.Input("samples"), io.Vae.Input("vae")],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, samples, vae):
        audio = _audio(samples)
        if audio is None:
            raise ValueError("Connect the audio/video latent from the first H3 sampler.")
        waveform = vae.decode(audio).movedim(-1, 1)
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        scale = (waveform.std(dim=(1, 2), keepdim=True) * 5).nan_to_num(
            nan=1.0, posinf=1.0, neginf=1.0,
        ).clamp_min(1.0)
        sample_rate = samples.get("sample_rate", getattr(vae, "audio_sample_rate_output",
                                                         getattr(vae, "audio_sample_rate", 44100)))
        return io.NodeOutput({"waveform": waveform / scale, "sample_rate": sample_rate})


class H3LatentRepairExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3AdaptiveLatentRepair, H3RepairSchedule, H3ExactVAEDecode, H3RepairDecodeAudio]


async def comfy_entrypoint():
    return H3LatentRepairExtension()
