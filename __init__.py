from itertools import accumulate
import json
import math
from pathlib import Path
import sys

import folder_paths
import torch
from comfy import model_management
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import ComfyExtension, io

TEMPORAL_MODELS = "h3_temporal_models"
folder_paths.folder_names_and_paths.setdefault(TEMPORAL_MODELS, ([], {".safetensors"}))
folder_paths.add_model_folder_path(TEMPORAL_MODELS, str(Path(folder_paths.models_dir) / TEMPORAL_MODELS))

PREFERRED_CHECKPOINT = "minimax_h3_latent_upscaler_3d_conv_v2_temporal_bf16.safetensors"


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


def _auto_block_scores(video):
    """Score each 17-frame H3 block's foreground latent velocity without VAE decode."""
    vid = video.detach().float()
    token_count = vid.shape[2]
    world_len = (token_count - 2) // 5 * 17 + 5
    block_count = (token_count - 2) // 5 + 1
    scores = []
    global_means = []
    for block_idx in range(block_count):
        base = 5 * block_idx
        if block_idx < block_count - 1:
            diff_4f = (vid[:, :, base + 2:base + 5] - vid[:, :, base + 1:base + 4]).pow(2).mean(dim=1).sqrt()
            excess = [
                float(torch.quantile(diff_4f[:, step].flatten(), 0.98) - torch.quantile(diff_4f[:, step].flatten(), 0.50))
                for step in range(3)
            ]
            scores.append(max(excess))
            global_means.append(float(diff_4f.mean()))
        else:
            diff_tail = (vid[:, :, base + 1:base + 2] - vid[:, :, max(0, base - 1):max(1, base)]).pow(2).mean(dim=1).sqrt()
            diff_tail *= 0.8 if base else 1.6
            flat = diff_tail.flatten()
            scores.append(float(torch.quantile(flat, 0.98) - torch.quantile(flat, 0.50)))
            global_means.append(float(diff_tail.mean()))
    return world_len, scores, global_means


def _auto_motion_windows(video, abs_thresh=0.45, rel_ratio=0.55):
    """Detect high-motion 17-frame H3 blocks directly in latent space without VAE decode."""
    world_len, scores, global_means = _auto_block_scores(video)
    peak = max(scores) if scores else 0.0
    gate = max(abs_thresh, peak * rel_ratio)
    windows = []
    for block_idx, (score, gmean) in enumerate(zip(scores, global_means)):
        if score >= gate or gmean >= 0.55:
            start = block_idx * 17
            end = min(world_len, (block_idx + 1) * 17)
            if windows and windows[-1][1] == start:
                windows[-1] = (windows[-1][0], end)
            else:
                windows.append((start, end))
    return windows, [round(score, 3) for score in scores], round(gate, 3)


def _auto_adaptive_block_dilations(video, low_thresh=0.45, high_thresh=0.75, rel_ratio=0.55):
    """Assign 1x (calm), 2x (moderate motion), or 3x (extreme motion) per 17-frame block."""
    world_len, scores, global_means = _auto_block_scores(video)
    peak = max(scores) if scores else 0.0
    low_gate = max(low_thresh, peak * rel_ratio)
    high_gate = max(high_thresh, low_gate + 0.10)
    block_dilations = []
    windows = []
    for block_idx, (score, gmean) in enumerate(zip(scores, global_means)):
        if score >= high_gate or gmean >= 0.70:
            dil = 3
        elif score >= low_gate or gmean >= 0.55:
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
    """Simplified production node for H3 latent auto-motion repair & joint spatial+temporal upscale."""

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
        models = folder_paths.get_filename_list(TEMPORAL_MODELS)
        default_ckpt = PREFERRED_CHECKPOINT if PREFERRED_CHECKPOINT in models or not models else models[0]
        return io.Schema(
            node_id="H3AdaptiveLatentRepair",
            display_name="H3 Latent Auto Repair & Upscale",
            category="latent/minimax/motion",
            description="Clean production node: auto-detects fast-motion 17-frame blocks in latent space (no Pass-1 VAE decode), time-aligns Pass-1 audio latents for refinement, and optionally applies joint spatial upscaling.",
            inputs=[
                io.Latent.Input("samples"),
                io.Conditioning.Input("conditioning"),
                io.Combo.Input("checkpoint", options=models or [PREFERRED_CHECKPOINT], default=default_ckpt,
                               tooltip="Joint spatial+temporal latent resizer (George0667 v2 temporal BF16 or FP16)."),
                io.Combo.Input("motion_mode", options=cls.MOTION_MODES, default="auto_adaptive (1x/2x/3x)",
                               tooltip="auto_adaptive: 1x calm / 2x moderate / 3x extreme bursts. auto_2x/auto_3x: uniform dilation on hot blocks only. full_2x/full_3x: whole clip. manual: use manual_ranges."),
                io.Float.Input("spatial_scale", default=1.0, min=1.0, max=2.0, step=0.25,
                               tooltip="1.0 = same-res repair; 1.5 = 640x384->960x576; 2.0 = 640x384->1280x768."),
                io.Float.Input("untouched_spatial_denoise", default=0.7, min=0.0, max=1.0, step=0.05,
                               tooltip="Denoise mask on untouched 1x blocks when spatial_scale > 1.0 (0.7 preserves faces/backgrounds while sharpening)."),
                io.String.Input("manual_ranges", default="", optional=True,
                                tooltip="Used when motion_mode='manual' (e.g. '17-50') or optional threshold override (e.g. '0.45,0.75' for adaptive, '0.40' for auto_2x/3x)."),
                io.String.Input("oracle_hold_map", force_input=True, optional=True,
                                tooltip="Optional external hold_map (e.g. from H3 Jerk Oracle) to override motion_mode."),
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
    def execute(cls, samples, conditioning, checkpoint, motion_mode="auto_adaptive (1x/2x/3x)",
                spatial_scale=1.0, untouched_spatial_denoise=0.7,
                manual_ranges="", oracle_hold_map=None):
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
        resizer = None
        if needs_temporal or needs_spatial:
            from .temporal_checkpoint import expand_video, load_resizer

            if not checkpoint.endswith(".safetensors"):
                raise ValueError("Select the George0667 v2 temporal .safetensors checkpoint.")
            path = folder_paths.get_full_path_or_raise(TEMPORAL_MODELS, checkpoint)
            device = model_management.get_torch_device()
            model_management.unload_all_models()
            resizer = load_resizer(path, device)
            uniform = {}
            for factor in sorted(set(factors)):
                if factor == 1 and not needs_spatial:
                    uniform[factor] = video
                    continue
                if factor == 1:
                    target_tokens = video.shape[2]
                else:
                    _, _, _, target_tokens, _ = motion.temporal_insert_map([factor] * world_len)
                uniform[factor] = expand_video(video, target_tokens, factor, device, motion,
                                               spatial_scale, target_hw, resizer)
            base_video = uniform.get(1, video)
            pieces = []
            for block_index, factor in enumerate(factors):
                first = block_index * 5 * factor
                end = first + 5 * factor if block_index + 1 < len(factors) else None
                pieces.append(uniform[factor][:, :, first:end])
            repaired = torch.cat(pieces, dim=2)
            if repaired.shape[2] != expanded_video.shape[2]:
                raise RuntimeError("H3 working timeline and checkpoint token count differ.")
        else:
            repaired = video.clone()

        output["samples"] = NestedTensor((repaired, audio))
        mask_scale = spatial_scale if needs_spatial else 1.0
        _block_mask(motion, base_video, output, holds, mask_scale, untouched_spatial_denoise)

        starts = list(accumulate([0, *holds]))
        resized_guides = {}
        adjusted = []
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
                        guide_input = keyframe.repeat(1, 1, 5, 1, 1) if keyframe.shape[2] == 1 else keyframe
                        resized = expand_video(guide_input, guide_input.shape[2], 1, device, motion,
                                               spatial_scale, target_hw, resizer)
                        resized_guides[cache_key] = resized[:, :, :keyframe.shape[2]].clone()
                    shifted["latent"] = resized_guides[cache_key]
                remapped.append(shifted)
            if remapped:
                metadata = {**metadata, "minimax_keyframes": sorted(remapped, key=lambda guide: guide["resolved_frame_index"])}
            adjusted.append([embedding, metadata])

        if resizer is not None:
            del resizer, pieces, uniform, base_video
            model_management.soft_empty_cache()

        used.update(repair_ranges=ranges, block_dilations=factors, unpadded_expanded_frames=unpadded)
        report = f"H3 block dilation: {factors}. Working timeline: {expanded} frames. Final timeline: {world_len} frames."
        return io.NodeOutput(output, adjusted, json.dumps(used), expanded, report)


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
        return [H3AdaptiveLatentRepair, H3RepairDecodeAudio]


async def comfy_entrypoint():
    return H3LatentRepairExtension()
