#!/usr/bin/env python3
"""
Simple serving benchmark client for ComfyUI's HTTP API.

This script is inspired by diffusion serving benchmarks and is designed to:
  - submit prompts to ComfyUI (/prompt or /bench/prompt),
  - optionally shape request arrivals (fixed rate or Poisson),
  - poll completion via /history/{prompt_id},
  - report latency/throughput/error metrics.

Usage — Wan 2.2 I2V benchmark
==============================

Step 1 — Generate prompt files (downloads images, writes JSONs, then exits):

  # Minimal: uses synthetic images, writes to prompts/wan22_i2v/
  python3 benchmarks/benchmark_comfyui_serving.py \\
    --generate-wan22-prompts \\
    --num-requests 50

  # With model download (needs ComfyUI root):
  python3 benchmarks/benchmark_comfyui_serving.py \\
    --generate-wan22-prompts \\
    --download-models \\
    --comfyui-base-dir /path/to/ComfyUI \\
    --num-requests 50

  # Custom image/output dirs:
  python3 benchmarks/benchmark_comfyui_serving.py \\
    --generate-wan22-prompts \\
    --wan22-input-dir /data/images \\
    --wan22-output-dir /data/prompts/wan22 \\
    --wan22-num-images 30 \\
    --num-requests 50

Step 2 — Run the benchmark (point at any one of the generated prompt files):

  python3 benchmarks/benchmark_comfyui_serving.py \\
    --prompt-file prompts/wan22_i2v/wan22_i2v_prompt_0000.json \\
    --num-requests 50 \\
    --max-concurrency 4 \\
    --host http://127.0.0.1:8188

The setup step also prints the exact run command at the end, so you can copy it directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import aiohttp


# ──────────────────────────────────────────────────────────────────────────────
# Wan 2.2 I2V benchmark setup helpers
# ──────────────────────────────────────────────────────────────────────────────

_WAN22_MODELS: list[tuple[str, str]] = [
    (
        "models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    ),
    (
        "models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    ),
    (
        "models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    ),
    (
        "models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
    ),
    (
        "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
    (
        "models/vae/wan_2.1_vae.safetensors",
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
    ),
]

# Placeholder sentinel replaced by generate_prompt_file.
_IMAGE_PLACEHOLDER = "__INPUT_IMAGE__"

_WAN22_I2V_GRAPH: dict[str, Any] = {
    "97": {
        "inputs": {"image": _IMAGE_PLACEHOLDER},
        "class_type": "LoadImage",
        "_meta": {"title": "Start Frame Image"},
    },
    "108": {
        "inputs": {
            "filename_prefix": "video/Wan2.2_image_to_video",
            "format": "auto",
            "codec": "auto",
            "video-preview": "",
            "video": ["130:117", 0],
        },
        "class_type": "SaveVideo",
        "_meta": {"title": "Save Video"},
    },
    "130:105": {
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default",
        },
        "class_type": "CLIPLoader",
        "_meta": {"title": "Load CLIP"},
    },
    "130:106": {
        "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "130:107": {
        "inputs": {
            "text": "A felt-style little eagle cashier greeting, waving, and smiling at the camera.",
            "clip": ["130:105", 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
    },
    "130:109": {
        "inputs": {"shift": 5.000000000000001, "model": ["130:126", 0]},
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "ModelSamplingSD3"},
    },
    "130:110": {
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 636787045983965,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 0,
            "end_at_step": 2,
            "return_with_leftover_noise": "enable",
            "model": ["130:109", 0],
            "positive": ["130:128", 0],
            "negative": ["130:128", 1],
            "latent_image": ["130:128", 2],
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "KSampler (Advanced)"},
    },
    "130:111": {
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": 4,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 2,
            "end_at_step": 4,
            "return_with_leftover_noise": "disable",
            "model": ["130:124", 0],
            "positive": ["130:128", 0],
            "negative": ["130:128", 1],
            "latent_image": ["130:110", 0],
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "KSampler (Advanced)"},
    },
    "130:117": {
        "inputs": {"fps": 16, "images": ["130:129", 0]},
        "class_type": "CreateVideo",
        "_meta": {"title": "Create Video"},
    },
    "130:122": {
        "inputs": {
            "unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "default",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"},
    },
    "130:123": {
        "inputs": {
            "unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            "weight_dtype": "default",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"},
    },
    "130:124": {
        "inputs": {"shift": 5.000000000000001, "model": ["130:127", 0]},
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "ModelSamplingSD3"},
    },
    "130:125": {
        "inputs": {
            "text": (
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
                "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
                "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            ),
            "clip": ["130:105", 0],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
    },
    "130:126": {
        "inputs": {
            "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            "strength_model": 1.0000000000000002,
            "model": ["130:122", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "Load LoRA"},
    },
    "130:127": {
        "inputs": {
            "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            "strength_model": 1.0000000000000002,
            "model": ["130:123", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "Load LoRA"},
    },
    "130:128": {
        "inputs": {
            "width": 720,
            "height": 720,
            "length": 81,
            "batch_size": 1,
            "positive": ["130:107", 0],
            "negative": ["130:125", 0],
            "vae": ["130:106", 0],
            "start_image": ["97", 0],
        },
        "class_type": "WanImageToVideo",
        "_meta": {"title": "WanImageToVideo"},
    },
    "130:129": {
        "inputs": {"samples": ["130:111", 0], "vae": ["130:106", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
}

# Google Drive file IDs from VBench's vbench2_beta_i2v/download_data.sh
_VBENCH_ORIGIN_ZIP_GDRIVE_ID = "1qhkLCSBkzll0dkKpwlDTwLL0nxdQ4nrY"


def download_wan22_models(base_dir: Path) -> None:
    """Download Wan 2.2 I2V model files into *base_dir* using wget."""
    for rel_path, url in _WAN22_MODELS:
        dest = base_dir / rel_path
        if dest.exists():
            print(f"[setup] already exists, skipping: {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[setup] downloading {dest.name} ...")
        subprocess.run(["wget", "-O", str(dest), url], check=True)


def _try_download_vbench_i2v(input_dir: Path) -> list[str]:
    """
    Download VBench I2V origin images from Google Drive via gdown (pip install gdown).
    Returns image basenames placed in *input_dir*, or [] on failure.
    """
    try:
        import gdown  # type: ignore
    except ImportError:
        print("[setup] gdown not available; skipping VBench download. Install with: pip install gdown")
        return []

    import zipfile

    zip_path = input_dir / "origin.zip"
    try:
        if not zip_path.exists():
            print("[setup] downloading VBench I2V origin images from Google Drive ...")
            gdown.download(id=_VBENCH_ORIGIN_ZIP_GDRIVE_ID, output=str(zip_path), quiet=False)
        print("[setup] extracting origin.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(input_dir))
        zip_path.unlink()
    except Exception as exc:
        print(f"[setup] VBench I2V download failed: {exc}")
        if zip_path.exists():
            zip_path.unlink()
        return []

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    filenames = sorted(
        p.name for p in input_dir.rglob("*") if p.suffix.lower() in image_exts
    )
    print(f"[setup] prepared {len(filenames)} VBench I2V images in {input_dir}")
    return filenames


def _generate_synthetic_images(input_dir: Path, num_images: int) -> list[str]:
    """Generate synthetic 720×720 white PNG placeholders; returns filenames."""
    try:
        from PIL import Image as PILImage  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Pillow is required for synthetic image generation. "
            "Install it with: pip install Pillow"
        )

    filenames: list[str] = []
    for i in range(num_images):
        fname = f"benchmark_input_{i:04d}.png"
        dest = input_dir / fname
        if not dest.exists():
            PILImage.new("RGB", (720, 720), color=(255, 255, 255)).save(str(dest))
        filenames.append(fname)
    return filenames


def prepare_input_images(input_dir: Path, num_images: int = 20) -> list[str]:
    """
    Prepare benchmark input images in *input_dir*.

    Priority:
      1. Reuse any images already present in the directory.
      2. Download Vchitect/VBench_I2V dataset via huggingface_hub.
      3. Generate synthetic 720×720 white PNG placeholders with Pillow.

    Returns a list of image basenames (not full paths).
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}

    existing = sorted(
        p.name for p in input_dir.iterdir() if p.suffix.lower() in image_exts
    )
    if existing:
        print(f"[setup] found {len(existing)} existing images in {input_dir}")
        return existing

    filenames = _try_download_vbench_i2v(input_dir)
    if filenames:
        return filenames

    print(f"[setup] generating {num_images} synthetic 720×720 placeholder images ...")
    return _generate_synthetic_images(input_dir, num_images)


def generate_prompt_file(
    output_path: Path,
    image_filename: str,
    positive_prompt: str | None = None,
) -> None:
    """
    Write a single Wan 2.2 I2V ComfyUI prompt JSON to *output_path*.

    *image_filename* is substituted into the LoadImage node (node "97").
    *positive_prompt* overrides the default positive text if provided.
    """
    graph: dict[str, Any] = json.loads(json.dumps(_WAN22_I2V_GRAPH))
    graph["97"]["inputs"]["image"] = image_filename
    if positive_prompt is not None:
        graph["130:107"]["inputs"]["text"] = positive_prompt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"prompt": graph}, indent=2))


def generate_prompt_files(
    output_dir: Path,
    input_dir: Path,
    num_prompts: int = 50,
    num_images: int = 20,
    download_models: bool = False,
    comfyui_base_dir: Path | None = None,
) -> list[Path]:
    """
    Full Wan 2.2 I2V benchmark setup:

      1. Optionally download model weights into *comfyui_base_dir*.
      2. Prepare input images in *input_dir* (VBench I2V or synthetic).
      3. Generate *num_prompts* prompt JSON files in *output_dir*, cycling
         through the available images.

    Returns the list of generated prompt file paths.
    """
    if download_models:
        if comfyui_base_dir is None:
            raise ValueError("--comfyui-base-dir is required when --download-models is set")
        download_wan22_models(comfyui_base_dir)

    image_filenames = prepare_input_images(input_dir, num_images=num_images)
    if not image_filenames:
        raise RuntimeError(f"No input images available in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for i in range(num_prompts):
        image_name = image_filenames[i % len(image_filenames)]
        prompt_path = output_dir / f"wan22_i2v_prompt_{i:04d}.json"
        generate_prompt_file(prompt_path, image_name)
        generated.append(prompt_path)

    print(f"[setup] generated {len(generated)} prompt files in {output_dir}")
    print(f"[setup] example run:")
    print(
        f"  python benchmark_comfyui_serving.py"
        f" --prompt-file {generated[0]}"
        f" --num-requests {num_prompts}"
    )
    return generated


# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    request_index: int
    prompt_id: str | None
    ok: bool
    error: str | None
    queued_at: float
    started_at: float
    finished_at: float
    end_to_end_s: float
    queue_wait_ms: float | None
    execution_ms: float | None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (len(values) - 1) * (pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def patch_seed_in_prompt(prompt: dict[str, Any], seed: int, seed_path: str | None) -> dict[str, Any]:
    """
    Patch prompt seed in-place for common sampler nodes.
    seed_path format: "<node_id>.<input_name>".
    """
    if seed_path:
        try:
            node_id, input_name = seed_path.split(".", 1)
            prompt[node_id]["inputs"][input_name] = seed
            return prompt
        except Exception as exc:
            raise ValueError(f"Invalid --seed-path '{seed_path}': {exc}") from exc

    # Best-effort fallback: update any input key named 'seed' or 'noise_seed'
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if "seed" in inputs:
            inputs["seed"] = seed
        if "noise_seed" in inputs:
            inputs["noise_seed"] = seed
    return prompt


def load_prompt_template(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "prompt" in data and isinstance(data["prompt"], dict):
        return data
    if isinstance(data, dict):
        return {"prompt": data}
    raise ValueError("Prompt file must be a JSON object (prompt graph or wrapper with 'prompt').")


async def submit_prompt(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> str:
    url = f"{base_url}{endpoint}"
    async with session.post(url, json=payload, timeout=timeout_s) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"submit failed [{resp.status}] {text}")
        body = json.loads(text)
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"missing prompt_id in response: {body}")
        return prompt_id


async def wait_for_prompt_done(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt_id: str,
    poll_interval_s: float,
    timeout_s: float,
) -> tuple[float | None, float | None]:
    """
    Returns (queue_wait_ms, execution_ms) when available from history status messages.
    Falls back to (None, None) if unavailable.
    """
    deadline = time.perf_counter() + timeout_s
    history_url = f"{base_url}/history/{prompt_id}"

    while time.perf_counter() < deadline:
        async with session.get(history_url, timeout=timeout_s) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"history failed [{resp.status}] {text}")

            payload = await resp.json()
            if not payload:
                await asyncio.sleep(poll_interval_s)
                continue

            history_item = payload.get(prompt_id)
            if history_item is None:
                await asyncio.sleep(poll_interval_s)
                continue

            status = history_item.get("status", {})
            status_str = status.get("status_str")
            messages = status.get("messages", [])
            if status_str not in ("success", "error"):
                await asyncio.sleep(poll_interval_s)
                continue

            queue_wait_ms = None
            execution_ms = None
            try:
                timestamp_map: dict[str, int] = {}
                for event, msg in messages:
                    if isinstance(msg, dict) and "timestamp" in msg:
                        timestamp_map[event] = int(msg["timestamp"])
                start_ts = timestamp_map.get("execution_start")
                end_ts = timestamp_map.get("execution_success") or timestamp_map.get("execution_error")
                if start_ts is not None and end_ts is not None:
                    execution_ms = max(0.0, end_ts - start_ts)
            except Exception:
                execution_ms = None

            return queue_wait_ms, execution_ms

        await asyncio.sleep(poll_interval_s)

    raise TimeoutError(f"timed out waiting for prompt_id={prompt_id}")


def build_arrival_schedule(num_requests: int, request_rate: float, poisson: bool, seed: int) -> list[float]:
    """
    Returns absolute offsets (seconds from benchmark start) for each request.
    """
    if request_rate <= 0:
        return [0.0] * num_requests

    rnd = random.Random(seed)
    offsets: list[float] = []
    t = 0.0
    for _ in range(num_requests):
        if poisson:
            delta = rnd.expovariate(request_rate)
        else:
            delta = 1.0 / request_rate
        t += delta
        offsets.append(t)
    return offsets


async def run_request(
    idx: int,
    start_time: float,
    scheduled_offset_s: float,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    prompt_wrapper_template: dict[str, Any],
) -> RequestResult:
    await asyncio.sleep(max(0.0, (start_time + scheduled_offset_s) - time.perf_counter()))
    queued_at = time.perf_counter()

    async with semaphore:
        started_at = time.perf_counter()
        prompt_id = None
        try:
            payload = json.loads(json.dumps(prompt_wrapper_template))
            payload.setdefault("extra_data", {})
            payload["client_id"] = args.client_id

            seed = args.base_seed + idx
            payload["prompt"] = patch_seed_in_prompt(payload["prompt"], seed, args.seed_path)

            prompt_id = await submit_prompt(
                session=session,
                base_url=args.host,
                endpoint=args.endpoint,
                payload=payload,
                timeout_s=args.request_timeout_s,
            )

            queue_wait_ms, execution_ms = await wait_for_prompt_done(
                session=session,
                base_url=args.host,
                prompt_id=prompt_id,
                poll_interval_s=args.poll_interval_s,
                timeout_s=args.request_timeout_s,
            )
            finished_at = time.perf_counter()
            return RequestResult(
                request_index=idx,
                prompt_id=prompt_id,
                ok=True,
                error=None,
                queued_at=queued_at,
                started_at=started_at,
                finished_at=finished_at,
                end_to_end_s=finished_at - queued_at,
                queue_wait_ms=queue_wait_ms,
                execution_ms=execution_ms,
            )
        except Exception as exc:
            finished_at = time.perf_counter()
            return RequestResult(
                request_index=idx,
                prompt_id=prompt_id,
                ok=False,
                error=repr(exc),
                queued_at=queued_at,
                started_at=started_at,
                finished_at=finished_at,
                end_to_end_s=finished_at - queued_at,
                queue_wait_ms=None,
                execution_ms=None,
            )


def print_summary(results: list[RequestResult], wall_s: float) -> None:
    success = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    lat_s = [r.end_to_end_s for r in success]
    queue_wait_ms = [r.queue_wait_ms for r in success if r.queue_wait_ms is not None]
    exec_ms = [r.execution_ms for r in success if r.execution_ms is not None]

    throughput = (len(success) / wall_s) if wall_s > 0 else 0.0
    print("\n=== ComfyUI Serving Benchmark Summary ===")
    print(f"requests_total:   {len(results)}")
    print(f"requests_success: {len(success)}")
    print(f"requests_failed:  {len(fail)}")
    print(f"wall_time_s:      {wall_s:.3f}")
    print(f"throughput_req_s: {throughput:.3f}")

    if lat_s:
        print(f"latency_p50_s:    {percentile(lat_s, 50):.3f}")
        print(f"latency_p90_s:    {percentile(lat_s, 90):.3f}")
        print(f"latency_p95_s:    {percentile(lat_s, 95):.3f}")
        print(f"latency_p99_s:    {percentile(lat_s, 99):.3f}")
        print(f"latency_mean_s:   {statistics.mean(lat_s):.3f}")
        print(f"latency_max_s:    {max(lat_s):.3f}")

    if queue_wait_ms:
        print(f"queue_wait_mean_ms: {statistics.mean(queue_wait_ms):.2f}")
        print(f"queue_wait_p95_ms:  {percentile(queue_wait_ms, 95):.2f}")

    if exec_ms:
        print(f"execution_mean_ms:  {statistics.mean(exec_ms):.2f}")
        print(f"execution_p95_ms:   {percentile(exec_ms, 95):.2f}")

    if fail:
        print("\nSample failures:")
        for r in fail[:5]:
            print(f"  idx={r.request_index} prompt_id={r.prompt_id} error={r.error}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ComfyUI request serving.")
    p.add_argument("--host", type=str, default="http://127.0.0.1:8188", help="ComfyUI base URL.")
    p.add_argument(
        "--endpoint",
        type=str,
        default="/prompt",
        choices=("/prompt", "/bench/prompt"),
        help="Submission endpoint.",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Path to prompt JSON. Required unless --generate-wan22-prompts is set.",
    )
    p.add_argument(
        "--generate-wan22-prompts",
        action="store_true",
        help="Generate Wan 2.2 I2V prompt files (steps: prepare images, write JSONs) then exit.",
    )
    p.add_argument(
        "--wan22-input-dir",
        type=Path,
        default=Path("inputs"),
        help="Directory for benchmark input images (default: inputs/).",
    )
    p.add_argument(
        "--wan22-output-dir",
        type=Path,
        default=Path("prompts/wan22_i2v"),
        help="Directory where generated prompt JSON files are written (default: prompts/wan22_i2v/).",
    )
    p.add_argument(
        "--wan22-num-images",
        type=int,
        default=20,
        help="Number of synthetic images to generate when VBench download is unavailable (default: 20).",
    )
    p.add_argument(
        "--download-models",
        action="store_true",
        help="Download Wan 2.2 model weights before generating prompts (requires --comfyui-base-dir).",
    )
    p.add_argument(
        "--comfyui-base-dir",
        type=Path,
        default=None,
        help="ComfyUI root directory used as the base for model downloads.",
    )
    p.add_argument("--num-requests", type=int, default=50)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--request-rate", type=float, default=0.0, help="Requests/sec. 0 = fire immediately.")
    p.add_argument("--poisson", action="store_true", help="Use Poisson inter-arrival when request-rate > 0.")
    p.add_argument("--base-seed", type=int, default=1234)
    p.add_argument(
        "--seed-path",
        type=str,
        default=None,
        help="Optional path to seed field in prompt: <node_id>.<input_name> (e.g. 3.seed).",
    )
    p.add_argument("--client-id", type=str, default=f"bench-{uuid.uuid4().hex[:12]}")
    p.add_argument("--request-timeout-s", type=float, default=600.0)
    p.add_argument("--poll-interval-s", type=float, default=0.2)
    p.add_argument("--output-json", type=Path, default=None, help="Write detailed result JSON.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for schedule generation.")
    return p.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    if args.prompt_file is None:
        raise SystemExit("error: --prompt-file is required (or use --generate-wan22-prompts to create one)")
    prompt_template = load_prompt_template(args.prompt_file)
    schedule = build_arrival_schedule(
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        poisson=args.poisson,
        seed=args.seed,
    )
    semaphore = asyncio.Semaphore(args.max_concurrency)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency * 2, 32))

    started = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                run_request(
                    idx=i,
                    start_time=started,
                    scheduled_offset_s=schedule[i],
                    semaphore=semaphore,
                    session=session,
                    args=args,
                    prompt_wrapper_template=prompt_template,
                )
            )
            for i in range(args.num_requests)
        ]
        results = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - started

    print_summary(results, wall_s)

    if args.output_json is not None:
        out = {
            "config": vars(args),
            "wall_time_s": wall_s,
            "results": [asdict(r) for r in sorted(results, key=lambda x: x.request_index)],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2))
        print(f"\nWrote results to: {args.output_json}")


def main() -> None:
    args = parse_args()
    if args.generate_wan22_prompts:
        generate_prompt_files(
            output_dir=args.wan22_output_dir,
            input_dir=args.wan22_input_dir,
            num_prompts=args.num_requests,
            num_images=args.wan22_num_images,
            download_models=args.download_models,
            comfyui_base_dir=args.comfyui_base_dir,
        )
        return
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
