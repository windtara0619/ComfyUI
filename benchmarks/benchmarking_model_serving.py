from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Make the repo root importable when running directly from the benchmarks/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comfy.model_management
import comfy.sd


# -----------------------------
# Data models
# -----------------------------

@dataclasses.dataclass
class RequestSpec:
    profile_name: str
    batch_size: int
    width: int
    height: int
    num_frames: int
    steps: int
    cfg_scale: float
    seed: int
    timeout_s: float = 180.0
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RequestResult:
    request_id: int
    profile_name: str
    ok: bool
    error: Optional[str]
    latency_ms: float
    queue_wait_ms: float
    step_latencies_ms: List[float]
    ttfs_ms: float          # time to first (denoising) step
    peak_vram_mb: float
    est_mem_mb: Optional[float]
    started_at: float
    ended_at: float


@dataclasses.dataclass
class RunSummary:
    total_requests: int
    success: int
    failed: int
    throughput_req_s: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    ttfs_p50_ms: float
    ttfs_p99_ms: float
    step_mean_ms: float
    step_p99_ms: float
    max_vram_mb: float


# -----------------------------
# Helpers
# -----------------------------

def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def now() -> float:
    return time.perf_counter()


def gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_request_stream(
    num_requests: int,
    base_seed: int,
    profiles: List[RequestSpec],
    weighted: Optional[List[float]] = None,
) -> List[RequestSpec]:
    rnd = random.Random(base_seed)
    out: List[RequestSpec] = []
    for i in range(num_requests):
        p = rnd.choices(profiles, weights=weighted, k=1)[0]
        out.append(dataclasses.replace(p, seed=base_seed + i))
    return out


# -----------------------------
# Model adapter
# -----------------------------

class WanRunner:
    """
    Thin adapter around ComfyUI model loading + the BaseModel.apply_model call path.

    Only the DiT denoiser is timed — no VAE encode/decode, no CLIP, no scheduler
    overhead — so measurements reflect true model inference cost.

    Latent shape convention (WAN):  [B, 16, T, H//8, W//8]
    Text conditioning shape (UMT5): [B, text_seq_len, text_dim]  (zeros for benchmarking)
    Sigma schedule (flow-matching):  linspace(1.0 → 1/steps, steps)
    """

    def __init__(
        self,
        checkpoint: str,
        device: str,
        dtype_str: str,
        text_seq_len: int = 512,
        text_dim: int = 4096,
    ):
        self.checkpoint = checkpoint
        self.device_str = device
        self.dtype_str = dtype_str
        self.text_seq_len = text_seq_len
        self.text_dim = text_dim
        self.patcher, self.model = self._load_model()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self):
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(self.dtype_str)
        model_opts = {"dtype": dtype} if dtype is not None else {}

        patcher = comfy.sd.load_diffusion_model(self.checkpoint, model_options=model_opts)
        # force_full_load=True keeps the whole model resident on GPU rather than
        # streaming weights on demand (important for latency benchmarking).
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)
        return patcher, patcher.model

    def _estimate_mem_mb(self, latent_shape: tuple, text_seq_len: int) -> Optional[float]:
        cond_shapes = {
            "c_crossattn": [(latent_shape[0], text_seq_len, self.text_dim)],
        }
        try:
            return self.model.memory_required(latent_shape, cond_shapes) / (1024 ** 2)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Single-request execution
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run_one(self, req: RequestSpec) -> RequestResult:
        start = now()
        reset_gpu_peak()

        step_latencies: List[float] = []
        ttfs_ms = float("nan")
        est_mem_mb: Optional[float] = None
        ok = True
        err = None

        try:
            device = comfy.model_management.get_torch_device()
            dtype = self.model.get_dtype_inference()

            # Latent noise tensor: [B, 16 channels, T frames, H/8, W/8]
            latent_shape = (
                req.batch_size, 16,
                req.num_frames,
                req.height // 8,
                req.width // 8,
            )
            x = torch.randn(latent_shape, dtype=dtype, device=device)
            est_mem_mb = self._estimate_mem_mb(latent_shape, self.text_seq_len)

            # Fake text conditioning — zeros have the right shape, non-zero
            # values are not needed for throughput/latency benchmarking.
            cross_attn = torch.zeros(
                req.batch_size, self.text_seq_len, self.text_dim,
                dtype=dtype, device=device,
            )

            # Linear sigma schedule: 1.0 → 1/steps  (flow-matching, noise→clean)
            sigmas = torch.linspace(1.0, 1.0 / req.steps, req.steps, device=device)

            for step_i, sigma_val in enumerate(sigmas):
                sigma_t = sigma_val.expand(req.batch_size)
                t0 = now()
                x = self.model.apply_model(x, sigma_t, c_crossattn=cross_attn)
                sync_cuda()
                elapsed_ms = (now() - t0) * 1000.0
                step_latencies.append(elapsed_ms)
                if step_i == 0:
                    ttfs_ms = elapsed_ms

        except Exception as e:
            ok = False
            err = repr(e)

        end = now()
        return RequestResult(
            request_id=-1,
            profile_name=req.profile_name,
            ok=ok,
            error=err,
            latency_ms=(end - start) * 1000.0,
            queue_wait_ms=0.0,     # filled in by the scheduler
            step_latencies_ms=step_latencies,
            ttfs_ms=ttfs_ms,
            peak_vram_mb=gpu_peak_mb(),
            est_mem_mb=est_mem_mb,
            started_at=start,
            ended_at=end,
        )


# -----------------------------
# Serving-style scheduler
# -----------------------------

async def run_closed_loop(
    runner: WanRunner,
    requests: List[RequestSpec],
    concurrency: int,
    request_rate: float = float("inf"),
) -> List[RequestResult]:
    """
    Closed-loop scheduler (default) or Poisson open-loop when request_rate is finite.

    Each request is dispatched to a thread so the asyncio event loop stays
    free to issue the next request while the GPU is busy.
    """
    sem = asyncio.Semaphore(concurrency)
    results: List[Optional[RequestResult]] = [None] * len(requests)

    async def worker(i: int, req: RequestSpec) -> None:
        async with sem:
            t_enq = now()
            res = await asyncio.to_thread(runner.run_one, req)
            res.request_id = i
            res.queue_wait_ms = max(0.0, (res.started_at - t_enq) * 1000.0)
            results[i] = res

    if request_rate == float("inf") or request_rate <= 0:
        await asyncio.gather(*(worker(i, r) for i, r in enumerate(requests)))
    else:
        tasks: List[asyncio.Task] = []
        for i, req in enumerate(requests):
            if i > 0:
                await asyncio.sleep(random.expovariate(request_rate))
            tasks.append(asyncio.create_task(worker(i, req)))
        await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


def summarize(results: List[RequestResult], wall_s: float) -> RunSummary:
    lat = [r.latency_ms for r in results if r.ok]
    ttfs = [r.ttfs_ms for r in results if r.ok and math.isfinite(r.ttfs_ms)]
    all_steps = [s for r in results if r.ok for s in r.step_latencies_ms]
    succ = sum(1 for r in results if r.ok)
    fail = len(results) - succ
    return RunSummary(
        total_requests=len(results),
        success=succ,
        failed=fail,
        throughput_req_s=(succ / wall_s) if wall_s > 0 else 0.0,
        p50_ms=percentile(lat, 50),
        p90_ms=percentile(lat, 90),
        p95_ms=percentile(lat, 95),
        p99_ms=percentile(lat, 99),
        mean_ms=(statistics.mean(lat) if lat else float("nan")),
        ttfs_p50_ms=percentile(ttfs, 50),
        ttfs_p99_ms=percentile(ttfs, 99),
        step_mean_ms=(statistics.mean(all_steps) if all_steps else float("nan")),
        step_p99_ms=percentile(all_steps, 99),
        max_vram_mb=max((r.peak_vram_mb for r in results), default=0.0),
    )


def print_summary(
    args: argparse.Namespace,
    summ: RunSummary,
    total_requests: int,
    wall_s: float,
) -> None:
    w = 60
    sep = "-" * w
    print("\n" + "=" * w)
    print("{s:^{n}}".format(s=" WAN Benchmark Result ", n=w))
    print("=" * w)
    print("{:<40} {:<}".format("Checkpoint:", Path(args.checkpoint).name))
    print("{:<40} {:<}".format("Device / dtype:", f"{args.device}/{args.dtype}"))
    print("{:<40} {:<}".format("Concurrency:", args.concurrency))
    rate_str = f"{args.request_rate:.1f} req/s" if args.request_rate != float("inf") else "inf (closed-loop)"
    print("{:<40} {:<}".format("Request rate:", rate_str))
    print(sep)
    print("{:<40} {:<.2f}".format("Benchmark duration (s):", wall_s))
    print("{:<40} {}/{}".format("Successful requests:", summ.success, total_requests))
    if summ.failed:
        print("{:<40} {:<}".format("Failed requests:", summ.failed))
    print(sep)
    print("{:<40} {:<.3f}".format("Throughput (req/s):", summ.throughput_req_s))
    print("{:<40} {:<.1f}".format("Latency mean (ms):", summ.mean_ms))
    print("{:<40} {:<.1f}".format("Latency p50  (ms):", summ.p50_ms))
    print("{:<40} {:<.1f}".format("Latency p90  (ms):", summ.p90_ms))
    print("{:<40} {:<.1f}".format("Latency p95  (ms):", summ.p95_ms))
    print("{:<40} {:<.1f}".format("Latency p99  (ms):", summ.p99_ms))
    print(sep)
    print("{:<40} {:<.1f}".format("TTFS p50 (ms):", summ.ttfs_p50_ms))
    print("{:<40} {:<.1f}".format("TTFS p99 (ms):", summ.ttfs_p99_ms))
    print("{:<40} {:<.1f}".format("Step latency mean (ms):", summ.step_mean_ms))
    print("{:<40} {:<.1f}".format("Step latency p99  (ms):", summ.step_p99_ms))
    print(sep)
    print("{:<40} {:<.1f}".format("Peak VRAM (MB):", summ.max_vram_mb))
    print("=" * w)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark ComfyUI WAN diffusion model denoising throughput and latency."
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to the WAN diffusion-model checkpoint (.safetensors / .pt).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--num-requests", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max number of in-flight requests (semaphore width).")
    p.add_argument(
        "--request-rate", type=float, default=float("inf"),
        help="Poisson arrival rate in req/s.  inf = closed-loop (default).",
    )
    p.add_argument("--warmup-requests", type=int, default=2,
                   help="Warmup iterations excluded from metrics.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--text-seq-len", type=int, default=512,
                   help="Cross-attention sequence length (UMT5 default: 512).")
    p.add_argument("--text-dim", type=int, default=4096,
                   help="Text embedding width (UMT5-XXL: 4096).")
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/out"))
    p.add_argument("--output-file", type=Path, default=None,
                   help="Override path for the summary JSON output.")
    return p.parse_args()


def default_profiles() -> List[RequestSpec]:
    return [
        RequestSpec("wan21_t2v_720p_16f_30s", 1, 1280, 720, 16, 30, 6.0, 0),
        RequestSpec("wan21_t2v_720p_32f_30s", 1, 1280, 720, 32, 30, 6.0, 0),
        RequestSpec("wan21_t2v_480p_32f_20s", 1,  854, 480, 32, 20, 6.0, 0),
    ]


async def main_async() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runner = WanRunner(
        checkpoint=args.checkpoint,
        device=args.device,
        dtype_str=args.dtype,
        text_seq_len=args.text_seq_len,
        text_dim=args.text_dim,
    )

    all_reqs = build_request_stream(
        args.num_requests + args.warmup_requests,
        args.seed,
        default_profiles(),
    )
    warmup_reqs = all_reqs[: args.warmup_requests]
    bench_reqs = all_reqs[args.warmup_requests :]

    if warmup_reqs:
        print(f"Running {len(warmup_reqs)} warmup request(s)...")
        for req in warmup_reqs:
            runner.run_one(req)
        print("Warmup complete.")

    print(f"Benchmarking {len(bench_reqs)} requests (concurrency={args.concurrency})...")
    t0 = now()
    results = await run_closed_loop(runner, bench_reqs, args.concurrency, args.request_rate)
    wall_s = now() - t0

    summ = summarize(results, wall_s)
    print_summary(args, summ, len(bench_reqs), wall_s)

    out_file = args.output_file or (args.out_dir / "summary.json")
    with open(args.out_dir / "requests.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")
    with open(out_file, "w") as f:
        json.dump(dataclasses.asdict(summ), f, indent=2)
    print(f"\nResults written to {args.out_dir}/")


if __name__ == "__main__":
    asyncio.run(main_async())
