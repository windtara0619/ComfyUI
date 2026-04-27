#!/usr/bin/env python3
"""
Simple serving benchmark client for ComfyUI's HTTP API.

This script is inspired by diffusion serving benchmarks and is designed to:
  - submit prompts to ComfyUI (/prompt or /bench/prompt),
  - optionally shape request arrivals (fixed rate or Poisson),
  - poll completion via /history/{prompt_id},
  - report latency/throughput/error metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import aiohttp


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
    p.add_argument("--prompt-file", type=Path, required=True, help="Path to prompt JSON.")
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
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
