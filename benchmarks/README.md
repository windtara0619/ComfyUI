# ComfyUI Serving Benchmarks

Measures latency and throughput of a running ComfyUI server by submitting
concurrent prompt requests and collecting results from the history API.

## Dependencies

```bash
pip install aiohttp tqdm gdown
```

## Supported models / tasks

| Model | Task | Description |
|-------|------|-------------|
| `wan22` | `i2v` | Wan 2.2 Image-to-Video — LightX2V 4-step, 720×720, 81 frames |

To add a new model/task: drop a workflow JSON in `workflows/` (with
`__INPUT_IMAGE__` as the image placeholder) and add an entry to
`_MODEL_REGISTRY` in `benchmark_comfyui_serving.py`.

## How it works

On each run the script:

1. Downloads model weights into the ComfyUI `models/` directory (only if
   `--download-models` is passed).
2. Downloads the [VBench I2V](https://github.com/Vchitect/VBench) image
   dataset via `gdown` into ComfyUI's `input/` folder.
3. Generates one prompt JSON per input image under
   `benchmarks/prompts/<model>_<task>/`.
4. Submits `--num-requests` prompts to the server, cycling through the
   generated prompt files in round-robin order.
5. Polls `/history/{prompt_id}` for completion and prints a latency /
   throughput summary.

Per-node execution times are available when the server is started with
`--benchmark-server-only`.

## Usage

### Start the server

```bash
python main.py --listen 127.0.0.1 --port 8188 --benchmark-server-only
```

### Run the benchmark

```bash
# From the ComfyUI root directory:
python3 benchmarks/benchmark_comfyui_serving.py \
  --model wan22 --task i2v \
  --num-requests 50 --max-concurrency 4 \
  --host http://127.0.0.1:8188
```

Include model weight download on first run:

```bash
python3 benchmarks/benchmark_comfyui_serving.py \
  --model wan22 --task i2v \
  --download-models --comfyui-base-dir /path/to/ComfyUI \
  --num-requests 50 --max-concurrency 4 \
  --host http://127.0.0.1:8188
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Model name (e.g. `wan22`) |
| `--task` | *(required)* | Task type (e.g. `i2v`) |
| `--host` | `http://127.0.0.1:8188` | ComfyUI base URL |
| `--num-requests` | `50` | Total requests to submit |
| `--max-concurrency` | `8` | Max in-flight requests |
| `--request-rate` | `0` | Requests/sec; `0` = fire immediately |
| `--poisson` | off | Poisson inter-arrival when `--request-rate > 0` |
| `--num-images` | `20` | Synthetic images if VBench download unavailable |
| `--prompts-dir` | `benchmarks/prompts/<model>_<task>/` | Prompt JSON output directory |
| `--download-models` | off | Download model weights before benchmarking |
| `--comfyui-base-dir` | — | ComfyUI root (required with `--download-models`) |
| `--output-json` | — | Write full per-request results to a JSON file |

## Output

```
benchmark: 100%|████████████| 50/50 [req, succeeded=50]

=== ComfyUI Serving Benchmark Summary ===
requests_total:   50
requests_success: 50
requests_failed:  0
wall_time_s:      412.341
throughput_req_s: 0.121
latency_p50_s:    38.201
latency_p90_s:    52.110
latency_p95_s:    55.837
latency_p99_s:    60.012
latency_mean_s:   39.445
latency_max_s:    61.203
execution_mean_ms: 35210.44
execution_p95_ms:  51200.11

--- Per-node execution time (mean ms across successful requests) ---
  KSampler (Advanced) (130:110): mean=18200.1  p95=22100.3  n=50
  KSampler (Advanced) (130:111): mean=16900.4  p95=20800.7  n=50
  VAEDecode (130:129):           mean=420.2    p95=510.1    n=50
  ...
```
