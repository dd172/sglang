"""Run a single-GPU RadixAttention on/off experiment against SGLang v0.5.6."""

import argparse
import csv
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib.metadata import version
from pathlib import Path


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def workload():
    rng = random.Random(172)

    def tokens(count):
        return rng.sample(range(10, 150_000), count)

    base = tokens(2048)
    seed = base + [21] + tokens(127)
    return [
        ("cold_seed", seed, 0),
        ("shared_2048", base + [22] + tokens(127), 2048),
        ("shared_1024", base[:1024] + [23] + tokens(1151), 1024),
        ("different_prefix_same_suffix", [24] + seed[1:], 0),
        ("identical_replay", seed, len(seed) - 1),
    ]


def choose_gpu(requested):
    fields = "index,uuid,name,memory.used,driver_version"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    )
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        index, uuid, name, used_mib, driver = row
        selected = requested in (index, uuid) if requested else "NVIDIA A40" in name
        if selected and int(used_mib) < 512:
            return {"index": index, "uuid": uuid, "name": name, "driver": driver}
    raise RuntimeError("No eligible GPU has less than 512 MiB in use.")


def generate(url, input_ids):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
        "stream": True,
    }
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first_token_ms = None
    output_ids = []
    meta = None
    with HTTP.open(req, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            received = time.perf_counter()
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            output_ids = event["output_ids"]
            meta = event["meta_info"]
            if output_ids and first_token_ms is None:
                first_token_ms = (received - start) * 1000
        else:
            raise RuntimeError("The generation stream ended before [DONE].")
    if first_token_ms is None or len(output_ids) != 1:
        raise RuntimeError(f"Expected one generated token, got {output_ids!r}.")
    return meta, output_ids[0], first_token_ms


def wait_ready(server, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"SGLang exited with status {server.returncode}.")
        try:
            with HTTP.open(url + "/health", timeout=2):
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    raise TimeoutError(f"SGLang did not become ready in {timeout} seconds.")


def stop_server(server):
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=10)
    except ProcessLookupError:
        server.wait()
    except subprocess.TimeoutExpired:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        server.wait()


def record_trace(url, trace, records, enabled, trial):
    with HTTP.open(urllib.request.Request(url + "/flush_cache", data=b""), timeout=30):
        pass
    rows = []
    for name, prompt, expected in trace:
        meta, token, ttft = generate(url, prompt)
        cached = meta["cached_tokens"]
        row = {
            "cache_enabled": enabled,
            "trial": trial,
            "case": name,
            "prompt_tokens": meta["prompt_tokens"],
            "expected_cached_tokens": expected if enabled else 0,
            "cached_tokens": cached,
            "prefill_tokens": len(prompt) - cached,
            "ttft_ms": ttft,
            "output_token_id": token,
        }
        records.write(json.dumps(row) + "\n")
        records.flush()
        if row["prompt_tokens"] != len(prompt) or cached != row["expected_cached_tokens"]:
            raise RuntimeError(f"Unexpected token counts: {row!r}")
        rows.append(row)
    return rows


def run_mode(args, gpu, trace, records, enabled):
    mode = "on" if enabled else "off"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model_path, "--host", "127.0.0.1", "--port", str(port),
        "--dtype", "float16", "--attention-backend", "triton",
        "--sampling-backend", "pytorch", "--page-size", "1",
        "--disable-cuda-graph", "--schedule-policy", "fcfs",
        "--context-length", "4096", "--max-total-tokens", "16384",
        "--chunked-prefill-size", "-1", "--max-running-requests", "1",
        "--skip-tokenizer-init",
    ]
    if args.model_path == MODEL:
        command.extend(["--revision", REVISION])
    if not enabled:
        command.append("--disable-radix-cache")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"])
    env.setdefault("HF_HOME", str(Path.home() / ".cache" / "radix-attention" / "huggingface"))
    env.setdefault("TRITON_CACHE_DIR", str(Path.home() / ".cache" / "radix-attention" / "triton"))
    log_path = args.output / f"server-{mode}.log"
    print(f"RadixAttention {mode}, server log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        server = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            wait_ready(server, url, args.startup_timeout)
            for trial in range(-2, 0):
                record_trace(url, trace, records, enabled, trial)
            rows = []
            for trial in range(args.rounds):
                rows.extend(record_trace(url, trace, records, enabled, trial))
            return rows
        finally:
            stop_server(server)


def summarize(trace, on, off):
    summary = []
    for name, _, _ in trace:
        hits = [row for row in on if row["case"] == name]
        misses = [row for row in off if row["case"] == name]
        row = {
            "case": name,
            "cached_tokens_on": hits[0]["cached_tokens"],
            "cached_tokens_off": misses[0]["cached_tokens"],
            "ttft_ms_on_p50": statistics.median(r["ttft_ms"] for r in hits),
            "ttft_ms_off_p50": statistics.median(r["ttft_ms"] for r in misses),
            "greedy_token_agreements": sum(
                a["output_token_id"] == b["output_token_id"] for a, b in zip(hits, misses)
            ),
        }
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", help="Idle GPU index or UUID. Defaults to the first idle A40.")
    parser.add_argument("--model-path", default=MODEL, help="Optional local Qwen checkpoint.")
    parser.add_argument("--rounds", default=10, type=int)
    parser.add_argument("--startup-timeout", default=900, type=int)
    parser.add_argument("--output", type=Path, required=True, help="New results directory.")
    args = parser.parse_args()
    if args.rounds < 1 or args.startup_timeout < 1:
        parser.error("--rounds and --startup-timeout must be positive")
    sglang_version = version("sglang")
    if sglang_version != "0.5.6":
        parser.error(f"Expected SGLang 0.5.6, found {sglang_version}")

    gpu = choose_gpu(args.gpu)
    args.output.mkdir(parents=True, exist_ok=False)
    trace = workload()
    manifest = {
        "gpu": gpu,
        "python": sys.version,
        "sglang": sglang_version,
        "torch": version("torch"),
        "model": args.model_path,
        "model_revision": REVISION if args.model_path == MODEL else None,
        "rounds_per_mode": args.rounds,
        "trial_numbers_excluded_from_summary": [-2, -1],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.output / "requests.jsonl").open("x", encoding="utf-8") as records:
        on = run_mode(args, gpu, trace, records, enabled=True)
        off = run_mode(args, gpu, trace, records, enabled=False)
    summary = summarize(trace, on, off)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
