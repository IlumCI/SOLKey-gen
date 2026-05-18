import argparse
import json
import multiprocessing as mp
import os
import re
import secrets
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ALPHABET_BYTES = ALPHABET.encode("ascii")


def b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    out = bytearray()
    while n:
        n, rem = divmod(n, 58)
        out.append(ALPHABET_BYTES[rem])
    pad = 0
    for byte in data:
        if byte == 0:
            pad += 1
        else:
            break
    out.extend(ALPHABET_BYTES[0] for _ in range(pad))
    out.reverse()
    return out.decode("ascii") if out else "1" * pad


def is_valid_base58(value: str) -> bool:
    return all(ch in ALPHABET for ch in value)


@dataclass
class WorkerReport:
    attempts: int
    elapsed: float


def worker(
    worker_id: int,
    targets: tuple[str, ...],
    stop_event: mp.Event,
    queue: mp.Queue,
    report_batch: int,
) -> None:
    # Pull entropy once per worker to ensure independent streams in long runs.
    secrets.token_bytes(32)

    batch_attempts = 0
    started = time.perf_counter()

    while not stop_event.is_set():
        sk = ed25519.Ed25519PrivateKey.generate()
        pk = sk.public_key().public_bytes_raw()
        address = b58encode(pk)
        batch_attempts += 1

        for target in targets:
            if address.endswith(target):
                private_raw = sk.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                queue.put(
                    {
                        "type": "hit",
                        "worker_id": worker_id,
                        "target": target,
                        "address": address,
                        "private_key_hex": private_raw.hex(),
                        "public_key_hex": pk.hex(),
                        "attempts": batch_attempts,
                        "elapsed": time.perf_counter() - started,
                    }
                )
                stop_event.set()
                return

        if batch_attempts >= report_batch:
            queue.put(
                {
                    "type": "progress",
                    "worker_id": worker_id,
                    "attempts": batch_attempts,
                    "elapsed": time.perf_counter() - started,
                }
            )
            batch_attempts = 0

    if batch_attempts:
        queue.put(
            {
                "type": "progress",
                "worker_id": worker_id,
                "attempts": batch_attempts,
                "elapsed": time.perf_counter() - started,
            }
        )


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_secure_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def to_solana_keypair_bytes(private_raw: bytes, public_raw: bytes) -> bytes:
    if len(private_raw) != 32 or len(public_raw) != 32:
        raise ValueError("Expected 32-byte private and public keys")
    return private_raw + public_raw


def load_solana_keypair_file(path: Path) -> bytes:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) != 64:
        raise ValueError("Expected 64-byte Solana keypair JSON array")
    if not all(isinstance(item, int) and 0 <= item <= 255 for item in values):
        raise ValueError("Keypair JSON must contain byte values 0..255")
    return bytes(values)


def run_gpu_grind(targets: tuple[str, ...], output_dir: Path, save_keypair: bool, keypair_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_outfile = output_dir / f"gpu_hit_{uuid.uuid4().hex}.json"
    command = ["solana-keygen", "grind", "--ignore-case", "--use-gpu", "--outfile", str(temp_outfile)]
    for target in targets:
        command.extend(["--ends-with", f"{target}:1"])

    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = max(time.perf_counter() - started, 1e-9)

    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
        raise RuntimeError(f"GPU grind failed: {message}")
    if not temp_outfile.exists():
        raise RuntimeError("GPU grind completed but no keypair outfile was produced")

    keypair_bytes = load_solana_keypair_file(temp_outfile)
    private_raw = keypair_bytes[:32]
    public_raw = keypair_bytes[32:]
    address = b58encode(public_raw)
    matched = next((target for target in targets if address.endswith(target)), "unknown")

    hit_payload = {
        "target": matched,
        "address": address,
        "private_key_hex": private_raw.hex(),
        "public_key_hex": public_raw.hex(),
        "secret_key_base58": b58encode(keypair_bytes),
        "solana_keypair": list(keypair_bytes),
        "worker_id": "gpu",
        "total_attempts": 0,
        "elapsed_seconds": elapsed,
        "created_at_unix": time.time(),
        "engine": "solana-keygen --use-gpu",
    }

    if save_keypair:
        write_secure_text(keypair_path, json.dumps(hit_payload["solana_keypair"]))
    temp_outfile.unlink(missing_ok=True)
    return hit_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-process Solana vanity suffix grinder")
    parser.add_argument(
        "--targets",
        nargs="+",
        required=False,
        help="Suffix targets to match (Base58 only)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Worker process count (default: logical cores)",
    )
    parser.add_argument(
        "--report-batch",
        type=int,
        default=25000,
        help="Attempts between worker progress reports",
    )
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=10.0,
        help="How often to write progress checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="grind_output",
        help="Directory for checkpoint and hit output",
    )
    parser.add_argument(
        "--save-keypair",
        action="store_true",
        help="Write Solana keypair file when a hit is found",
    )
    parser.add_argument(
        "--keypair-file",
        default="id_vanity.json",
        help="Keypair filename (placed under output dir by default)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a speed benchmark and exit",
    )
    parser.add_argument(
        "--benchmark-seconds",
        type=float,
        default=8.0,
        help="Benchmark duration in seconds",
    )
    parser.add_argument(
        "--benchmark-gpu",
        action="store_true",
        help="Try GPU benchmark via solana-keygen grind --use-gpu",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use solana-keygen GPU grinder for actual vanity generation",
    )
    return parser.parse_args()


def benchmark_worker(seconds: float, queue: mp.Queue) -> None:
    started = time.perf_counter()
    attempts = 0
    while True:
        sk = ed25519.Ed25519PrivateKey.generate()
        pk = sk.public_key().public_bytes_raw()
        _ = b58encode(pk)
        attempts += 1
        if (time.perf_counter() - started) >= seconds:
            break
    queue.put((attempts, time.perf_counter() - started))


def run_cpu_benchmark(processes: int, seconds: float) -> float:
    queue: mp.Queue = mp.Queue()
    workers = [mp.Process(target=benchmark_worker, args=(seconds, queue)) for _ in range(processes)]
    for process in workers:
        process.start()
    totals = [queue.get() for _ in workers]
    for process in workers:
        process.join()
    attempts = sum(item[0] for item in totals)
    elapsed = max(item[1] for item in totals)
    return attempts / max(elapsed, 1e-9)


def run_gpu_benchmark(seconds: float) -> tuple[bool, str]:
    target = "111"
    command = [
        "solana-keygen",
        "grind",
        "--ignore-case",
        "--starts-with",
        f"{target}:1",
        "--use-gpu",
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(3.0, seconds),
            check=False,
        )
    except FileNotFoundError:
        return False, "solana-keygen not found in PATH"
    except subprocess.TimeoutExpired:
        return True, "GPU grind started (timed out after benchmark window)"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    merged_output = "\n".join(part for part in (stdout, stderr) if part)
    rate_match = re.search(
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kKmM]?)\s*(?:keys|keypairs|attempts)\s*/\s*(?:s|sec|second)",
        merged_output,
        flags=re.IGNORECASE,
    )
    if rate_match:
        value = float(rate_match.group(1).replace(",", ""))
        scale = rate_match.group(2).lower()
        if scale == "k":
            value *= 1_000
        elif scale == "m":
            value *= 1_000_000
        return True, f"{value:,.0f} keys/sec (reported by solana-keygen)"

    if proc.returncode == 0:
        return True, "GPU grind command succeeded"
    message = stderr or stdout or f"exit code {proc.returncode}"
    return False, message


def main() -> None:
    args = parse_args()

    if args.benchmark:
        cpu_cores = max(1, os.cpu_count() or 1)
        cpu_kps = run_cpu_benchmark(cpu_cores, args.benchmark_seconds)
        print(f"CPU benchmark ({cpu_cores} proc, {args.benchmark_seconds:.1f}s): {cpu_kps:,.0f} keys/sec")
        if args.benchmark_gpu:
            ok, message = run_gpu_benchmark(args.benchmark_seconds)
            prefix = "GPU benchmark" if ok else "GPU benchmark unavailable"
            print(f"{prefix}: {message}")
        return

    cleaned = []
    for target in args.targets or []:
        value = target.strip()
        if not value:
            continue
        if not is_valid_base58(value):
            raise SystemExit(f"Invalid Base58 target: {value}")
        cleaned.append(value)

    if not cleaned:
        raise SystemExit("No valid targets provided")

    targets = tuple(dict.fromkeys(cleaned))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    hit_path = output_dir / "hit.json"
    keypair_path = Path(args.keypair_file)
    if not keypair_path.is_absolute():
        keypair_path = output_dir / keypair_path

    print(f"Targets: {', '.join(targets)}")
    print(f"Processes: {args.processes}")
    print(f"Output dir: {output_dir}")

    if args.gpu:
        try:
            hit_payload = run_gpu_grind(targets, output_dir, args.save_keypair, keypair_path)
        except FileNotFoundError:
            raise SystemExit("solana-keygen not found in PATH (required for --gpu)")
        except Exception as exc:
            raise SystemExit(str(exc))

        checkpoint = {
            "targets": list(targets),
            "processes": "gpu",
            "total_attempts": hit_payload["total_attempts"],
            "elapsed_seconds": hit_payload["elapsed_seconds"],
            "keys_per_second": None,
            "per_worker_attempts": {"gpu": hit_payload["total_attempts"]},
            "updated_at_unix": time.time(),
            "status": "hit",
            "engine": hit_payload["engine"],
        }
        atomic_write_json(hit_path, hit_payload)
        atomic_write_json(checkpoint_path, checkpoint)
        print(f"\nHIT: {hit_payload['address']} (target: {hit_payload['target']})")
        print(f"Hit saved: {hit_path}")
        print(f"Checkpoint: {checkpoint_path}")
        if args.save_keypair:
            print(f"Keypair file: {keypair_path}")
            print(f"Use it: solana config set --keypair {keypair_path}")
        return

    stop_event = mp.Event()
    queue: mp.Queue = mp.Queue()
    workers = [
        mp.Process(
            target=worker,
            args=(idx, targets, stop_event, queue, args.report_batch),
            daemon=True,
        )
        for idx in range(args.processes)
    ]

    def handle_signal(signum: int, _frame: object) -> None:
        print(f"\nReceived signal {signum}; stopping workers...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for process in workers:
        process.start()

    started = time.perf_counter()
    total_attempts = 0
    per_worker_attempts: dict[int, int] = {idx: 0 for idx in range(args.processes)}
    last_checkpoint = started
    hit_payload = None

    while True:
        all_stopped = all(not p.is_alive() for p in workers)
        if all_stopped and queue.empty():
            break

        try:
            msg = queue.get(timeout=0.5)
        except Exception:
            msg = None

        now = time.perf_counter()

        if msg is not None:
            if msg["type"] == "progress":
                attempts = int(msg["attempts"])
                worker_id = int(msg["worker_id"])
                total_attempts += attempts
                per_worker_attempts[worker_id] += attempts
            elif msg["type"] == "hit":
                attempts = int(msg["attempts"])
                worker_id = int(msg["worker_id"])
                total_attempts += attempts
                per_worker_attempts[worker_id] += attempts
                private_raw = bytes.fromhex(msg["private_key_hex"])
                public_raw = bytes.fromhex(msg["public_key_hex"])
                solana_keypair = to_solana_keypair_bytes(private_raw, public_raw)
                hit_payload = {
                    "target": msg["target"],
                    "address": msg["address"],
                    "private_key_hex": msg["private_key_hex"],
                    "public_key_hex": msg["public_key_hex"],
                    "secret_key_base58": b58encode(solana_keypair),
                    "solana_keypair": list(solana_keypair),
                    "worker_id": worker_id,
                    "total_attempts": total_attempts,
                    "elapsed_seconds": now - started,
                    "created_at_unix": time.time(),
                }
                atomic_write_json(hit_path, hit_payload)
                if args.save_keypair:
                    keypair_json = json.dumps(hit_payload["solana_keypair"])
                    write_secure_text(keypair_path, keypair_json)
                stop_event.set()
                print(f"\nHIT: {msg['address']} (target: {msg['target']})")

        elapsed = max(now - started, 1e-9)
        if (now - last_checkpoint) >= args.checkpoint_seconds:
            kps = total_attempts / elapsed
            checkpoint = {
                "targets": list(targets),
                "processes": args.processes,
                "total_attempts": total_attempts,
                "elapsed_seconds": elapsed,
                "keys_per_second": kps,
                "per_worker_attempts": per_worker_attempts,
                "updated_at_unix": time.time(),
            }
            atomic_write_json(checkpoint_path, checkpoint)
            print(
                f"progress: {total_attempts:,} attempts | {kps:,.0f} keys/sec | "
                f"elapsed {elapsed/60:.1f}m"
            )
            last_checkpoint = now

    for process in workers:
        process.join(timeout=1)

    elapsed = max(time.perf_counter() - started, 1e-9)
    final_kps = total_attempts / elapsed
    final_checkpoint = {
        "targets": list(targets),
        "processes": args.processes,
        "total_attempts": total_attempts,
        "elapsed_seconds": elapsed,
        "keys_per_second": final_kps,
        "per_worker_attempts": per_worker_attempts,
        "updated_at_unix": time.time(),
        "status": "hit" if hit_payload else "stopped",
    }
    atomic_write_json(checkpoint_path, final_checkpoint)

    print(f"\nDone. attempts={total_attempts:,}, avg_keys_per_sec={final_kps:,.0f}")
    print(f"Checkpoint: {checkpoint_path}")
    if hit_payload:
        print(f"Hit saved: {hit_path}")
        if args.save_keypair:
            print(f"Keypair file: {keypair_path}")
            print(f"Use it: solana config set --keypair {keypair_path}")
    else:
        print("No hit found in this run.")


if __name__ == "__main__":
    main()
