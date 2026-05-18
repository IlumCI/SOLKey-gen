import os
import time
import multiprocessing as mp
from cryptography.hazmat.primitives.asymmetric import ed25519


ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
TARGETS = ("euro", "europa", "europe", "swarms", "Lucre")


def b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    out = bytearray()
    while n:
        n, rem = divmod(n, 58)
        out.append(ALPHABET[rem])
    pad = 0
    for b in data:
        if b == 0:
            pad += 1
        else:
            break
    out.extend(ALPHABET[0] for _ in range(pad))
    out.reverse()
    return out.decode("ascii") if out else "1" * pad


def worker(seconds: float, q: mp.Queue) -> None:
    t0 = time.perf_counter()
    c = 0
    hits = 0
    while True:
        sk = ed25519.Ed25519PrivateKey.generate()
        pk = sk.public_key().public_bytes_raw()
        addr = b58encode(pk)
        c += 1
        for target in TARGETS:
            if addr.endswith(target):
                hits += 1
                break
        if (time.perf_counter() - t0) >= seconds:
            break
    q.put((c, hits, time.perf_counter() - t0))


def run(processes: int, seconds: float = 8.0) -> tuple[float, float]:
    q: mp.Queue = mp.Queue()
    workers = [mp.Process(target=worker, args=(seconds, q)) for _ in range(processes)]
    for process in workers:
        process.start()
    totals = [q.get() for _ in workers]
    for process in workers:
        process.join()
    keys = sum(item[0] for item in totals)
    hits = sum(item[1] for item in totals)
    elapsed = max(item[2] for item in totals)
    return keys / elapsed, hits / elapsed


def main() -> None:
    cores = os.cpu_count() or 1
    print(f"Logical CPU cores: {cores}")
    seen = set()
    for processes in [1, min(2, cores), min(4, cores), min(8, cores), min(16, cores), cores]:
        processes = max(1, processes)
        if processes in seen:
            continue
        seen.add(processes)
        keys_per_second, hits_per_second = run(processes)
        print(
            f"{processes:>2} proc -> {keys_per_second:,.0f} keys/sec, "
            f"{hits_per_second:.5f} hits/sec"
        )


if __name__ == "__main__":
    main()
