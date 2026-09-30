"""Measure indexed image/JSON parsing on a fixed selection of a local tar."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import DataLoader, Subset

from dl_core.datasets import IndexedTarDataset


def parse_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Decode an image and JSON without transferring the full image to the parent."""
    members = sample["members"]
    extension = next(name for name in ("jpg", "jpeg", "png") if name in members)
    image = cv2.imdecode(
        np.frombuffer(members[extension], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError(f"Could not decode {sample['key']}")
    metadata = json.loads(members["json"])
    return {
        "key": sample["key"],
        "pixels": int(image.shape[0] * image.shape[1]),
        "metadata_fields": len(metadata),
    }


def configure_worker(worker_id: int) -> None:
    """Use one decoder thread per worker so worker counts remain comparable."""
    cv2.setNumThreads(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tar", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    args = parser.parse_args()
    if args.samples < 1 or any(workers < 0 for workers in args.workers):
        parser.error("samples must be positive and worker counts nonnegative")
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix="indexed-tar-benchmark-") as temporary:
        started = time.perf_counter()
        dataset = IndexedTarDataset(
            [args.tar],
            transform=parse_sample,
            required_extensions=["json"],
            index_dir=temporary,
        )
        count = min(args.samples, len(dataset))
        if not count:
            raise ValueError("Tar has no eligible samples")
        print(
            json.dumps(
                {
                    "tar": str(args.tar),
                    "indexed_samples": len(dataset),
                    "index_build_seconds": round(time.perf_counter() - started, 3),
                    "benchmark_samples": count,
                }
            ),
            flush=True,
        )
        # Warm the selected payloads once for all worker configurations.
        for index in range(count):
            dataset[index]
        dataset.close()
        for workers in args.workers:
            loader = DataLoader(
                Subset(dataset, range(count)),
                batch_size=16,
                num_workers=workers,
                worker_init_fn=configure_worker,
                prefetch_factor=2 if workers else None,
                persistent_workers=workers > 0,
            )
            started = time.perf_counter()
            for batch in loader:
                pass
            startup = time.perf_counter() - started
            started = time.perf_counter()
            keys = []
            for batch in loader:
                keys.extend(batch["key"])
            elapsed = time.perf_counter() - started
            assert len(keys) == count and len(set(keys)) == count
            print(
                json.dumps(
                    {
                        "workers": workers,
                        "first_pass_seconds": round(startup, 3),
                        "steady_pass_seconds": round(elapsed, 3),
                        "samples_per_second": round(count / elapsed, 2),
                    }
                ),
                flush=True,
            )
            # Dropping the loader terminates its persistent worker pool before
            # the next configuration and before the temporary index is removed.
            del loader
            dataset.close()


if __name__ == "__main__":
    main()
