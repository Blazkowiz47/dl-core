"""Measure index construction and image/JSON reading across local plain tars."""

from __future__ import annotations

import argparse
import json
import logging
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
    parser.add_argument("tar", type=Path, nargs="+")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--index-workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or any(workers < 0 for workers in args.workers):
        parser.error("samples must be positive and worker counts nonnegative")
    if any(workers < 1 for workers in args.index_workers):
        parser.error("index worker counts must be positive")
    logging.basicConfig(level=logging.INFO)
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix="indexed-tar-benchmark-") as temporary:
        for position, workers in enumerate(args.index_workers):
            root = Path(temporary) / str(position)
            for cache in ("cold", "warm"):
                dataset = IndexedTarDataset(
                    args.tar,
                    transform=parse_sample,
                    required_extensions=["json"],
                    index_dir=root,
                    index_workers=workers,
                )
                print(
                    json.dumps(
                        {
                            "stage": "index",
                            "cache": cache,
                            "index_workers": workers,
                            **dataset.index_stats,
                        }
                    ),
                    flush=True,
                )
                dataset.close()
        if args.index_only:
            return
        count = min(args.samples, len(dataset))
        if not count:
            raise ValueError("Tar has no eligible samples")
        print(
            json.dumps(
                {
                    "tars": [str(path) for path in args.tar],
                    "indexed_samples": len(dataset),
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
