
"""
Step 11: GPU feasibility benchmark for the experimental
VDMamba-inspired image restoration model.

Measures:
- Forward-pass execution time
- Forward + backward execution time
- Peak PyTorch CUDA allocation
- Image shape preservation
- Gradient finiteness

No training dataset or checkpoints are used.
No project files are modified.
Results are saved in a timestamped JSON report.
"""

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_size(size, device, channels=8, state_size=4):
    model = None
    image = None
    output = None
    loss = None

    try:
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        model = ExperimentalVDMambaImageModel(
            channels=channels,
            state_size=state_size,
        ).to(device)

        model.train()

        image = torch.rand(
            1, 3, size, size,
            device=device,
        )

        # Warmup forward pass, excluded from measurements.
        with torch.no_grad():
            warmup = model(image)
            assert warmup.shape == image.shape
            del warmup

        synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()

        output = model(image)

        synchronize(device)
        forward_seconds = time.perf_counter() - start

        if output.shape != image.shape:
            raise RuntimeError("Output shape mismatch")

        if not torch.isfinite(output).all().item():
            raise RuntimeError("Non-finite model output")

        model.zero_grad(set_to_none=True)

        loss = (output - image).square().mean()

        backward_start = time.perf_counter()
        loss.backward()
        synchronize(device)

        backward_seconds = time.perf_counter() - backward_start

        gradients_valid = all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        if not gradients_valid:
            raise RuntimeError("Invalid parameter gradients")

        peak_memory_mib = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else None
        )

        result = {
            "size": size,
            "status": "passed",
            "input_shape": list(image.shape),
            "forward_seconds": round(forward_seconds, 4),
            "backward_seconds": round(backward_seconds, 4),
            "total_seconds": round(
                forward_seconds + backward_seconds, 4
            ),
            "peak_allocated_mib": (
                round(peak_memory_mib, 2)
                if peak_memory_mib is not None else None
            ),
            "gradients_valid": gradients_valid,
        }

    except (RuntimeError, MemoryError) as exc:
        result = {
            "size": size,
            "status": "failed",
            "error": str(exc)[:1000],
        }

    finally:
        del loss, output, image, model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32],
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--state-size",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if any(size < 2 or size > 64 for size in args.sizes):
        parser.error("Sizes must be between 2 and 64.")

    if args.channels < 1 or args.state_size < 1:
        parser.error("Channels and state size must be positive.")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available.")

    device = torch.device(args.device)

    print("=" * 65)
    print("EXPERIMENTAL VDMAMBA FEASIBILITY BENCHMARK")
    print("=" * 65)

    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print("CUDA version:", torch.version.cuda)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    results = []

    for size in sorted(set(args.sizes)):
        print(f"\nTesting {size} x {size} ...", flush=True)

        result = benchmark_size(
            size,
            device,
            channels=args.channels,
            state_size=args.state_size,
        )

        results.append(result)

        print(json.dumps(result, indent=2), flush=True)

        # Stop after failure; larger sizes may be unsafe.
        if result["status"] != "passed":
            print("Stopping after failure.")
            break

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_dir = Path("outputs/vdmamba/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"benchmark_{timestamp}.json"

    report = {
        "model": "ExperimentalVDMambaImageModel",
        "model_status": "untrained_research_prototype",
        "device": str(device),
        "pytorch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "channels": args.channels,
        "state_size": args.state_size,
        "batch_size": 1,
        "warmup": "one forward pass per size",
        "results": results,
    }

    with output_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nBenchmark report:", output_path)


if __name__ == "__main__":
    main()
