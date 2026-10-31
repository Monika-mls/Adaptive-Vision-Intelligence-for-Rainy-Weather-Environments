
"""
Phase 3 Step 9: MobileNetV3-Small verification.

Modes:
    --status             Check local weights only (default).
    --download-weights   Explicitly download and verify official weights.
    --benchmark          Run inference-only GPU benchmarks.

No model training, optimizer updates or backward passes.
No dataset images are downloaded.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import torch
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights

from vision.severity.model import (
    CLASS_NAMES,
    create_severity_model,
)


WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1
WEIGHTS_URL = WEIGHTS.url

REPORT_PATH = Path("docs/pretrained_verification.json")
BATCH_SIZES = (1, 2, 4)

EXPECTED_SHA_PREFIX = "047dcff4"


def weight_file():
    """Return Torch's standard local checkpoint path."""

    filename = Path(urlparse(WEIGHTS_URL).path).name

    return (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / filename
    )


def file_sha256(path):
    """Calculate a file's complete SHA256 digest."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_local_weights():
    """
    Verify the downloaded checkpoint's hash prefix
    and confirm it can be loaded as a state dictionary.

    No network access is performed.
    """

    path = weight_file()

    result = {
        "path": str(path),
        "exists": path.is_file(),
        "verified": False,
        "sha256": None,
        "size_bytes": None,
        "tensor_count": None,
        "error": None,
    }

    if not path.is_file():
        result["error"] = "Checkpoint not downloaded."
        return result

    try:
        digest = file_sha256(path)
        result["sha256"] = digest
        result["size_bytes"] = path.stat().st_size

        if not digest.startswith(EXPECTED_SHA_PREFIX):
            raise ValueError(
                "Checkpoint SHA256 does not match "
                "the published filename hash prefix."
            )

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(state, dict) or not state:
            raise ValueError(
                "Checkpoint does not contain a valid "
                "state dictionary."
            )

        result["tensor_count"] = len(state)
        result["verified"] = True

    except Exception as error:
        result["error"] = str(error)

    return result


def download_official_weights():
    """
    Download the official Torchvision checkpoint.

    Uses hash verification and Torch's cache.
    Requires explicit --download-weights.
    """

    print("\nDownloading/verifying official ImageNet weights...")

    # Torch will reuse a cached matching checkpoint.
    torch.hub.load_state_dict_from_url(
        WEIGHTS_URL,
        model_dir=str(weight_file().parent),
        progress=True,
        check_hash=True,
        map_location="cpu",
        weights_only=True,
    )

    result = verify_local_weights()

    if not result["verified"]:
        raise RuntimeError(
            f"Weight verification failed: {result['error']}"
        )

    return result


def mib(value):
    """Convert bytes to mebibytes."""

    return round(value / (1024 ** 2), 2)


def synchronize():
    """Wait for outstanding GPU work to finish."""

    torch.cuda.synchronize()


def cuda_memory():
    """Return currently free and total CUDA device memory."""

    free, total = torch.cuda.mem_get_info()

    return {
        "free_mib": mib(free),
        "total_mib": mib(total),
    }


def benchmark_batch(model, batch_size, device):
    """
    Benchmark inference without gradients.

    Inputs are synthetic normalized tensors.
    Results must not be interpreted as classification
    accuracy or real rain-severity predictions.
    """

    result = {
        "batch_size": batch_size,
        "input_shape": [
            batch_size, 3, 224, 224
        ],
        "status": "pending",
    }

    images = None

    try:
        # Measure PyTorch allocator peaks from this batch.
        torch.cuda.reset_peak_memory_stats()

        memory_before = cuda_memory()

        images = torch.randn(
            batch_size,
            3,
            224,
            224,
            device=device,
            dtype=torch.float32,
        )

        # Warm-up passes are excluded from timing.
        with torch.inference_mode():
            for _ in range(3):
                output = model(images)

            synchronize()

            durations = []

            for _ in range(10):
                synchronize()

                start = time.perf_counter()
                output = model(images)
                synchronize()

                durations.append(
                    (time.perf_counter() - start)
                    * 1000.0
                )

        if output.shape != (batch_size, 2):
            raise RuntimeError(
                f"Unexpected output shape: {output.shape}"
            )

        if not torch.isfinite(output).all().item():
            raise RuntimeError(
                "Model produced non-finite logits."
            )

        result.update({
            "status": "passed",
            "output_shape": list(output.shape),
            "mean_forward_ms": round(
                sum(durations) / len(durations), 3
            ),
            "min_forward_ms": round(
                min(durations), 3
            ),
            "max_forward_ms": round(
                max(durations), 3
            ),
            "peak_allocated_mib": mib(
                torch.cuda.max_memory_allocated()
            ),
            "peak_reserved_mib": mib(
                torch.cuda.max_memory_reserved()
            ),
            "cuda_memory_before": memory_before,
            "cuda_memory_after": cuda_memory(),
        })

    except torch.cuda.OutOfMemoryError as error:
        result["status"] = "out_of_memory"
        result["error"] = str(error)

    except RuntimeError as error:
        result["status"] = "failed"
        result["error"] = str(error)

    finally:
        if images is not None:
            del images

        # Release unused cached memory between batches.
        torch.cuda.empty_cache()

    return result


def run_benchmark():
    """
    Load the verified pretrained model and test
    batches 1, 2, and 4 on CUDA.
    """

    verification = verify_local_weights()

    if not verification["verified"]:
        raise RuntimeError(
            "Verified pretrained weights are required. "
            "Run --download-weights first."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. GPU benchmark cannot run."
        )

    device = torch.device("cuda")

    print("\nLoading MobileNetV3-Small...")
    print("Pretrained backbone: ImageNet")
    print("Binary classes:", CLASS_NAMES)

    # The checkpoint was verified in the local cache.
    # No additional download should be necessary.
    model = create_severity_model(
        pretrained=True,
        freeze_backbone=True,
        device="cuda",
    )

    model.eval()

    if model.model.classifier[3].out_features != 2:
        raise RuntimeError(
            "Binary classification head is incorrect."
        )

    for parameter in model.model.features.parameters():
        if parameter.requires_grad:
            raise RuntimeError(
                "Expected a frozen feature extractor."
            )

    print("Model loaded successfully.")

    synchronize()

    report = {
        "project": "Adaptive Vision Intelligence",
        "phase": "3 - Step 9",
        "verified_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "python_pytorch": {
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_memory": cuda_memory(),
        },
        "model": {
            "architecture": "mobilenet_v3_small",
            "weights": "IMAGENET1K_V1",
            "weights_url": WEIGHTS_URL,
            "weights_verification": verification,
            "classes": list(CLASS_NAMES),
            "num_classes": 2,
            "backbone_frozen": True,
            "trained_for_rain": False,
        },
        "benchmark": {
            "input": "synthetic_normalized_tensor",
            "shape_per_image": [3, 224, 224],
            "mode": "eval_inference_only",
            "warmup_iterations": 3,
            "timed_iterations": 10,
            "batches": [],
        },
        "limitations": [
            "Synthetic tensor benchmark only.",
            "No rain classification accuracy measured.",
            "No gradients or training performed.",
            "Training uses additional GPU memory.",
            "PyTorch allocator metrics do not include "
            "all CUDA context and external allocations.",
        ],
    }

    for batch_size in BATCH_SIZES:
        print(f"\nBenchmarking batch size {batch_size}...")

        result = benchmark_batch(
            model,
            batch_size,
            device,
        )

        report["benchmark"]["batches"].append(result)

        print("Status:", result["status"])

        if result["status"] == "passed":
            print(
                "Mean forward time:",
                result["mean_forward_ms"],
                "ms",
            )
            print(
                "Peak allocated:",
                result["peak_allocated_mib"],
                "MiB",
            )
            print(
                "Peak reserved:",
                result["peak_reserved_mib"],
                "MiB",
            )
        else:
            print("Error:", result.get("error"))

            # Avoid continuing after CUDA failure.
            break

    save_report(report)

    print("\nVerification report:", REPORT_PATH)
    print("Benchmark finished. No training performed.")


def save_report(report):
    """Write the verification report as JSON."""

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            default=str,
        )


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--status",
        action="store_true",
        help="Check locally cached pretrained weights.",
    )

    group.add_argument(
        "--download-weights",
        action="store_true",
        help="Download official pretrained weights.",
    )

    group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference-only CUDA memory benchmarks.",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("MOBILENETV3-SMALL PRETRAINED WEIGHT VERIFICATION")
    print("=" * 65)

    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Weights:", WEIGHTS_URL)
    print("Cache path:", weight_file())

    if args.download_weights:
        result = download_official_weights()
        print(json.dumps(result, indent=2))

    elif args.benchmark:
        run_benchmark()

    else:
        result = verify_local_weights()
        print(json.dumps(result, indent=2))
        print("\nNo downloads or training performed.")


if __name__ == "__main__":
    main()
