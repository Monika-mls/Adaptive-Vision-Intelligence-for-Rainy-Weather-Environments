
"""
MobileNetV3-Small training memory smoke test.

Uses synthetic images and labels.
Runs forward + backward passes without optimizer steps.
Does not update model weights or train on dataset images.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from vision.severity.model import create_severity_model


OUTPUT = Path("docs/training_memory_report.json")
BATCH_SIZES = (1, 2, 4)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda")

    print("=" * 60)
    print("MOBILENETV3 TRAINING MEMORY TEST")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))

    # Uses the already downloaded ImageNet checkpoint.
    model = create_severity_model(
        pretrained=True,
        freeze_backbone=True,
        device="cuda",
    )

    model.train()

    criterion = nn.CrossEntropyLoss()

    results = []

    for batch_size in BATCH_SIZES:
        print(f"\nTesting batch size: {batch_size}")

        images = None
        labels = None
        logits = None
        loss = None

        try:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            images = torch.randn(
                batch_size, 3, 224, 224,
                device=device,
            )

            labels = torch.tensor(
                [i % 2 for i in range(batch_size)],
                device=device,
                dtype=torch.long,
            )

            logits = model(images)
            loss = criterion(logits, labels)

            # Compute gradients, but DO NOT update weights.
            loss.backward()

            torch.cuda.synchronize()

            allocated = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )

            reserved = (
                torch.cuda.max_memory_reserved()
                / (1024 ** 2)
            )

            result = {
                "batch_size": batch_size,
                "status": "passed",
                "peak_allocated_mib": round(allocated, 2),
                "peak_reserved_mib": round(reserved, 2),
            }

            print("Status: PASSED")
            print(f"Peak allocated: {allocated:.2f} MiB")
            print(f"Peak reserved: {reserved:.2f} MiB")

        except torch.cuda.OutOfMemoryError as error:
            result = {
                "batch_size": batch_size,
                "status": "out_of_memory",
                "error": str(error),
            }

            print("Status: OUT OF MEMORY")

        finally:
            results.append(result)
            del images, labels, logits, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if result["status"] != "passed":
            break

    report = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "model": "MobileNetV3-Small",
        "classes": ["Light", "Heavy"],
        "pretrained_backbone": True,
        "backbone_frozen": True,
        "optimizer_steps_performed": 0,
        "dataset_images_used": 0,
        "results": results,
        "limitations": [
            "Synthetic images only.",
            "No optimizer step was executed.",
            "Optimizer-state memory was not measured.",
            "Actual training memory may differ.",
        ],
    }

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nReport saved:", OUTPUT)
    print("No model weights were updated.")


if __name__ == "__main__":
    main()
