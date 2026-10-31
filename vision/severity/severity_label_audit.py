
"""
Phase 3: RealRain-1k severity label audit.

Audits original Light and Heavy image-pair metadata,
checks matched scene IDs and split consistency,
and creates a provenance-preserving label manifest.

No image downloads, synthetic generation or training.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SOURCE = Path("docs/dataset_manifest.csv")
OUTPUT = Path("docs/severity_label_audit.json")
LABEL_MANIFEST = Path("docs/severity_labels.csv")

SPLITS = {"train", "validation", "test"}
SEVERITIES = {"Light", "Heavy"}


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing {SOURCE}. Complete Phase 1 first."
        )

    with SOURCE.open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Dataset manifest is empty.")

    errors = []
    scenes = defaultdict(dict)
    label_rows = []
    seen_paths = set()

    for row in rows:
        severity = row["severity"]
        split = row["split"]
        image_id = row["image_id"]
        rainy_path = row["rainy_path"]
        clean_path = row["clean_path"]

        if severity not in SEVERITIES:
            errors.append(
                f"Unexpected severity: {severity}"
            )
            continue

        if split not in SPLITS:
            errors.append(f"Unexpected split: {split}")
            continue

        if row["pair_valid"].lower() != "true":
            errors.append(
                f"Invalid image pair: {rainy_path}"
            )
            continue

        if not rainy_path or not clean_path:
            errors.append(
                f"Missing path for {image_id}"
            )
            continue

        if rainy_path in seen_paths:
            errors.append(
                f"Duplicate rainy path: {rainy_path}"
            )
            continue

        seen_paths.add(rainy_path)

        # Pair Light and Heavy variants by scene ID.
        scene_key = (split, image_id)

        if severity in scenes[scene_key]:
            errors.append(
                f"Duplicate severity for {scene_key}"
            )
            continue

        scenes[scene_key][severity] = row

        label_rows.append({
            "image_id": image_id,
            "scene_group_id": image_id,
            "split": split,
            "severity": severity,
            "label_source": "RealRain-1k original track",
            "label_status": "original_track_label",
            "rainy_path": rainy_path,
            "clean_path": clean_path,
        })

    paired_scenes = []
    incomplete_scenes = []
    target_path_mismatches = []

    for (split, image_id), variants in sorted(
        scenes.items()
    ):
        if set(variants) == SEVERITIES:
            paired_scenes.append((split, image_id))

            light_target = variants["Light"]["clean_path"]
            heavy_target = variants["Heavy"]["clean_path"]

            # Different paths are expected across tracks.
            # Content equality requires hashes and is not
            # established by comparing filenames alone.
            if Path(light_target).name != Path(
                heavy_target
            ).name:
                target_path_mismatches.append({
                    "split": split,
                    "image_id": image_id,
                })
        else:
            incomplete_scenes.append({
                "split": split,
                "image_id": image_id,
                "available": sorted(variants),
            })

    image_id_splits = defaultdict(set)

    for split, image_id in scenes:
        image_id_splits[image_id].add(split)

    cross_split_ids = {
        image_id: sorted(split_set)
        for image_id, split_set in image_id_splits.items()
        if len(split_set) > 1
    }

    counts = Counter(
        (row["split"], row["severity"])
        for row in label_rows
    )

    summary = {
        "dataset": "RealRain-1k",
        "label_classes_available": [
            "Light",
            "Heavy",
        ],
        "medium_labels_available": False,
        "total_original_labels": len(label_rows),
        "matched_light_heavy_scene_groups": len(
            paired_scenes
        ),
        "incomplete_scene_groups": incomplete_scenes,
        "cross_split_image_ids": cross_split_ids,
        "target_filename_mismatches": (
            target_path_mismatches
        ),
        "errors": errors,
        "counts": {
            f"{split}/{severity}": count
            for (split, severity), count
            in sorted(counts.items())
        },
        "limitations": [
            "Scene grouping assumes matching IDs "
            "identify corresponding scenes.",
            "Scene content is not independently verified.",
            "Clean target hashes are not verified "
            "for every scene.",
            "Medium is not an original dataset label.",
            "No synthetic images were generated.",
            "No classifier was trained.",
        ],
    }

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    with LABEL_MANIFEST.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "image_id",
            "scene_group_id",
            "split",
            "severity",
            "label_source",
            "label_status",
            "rainy_path",
            "clean_path",
        ]

        writer = csv.DictWriter(
            file, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(label_rows)

    print("=" * 60)
    print("REALRAIN-1K SEVERITY LABEL AUDIT")
    print("=" * 60)

    print("Original labels:", len(label_rows))
    print("Matched scene groups:", len(paired_scenes))
    print("Incomplete groups:", len(incomplete_scenes))
    print("Cross-split IDs:", len(cross_split_ids))
    print("Audit errors:", len(errors))

    print("\nOriginal label distribution:")

    for key, value in sorted(
        summary["counts"].items()
    ):
        print(f"  {key}: {value}")

    print("\nMedium labels: NOT AVAILABLE")
    print("Synthetic images generated: 0")
    print("Training performed: NO")

    print("\nAudit report:", OUTPUT)
    print("Label manifest:", LABEL_MANIFEST)

    if errors or incomplete_scenes or cross_split_ids:
        raise RuntimeError(
            "Severity audit found issues. "
            "Review the generated JSON report."
        )

    print("\nMetadata audit passed.")


if __name__ == "__main__":
    main()
