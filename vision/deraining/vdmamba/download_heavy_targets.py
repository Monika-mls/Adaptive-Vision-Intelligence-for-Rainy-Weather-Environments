
"""
Download only missing RealRain-1k Heavy clean targets.

Safety:
- No test-set downloads
- No modification to existing input images
- No overwriting existing clean targets
- No modification to checkpoints or model code
- Checks remote paths against the manifest

Set PINNED_REVISION to the previously recorded
dataset commit if available.
"""

import csv
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "dronefreak/RealRain-1k"
PINNED_REVISION = None  # Replace with recorded commit SHA

ROOT = Path.cwd()
DATASETS = ROOT / "datasets"
MANIFEST = ROOT / "docs/severity_labels.csv"


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    api = HfApi()

    info = api.dataset_info(
        REPO_ID,
        revision=PINNED_REVISION,
    )
    revision = info.sha

    files = set(
        api.list_repo_files(
            REPO_ID,
            repo_type="dataset",
            revision=revision,
        )
    )

    required = []

    with MANIFEST.open(
        newline="",
        encoding="utf-8-sig",
    ) as file:
        for row in csv.DictReader(file):
            if row["severity"].strip().lower() != "heavy":
                continue

            split = row["split"].strip().lower()

            if split not in ("train", "validation"):
                continue

            relative = Path(
                row["clean_path"].strip().replace("\\", "/")
            )

            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe path: {relative}")

            filename = relative.as_posix()

            if not filename.startswith(
                f"RealRain-1k-H/{split}/target/"
            ):
                raise ValueError(
                    f"Unexpected clean target: {filename}"
                )

            required.append(filename)

    expected = 784 + 112

    if len(required) != expected or len(set(required)) != expected:
        raise RuntimeError(
            f"Expected {expected} unique target paths; "
            f"found {len(set(required))}"
        )

    unavailable = sorted(set(required) - files)

    print("Dataset revision:", revision)
    print("Required target files:", len(required))
    print("Unavailable remote files:", len(unavailable))

    if unavailable:
        print("Examples:", unavailable[:10])
        raise RuntimeError(
            "Remote file layout differs from the manifest. "
            "No files downloaded."
        )

    existing = sum(
        (DATASETS / filename).is_file()
        for filename in required
    )

    print("Already present:", existing)
    print("Missing:", expected - existing)

    for index, filename in enumerate(sorted(required), 1):
        destination = DATASETS / filename

        if destination.is_file():
            continue

        # Downloads into the dataset folder while
        # maintaining repository-relative paths.
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            local_dir=DATASETS,
        )

        if index % 100 == 0:
            print(f"Checked {index}/{expected} files")

    missing_after = [
        filename
        for filename in required
        if not (DATASETS / filename).is_file()
    ]

    if missing_after:
        raise RuntimeError(
            f"{len(missing_after)} target files still missing."
        )

    print("Download completed successfully.")
    print("Clean training targets: 784")
    print("Clean validation targets: 112")
    print("Test split: untouched")
    print("Dataset revision:", revision)


if __name__ == "__main__":
    main()
