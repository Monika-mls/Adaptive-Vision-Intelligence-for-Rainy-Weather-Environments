"""Fail-closed integration contract for the EXACT single-image VDMamba (IEEE TMM 2026).

This is NOT an implementation of VDMamba. It audits source/checkpoint availability
and provides one explicit adapter boundary. No training or fake restoration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PAPER_TITLE = "VDMamba: Vector Decomposition in Vision Mamba for Image Deraining and Beyond"
PAPER_DOI = "10.1109/TMM.2026.3651113"
EXPECTED_AUTHORS = ("Kui Jiang", "Junjun Jiang", "Shiqi Wang", "Wenqi Ren", "Chia-Wen Lin", "Zhengguo Li")


@dataclass(frozen=True)
class HeavyModelFiles:
    source_dir: Path = Path("external_models/VDMamba_image_deraining")
    checkpoint: Path = Path("checkpoints/deraining/vdmamba_official.pth")
    provenance: Path = Path("docs/vdmamba_model_provenance.json")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(config: HeavyModelFiles) -> dict:
    """Pure read-only audit. No code imports, no checkpoint deserialization."""
    source_exists = config.source_dir.is_dir()
    weights_exist = config.checkpoint.is_file()
    provenance_exists = config.provenance.is_file()
    problems = []
    meta = None

    if not source_exists:
        problems.append("Exact paper implementation is not installed")
    if not weights_exist:
        problems.append("Verified pretrained deraining checkpoint is missing")
    if not provenance_exists:
        problems.append("Official checkpoint provenance record is missing")
    else:
        try:
            meta = json.loads(config.provenance.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"Cannot read provenance: {exc}")
        else:
            if meta.get("paper_doi") != PAPER_DOI:
                problems.append("Provenance DOI does not match target paper")
            if meta.get("source_type") != "authors_official_release":
                problems.append("Source has not been identified as authors' official release")
            if not meta.get("source_url"):
                problems.append("Provenance source_url missing")
            if not meta.get("checkpoint_url"):
                problems.append("Provenance checkpoint_url missing")
            if not meta.get("license_or_permission"):
                problems.append("License or explicit usage permission is unrecorded")
            if weights_exist and meta.get("checkpoint_sha256") != file_hash(config.checkpoint):
                problems.append("Checkpoint SHA256 differs from recorded value")

    # Even a complete file audit cannot prove model API/checkpoint architecture.
    return {
        "paper": PAPER_TITLE,
        "doi": PAPER_DOI,
        "source_exists": source_exists,
        "checkpoint_exists": weights_exist,
        "provenance_exists": provenance_exists,
        "provenance_verified": len(problems) == 0,
        "inference_verified": False,
        "status": "ready_for_architecture_test" if not problems else "model_unavailable",
        "problems": problems,
        "no_training_performed": True,
        "no_restoration_performed": True,
    }


def restore_heavy(*, input_path: Path, output_path: Path, config: HeavyModelFiles | None = None):
    """Explicit integration boundary; never manufacture a result."""
    config = config or HeavyModelFiles()
    result = audit(config)
    if result["status"] != "ready_for_architecture_test":
        raise RuntimeError("VDMamba unavailable: " + "; ".join(result["problems"]))
    raise NotImplementedError(
        "Artifacts were verified, but the official model's constructor, "
        "checkpoint loading and tensor I/O must pass an architecture test "
        "before real restoration can be enabled."
    )


def main():
    parser = argparse.ArgumentParser(description="Audit exact VDMamba Heavy branch (read only)")
    parser.add_argument("--source-dir", type=Path, default=HeavyModelFiles.source_dir)
    parser.add_argument("--checkpoint", type=Path, default=HeavyModelFiles.checkpoint)
    parser.add_argument("--provenance", type=Path, default=HeavyModelFiles.provenance)
    args = parser.parse_args()
    print(json.dumps(audit(HeavyModelFiles(args.source_dir,args.checkpoint,args.provenance)),indent=2))


if __name__ == "__main__":
    main()
