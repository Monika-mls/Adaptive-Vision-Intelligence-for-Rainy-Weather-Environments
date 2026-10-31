
"""
Tests for binary Light/Heavy severity dataset metadata.

No model training or image downloads.
"""

import csv
from collections import Counter
from pathlib import Path

import pytest


MANIFEST = Path("docs/severity_labels.csv")

CLASSES = ("Light", "Heavy")

CLASS_TO_INDEX = {
    "Light": 0,
    "Heavy": 1,
}


@pytest.fixture(scope="module")
def dataset_rows():
    assert MANIFEST.exists(), (
        "Run severity_label_audit.py first."
    )

    with MANIFEST.open(
        newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def test_only_two_classes(dataset_rows):
    classes = {
        row["severity"] for row in dataset_rows
    }

    assert classes == set(CLASSES)


def test_class_mapping():
    assert CLASS_TO_INDEX == {
        "Light": 0,
        "Heavy": 1,
    }


def test_total_images(dataset_rows):
    assert len(dataset_rows) == 2240


def test_class_balance(dataset_rows):
    counts = Counter(
        row["severity"] for row in dataset_rows
    )

    assert counts["Light"] == 1120
    assert counts["Heavy"] == 1120


def test_expected_splits(dataset_rows):
    counts = Counter(
        (row["split"], row["severity"])
        for row in dataset_rows
    )

    expected = {
        ("train", "Light"): 784,
        ("train", "Heavy"): 784,
        ("validation", "Light"): 112,
        ("validation", "Heavy"): 112,
        ("test", "Light"): 224,
        ("test", "Heavy"): 224,
    }

    assert counts == expected


def test_no_scene_split_leakage(dataset_rows):
    scene_splits = {}

    for row in dataset_rows:
        scene = row["scene_group_id"]
        split = row["split"]

        if scene in scene_splits:
            assert scene_splits[scene] == split
        else:
            scene_splits[scene] = split


def test_original_label_provenance(dataset_rows):
    for row in dataset_rows:
        assert row["label_status"] == (
            "original_track_label"
        )


def test_no_medium_class(dataset_rows):
    assert all(
        row["severity"] != "Medium"
        for row in dataset_rows
    )
