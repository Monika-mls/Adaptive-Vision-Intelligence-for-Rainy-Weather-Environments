import hashlib
import json
from pathlib import Path
import pytest
from vision.deraining.vdmamba_contract import HeavyModelFiles, PAPER_DOI, audit, restore_heavy


def test_missing_weights_fails_closed(tmp_path):
    config = HeavyModelFiles(tmp_path / "src",tmp_path / "model.pth",tmp_path / "provenance.json")
    result = audit(config)
    assert result["status"] == "model_unavailable"
    assert result["inference_verified"] is False
    with pytest.raises(RuntimeError, match="unavailable"):
        restore_heavy(input_path=tmp_path / "rainy.png",output_path=tmp_path / "restored.png",config=config)
    assert not (tmp_path / "restored.png").exists()


def test_complete_audit_still_does_not_claim_inference(tmp_path):
    source=tmp_path / "src"; source.mkdir()
    checkpoint=tmp_path / "model.pth"; checkpoint.write_bytes(b"dummy not actual pretrained weights")
    meta=tmp_path / "provenance.json"
    meta.write_text(json.dumps({"paper_doi":PAPER_DOI,"source_type":"authors_official_release","source_url":"https://example.org/source","checkpoint_url":"https://example.org/model","license_or_permission":"test-only placeholder","checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}),encoding="utf-8")
    config = HeavyModelFiles(source,checkpoint,meta)
    result=audit(config)
    assert result["status"] == "ready_for_architecture_test"
    assert result["inference_verified"] is False
    with pytest.raises(NotImplementedError):
        restore_heavy(input_path=tmp_path / "rainy.png",output_path=tmp_path / "restored.png",config=config)


def test_hash_mismatch_rejected(tmp_path):
    source=tmp_path / "src";source.mkdir()
    ck=tmp_path/"checkpoint.pth";ck.write_bytes(b"actual")
    prov=tmp_path/"metadata.json"
    prov.write_text(json.dumps({"paper_doi":PAPER_DOI,"source_type":"authors_official_release","source_url":"https://example.org/source","checkpoint_url":"https://example.org/checkpoint","license_or_permission":"test","checkpoint_sha256":"bad"}),encoding="utf-8")
    assert "Checkpoint SHA256 differs from recorded value" in audit(HeavyModelFiles(source,ck,prov))["problems"]
