"""Local API tests. Uses a fake restoration function (no model/GPU)."""
import io
from pathlib import Path

import pytest
from PIL import Image

import backend.app as module


def png_bytes():
    buffer = io.BytesIO()
    Image.new('RGB', (80, 64), (120, 160, 180)).save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


@pytest.fixture()
def client(tmp_path, monkeypatch):
    app = module.create_app({'TESTING': True, 'UPLOAD_ROOT': str(tmp_path)})

    def fake_run(input_path, output_path, max_side):
        with Image.open(input_path) as image:
            image.save(output_path)
        return {
            'original_size': [80, 64], 'processed_size': [80, 64],
            'output_size': [80, 64], 'device': 'cpu', 'recurrent_stages': 6,
            'inference_seconds': 0.01, 'peak_gpu_allocated_mib': None,
        }

    monkeypatch.setattr(module, 'run_prenet', fake_run)
    return app.test_client()


def test_health(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json['mode'] == 'direct_prenet'


def test_restoration_and_output(client):
    response = client.post('/api/restore', data={
        'image': (png_bytes(), 'rain.png'), 'max_side': '256',
    })
    assert response.status_code == 200
    data = response.json
    assert data['model'].startswith('PReNet6')
    assert data['severity_classification_performed'] is False
    assert client.get(data['input_url']).status_code == 200
    assert client.get(data['restored_url']).status_code == 200


def test_reject_missing_image(client):
    assert client.post('/api/restore', data={}).status_code == 400


def test_reject_wrong_extension(client):
    result = client.post('/api/restore', data={'image': (png_bytes(), 'rain.txt')})
    assert result.status_code == 400


def test_reject_invalid_resolution(client):
    result = client.post('/api/restore', data={'image': (png_bytes(), 'rain.png'), 'max_side': '999'})
    assert result.status_code == 400


def test_reject_forged_result(client):
    assert client.get('/api/results/notanid/input.png').status_code == 404
    assert client.get('/api/results/' + 'a' * 32 + '/secret.txt').status_code == 404
