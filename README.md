# Adaptive Vision Intelligence for Rainy Weather Environments

**B.Tech Final-Year Major Project | Phase 1 Implementation**  
**Domain:** Computer Vision and Deep Learning  
**Team:** 11 | **Project Guide:** Dr. G. Ramesh Chandra

An adaptive image-restoration research prototype that predicts the severity of rain in an uploaded photograph and selects one of two restoration models. The application includes a local Flask API and a React web interface for uploading images, inspecting predictions, comparing before/after images, and downloading restored outputs.

> **Project status (Phase 1):** Severity prediction and Light/Heavy restoration are implemented. Automatic routing includes an uncertainty gate, with manual restoration available. **Object detection, integrated detector evaluation, and the final end-to-end pipeline are planned for Phase 2.** This is a research prototype, not a production safety system.

## Contents

1. [Features and architecture](#features-and-architecture)
2. [Hardware and software requirements](#hardware-and-software-requirements)
3. [Clone and install](#clone-and-install)
4. [Obtain the model code and checkpoints](#obtain-the-model-code-and-checkpoints)
5. [Dataset setup (for experiments)](#dataset-setup-for-experiments)
6. [Run the application](#run-the-application)
7. [API usage](#api-usage)
8. [Testing and research commands](#testing-and-research-commands)
9. [Folder structure](#folder-structure)
10. [Troubleshooting](#troubleshooting)
11. [Known limitations](#known-limitations)
12. [Reproducibility, licensing and Phase 2](#reproducibility-licensing-and-phase-2)

## Features and architecture

```text
                    Rainy image (PNG / JPG / WEBP)
                                  |
                                  v
                   MobileNetV3-Small classifier
                         Light / Heavy
                                  |
                     Confidence + margin gate
                       /                  \
                 accepted               uncertain
                    |                       |
              Select by class         Block automatic
              /           \           restoration;
          Light           Heavy        allow manual choice
            |               |
         PReNet6       Fine-tuned Restormer
            |          (overlapping tiles)
            \               /
             Restored image + processing metrics
                           |
                    React results page
```

- **Automatic mode:** Predict rain severity and select **PReNet6 (Light)** or **fine-tuned Restormer (Heavy)** only if the classifier accepts the prediction.
- **Manual mode:** Choose Light/PReNet6 or Heavy/Restormer explicitly, including when the automatic route is uncertain.
- **Light model:** PReNet6, using a pretrained Rain100L checkpoint; inference resizes to a selectable maximum dimension.
- **Heavy model:** Official Restormer architecture, initialized from official pretrained deraining weights and fine-tuned on the Heavy subset of RealRain-1k. Heavy inference processes overlapping tiles and preserves image dimensions.
- **UI:** Drag-and-drop upload, original/restored comparison slider, downloadable output, prediction confidence (in automatic mode), execution model, time, resolution, and GPU memory metrics.
- **Research-only comparison:** An experimental *VDMamba-inspired* implementation and CNN baseline exist in the repository. **They are not official VDMamba implementations and are not the selected production-path Heavy model.**

## Hardware and software requirements

### Tested development environment

| Component | Tested configuration |
|---|---|
| Operating system | Windows |
| Python | **3.11.9** |
| GPU | NVIDIA GeForce MX570 A, **2 GB VRAM** |
| RAM | **16 GB** |
| PyTorch | CUDA-enabled build (project environment reported `2.11.0+cu128`) |
| Frontend | React + Vite |
| Backend | Flask + PyTorch |
| Editor | VS Code (optional) |

**Required tools:** Git, Python 3.11, `pip`, Node.js and npm. An NVIDIA GPU is recommended for practical inference; CPU may work but will be slower. CUDA builds must match the installed NVIDIA driver and supported PyTorch installation options. For an authoritative platform-specific installation command, use the [PyTorch installation selector](https://pytorch.org/get-started/locally/). Docker is **not required**.

**Storage:** Reserve several gigabytes for the paired dataset, checkpoints, dependencies, and generated outputs. The Restormer deraining checkpoint alone is approximately **105 MB**; total space depends on the dataset portions downloaded. Avoid running multiple GPU-heavy processes simultaneously on the 2 GB GPU.

## Clone and install

Open **PowerShell** in a directory where you want the project.

### 1. Clone the project

```powershell
git clone https://github.com/YOUR_USERNAME/Adaptive-Vision-Intelligence.git
cd Adaptive-Vision-Intelligence
```

Replace `YOUR_USERNAME` and the repository name with the actual GitHub URL. If you already cloned it, open that folder instead.

### 2. Create and activate a Python virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell blocks activation, use Command Prompt with `.venv\Scripts\activate.bat`, or allow scripts for your **current PowerShell process only**:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

Use the root project's dependency file **if it is present**:

```powershell
python -m pip install --upgrade pip
if (Test-Path requirements.txt) { python -m pip install -r requirements.txt }
```

A CUDA-enabled PyTorch build may need to be installed separately using the [official PyTorch selector](https://pytorch.org/get-started/locally/). **Do not blindly overwrite an already working CUDA installation.** For core inference, the project uses PyTorch/torchvision, Flask, Pillow, NumPy, OpenCV (`opencv-python`), and `einops`; tests use `pytest`. If the repository does not yet provide a complete dependency lockfile, install any missing dependency reported by Python, then record the tested versions in `requirements.txt`.

For example, if the above packages are missing (and **after** installing PyTorch appropriate to your system):

```powershell
python -m pip install flask pillow numpy opencv-python einops pytest
```

Check GPU support:

```powershell
python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> **Reproducibility note:** Package versions must ultimately be pinned to the versions used in the committed project environment. The commands above are installation guidance, not a claim that all third-party versions have been verified together on another machine.

### 4. Install frontend dependencies

```powershell
cd frontend
npm ci
```

If the repository has no `frontend/package-lock.json`, use `npm install` instead. Then return to the repository root:

```powershell
cd ..
```

## Obtain the model code and checkpoints

**Important:** Git normally excludes `.pt`, `.pth`, `checkpoints/`, `external_models/`, datasets, and `outputs/`. Cloning the code alone is **not enough** to run inference. Restore or download all model dependencies below before starting the backend.

### A. Rain-severity classifier (required for Automatic mode)

Place the project-trained checkpoint at:

```text
checkpoints/severity/best.pt
```

This checkpoint belongs to the project and is **not** downloadable from the original ImageNet or torchvision model catalog. Obtain it from an authorized project-team copy or a future project release; do not substitute generic MobileNetV3 weights. The inference code is `vision/severity/predict.py`.

If the checkpoint is absent, manual PReNet/Restormer modes can still be used, but **Automatic mode cannot classify images**.

### B. PReNet6 for Light rain (required for Light mode)

The existing PReNet adapter expects its pretrained model implementation and Rain100L model weights at paths equivalent to:

```text
external_models/PReNet/networks.py
external_models/PReNet/logs/Rain100L/PReNet6/net_latest.pth
```

These files are not generated by `pip install`. Obtain the PReNet implementation and trained checkpoint from their legitimate original distribution or an authorized team copy, preserving the directory layout expected by `vision/deraining/prenet_adapter.py`.

Check the adapter's `DEFAULT_CHECKPOINT` and model import paths if your local layout differs. **PReNet source/weights redistribution permissions must be checked before publishing them.**

### C. Restormer for Heavy rain (required for Heavy mode)

Clone the official architecture:

```powershell
git clone https://github.com/swz30/Restormer.git external_models/Restormer
```

Install `einops` if necessary:

```powershell
python -m pip install einops
```

The official pretrained deraining checkpoint can be obtained from the [Restormer v1.0 release](https://github.com/swz30/Restormer/releases/tag/v1.0):

```powershell
New-Item -ItemType Directory -Force "external_models/Restormer/Deraining/pretrained_models" | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/swz30/Restormer/releases/download/v1.0/deraining.pth" `
  -OutFile "external_models/Restormer/Deraining/pretrained_models/deraining.pth"
```

The official file is the **initialization baseline**. The application's selected Heavy-rain weights are the project-fine-tuned checkpoint at:

```text
outputs/restormer/finetune/20261030_210003_966091/best.pt
```

Obtain this project-specific fine-tuned checkpoint from an authorized project-team copy or a separately published release. **Downloading only `deraining.pth` does not reproduce the fine-tuned Heavy-rain model used in the application.** Alternatively, researchers can re-create a fine-tuned checkpoint from the training dataset using `vision/deraining/finetune_restormer.py`, but the resulting dated output folder will differ; update the checkpoint path in `backend/app.py` and/or `vision/deraining/restormer_tiled.py` accordingly.

Verify checkpoints before using them. Only load model weights from trusted sources.

### D. Quick model file check

From the project root:

```powershell
@'
from pathlib import Path
paths = {
    'Severity classifier': 'checkpoints/severity/best.pt',
    'PReNet architecture': 'external_models/PReNet/networks.py',
    'PReNet weights': 'external_models/PReNet/logs/Rain100L/PReNet6/net_latest.pth',
    'Restormer architecture': 'external_models/Restormer/basicsr/models/archs/restormer_arch.py',
    'Original Restormer weights': 'external_models/Restormer/Deraining/pretrained_models/deraining.pth',
    'Fine-tuned Restormer': 'outputs/restormer/finetune/20261030_210003_966091/best.pt',
}
for name, path in paths.items():
    print(f"{'OK' if Path(path).is_file() else 'MISSING':7} {name}: {path}")
'@ | python -
```

If something is missing, resolve it before running the relevant model. Keeping weights out of Git is deliberate; it does **not** mean the trained model is publicly reproducible until the weights or a complete training procedure are made available.

## Dataset setup (for experiments)

The application can run inference on personal images **without** downloading RealRain-1k, provided the required checkpoints are available. The dataset is needed for retraining, validation and research experiments.

- Dataset: [RealRain-1k on Hugging Face](https://huggingface.co/datasets/dronefreak/RealRain-1k)
- Task: paired rainy-to-clean image restoration
- Project severity classes: **Light** and **Heavy** (no Medium class in Phase 1)
- Project split: 784 Heavy training pairs, 112 Heavy validation pairs, 224 Heavy test pairs; analogous Light splits
- Dataset license is listed by the dataset provider; confirm terms on the dataset card before redistribution.

One way to obtain the original dataset files:

```powershell
python -m pip install huggingface_hub
@'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='dronefreak/RealRain-1k',
    repo_type='dataset',
    local_dir='datasets/RealRain-1k',
)
'@ | python -
```

**Important:** This downloads the provider's dataset structure. The repository's `HeavyRainDataset` and severity loader expect the project-specific paired-file layout and split manifest. Use the repository's existing preparation/manifest workflow to organize the downloaded files; simply downloading them into a new directory may not satisfy the loader's expected paths. Check:

```text
vision/deraining/vdmamba/heavy_dataset.py
docs/severity_labels.csv
```

If `docs/severity_labels.csv` is not included in the Git repository, recover or regenerate the validated manifest before running experiments. Do not fabricate file paths, merge splits, or use **test** images for training or model selection.

## Run the application

Open **two terminals** at the project root. Activate `.venv` in the Flask terminal.

### Terminal 1 — Flask backend

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.app
```

Backend runs locally at:

```text
http://127.0.0.1:5000
http://127.0.0.1:5000/api/health
```

The health endpoint should report `status: ok`, `mode: adaptive_restoration`, and whether the severity and Restormer checkpoints are present. If it reports `direct_prenet` or `manual_restoration`, the previously running backend is still active; stop it and launch the current `backend/app.py`.

### Terminal 2 — React frontend

```powershell
cd frontend
npm run dev
```

Open the Vite URL printed by npm, typically:

```text
http://127.0.0.1:5173
```

The provided Vite configuration proxies `/api/` requests to Flask on port 5000. Keep **both** terminals running.

### Use the interface

1. Upload a PNG, JPG/JPEG or WEBP image (maximum **12 MB** per request; source images up to **20 megapixels** are decoded/checked).
2. Choose **Automatic**, **Light – PReNet6**, or **Heavy – Fine-tuned Restormer**.
3. In Automatic mode, the MobileNetV3-Small classifier predicts Light/Heavy and checks its existing confidence/margin gate. **Uncertain predictions are blocked**; select a model manually if needed.
4. For PReNet, choose the maximum processing dimension (**128 / 256 / 384 / 512 px**). Start at **256** on a 2 GB GPU.
5. For Restormer, inference uses overlapping **128 × 128 tiles** with **32 px overlap**, preserving full input dimensions. The current Flask Heavy-mode limit is **1.2 megapixels**.
6. Click **Analyze and Restore** or **Restore**, inspect the before/after comparison, and download the restored PNG.

**Processing time:** Heavy-rain tiled inference is substantially slower than PReNet. On the tested MX570 A, a 1280 × 720 image required **104 tiles** and about **29.4 seconds of model inference**, excluding some request overhead. Timing varies across systems.

Results are written locally beneath:

```text
outputs/web_restoration/<request-id>/
    input.png
    restored.png
    result.json
```

These are local files, not a database or cloud upload. Delete old result folders manually when they are no longer needed.

## API usage

### Health

```powershell
curl.exe http://127.0.0.1:5000/api/health
```

### Automatic classification and routing

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/restore `
  -F "image=@inputs/heavy_rain_scene.jpg" `
  -F "model=auto"
```

### Manual Light rain (PReNet6)

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/restore `
  -F "image=@inputs/light_rain.jpg" `
  -F "model=prenet" `
  -F "max_side=256"
```

### Manual Heavy rain (fine-tuned Restormer)

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/restore `
  -F "image=@inputs/heavy_rain_scene.jpg" `
  -F "model=restormer"
```

Replace the example `inputs/...` paths with real images on your machine; **the example input files are not included in Git**.

An accepted response reports the selected model, image URLs, processing time and other metrics. When Automatic mode is uncertain, the API returns `status: blocked_uncertain` without a restored image; this is expected behavior. To access a restored image, open its returned `/api/results/<request-id>/restored.png` URL on the local Flask server.

## Testing and research commands

### Python tests

From the root:

```powershell
python -m pytest tests/ -q
```

> Some research tests have additional dataset or checkpoint prerequisites. Missing checkpoints are setup problems; test success alone does not establish restoration quality.

### Frontend build

```powershell
cd frontend
npm run build
cd ..
```

### Restormer — original pretrained preview

```powershell
python -m vision.deraining.restormer_adapter `
  --input "PATH_TO_RAINY_IMAGE.jpg" `
  --output "outputs/restormer/restored_preview.png" `
  --max-side 128
```

### Restormer — fine-tuned full-resolution tiled inference

```powershell
python -m vision.deraining.restormer_tiled `
  --input "PATH_TO_RAINY_IMAGE.jpg" `
  --output "outputs/restormer/full_resolution_restored.png" `
  --device cuda --tile-size 128 --overlap 32
```

This uses the fine-tuned checkpoint defined in `restormer_tiled.py`; override `--checkpoint` if necessary.

### Restormer fine-tuning (optional research reproduction)

```powershell
python -m vision.deraining.finetune_restormer `
  --device cuda `
  --train-count 784 `
  --val-count 112 `
  --epochs 2 `
  --patch-size 64 `
  --lr 0.00001
```

This experiment initializes from the **original official checkpoint** and fine-tunes the **final output convolution only** (2,592 parameters), not all 26.1 million parameters. It requires the correctly prepared Heavy-rain training/validation files. It will create a new dated checkpoint under `outputs/restormer/finetune/` and does **not** automatically update Flask's checkpoint path. Do not train on the test split.

### Model evaluation scope

The existing scripts in `vision/deraining/` and `vision/deraining/vdmamba/` contain paired-image quality evaluation and comparison experiments. Use the same split, patch size, crop policy and reference image when comparing PSNR/SSIM across models. **PSNR/SSIM cannot be honestly computed for an ordinary uploaded photograph without a matching clean reference.**

## Folder structure

```text
Adaptive-Vision-Intelligence/
├── backend/
│   └── app.py                     # Flask API
├── frontend/
│   ├── src/                       # React interface and styles
│   ├── package.json
│   └── vite.config.js             # Development API proxy
├── vision/
│   ├── severity/                  # MobileNetV3-Small training/inference
│   └── deraining/
│       ├── prenet_adapter.py      # Light restoration
│       ├── restormer_adapter.py   # Official pretrained Restormer loader
│       ├── restormer_tiled.py     # Full-resolution Heavy inference
│       ├── finetune_restormer.py  # Heavy dataset adaptation
│       ├── adaptive_router.py     # Earlier router experiment
│       └── vdmamba/              # Experimental VDMamba-inspired research
├── tests/                         # Unit/integration tests
├── docs/                          # Project notes and manifests (if tracked)
├── checkpoints/                   # LOCAL, not committed
├── external_models/              # LOCAL, not committed
├── datasets/                     # LOCAL, not committed
├── outputs/                      # LOCAL, not committed
├── .gitignore
└── README.md
```

**Note:** `vision/deraining/adaptive_router.py` is an earlier research router that originally marked Heavy/VDMamba as unavailable. The **current web application's routing policy lives in `backend/app.py`** and routes accepted Heavy predictions to Restormer. Do not confuse the two entry points.

## Troubleshooting

| Problem | Likely reason / action |
|---|---|
| `python` selects the wrong version | Activate `.venv`; confirm `python --version` is 3.11.x. |
| `ModuleNotFoundError` | Activate `.venv`, install requirements, check the missing package. |
| `torch.cuda.is_available()` is `False` | Check NVIDIA driver and installed CUDA-enabled PyTorch build; CPU may work but is slower. |
| PowerShell activation blocked | Use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or Command Prompt. |
| `npm` not recognized | Install Node.js (supported LTS) and reopen your terminal. |
| Browser says backend unavailable | Start Flask, verify `/api/health`, and verify Vite's proxy configuration. |
| Health says `direct_prenet` | Old backend is still running; stop it and restart `python -m backend.app`. |
| Restormer checkpoint unavailable | Restore the fine-tuned `best.pt` at the exact configured path; official weights alone are insufficient for the selected model. |
| PReNet import or checkpoint error | Check both external PReNet source and Rain100L `net_latest.pth` paths. |
| Automatic restoration blocked | Confidence gate rejected prediction. This is expected; choose Light or Heavy manually. |
| Heavy-rain image rejected | Current endpoint allows up to 1.2 megapixels for Restormer; use a smaller image or adjust carefully after memory testing. |
| CUDA out-of-memory | Close other GPU apps, try smaller tiles or CPU, and avoid running multiple model jobs concurrently. |
| Rain streaks remain | Results vary by dataset and image characteristics. Numerical PSNR gain is not a guarantee of visible streak removal. |
| `git clone` includes no weights/data | They were intentionally excluded. Follow checkpoint and dataset sections above. |

## Known limitations

- The classifier predicts only **Light** and **Heavy**, not Medium. Earlier validation accuracy was approximately **72.77%**, with many uncertain predictions correctly gated rather than automatically routed.
- PReNet resizes the image, so Light-rain outputs may be lower resolution than input photos.
- Fine-tuned Restormer uses original-resolution tiled inference, which can be noticeably slower on a laptop GPU.
- Fine-tuning improved Restormer **pooled PSNR from 20.7151 dB to 21.4954 dB** on **112 validation crops of 64 × 64** (+0.7803 dB). This is a validation result, **not** a full-resolution or held-out test-set score.
- Both pretrained and fine-tuned restoration models may leave visible rain streaks on difficult or out-of-distribution real photographs.
- The project-specific classifier and fine-tuned weights are not included in standard Git commits; availability and licensing govern whether others can reproduce the app exactly.
- The backend serves local results and is designed for local development, **not** public deployment. Do not expose this Flask development server directly to the internet.
- **Phase 1 does not yet implement the final non-YOLO object-detection stage.** Do not claim end-to-end object-detection performance until Phase 2 is implemented and evaluated.

## Reproducibility, licensing and Phase 2

### Git repository vs. full runnable application

Source code is committed to GitHub, but the application also depends on separately obtained **model source code, pretrained weights and project-specific checkpoints**. To make the repository fully reproducible by other students or evaluators, publish (when permitted) a versioned model-weights manifest and authorized download instructions; pin dependency versions and third-party source revisions; provide dataset preparation steps and exact evaluation commands.

### Third-party acknowledgments

- **PReNet:** *Progressive Image Deraining Networks: A Better and Simpler Baseline* (CVPR 2019). Respect the original source and weights licensing/redistribution terms.
- **Restormer:** *Restormer: Efficient Transformer for High-Resolution Image Restoration* (CVPR 2022), [official repository](https://github.com/swz30/Restormer). Check the repository's current license and model-use terms.
- **RealRain-1k:** [Dataset repository](https://huggingface.co/datasets/dronefreak/RealRain-1k). Review its dataset card and license before using or redistributing data.
- **Experimental VDMamba-inspired code:** A local research prototype, **not the official published VDMamba model**. Cite the original paper in academic reports without presenting this prototype as a faithful reproduction.

### Phase 2 roadmap

1. Integrate a pretrained **non-YOLO object detector** (candidate: Faster R-CNN) with the restored output.
2. Display bounding boxes, class names, confidence scores and detector latency.
3. Measure downstream detection on rainy vs. restored images using valid labels and evaluation splits.
4. Complete systematic Light/Heavy restoration evaluation, final test-set reporting, inference optimization and error analysis.
5. Prepare final architecture, experiment tables, screenshots and project report.

---

**Project:** Adaptive Vision Intelligence for Rainy Weather Environments  
**Scope:** Local academic research prototype; Phase 1 restoration and severity-routing implementation.
