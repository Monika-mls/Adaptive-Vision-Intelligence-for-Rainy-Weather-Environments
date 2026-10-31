
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vision.deraining.restormer_adapter import load_model
from vision.deraining.finetune_restormer import evaluate
from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from torch.utils.data import DataLoader


def restore_crop(model, image_path, output_path, device):
    image = Image.open(image_path).convert("RGB")
    data = np.asarray(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(data).permute(
        2, 0, 1
    ).unsqueeze(0).to(device)

    h, w = tensor.shape[-2:]
    padded = F.pad(
        tensor,
        (0, (-w) % 8, 0, (-h) % 8),
        mode="replicate",
    )

    with torch.inference_mode():
        prediction = model(padded)[..., :h, :w].clamp(0, 1)

    array = (
        prediction[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    output = np.uint8(np.round(array * 255))
    Image.fromarray(output).save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = HeavyRainDataset(
        split="validation",
        patch_size=64,
        augment=False,
        limit=112,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    output_dir = Path("outputs/restormer/finetune_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_path = Path("outputs/restormer/original_crop.png")

    if not crop_path.is_file():
        raise FileNotFoundError(
            "First create outputs/restormer/original_crop.png"
        )

    model = load_model(device)

    print("Evaluating original pretrained Restormer...", flush=True)
    original = evaluate(model, loader, device)

    restore_crop(
        model,
        crop_path,
        output_dir / "original_weights.png",
        device,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["params"], strict=True)
    model.eval()

    print("Evaluating fine-tuned Restormer...", flush=True)
    finetuned = evaluate(model, loader, device)

    restore_crop(
        model,
        crop_path,
        output_dir / "finetuned_weights.png",
        device,
    )

    print("\nFINAL VALIDATION COMPARISON")
    print("Validation images:", len(dataset))
    print(
        f"Original PSNR:   {original['restored_psnr']:.4f} dB"
    )
    print(
        f"Fine-tuned PSNR: {finetuned['restored_psnr']:.4f} dB"
    )
    print(
        f"Difference:      "
        f"{finetuned['restored_psnr'] - original['restored_psnr']:+.4f} dB"
    )
    print(
        "Original improved patches:",
        original["improved_patches"],
    )
    print(
        "Fine-tuned improved patches:",
        finetuned["improved_patches"],
    )
    print("Visual comparison saved to:", output_dir)
    print("Test split used: No")


if __name__ == "__main__":
    main()
