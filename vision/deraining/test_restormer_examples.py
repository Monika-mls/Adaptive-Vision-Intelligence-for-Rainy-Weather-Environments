
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from vision.deraining.restormer_adapter import load_model
from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset


CHECKPOINT = Path(
    "outputs/restormer/finetune/"
    "20261030_210003_966091/best.pt"
)

OUTPUT = Path("outputs/restormer/visual_examples")


def to_image(tensor):
    array = (
        tensor.detach().cpu().clamp(0, 1)
        .permute(1, 2, 0).numpy()
    )
    return Image.fromarray(
        (array * 255).round().astype(np.uint8)
    )


def psnr(a, b):
    mse = (a - b).square().mean().item()
    return float("inf") if mse == 0 else -10 * np.log10(mse)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = HeavyRainDataset(
        split="validation",
        patch_size=128,
        augment=False,
        limit=112,
    )

    # Four separate validation images.
    indices = [0, 3, 7, 11]

    print("Loading original Restormer...", flush=True)
    model = load_model(device)

    examples = []

    with torch.inference_mode():
        for index in indices:
            item = dataset[index]
            rainy = item["rainy"].unsqueeze(0).to(device)
            clean = item["clean"].unsqueeze(0).to(device)

            original = model(rainy).clamp(0, 1)

            examples.append({
                "index": index,
                "rainy": rainy.cpu(),
                "clean": clean.cpu(),
                "original": original.cpu(),
            })

    print("Loading fine-tuned checkpoint...", flush=True)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["params"],
        strict=True,
    )
    model.eval()

    with torch.inference_mode():
        for example in examples:
            rainy = example["rainy"].to(device)
            clean = example["clean"].to(device)

            restored = model(rainy).clamp(0, 1)

            images = [
                to_image(example["rainy"][0]),
                to_image(example["original"][0]),
                to_image(restored[0]),
                to_image(example["clean"][0]),
            ]

            labels = [
                "Rainy",
                "Original Restormer",
                "Fine-tuned Restormer",
                "Ground Truth",
            ]

            comparison = Image.new(
                "RGB", (512, 152), "white"
            )

            draw = ImageDraw.Draw(comparison)

            for i, (image, label) in enumerate(
                zip(images, labels)
            ):
                comparison.paste(image, (i * 128, 24))
                draw.text(
                    (i * 128 + 4, 6),
                    label,
                    fill="black",
                )

            path = OUTPUT / (
                f"comparison_{example['index']:02d}.png"
            )
            comparison.save(path)

            print(
                f"Image {example['index']} | "
                f"Rainy: {psnr(rainy, clean):.2f} | "
                f"Original: {psnr(example['original'], example['clean']):.2f} | "
                f"Fine-tuned: {psnr(restored, clean):.2f}"
            )

    print("\nComparisons saved:", OUTPUT)
    print("Test split used: No")


if __name__ == "__main__":
    main()
