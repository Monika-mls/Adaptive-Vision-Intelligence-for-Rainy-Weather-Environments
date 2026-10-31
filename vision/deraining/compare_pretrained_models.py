
import csv
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from vision.deraining.restormer_adapter import load_model
from vision.deraining.prenet_adapter import restore_image


OUTPUT = Path("outputs/pretrained_comparison")
OUTPUT.mkdir(parents=True, exist_ok=True)


def save_image(tensor, path):
    array = (
        tensor.squeeze(0)
        .detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(
        (array * 255).round().astype(np.uint8)
    ).save(path)


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse == 0 else -10 * np.log10(mse)


def restormer_predict(model, rainy):
    height, width = rainy.shape[-2:]
    padded = F.pad(
        rainy,
        (0, (-width) % 8, 0, (-height) % 8),
        mode="replicate",
    )

    with torch.inference_mode():
        output = model(padded)

    return output[..., :height, :width].clamp(0, 1)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = HeavyRainDataset(
        split="validation",
        patch_size=128,
        augment=False,
        limit=8,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Loading pretrained Restormer...")
    restormer = load_model(device)

    rows = []

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)

        for index, batch in enumerate(loader, start=1):
            rainy = batch["rainy"]
            clean = batch["clean"]

            input_path = folder / "rainy.png"
            prenet_path = folder / "prenet.png"

            save_image(rainy, input_path)

            print(f"Processing pair {index}/8...", flush=True)

            restore_image(
                input_path=input_path,
                output_path=prenet_path,
                device="auto",
                max_side=128,
            )

            with Image.open(prenet_path) as image:
                array = np.asarray(
                    image.convert("RGB"), dtype=np.float32
                ).copy() / 255.0

            prenet_output = (
                torch.from_numpy(array)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            restored = restormer_predict(
                restormer,
                rainy.to(device),
            ).cpu()

            rainy_psnr = psnr(rainy, clean)
            prenet_psnr = psnr(prenet_output, clean)
            restormer_psnr = psnr(restored, clean)

            rows.append({
                "pair": index,
                "rainy_psnr": rainy_psnr,
                "prenet_psnr": prenet_psnr,
                "restormer_psnr": restormer_psnr,
                "prenet_gain": prenet_psnr - rainy_psnr,
                "restormer_gain": restormer_psnr - rainy_psnr,
            })

            comparison = Image.new("RGB", (128 * 4, 128))

            for position, tensor in enumerate(
                [rainy, prenet_output, restored, clean]
            ):
                path = folder / f"temp_{position}.png"
                save_image(tensor, path)

                with Image.open(path) as image:
                    comparison.paste(image, (position * 128, 0))

            comparison.save(
                OUTPUT / f"comparison_{index:02d}.png"
            )

            print(
                f"  Rainy: {rainy_psnr:.2f} | "
                f"PReNet: {prenet_psnr:.2f} | "
                f"Restormer: {restormer_psnr:.2f}"
            )

    with open(OUTPUT / "results.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nAVERAGE PSNR GAIN")
    for model in ["prenet_gain", "restormer_gain"]:
        values = [row[model] for row in rows]
        print(f"{model}: {np.mean(values):+.4f} dB")

    print("\nResults:", OUTPUT)
    print("Test split used: No")


if __name__ == "__main__":
    main()
