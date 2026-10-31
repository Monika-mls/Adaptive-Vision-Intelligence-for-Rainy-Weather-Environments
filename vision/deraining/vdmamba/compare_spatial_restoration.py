
import torch
from torch.utils.data import DataLoader

from vision.deraining.vdmamba.heavy_dataset import HeavyRainDataset
from vision.deraining.vdmamba.image_model import (
    ExperimentalVDMambaImageModel,
)

CHECKPOINT = (
    "outputs/vdmamba/multi_train/"
    "run_20261028_093414_860421/best.pt"
)

OFFSET = 12.9815 / 255.0


def gradients(image):
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx, dy


def gradient_error(prediction, target):
    pred_dx, pred_dy = gradients(prediction)
    true_dx, true_dy = gradients(target)

    error_x = (pred_dx - true_dx).abs().mean()
    error_y = (pred_dy - true_dy).abs().mean()

    return ((error_x + error_y) / 2).item()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]

    model = ExperimentalVDMambaImageModel(
        channels=int(config["channels"]),
        state_size=int(config["state_size"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

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

    methods = ["rainy", "brightness", "vdmamba"]

    gradient_errors = {name: [] for name in methods}
    correction_errors = {name: [] for name in methods}
    residual_variations = {name: [] for name in methods}

    with torch.inference_mode():
        for batch in loader:
            rainy = batch["rainy"].to(device)
            clean = batch["clean"].to(device)

            predictions = {
                "rainy": rainy,
                "brightness": (rainy - OFFSET).clamp(0, 1),
                "vdmamba": model(rainy).clamp(0, 1),
            }

            target_correction = clean - rainy

            for name, prediction in predictions.items():
                if prediction.shape != clean.shape:
                    raise RuntimeError("Unexpected output shape")

                if not torch.isfinite(prediction).all().item():
                    raise RuntimeError("Non-finite output")

                correction = prediction - rainy

                gradient_errors[name].append(
                    gradient_error(prediction, clean)
                )

                correction_errors[name].append(
                    (correction - target_correction)
                    .abs().mean().item()
                )

                residual_variations[name].append(
                    correction.std(unbiased=False).item()
                )

    print("=" * 65)
    print("SPATIAL RESTORATION DIAGNOSTICS")
    print("=" * 65)

    print("Validation samples:", len(dataset))

    for name in methods:
        print(f"\n{name.upper()}")
        print(
            "Mean gradient MAE:",
            round(
                sum(gradient_errors[name])
                / len(gradient_errors[name]),
                6,
            ),
        )
        print(
            "Mean correction error:",
            round(
                sum(correction_errors[name])
                / len(correction_errors[name]),
                6,
            ),
        )
        print(
            "Mean residual variation:",
            round(
                sum(residual_variations[name])
                / len(residual_variations[name]),
                6,
            ),
        )

    print("\nLower gradient and correction error is better.")
    print("Residual variation alone is not a quality score.")
    print("Test split not used.")


if __name__ == "__main__":
    main()
