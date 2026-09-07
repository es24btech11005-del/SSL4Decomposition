from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from patch_dataset import SuperpositionPatchDataset
from unet import UNet


def run_epoch(
    model: UNet,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    for inputs, target_one, target_two in loader:
        inputs = inputs.to(device)
        targets = torch.cat((target_one, target_two), dim=1).to(device)
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a two-output U-Net on summed patch pairs.")
    parser.add_argument("--split-root", type=Path, default=Path(__file__).parent / "split")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "unet.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--samples-per-epoch", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = SuperpositionPatchDataset(
        args.split_root / "train",
        crop_size=args.crop_size,
        samples_per_epoch=args.samples_per_epoch,
        augment=True,
    )
    validation_dataset = SuperpositionPatchDataset(
        args.split_root / "validation",
        crop_size=args.crop_size,
        samples_per_epoch=max(args.batch_size, args.samples_per_epoch // 5),
        augment=False,
        seed=0,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_function, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, loss_function, device, None)
        print(f"epoch {epoch:03d}: train_mse={train_loss:.6f} validation_mse={validation_loss:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()