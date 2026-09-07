from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile


STRUCTURES = ("T1", "T2", "T3")


def as_zcyx(array: np.ndarray, axes: str) -> np.ndarray:
    """Normalize a single-channel ZYX or ZCYX TIFF to ZCYX."""
    axes = axes.upper()
    if axes in {"ZYX", "IYX"}:
        return array[:, np.newaxis, :, :]
    if axes == "ZCYX":
        return array
    raise ValueError(f"Expected ZYX or ZCYX TIFF data, got shape {array.shape} with axes {axes!r}")


def read_zcyx(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        array = series.asarray()
        return as_zcyx(array, series.axes)


def write_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, frame)


def create_split(
    input_root: Path,
    output_root: Path,
    reserved_z: int,
    reserved_c: int,
) -> list[dict[str, str | int]]:
    manifest: list[dict[str, str | int]] = []

    for structure in STRUCTURES:
        structure_root = input_root / structure
        image = read_zcyx(structure_root / "Channel.tif")
        mask = read_zcyx(structure_root / "Seg.tif")

        if image.shape != mask.shape:
            raise ValueError(
                f"{structure}: image shape {image.shape} does not match mask shape {mask.shape}"
            )
        z_count, channel_count, height, width = image.shape
        if not 0 <= reserved_z < z_count or not 0 <= reserved_c < channel_count:
            raise ValueError(
                f"{structure}: reserved frame (z={reserved_z}, c={reserved_c}) is outside {image.shape[:2]}"
            )

        for z_index in range(z_count):
            for channel_index in range(channel_count):
                image_frame = image[z_index, channel_index]
                mask_frame = mask[z_index, channel_index]
                frame_name = f"{structure}_z{z_index:03d}_c{channel_index:03d}"

                if (z_index, channel_index) == (reserved_z, reserved_c):
                    midpoint = width // 2
                    split_parts = (
                        ("validation", slice(None), slice(0, midpoint)),
                        ("test", slice(None), slice(midpoint, width)),
                    )
                    for split_name, y_slice, x_slice in split_parts:
                        write_frame(
                            output_root / split_name / structure / f"{frame_name}_image.tif",
                            image_frame[y_slice, x_slice],
                        )
                        write_frame(
                            output_root / split_name / structure / f"{frame_name}_mask.tif",
                            mask_frame[y_slice, x_slice],
                        )
                        manifest.append(
                            {
                                "structure": structure,
                                "split": split_name,
                                "z": z_index,
                                "c": channel_index,
                                "image": str(Path(split_name) / structure / f"{frame_name}_image.tif"),
                                "mask": str(Path(split_name) / structure / f"{frame_name}_mask.tif"),
                            }
                        )
                else:
                    write_frame(
                        output_root / "train" / structure / f"{frame_name}_image.tif",
                        image_frame,
                    )
                    write_frame(
                        output_root / "train" / structure / f"{frame_name}_mask.tif",
                        mask_frame,
                    )
                    manifest.append(
                        {
                            "structure": structure,
                            "split": "train",
                            "z": z_index,
                            "c": channel_index,
                            "image": str(Path("train") / structure / f"{frame_name}_image.tif"),
                            "mask": str(Path("train") / structure / f"{frame_name}_mask.tif"),
                        }
                    )

    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ZCYX TIFF stacks into train/validation/test 2D frames.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).parent / "NatureMethods_segmentation_data" / "segmentation_data",
    )
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "split")
    parser.add_argument("--reserved-z", type=int, default=2)
    parser.add_argument("--reserved-c", type=int, default=0)
    args = parser.parse_args()

    manifest = create_split(args.input_root, args.output_root, args.reserved_z, args.reserved_c)
    counts = {split: sum(row["split"] == split for row in manifest) for split in ("train", "validation", "test")}
    print(f"Created split in {args.output_root}")
    print(f"Reserved frame: z={args.reserved_z}, c={args.reserved_c}")
    print(f"Frames: train={counts['train']}, validation={counts['validation']}, test={counts['test']}")


if __name__ == "__main__":
    main()