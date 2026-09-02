"""Convert a representative Z-slice from an Allen Cell TIFF stack to PNG."""

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def tiff_to_png(
    input_path: str,
    output_path: str,
    channel_index: int = 0,
    z_offset: int | None = None,
    z_index: int | None = None,
) -> None:
    """Save a representative Z-slice of a selected TIFF channel as a PNG.

    When neither ``z_index`` nor ``z_offset`` is provided, the function uses the
    exact middle slice. ``z_offset`` can be ``-1``, ``0``, or ``1`` to select the
    middle slice or the adjacent ones.
    """
    with tifffile.TiffFile(input_path) as tif:
        series = tif.series[0]
        axes = series.axes
        if len(series.shape) < 2 or not axes.endswith("YX"):
            raise ValueError(f"Unsupported TIFF dimensions: {series.shape} ({series.axes})")

        try:
            image = series.asarray()
        except (IndexError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Unable to read TIFF stack {input_path}: {exc}") from exc

        if "C" in axes:
            if not 0 <= channel_index < series.shape[axes.index("C")]:
                raise IndexError(f"Channel index {channel_index} is out of range for {series.shape}")
            channel_axis = axes.index("C")
            image = np.take(image, channel_index, axis=channel_axis)
            axes = axes.replace("C", "")
        elif channel_index != 0:
            raise ValueError(f"TIFF has no channel axis: {series.shape} ({series.axes})")

        if "Z" in axes:
            z_axis = axes.index("Z")
            z_count = image.shape[z_axis]
            if z_count == 0:
                raise ValueError(f"TIFF has no Z slices: {series.shape} ({series.axes})")

            middle_z = z_count // 2
            if z_index is not None:
                selected_z = z_index
            elif z_offset is not None:
                selected_z = max(0, min(z_count - 1, middle_z + int(z_offset)))
            else:
                selected_z = middle_z

            # Clamp to valid bounds for short stacks and extreme offsets.
            selected_z = max(0, min(z_count - 1, selected_z))
            if not 0 <= selected_z < z_count:
                raise IndexError(
                    f"Z slice index {selected_z} is out of range for shape {image.shape}"
                )
            image = np.take(image, selected_z, axis=z_axis)

        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image after selection, got {image.shape} ({axes})")

    image = image.astype(np.float32)
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        normalized = np.zeros_like(image, dtype=np.uint8)
    else:
        normalized = np.clip((image - low) * 255 / (high - low), 0, 255).astype(np.uint8)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalized).save(destination)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    arguments = parser.parse_args()
    tiff_to_png(arguments.input_path, arguments.output_path)