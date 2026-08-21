"""Convert a representative Z-slice from an Allen Cell TIFF stack to PNG."""

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def tiff_to_png(input_path: str, output_path: str) -> None:
    """Save the middle Z-slice of a TIFF stack as an 8-bit grayscale PNG."""
    with tifffile.TiffFile(input_path) as tif:
        series = tif.series[0]
        if len(series.shape) < 2 or not series.axes.endswith("YX"):
            raise ValueError(f"Unsupported TIFF dimensions: {series.shape} ({series.axes})")
        z_axis = series.axes.find("Z")
        if z_axis == -1:
            image = series.pages[0].asarray()
        else:
            z_count = series.shape[z_axis]
            pages_per_z = len(series.pages) // z_count
            middle_z = z_count // 2
            image = series.pages[middle_z * pages_per_z].asarray()

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