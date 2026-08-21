"""Download Allen Cell image stacks and segmentation masks."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import quilt3 as q3


PACKAGE_NAME = "aics/hipsc_single_cell_image_dataset"
REGISTRY = "s3://allencell"


def download_structures(structure_type: str, num_stacks: int, output_dir: str):
    """
    Download image stacks and segmentation masks for one Allen Cell structure.

    The selected files are written directly into ``output_dir`` using the file
    names stored in the dataset's ``crop_raw`` and ``crop_seg`` columns.
    """
    if not structure_type:
        raise ValueError("structure_type must not be empty")
    if num_stacks < 0:
        raise ValueError("num_stacks must be non-negative")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    package = q3.Package.browse(PACKAGE_NAME, registry=REGISTRY)
    with TemporaryDirectory() as temporary_dir:
        metadata_path = Path(temporary_dir) / "metadata.csv"
        package["metadata.csv"].fetch(str(metadata_path))
        metadata = pd.read_csv(metadata_path)

    structure_column = "structure" if "structure" in metadata.columns else "structure_name"
    required_columns = {structure_column, "crop_raw", "crop_seg"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"metadata.csv is missing required columns: {missing}")

    selected = metadata.loc[metadata[structure_column] == structure_type].head(num_stacks)
    for row in selected.itertuples(index=False):
        raw_path = Path(row.crop_raw)
        seg_path = Path(row.crop_seg)
        package[str(row.crop_raw)].fetch(str(destination / raw_path.name))
        package[str(row.crop_seg)].fetch(str(destination / seg_path.name))


if __name__ == "__main__":
    raise SystemExit("Import download_structures and provide a structure type, count, and output directory.")