"""Download Allen Cell image stacks and segmentation masks."""

import argparse
from pathlib import Path

import pandas as pd
import quilt3 as q3


PACKAGE_NAME = "aics/hipsc_single_cell_image_dataset"
REGISTRY = "s3://allencell"
METADATA_PATH = Path(__file__).with_name("metadata.csv")
STRUCTURE_NAMES_PATH = Path(__file__).with_name("structure_names.txt")


def load_metadata(package: q3.Package) -> pd.DataFrame:
    """Load cached metadata, downloading it once when it is not present."""
    if not METADATA_PATH.exists():
        package["metadata.csv"].fetch(str(METADATA_PATH))
        print(f"Downloaded metadata to {METADATA_PATH}")

    return pd.read_csv(METADATA_PATH)


def get_structure_names(metadata: pd.DataFrame) -> list[str]:
    """Return unique structure names and save them for later reference."""
    structure_column = "structure" if "structure" in metadata.columns else "structure_name"
    if structure_column not in metadata.columns:
        raise ValueError("metadata.csv must contain a structure or structure_name column")

    structure_names = sorted(
        metadata[structure_column].dropna().astype(str).unique()
    )
    STRUCTURE_NAMES_PATH.write_text("\n".join(structure_names), encoding="utf-8")
    return structure_names


def download_structures(
    structure_type: str,
    num_stacks: int,
    output_dir: str,
    package: q3.Package | None = None,
    metadata: pd.DataFrame | None = None,
):
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

    package = package or q3.Package.browse(PACKAGE_NAME, registry=REGISTRY)
    metadata = metadata if metadata is not None else load_metadata(package)
    structure_column = "structure_name"
    required_columns = {structure_column, "crop_raw", "crop_seg"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"metadata.csv is missing required columns: {missing}")

    selected = metadata.loc[metadata[structure_column] == structure_type].head(num_stacks)
    for row in selected.itertuples(index=False):
        raw_path = Path(row.crop_raw)
        seg_path = Path(row.crop_seg)
        
        package[str(row.crop_raw)].fetch(str(destination / row.structure_name / raw_path.name))
        package[str(row.crop_seg)].fetch(str(destination / row.structure_name / seg_path.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure",
        nargs="+",
        help="Structure name(s) to download; defaults to every structure in metadata.csv.",
    )
    parser.add_argument(
        "--num-stacks",
        type=int,
        default=5,
        help="Number of image pairs per structure (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads",
        help="Directory for downloaded files (default: downloads).",
    )
    arguments = parser.parse_args()

    package = q3.Package.browse(PACKAGE_NAME, registry=REGISTRY)
    metadata = load_metadata(package)
    all_structures = get_structure_names(metadata)
    structures = arguments.structure or all_structures
    print(f"Saved {len(all_structures)} unique structure names to {STRUCTURE_NAMES_PATH}")

    for structure in structures:
        print(f"Downloading {arguments.num_stacks} stack(s) for {structure}...")
        download_structures(
            structure,
            arguments.num_stacks,
            arguments.output_dir,
            package=package,
            metadata=metadata,
        )