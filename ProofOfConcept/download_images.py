"""Download Allen Cell image stacks and segmentation masks."""

import argparse
from pathlib import Path

import pandas as pd
import quilt3 as q3


PACKAGE_NAME = "aics/hipsc_single_cell_image_dataset"
REGISTRY = "s3://allencell"
METADATA_PATH = Path(__file__).with_name("metadata.csv")
STRUCTURE_NAMES_PATH = Path(__file__).with_name("structure_names.txt")
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "structure_downloads"


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
    output_dir: str | Path,
    package: q3.Package | None = None,
    metadata: pd.DataFrame | None = None,
):
    """
    Download up to ``num_stacks`` raw stack/segmentation pairs for one structure.

    Files are cached by structure and only downloaded when missing, which keeps
    repeated notebook runs from re-fetching the same TIFFs.
    """
    if not structure_type:
        raise ValueError("structure_type must not be empty")
    if num_stacks < 1:
        raise ValueError("num_stacks must be at least 1")

    destination = Path(output_dir) / str(structure_type)
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
        raw_name = Path(row.crop_raw).name
        seg_name = Path(row.crop_seg).name
        raw_dest = destination / raw_name
        seg_dest = destination / seg_name
        if raw_dest.exists() and seg_dest.exists():
            continue

        if not raw_dest.exists():
            package[str(row.crop_raw)].fetch(str(raw_dest))
        if not seg_dest.exists():
            package[str(row.crop_seg)].fetch(str(seg_dest))


def download_structure_samples(
    structure_names: list[str] | None = None,
    num_stacks: int = 3,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    package: q3.Package | None = None,
    metadata: pd.DataFrame | None = None,
) -> list[Path]:
    """Ensure a small number of TIFF pairs exist for each requested structure."""
    package = package or q3.Package.browse(PACKAGE_NAME, registry=REGISTRY)
    metadata = metadata if metadata is not None else load_metadata(package)
    all_structures = get_structure_names(metadata)
    requested = structure_names or all_structures
    for structure in requested:
        print(f"Ensuring {num_stacks} stack(s) are available for {structure}...")
        download_structures(
            structure,
            num_stacks,
            output_dir,
            package=package,
            metadata=metadata,
        )
    return [Path(output_dir) / str(structure) for structure in requested]


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
        default=3,
        help="Number of image pairs per structure (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for downloaded files (default: ProofOfConcept/structure_downloads).",
    )
    arguments = parser.parse_args()

    package = q3.Package.browse(PACKAGE_NAME, registry=REGISTRY)
    metadata = load_metadata(package)
    structures = arguments.structure or get_structure_names(metadata)
    print(f"Saved {len(structures)} unique structure names to {STRUCTURE_NAMES_PATH}")

    download_structure_samples(
        structures,
        arguments.num_stacks,
        arguments.output_dir,
        package=package,
        metadata=metadata,
    )