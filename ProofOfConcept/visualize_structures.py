"""Create middle-slice image and segmentation previews for each structure."""

import argparse
from pathlib import Path

import pandas as pd

try:
    from ProofOfConcept.preview_tiff import tiff_to_png
except ModuleNotFoundError:
    from preview_tiff import tiff_to_png


METADATA_PATH = Path(__file__).with_name("metadata.csv")
DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "downloads"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "visualizations"


def visualize_structures(
    metadata_path: Path,
    input_dir: Path,
    output_dir: Path,
    files_per_structure: int = 2,
) -> None:
    """Create image and mask previews for each structure in the metadata."""
    if files_per_structure < 1:
        raise ValueError("files_per_structure must be at least 1")

    metadata = pd.read_csv(metadata_path)
    structure_column = "structure" if "structure" in metadata.columns else "structure_name"
    required_columns = {structure_column, "crop_raw", "crop_seg"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"metadata.csv is missing required columns: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    structure_names = sorted(metadata[structure_column].dropna().astype(str).unique())
    missing_files: list[Path] = []

    for structure_name in structure_names:
        selected = metadata.loc[
            metadata[structure_column].astype(str) == structure_name
        ].head(files_per_structure)

        for index, row in enumerate(selected.itertuples(index=False), start=1):
            raw_path = input_dir / Path(row.crop_raw).name
            mask_path = input_dir / Path(row.crop_seg).name
            if not raw_path.exists():
                missing_files.append(raw_path)
            if not mask_path.exists():
                missing_files.append(mask_path)
            if not raw_path.exists() or not mask_path.exists():
                continue

            tiff_to_png(str(raw_path), str(output_dir / f"{structure_name}_image_{index}.png"))
            tiff_to_png(str(mask_path), str(output_dir / f"{structure_name}_mask_{index}.png"))

    if missing_files:
        missing = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Some selected TIFF files are missing from the input directory:\n" + missing
        )

    print(
        f"Created {len(structure_names) * files_per_structure * 2} previews "
        f"for {len(structure_names)} structures in {output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--files-per-structure",
        type=int,
        default=2,
        help="Number of image/mask pairs per structure (default: 2).",
    )
    arguments = parser.parse_args()
    visualize_structures(
        arguments.metadata,
        arguments.input_dir,
        arguments.output_dir,
        arguments.files_per_structure,
    )
