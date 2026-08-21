# Allen Cell Image Downloader

This project downloads raw 3D image stacks and segmentation masks from the Allen Cell dataset using Quilt3. The downloaded files are `.ome.tif` Z-stacks: each file contains multiple 2D image slices along the Z axis.

## Setup

Install Python 3.10 or newer, then open PowerShell in this project directory:

```powershell
python -m pip install -r ProofOfConcept/requirements.txt
```

If you use the supplied conda environment, replace `python` with its Python executable.

## Download images

The downloader is a function, so it must be called with arguments. This example downloads five raw stacks and five segmentation masks for TOMM20:

```powershell
python -c "from ProofOfConcept.download_images import download_structures; download_structures('TOMM20', 5, 'downloads')"
```

The files are written to `downloads/`. To download another structure, replace `TOMM20` with a label from the dataset, such as `ACTB`, `ACTN1`, `ATP2A2`, `CETN2`, `DSP`, `FBL`, `LAMP1`, `LMNB1`, `MYH10`, `NPM1`, `NUP153`, `SEC61B`, `ST6GAL1`, `TJP1`, or `TUBA1B`.

The first Quilt3 call may take several minutes because it loads the Allen Cell package manifest. The image files can also be several megabytes each, so allow the download to finish before closing the terminal.

## Create viewable previews

To convert one downloaded Z-stack to a PNG, run:

```powershell
python ProofOfConcept/preview_tiff.py downloads/<file-name>.ome.tif previews/example.png
```

The preview is the middle Z-slice, contrast-normalized for viewing. It is only a 2D representation; the original TIFF remains the complete 3D dataset.

Example previews generated from the local downloads:

![Raw image middle Z-slice](previews/raw_example_1.png)

![Segmentation mask middle Z-slice](previews/segmentation_example.png)

## Repository contents

- `ProofOfConcept/download_images.py`: downloads raw and segmentation stacks.
- `ProofOfConcept/preview_tiff.py`: converts a representative TIFF slice to PNG.
- `previews/`: small example images suitable for viewing on GitHub.
- `downloads/`: local downloaded data; ignored by Git because the files are large.

.ome.tif