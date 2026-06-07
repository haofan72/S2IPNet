# S2IPNet

This is the code repository for the paper "S2IPNet: SAM 2 Guided Information Preservation Network for Infrared Small Target Detection".

## Dataset Split Files

The `dataset/` directory provides the train/test split files for three datasets: `IRSTD-Air`, `NUDT-SIRST`, and `SIRST`. Each dataset folder contains two split files, one for training and one for testing.

For each dataset, the image files and label files should be organized as follows:

- `images/`: infrared images
- `masks/`: ground-truth annotation masks

Each `.txt` file stores one sample ID per line. During training or evaluation, each sample ID should be mapped to its corresponding image in `images/` and mask in `masks/`.


## Usage

The basic usage is:

1. Read the train or test split file according to the current stage.
2. Iterate over the sample IDs and construct the image and mask paths from `images/` and `masks/`.

Example:

```python
image_path = dataset_root / "images" / f"{sample_id}.png"
mask_path = dataset_root / "masks" / f"{sample_id}.png"
```
