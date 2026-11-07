# S2IPNet

This is the code repository for the paper "S2IPNet: SAM 2 Guided Information Preservation Network for Infrared Small Target Detection".

## Dataset Split Files

The `dataset/` directory provides the train/test split files for three datasets: `IRSTD-Air`, `NUDT-SIRST`, and `SIRST`. Each dataset folder contains two split files, one for training and one for testing.

For each dataset, the image files and label files should be organized as follows:

- `images/`: infrared images
- `masks/`: ground-truth annotation masks

Each `.txt` file stores one sample ID per line. During training or evaluation, each sample ID should be mapped to its corresponding image in `images/` and mask in `masks/`.


## Requirements

Install the project's runtime dependencies in a Python environment, including
PyTorch, torchvision, OpenCV, Pillow, NumPy, and tqdm. The appropriate PyTorch
installation depends on the available CUDA version.

The SAM 2 checkpoint and matching configuration must exist under
`utils/sam2/` before training. The default mode is `s`. Download the required
checkpoints with the script provided by SAM 2:

```text
utils/sam2/Checkpoints/download_ckpts.sh
```

From the repository root, run:

```bash
bash utils/sam2/Checkpoints/download_ckpts.sh
```

The default mode then uses:

```text
utils/sam2/sam2_configs/sam2.1_hiera_s.yaml
```

The SAM 2 implementation is based on the official
[facebookresearch/sam2](https://github.com/facebookresearch/sam2) repository.

## Training


```bash
python train.py
```

For example, to train on SIRST with the base-plus SAM 2 model:

```bash
python train.py --dataset SIRST --SAM2_mode b+
```

Available SAM 2 modes are `s`, `b+`, `l`, and `t`.

## Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{11745917,
  author={Hao, Fan and Zhou, Feng and Lu, Pengfei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={S2IPNet: SAM 2-Guided Information Preservation Network for Infrared Small Target Detection},
  year={2026},
  volume={64},
  number={},
  pages={5012314-5012314},
  keywords={Modeling;Object detection;Signal detection;Measurement;Personal digital devices;Decoding;Accuracy;Foundation models;Conferences;Location awareness;Cross-attention;foundation model;information preservation;infrared small target detection (IRSTD);multiscale},
  doi={10.1109/TGRS.2026.3745740}
}
```
