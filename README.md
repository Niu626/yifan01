# LGHA-Net: Local-Global Hybrid Attention for Pixel-Wise Multi-Illuminant Estimation under Low-Bit-Depth Degradation

Official implementation of LGHA-Net for pixel-wise multi-illuminant
estimation under spatially varying illumination and low-bit-depth
degradation.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- OpenCV
- Pillow

## Baseline Method

Our implementation is developed based on and compared with the U+PP
method presented in:

S. Yue, M. Wei, et al., “Robust pixel-wise illuminant estimation
algorithm for images with a low bit-depth,” Optics Express,
vol. 32, no. 15, pp. 26708–26718, 2024.

Reference implementation:

https://github.com/shuwei666/Robust-pixel-wise-illuminant-estimation

## Single-Illuminant Benchmark Datasets

### Cube+

N. Banić, K. Koščević, and S. Lončarić,
“Unsupervised Learning for Color Constancy,”
arXiv:1712.00436, 2017.

- Paper: https://arxiv.org/abs/1712.00436
- Dataset: https://ipg.fer.hr/ipg/resources/color_constancy

### NUS 8-Camera

D. Cheng, D. K. Prasad, and M. S. Brown,
“Illuminant Estimation for Color Constancy: Why Spatial-Domain Methods
Work and the Role of the Color Distribution,”
Journal of the Optical Society of America A,
vol. 31, no. 5, pp. 1049–1058, 2014.

- Paper: https://doi.org/10.1364/JOSAA.31.001049
- Dataset: https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html

## Pretrained Models

The pretrained model can be downloaded from:

[Quark Drive](https://pan.quark.cn/s/2118ea3c447b)

After downloading, place the model checkpoint in the following directory:

```text
pretrained_models/
