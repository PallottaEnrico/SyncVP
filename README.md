<h1 align="center"> SyncVP </h1>

Official implementation of **CVPR 2025** paper:

**"SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction"**

> [Enrico Pallotta](https://pallottaenrico.github.io/), [Sina Mokhtarzadeh Azar](https://scholar.google.com/citations?user=kojTGo8AAAAJ&hl=en), [Shuai Li](https://derkbreeze.github.io), [Olga Zatsarynna](https://olga-zats.github.io), [Juergen Gall](https://pages.iai.uni-bonn.de/gall_juergen/)
> 
> [![arXiv](https://img.shields.io/badge/arXiv-2503.18933-b31b1b.svg)](https://arxiv.org/abs/2503.18933) [![Project Page Badge](https://img.shields.io/badge/CVPR'25-Coming%20soon-blue)]() [![Project Page Badge](https://img.shields.io/badge/Project%20Page-Visit%20Site-brightgreen)](https://syncvp.github.io/)
> 

## How to
### Training
To train your model you can use predefined [config files](configs/run/README.md) or define custom ones.

You will need to follow the next steps:
#### 1. Train autoencoder (ideally one per modality)
```bash
python3 main.py --config configs/run/train/ae_city_rgb.yaml
```
(Optional) GAN fine-tuning: A few iterations may increase autoencoder reconstruction performance.
#### 2. Train a single modality diffusion model
This can already be used as a standalone model for video prediction.
```bash
python3 main.py --config configs/run/train/ddpm_city_rgb.yaml
```
#### 3. Train a Multi-modal diffusion model (SyncVP).
You can either initialize this with pre-trained modality specific diffusion models or train it from scratch, we recommend the first option as discussed in the [paper](https://arxiv.org/pdf/2503.18933).
```bash
python3 main.py --config configs/run/train/sync_city.yaml
```
### Evaluation
Coming soon ...

## Model checkpoints
Cityscapes autoencoders and multi-modal model checkpoints can be downloaded using:
```bash
bash download.sh
```
## Datasets
Preprocessed version of Cityscapes at 128x128 resolution with disparity (depth) maps can be downloaded [here](https://uni-bonn.sciebo.de/s/H7ke289qsY4I3lV).

## 📋 TODO List

- [ ] Non 1:1 aspect ratio implementation
- [ ] Full evaluation code release
- [X] Training code released

## Cite
(Will be updated with @inproceedings of CVPR once available)
```bibtex
@article{pallotta2025syncvp,
  title={SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction},
  author={Pallotta, Enrico and Azar, Sina Mokhtarzadeh and Li, Shuai and Zatsarynna, Olga and Gall, Juergen},
  journal={arXiv preprint arXiv:2503.18933},
  year={2025}
}
```


## Reference
This repository is mainly based on the PVDM [codebase](https://github.com/sihyun-yu/PVDM).



