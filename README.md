# HouseDiffusion

Vector floorplan generation with a diffusion model (discrete + continuous denoising). Upstream implementation: [aminshabani/house_diffusion](https://github.com/aminshabani/house_diffusion). Paper: [HouseDiffusion (arXiv)](https://arxiv.org/abs/2211.13287). Builds on the [guided-diffusion](https://github.com/openai/guided-diffusion) style codebase. Place RPLAN data under `datasets/rplan` and processed tensors under `processed_rplan/` as in the original workflow.

## Setup and usage

Install dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e .
```

Train (example):

```bash
python scripts/image_train.py --dataset rplan --batch_size 32 --set_name train --target_set 8
```

Sample / evaluate (run from repo root; outputs go to `outputs/`; FID uses rasterized copies in `outputs/gt_png` and `outputs/pred_png`):

```bash
python scripts/image_sample.py --dataset rplan --batch_size 32 --set_name eval --target_set 8 --model_path /path/to/model250000.pt --num_samples 64 --max_length 2048
```

Optional: `--max_length` caps how many layouts are loaded from processed `.npz` files to save RAM.
