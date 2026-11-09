# Trans-TCDM

Anonymous main-experiment implementation for diffusion MRI super-resolution.
The shared model contains dual-path upsampling, a Transformer TDI encoder,
a Transformer diffusion encoder, and a convolutional decoder.

## Files

```text
model.py             Main conditional denoising network
diffusion.py         Noise-prediction loss and DDIM sampling with resampling
train.py             Shared training entry point
evaluate.py          Sampling, DWI metrics, and FA evaluation
datasets/            Original HCP, CHCP, and PD data loaders
configs/             Main-experiment settings and relative paths
requirements.txt     Runtime dependencies
```

## Environment

Use Python 3.10 and a CUDA-enabled PyTorch installation suitable for the machine.
Install PyTorch first, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The dependency file is a runtime dependency list, not an exported lockfile of
the original training environment. Run commands from this repository root.

## Prepared data

Provide the preprocessed HDF5 files and edit the paths in the selected JSON
configuration. A path named `train_data_dir` or `test_file_z` denotes an HDF5
**file**, including when its name has no extension. Image data, subject lists,
preprocessing scripts, checkpoints, and experimental outputs are not bundled.
Use separate training and test subjects; the JSON files do not create splits.

Let `N` be the number of single-channel training samples, `S` the number of
axial slices per subject, `C = 30` the number of diffusion directions, and
`H, W` the high-resolution in-plane dimensions. LR has half the in-plane
resolution. Supply `float32` arrays; LR and HR use the `[0, 1]` scale.

| Dataset | HR size | Training HDF5 | Test HDF5 |
| --- | --- | --- | --- |
| HCP | 112 x 144 | Root `hr: [N,H,W]`, `lr: [N,H/2,W/2]`, `fibre_density: [N,H,W]` | Subject groups |
| CHCP | 140 x 140 | Subject groups | Subject groups |
| PD | 88 x 88 | Root `hr: [N,H,W]`, `lr: [N,H/2,W/2]`, `fibre_density: [N,H,W]` | Subject groups |

Each test subject group contains `hr: [S,C,H,W]`,
`lr: [S,C,H/2,W/2]`, and `fibre_density: [S,1,H,W]`.
HCP also accepts `fibre_density: [S,H,W]`.
CHCP training uses the same grouped layout and expands each slice/direction
into a single-channel training sample. Evaluation processes one slice at a time.

For each subject key `<subject>`, the configured `affine_dir` must contain
`<subject>_data_affine`, a text file containing the 4 x 4 affine matrix for
the output volume. For evaluation with a brain mask and FA, `data_dir` contains:

```text
<subject>_mask.nii.gz
<subject>_b0.nii.gz
<subject>_bvals.txt
<subject>_bvecs.txt
<subject>_bvals_b0.txt
<subject>_bvecs_b0.txt
```

The existing evaluation path crops auxiliary volumes to the middle `S` slices
and resizes them in-plane when needed. Prepare aligned auxiliary volumes and
an affine appropriate for the saved volume. Without a mask, DWI metrics run
without masking; without the gradient/b0 files, FA evaluation is skipped.
These settings affect metric comparability.

## Train and evaluate

```bash
python train.py --config configs/hcp.json --gpu 0
python evaluate.py --config configs/hcp.json --gpu 0
```

Use `configs/chcp.json` or `configs/pd.json` for the other datasets.
Outputs go to `outputs/<dataset>/`. The defaults retain 51 epochs, batch size
1, AdamW with learning rate `1e-4` and weight decay `1e-4`, five warmup epochs
followed by cosine decay, 2,000 diffusion steps, 50 DDIM steps, and `U = 3`.
The noise-prediction objective is summed smooth L1 loss. The DDIM sampler
retains `eta = 1.0`. The random seed is 123.

Select a saved epoch explicitly when needed:

```bash
python evaluate.py --config configs/hcp.json --gpu 0 --checkpoint outputs/hcp/checkpoints/ckpt_0050.pth
```

`--subjects` selects test HDF5 subject keys. Existing SR volumes are reused;
pass `--force_resample` when switching checkpoints or sampling settings.

## Retained checkpoint behavior

The original trainer saves a checkpoint for every epoch. Its `best.pth` stores
the minimum-training-loss epoch metadata, but the referenced model tensors
continue updating until the end of training. Therefore, use `ckpt_XXXX.pth`
to evaluate the weights of a specific epoch. This behavior is retained here
to avoid silently changing the experimental implementation.
`--resume` reloads model weights and advances the epoch counter; it does not
restore the optimizer, scheduler, or random-number state.

This source release does not include trained weights or establish reproduced
paper metrics. Dataset access and preparation are required for a full run.

## Checks performed

Core model, diffusion, data-loader, and metric implementations were checked
against the supplied code. Syntax checks, CPU forwards at all three native
image sizes, a loss/backward check on 16 x 16 inputs, DDIM sampling with
50 steps and `U = 3` on 16 x 16 inputs, and synthetic HDF5 loading passed.
Full training/evaluation entry points were not run; the check environment
lacked `scikit-image` and `tensorboard`.
