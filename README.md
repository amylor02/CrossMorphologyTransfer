# Cross-Morphology Motion Transfer with Semantic Style Alignment

**Alexios Mylordos, Jose Luis Ponton, Nuria Pelechano, Andreas Aristidou**

## Overview

This repository contains the PyTorch implementation of the method presented in _Cross-Morphology Motion Transfer with Semantic Style Alignment_, published in **IEEE Transactions on Visualization and Computer Graphics (2026)**.

The method transfers motion between characters with different skeletal structures using morphology-agnostic control signals and morphology-specific learned motion representations.

[Paper](https://doi.org/10.1109/TVCG.2026.3729292) | [Project website](https://andreasaristidou.com/CrossMorphologyTransfer)

## Requirements

- Python 3.11 (tested with Python 3.11.5).
- PyTorch 2.1.0 with CUDA 12.1 (CUDA-capable GPU recommended).
- Additional dependencies listed in `python/requirements.txt`.

## Installation

1. Clone the repository and navigate to its `python` directory:

   ```powershell
   git clone https://github.com/amylor02/CrossMorphologyTransfer.git
   cd CrossMorphologyTransfer\python
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv env
   .\env\Scripts\Activate.ps1
   ```

3. Install the required packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

## Data and Pretrained Models

The pretrained models are available in `pretrained_models.zip` on the [GitHub Releases page](https://github.com/amylor02/CrossMorphologyTransfer/releases).

Download the archive and extract its `models/` directory into `python/`.

The datasets and quantitative evaluation inputs are not publicly distributed. **To request access, contact [alexymlordos@gmail.com](mailto:alexymlordos@gmail.com).** Availability of third-party data is subject to the applicable licenses.

After obtaining the files, the `python/` directory should be organized as follows:

```text
python/
├── data/
│   ├── dog/
│   │   ├── train/
│   │   └── eval/
│   ├── human/
│   │   ├── train/
│   │   └── eval/
│   └── ostrich/
│       ├── train/
│       └── eval/
├── models/
│   ├── model_dog/
│   ├── model_human/
│   ├── model_dog_contact/
│   ├── model_dog_sml_prior_net/
│   ├── model_human_sml_prior_net/
│   └── model_ostrich/
├── metric_inputs/
└── src_ortho_new/
```

Each model directory contains `generator.pt` and `data.pt`.

## Reproducing Quantitative Results

To reproduce the motion fidelity metrics in **Table I** and the contact-phase metrics in **Table II**, run:

```powershell
python src_ortho_new\reproduce_paper_metrics.py
```

The script uses the reference datasets in `data/` and the evaluation motions in `metric_inputs/`. To request these inputs, contact [alexymlordos@gmail.com](mailto:alexymlordos@gmail.com)

## Scripts for Generating Motions

After installing the dependencies, extracting the pretrained models, and obtaining the required BVH inputs, run any of these parameter-free PowerShell scripts from the `python/` directory:

```powershell
.\generate_dog_from_human.ps1
.\generate_human_from_dog.ps1
.\generate_dog_from_ostrich_fc.ps1
```

The scripts use the fixed inference settings shown in the section below and save the generated BVH files to `python/results/`. Run them one at a time.

## Usage Examples

Run the following commands from the `python/` directory.

### Human → Dog

```powershell
python src_ortho_new\eval_vq_vae.py `
    .\models\model_dog `
    .\data\dog\train\D1_055_KAN01_001.bvh `
    generator `
    --input_path2 .\data\human\train\S01_A01.bvh `
    --training-stage prior `
    --decoder-source predicted_codebook `
    --prior-stage-architecture `
    --inference-temperature 0.5 `
    --no-stationary-root-lock
```

### Dog → Human

```powershell
python src_ortho_new\eval_vq_vae.py `
    .\models\model_human `
    .\data\human\train\S01_A01.bvh `
    generator `
    --input_path2 .\data\dog\train\D1_053_KAN01_001.bvh `
    --training-stage prior `
    --decoder-source predicted_codebook `
    --prior-stage-architecture `
    --inference-temperature 0.5 `
    --no-stationary-root-lock
```

### Ostrich → Dog (Foot-Contact Conditioning)

```powershell
python src_ortho_new\eval_vq_vae.py `
    .\models\model_dog_contact `
    .\data\dog\train\D1_055_KAN01_001.bvh `
    generator `
    --input_path2 .\data\ostrich\train\idle_walk_idle_2.bvh `
    --training-stage prior `
    --decoder-source predicted_codebook `
    --prior-stage-architecture `
    --foot-contact-conditioning `
    --inference-temperature 0.5 `
    --no-stationary-root-lock
```

Generated BVH files are saved under `python/data/` with an `eval_` prefix.

## License and Acknowledgments

The code is distributed under the [MIT License](LICENSE). The original SparsePoser license terms for model weights and data are included in [LICENSE_data](LICENSE_data). Third-party datasets and assets remain subject to their respective licenses.

This implementation builds upon [SparsePoser](https://github.com/UPC-ViRVIG/SparsePoser) by Ponton et al. It also includes a modified version of [PyMotion](https://github.com/UPC-ViRVIG/pymotion); see the license and notice in `python/third_party/upc_pymotion_modified/`.
