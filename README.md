# Think Global, Look Focal: Dual-Stream Retrieval-Augmented Pathology Report Generation for Whole Slide Images

<div align="center">

**Liang Xiong<sup>1*</sup>**, **Yi Zhang<sup>1*</sup>**, **HaoJie Song<sup>2</sup>**, **QianYao Peng<sup>1</sup>**, **ShaoGuo Cui<sup>1</sup>**

<sup>1</sup> Chongqing Normal University &nbsp;&nbsp; <sup>2</sup> Shenzhen University

\* Equal contribution

[![Paper](https://img.shields.io/badge/Paper-MICCAI%202026-blue)](#citation)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

This repository contains the official implementation of **DR-Gen**, a dual-stream retrieval-augmented framework for generating pathology reports from gigapixel whole slide images (WSIs). DR-Gen follows the diagnostic workflow of pathologists: **think globally, look focally, and reason with reliable medical knowledge**.

## Overview

Existing WSI report generation methods face two major challenges:

- **Visual dilution:** diagnostically important focal regions can be overwhelmed by thousands of background patches.
- **Retrieval noise:** unstructured or low-confidence retrieved evidence may introduce non-diagnostic content and factual hallucinations.

DR-Gen addresses these problems with three main components:

1. **Global-focal dual-stream visual decomposition** separates slide-level tissue architecture from high-relevance focal regions.
2. **Visual Priming Adapter (VPA)** initializes visual queries with slide-specific global and focal anchors instead of purely random queries.
3. **Adaptive Confidence Gating (ACG)** calibrates global and focal retrieval confidence to suppress unreliable diagnostic evidence from a CAP-guided atomic knowledge bank.

The visual and textual representations are finally integrated by a query-driven cross-modal encoder for autoregressive pathology report generation.

## Framework

<p align="center">
  <img src="pics/framework.png" width="100%" alt="DR-Gen framework">
</p>

The global stream summarizes slide-level morphology through attention pooling, while the focal stream retains the most diagnostically relevant patches. The two streams provide both visual anchors for VPA and multi-granularity queries for knowledge retrieval. ACG reweights retrieved evidence according to calibrated confidence before cross-modal fusion.

## Main Results

DR-Gen is evaluated on the **PathText (BRCA)** benchmark, which contains 1,041 aligned WSI-report pairs: 845 for training, 98 for validation, and 98 for testing.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | Factent | Precision | Recall | F1 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MI-Gen | 0.408 | 0.262 | 0.171 | 0.117 | 0.164 | 0.287 | 0.470 | 0.865 | 0.805 | 0.834 |
| BiGen | 0.437 | 0.282 | 0.186 | 0.125 | 0.172 | 0.282 | 0.494 | 0.897 | 0.884 | 0.891 |
| **DR-Gen** | **0.441** | **0.291** | **0.198** | **0.138** | **0.173** | **0.298** | **0.519** | **0.958** | **0.896** | **0.926** |

DR-Gen improves BLEU-4 by **10.4%** relative to BiGen and achieves the best factual consistency and carcinoma subtype classification performance in the comparison.

## Data Preparation

### 1. WSI preprocessing

Following the paper, WSIs are processed as follows:

1. Segment tissue regions with [CLAM](https://github.com/mahmoodlab/CLAM).
2. Extract non-overlapping `256 x 256` patches at `10x` magnification.
3. Encode each patch with [CONCH](https://github.com/mahmoodlab/CONCH).
4. Save the patch features for each WSI as a PyTorch tensor with shape `[N, 512]` in a `.pt` file.

Feature filenames must begin with a TCGA case identifier such as `TCGA-A7-A0CJ`. The dataset loader uses the first three hyphen-separated fields to match features, reports, splits, and clinical attributes.

### 2. Report annotations

Organize the report annotations by case:

```text
annotations/
├── TCGA-A7-A0CJ-.../
│   └── annotation
├── TCGA-A8-A07E-.../
│   └── annotation
└── ...
```

Each `annotation` file should contain the corresponding pathology report as plain text. The vocabulary builder expects this exact filename.

### 3. Splits and clinical attributes

The repository provides the BRCA split files and seven clinical attribute labels under `ocr/dataset_csv/`:

```text
ocr/dataset_csv/
├── splits_0.csv
├── splits_3.csv
└── target_attributes.csv
```

The seven attributes are high grade, multifocality, necrosis, lymphovascular invasion, ductal subtype, lymph-node positivity, and ductal carcinoma in situ.

### 4. Diagnostic knowledge bank

`memory_short.pt` contains the 512-dimensional encoded diagnostic knowledge bank used by the retrieval module. A custom bank may be supplied with `--bank_path`; it must be either a tensor of shape `[M, 512]` or a dictionary containing that tensor under the `features` key.

## Installation

```bash
git clone https://github.com/jasonlightx/DR-Gen.git
cd DR-Gen

conda create -n drgen python=3.8 -y
conda activate drgen

pip install torch torchvision numpy pandas pillow tqdm einops thop pycocoevalcap
```

Install a CUDA-enabled PyTorch build compatible with your system. The experiments reported in the paper were conducted on a single NVIDIA RTX 5000 Ada GPU with 32 GB memory.

The METEOR evaluator in `pycocoevalcap` also requires a Java runtime.

The optional OCR preprocessing script additionally requires:

```bash
pip install pymupdf pytesseract
```

## Training

```bash
python main.py \
  --mode Train \
  --n_gpu 0 \
  --image_dir /path/to/conch_features \
  --ann_path /path/to/pathtext_brca_annotations \
  --split_path ocr/dataset_csv/splits_0.csv \
  --bank_path memory_short.pt \
  --save_dir results/drgen \
  --record_dir records/drgen
```

The paper uses the following default settings, which are also the defaults in `main.py`:

| Argument | Value | Description |
|:--|--:|:--|
| `--k` | `0.6` | Focal-stream patch retention ratio |
| `--v` | `5` | Retrieved knowledge entries per region |
| `--m` | `80` | Focal region size |
| `--d_model` | `512` | Transformer embedding dimension |
| `--num_layers` | `3` | Number of encoder/decoder layers |
| `--num_heads` | `4` | Number of attention heads |
| `--batch_size` | `1` | Training batch size |
| `--beam_size` | `3` | Beam-search width |
| `--lr_ed` | `1e-4` | Learning rate |
| `--weight_decay` | `5e-5` | Weight decay |

## Evaluation

Place the trained `model_best.pth` checkpoint in the directory supplied through `--checkpoint_dir`:

```bash
python main.py \
  --mode Test \
  --n_gpu 0 \
  --image_dir /path/to/conch_features \
  --ann_path /path/to/pathtext_brca_annotations \
  --split_path ocr/dataset_csv/splits_0.csv \
  --bank_path memory_short.pt \
  --checkpoint_dir results/drgen \
  --save_dir results/drgen_test
```

The included evaluation code computes BLEU, METEOR, and ROUGE scores through `pycocoevalcap`. Factent and clinical subtype metrics in the paper are reported using their corresponding external evaluation protocols.

## Repository Structure

```text
DR-Gen/
├── main.py                         # Training and evaluation entry point
├── memory_short.pt                 # Diagnostic knowledge bank
├── models/
│   └── r2gen.py                    # DR-Gen model wrapper
├── modules/
│   ├── encoder_decoder.py          # Dual streams, VPA, retrieval, and Transformer
│   ├── datasets.py                 # PathText/TCGA dataset loader
│   ├── dataloaders.py              # Distributed data loading and feature padding
│   ├── trainer.py                  # Training, validation, and testing loops
│   ├── tokenizers.py               # Report tokenizer
│   ├── metrics.py                  # NLG metrics
│   └── ...
├── baselines/                      # MIL baseline implementations
├── ocr/
│   ├── pdf2text.py                 # Optional pathology-report OCR utility
│   └── dataset_csv/                # Data splits and clinical attributes
└── pics/
    └── framework.png               # DR-Gen architecture
```

## Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{xiong2026drgen,
  title     = {Think Global, Look Focal: Dual-Stream Retrieval-Augmented Pathology Report Generation for Whole Slide Images},
  author    = {Xiong, Liang and Zhang, Yi and Song, HaoJie and Peng, QianYao and Cui, ShaoGuo},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

## Acknowledgements

This implementation benefits from [WsiCaption](https://github.com/cpystan/Wsi-Caption), [CLAM](https://github.com/mahmoodlab/CLAM), [CONCH](https://github.com/mahmoodlab/CONCH), and the open-source image-captioning and report-generation community.

## License

This project is released under the [MIT License](LICENSE).
