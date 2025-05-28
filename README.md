# CoLM
![Python 3.10](https://img.shields.io/badge/python-3.10-green)
![Pytorch 2.4.0](https://img.shields.io/badge/pytorch-2.4.0-green)
![License MIT](https://img.shields.io/badge/license-MIT-blue)

This repository is the official implementation of our ICLR 2025 paper [Mini-batch Coresets for Memory-efficient Language Model Training on Data Mixtures](https://arxiv.org/pdf/2407.19580).

## 🔗 Quick Links
- [CoLM](#colm)
  - [🔗 Quick Links](#-quick-links)
  - [Install Requirements](#install-requirements)
  - [Data Preparation](#data-preparation)
  - [Data Selection Pipeline](#data-selection-pipeline)
  - [Evaluation](#evaluation)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)


## Install Requirements
```bash
conda create -n colm python=3.10
conda activate colm
conda install -c nvidia cuda-python
pip install -r requirement.txt --no-cache-dir --no-build-isolation
git clone https://github.com/hsgser/vllm.git
cd vllm
VLLM_INSTALL_PUNICA_KERNELS=1 pip install -e .
cd ..
pip install traker[fast] --no-cache-dir
pip install flash-attn==2.5.7 --no-build-isolation
pip install -i https://pypi.org/simple/ bitsandbytes
git clone https://github.com/decile-team/submodlib.git
cd submodlib
pip install -e .
cd ..
pip install -e .
```

## Data Preparation
Please download MathInstruct dataset with additional annotations [here](https://drive.google.com/file/d/1kpYMJ0xrn0eLyv-uwhUZCTjFWT6Zlb-Q/view?usp=sharing) and store it under the following path `/data/MathInstruct.jsonl`.

## Data Selection Pipeline
```bash
bash scripts/run_math.sh
```

## Evaluation
```bash
cd math_eval
bash eval_finetuned.sh /path/to/your/model
```

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Dang Nguyen (nguyentuanhaidang@gmail.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{nguyen2024mini,
  title = {Mini-batch Coresets for Memory-efficient Language Model Training on Data Mixtures},
  author = {Nguyen, Dang and Yang, Wenhan and Anand, Rathul and Yang, Yu and Mirzasoleiman, Baharan},
  journal = {International Conference on Learning Representations (ICLR)},
  year = {2025}
}
```

## Acknowledgments  
The structure of this repository is largely based on the official implementation of [LESS](https://github.com/princeton-nlp/LESS) and [MeZO](https://github.com/princeton-nlp/MeZO). We are grateful for their open sources.