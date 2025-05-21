# Minimizing Quantization Error in Vision-Language Models Through Token-level Knowledge Distillation

This repository is the codebase of the master's thesis, successfully defened on 20 May 2025 at Eindhoven University of Technology.

The code utulizes two Vision-Language models: [ViLT]() and [METER]() whose implementations are under ```vilt/``` and ```meter/```. The original implementations were modified to make them compatible with Python 3.12.

## Download Pretrained Weights
The pretrained model weights can be accessed through the following links:
1. ViLT-B/32 Pretrained with MLM+ITM for 200k steps on GCC+SBU+COCO+VG (ViLT-B/32 200k) [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt)
2. ViLT-B/32 200k finetuned on VQAv2 [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_vqa.ckpt)
3. ViLT-B/32 200k finetuned on NLVR2 [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_nlvr2.ckpt)
4. METER-CLIP16-RoBERTa (resolution: 224^2) pre-trained on GCC+SBU+COCO+VG [link](https://github.com/zdou0830/METER/releases/download/checkpoint2/meter_clip16_224_roberta_pretrain.ckpt)
5. METER-CLIP16-RoBERTa fine-tuned on VQAv2 (resolution: 576^2) [link](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_vqa.ckpt)
6. METER-CLIP16-RoBERTa fine-tuned on NLVR2 (resolution: 288^2) [link](https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_nlvr2.ckpt)

Put the weights under your root model folder.

## Repo Structure
The structure of the codebase is as follows

    root
    ├── vilt/                           # ViLT model implementation and utilities
    │   ├── modules/                    # Core model architecture components
    │   ├── datasets/                   # Dataset loaders for ViLT
    │   ├── utils/                      # Utility functions for data processing
    │   │   ├── write_coco_karpathy.py  # Data conversion for COCO dataset
    │   │   ├── write_vqa.py            # Data conversion for VQA dataset
    │   │   ├── write_nlvr2.py          # Data conversion for NLVR2 dataset
    │   │   └── ...                     # Other data conversion utilities
    │   └── configs/                    # Configuration files for ViLT
    │
    ├── meter/                          # METER model implementation and utilities
    │   ├── modules/                    # Core model architecture components
    │   ├── datasets/                   # Dataset loaders for METER
    │   ├── utils/                      # Utility functions
    │   └── configs/                    # Configuration files for METER
    │
    ├── data/                           # VLUE benchmark annotations in json format
    │
    ├── experiments/                    # Experiment results and checkpoints
    │   ├── vilt/                       # ViLT experiment results
    │   └── meter/                      # METER experiment results
    │
    ├── run_vilt.py                     # Script to run ViLT model in FP32
    ├── run_vilt_ptq.py                 # Post-training quantization for ViLT
    ├── run_vilt_kd.py                  # Knowledge distillation retrainign for ViLT
    ├── run_vilt_kd_config.py           # Configuration for ViLT knowledge distillation
    │
    ├── run_meter.py                    # Script to run METER model FP32
    ├── run_meter_ptq.py                # Post-training quantization for METER
    ├── run_meter_kd.py                 # Knowledge distillation retrainign for METER
    ├── run_meter_kd_config.py          # Configuration for METER knowledge distillation
    │
    ├── run_block_sensitivity.py        # Block sensitivity analysis script
    ├── assess_vqa.py                   # VQA assessment script
    ├── arrow.py                        # Data conversion to arrow format
    ├── quantization_utils.py           # Utilities for quantization
    │
    ├── DATA.md                         # Instructions for dataset preparation
    ├── EVAL.md                         # Evaluation instructions
    ├── TRAIN.md                        # Training instructions
    ├── README.md                       # Main repository README
    ├── requirements.txt                # Python dependencies
    └── setup.py                        # Package installation script

Please note that model weights and data files not necessarliy have to be included under this root.

## Environment Setup


## Dataset
Please refer to the ```DATA.md``` file for instructions on how to download and prepare the datasets.

## Evaluation
Please refer to the ```EVAL.md``` file for the details of how to run the scripts