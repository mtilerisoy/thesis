# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "8"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "8"  # Set this to the number of CPUs you want to use

import copy
import torch
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule as MTDataModuleVILT

from quantization_utils import SmallMTDataModuleVILT, get_module_by_path, quantize_modules, freeze_except_layers
import configs
from vilt.modules.kd_module import KDLightningModule
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer
from time import time
import run_vilt_kd_config as CLI


print("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐")
print("│                                                                                                        │")
print(f"│  Running with: epochs={CLI.EPOCHS}, max_steps={CLI.MAX_STEPS}, learning_rate={CLI.LEARNING_RATE}    \n│")
print(f"│  dataset={CLI.DATASET}, percentage={CLI.PERCENTAGE}, alpha_kd={CLI.ALPHA_KD}                        \n│")
print(f"│  gpu={CLI.GPU}, kd_layer={CLI.KD_LAYER}, temperature={CLI.TEMPERATURE}                              \n│")
print(f"│  log_dir={CLI.LOG_DIR},                                                                         \n  │")
print("│                                                                                                        │")
if CLI.EPOCHS == -1:
    print("│  Running INFINTE Training Loop. Please stop the script manually.                                 \n│")
print("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘")

if __name__ == "__main__":
    if CLI.DATASET == "nlvr2_ood":
        _config = configs.vilt_config_nlvr2
    elif CLI.DATASET == "nlvr2_original":
        _config = configs.vilt_config_nlvr2_original
    else:
        raise ValueError(f"Unknown dataset: {CLI.DATASET}")
    
    # ========== Update the configuration ==========
    _config["batch_size"] = 32
    _config["per_gpu_batchsize"] = 16
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # ========== Initialize the datamodule for pl.Trainer ==========
    # dm = MTDataModule(_config, dist=False)
    dm = SmallMTDataModuleVILT(_config, dist=False, percentage=CLI.PERCENTAGE)
    dm.setup("", is_random=True)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    # test_dataloader = dm.test_dataloader()

    print("Dataloader Length: ", len(train_dataloader))

    print(f"Length of the first batch: {len(next(iter(train_dataloader))['answers'])}")
    print(f"Shape of the first batch: {next(iter(train_dataloader))['image_0'][0].shape}")


    # =============== Initialize Full Precision Model ==============
    model_teacher = ViLTransformerSS(_config)
    model_student = copy.deepcopy(model_teacher)
    model_student.kd_layer = CLI.KD_LAYER
    model_teacher.eval()

    # Define the modeules to train
    modules_to_train = CLI.modules_to_train

    qat_quantizer = Int8DynActInt4WeightQATQuantizer()
    model_student = qat_quantizer.prepare(model_student, **modules_to_train)
    
    # Freeze all layers except for the specified ones
    freeze_except_layers(model_student, modules_to_train['layer_names'])

    # Initialize the KD model
    kd_model = KDLightningModule(student_model=model_student, teacher_model=model_teacher, alpha_kd=CLI.ALPHA_KD, lr=CLI.LEARNING_RATE, config=_config, **modules_to_train)

    print("Model Scale Factor: ", model_student.scale_factor)
    # ========== Initialize the trainer for full precision ==========
    _config["exp_name"] = CLI.EXP_NAME
    _config["log_dir"] = CLI.LOG_DIR
    exp_name = f'{_config["exp_name"]}'
    os.makedirs(_config["log_dir"], exist_ok=True)

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=CLI.EXP_NAME,
        default_hp_metric=False
    )
    
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    print("Gradient Accumulation Steps: ", grad_steps)

    # =============== Testing Quantized Model ===============
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=CLI.GPU,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        benchmark=True,
        deterministic=True,
        max_epochs=CLI.EPOCHS,
        max_steps=CLI.MAX_STEPS,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=grad_steps,
        enable_checkpointing=False,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    # Store the initial weights before training
    fc2_weight = get_module_by_path(model_student, modules_to_train['layer_names'][-1]).weight.clone()
    

    print("Starting Full Precision Training")
    # Train the model with the quantization-aware training (QAT) quantizer
    trainer.fit(kd_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Store the weights after training before quantization
    fc2_weight_after_qat = get_module_by_path(model_student, modules_to_train['layer_names'][-1]).weight.clone()

    # Quantize the model
    model_quant = quantize_modules(model_student, modules_to_train['layer_names'], 4)

    # Store the weights after quantization
    fc2_weight_after_dyn_quant = get_module_by_path(model_quant, modules_to_train['layer_names'][-1]).weight().int_repr().clone()

    # Print the tensor information
    print("============================================================")
    print(f"Min, Max and Mean of the ORIGINAL WEIGHTS: \n{torch.min(fc2_weight)}, {torch.max(fc2_weight)}, mean: {torch.mean(fc2_weight)}")
    print(f"Min, Max and Mean of the QAT WEIGHTS: \n{torch.min(fc2_weight_after_qat)}, {torch.max(fc2_weight_after_qat)}, mean: {torch.mean(fc2_weight_after_qat)}")
    print(f"Min, Max and Mean of the QAT WEIGHTS: \n{torch.min(fc2_weight_after_dyn_quant)}, {torch.max(fc2_weight_after_dyn_quant)}")
    print("Model Scale Factor after training: \n", model_student.scale_factor)
    print("============================================================")


    # =============== Testing Quantized Model ===============
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        benchmark=True,
        deterministic=True,
        max_steps=CLI.MAX_STEPS,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=grad_steps,
        enable_checkpointing=False,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    # Initalize the ood dataset
    _config = configs.vilt_config_nlvr2
    dm = SmallMTDataModuleVILT(_config, dist=False, percentage=1)
    dm.setup("test", is_random=True)
    test_dataloader = dm.test_dataloader()


    trainer.test(model_student, dataloaders=test_dataloader)