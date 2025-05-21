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

from meter.modules import METERTransformerSS
from meter.modules.kd_module import KDLightningModule

from quantization_utils import SmallMTDataModuleMETER, quantize_modules, freeze_except_layers
import configs
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer
import run_meter_kd_config as CLI


print("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐")
print("│                                                                                                        │")
print(f"│  Running with: epochs={CLI.EPOCHS}, max_steps={CLI.MAX_STEPS}, learning_rate={CLI.LEARNING_RATE}    \n│")
print(f"│  dataset={CLI.DATASET}, percentage={CLI.PERCENTAGE}, alpha_kd={CLI.ALPHA_KD}                        \n│")
print(f"│  gpu={CLI.GPU}, kd_layer={CLI.KD_LAYER}, log_dir={CLI.LOG_DIR},                                     \n│")
print(f"│                                                                                                     \n│")
print("│                                                                                                        │")
if CLI.EPOCHS == -1:
    print("│  Running INFINTE Training Loop. Please stop the script manually.                                 \n│")
print("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘")

if __name__ == "__main__":
    if CLI.DATASET == "nlvr2_ood":
        _config = configs.meter_config_nlvr2_ood
    elif CLI.DATASET == "nlvr2_original":
        _config = configs.meter_config_nlvr2_id
    else:
        raise ValueError(f"Unknown dataset: {CLI.DATASET}")
    
    # ========== Update the configuration ==========
    _config["batch_size"] = 32
    _config["per_gpu_batchsize"] = 2
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # ========== Initialize the datamodule for pl.Trainer ==========
    # dm = MTDataModule(_config, dist=False)
    dm = SmallMTDataModuleMETER(_config, dist=False, percentage=1)
    dm.setup("train", is_random=True)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_dataloader = dm.test_dataloader()

    print("Dataloader Length: ", len(train_dataloader))
    print("Dataloader Length: ", len(val_dataloader))
    print("Dataloader Length: ", len(test_dataloader))


    # =============== Initialize Full Precision Model ==============
    model_teacher = METERTransformerSS(_config)
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

    print("Starting Full Precision Training")
    # Train the model with the quantization-aware training (QAT) quantizer
    # trainer.fit(kd_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.fit(kd_model, train_dataloaders=train_dataloader)

    # Quantize the model
    model_quant = quantize_modules(model_student, modules_to_train['layer_names'], 4)

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
    _config = configs.meter_config_nlvr2_ood
    dm = SmallMTDataModuleMETER(_config, dist=False, percentage=1)
    dm.setup("test", is_random=True)
    test_dataloader = dm.test_dataloader()
    print("Dataloader Length: ", len(test_dataloader))
    model_quant.eval()


    trainer.test(model_quant, dataloaders=test_dataloader)