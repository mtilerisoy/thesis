# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "9"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "9"  # Set this to the number of CPUs you want to use

import copy
import torch
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule as MTDataModuleVILT

from quantization_utils import SmallMTDataModuleVILT, get_module_by_path, quantize_modules
import configs
from vilt.modules.kd_module import KDLightningModule
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer


import argparse
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Custom QAT Script")
parser.add_argument("-e", "--epochs",           type=int,   default=2,                  help="Number of epochs to train the model")
parser.add_argument("-l", "--learning_rate",    type=float, default=2e-4,               help="Learning rate for the optimizer")
parser.add_argument("-d", "--dataset",          type=str,   default="nlvr2_ood",        help="Dataset to train the model on")
parser.add_argument("-p", "--percentage",       type=float, default=0.01,                help="Percentage of the dataset to use for fine-tuning")
parser.add_argument("-g", "--gpu",              type=int,   default=1,                  help="List of GPUs to use for training")
args = parser.parse_args()
# print("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐")
# print("│                                                                                                     │")
# print(f"│  Running with: epochs={args.epochs}, learning_rate={args.learning_rate}, percentage={args.percentage}    │")
# print("│                                                                                                     │")
# print("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘")



def freeze_except_layers(model, layers_to_unfreeze_names):
    """
    Freezes all parameters of a PyTorch model except for the layers specified by their names.

    Args:
        model (nn.Module): The PyTorch model to freeze parameters in.
        layers_to_unfreeze_names (list of str): A list of module names that should NOT be frozen.
                                             Parameters in modules whose names contain these strings will be unfrozen.
    """
    for name, param in model.named_parameters():
        freeze = True  # Initially assume we should freeze the parameter
        for layer_name_to_unfreeze in layers_to_unfreeze_names:
            if layer_name_to_unfreeze in name:
                freeze = False  # Unfreeze if the name contains a layer to unfreeze
                break  # No need to check other layer names if already unfrozen

        if freeze:
            param.requires_grad = False  # Freeze the parameter
        else:
            param.requires_grad = True   # Ensure it's unfrozen (explicitly set to True)

    # Print which layers are frozen and unfrozen for verification
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Frozen: {not param.requires_grad}")

if __name__ == "__main__":
    if args.dataset == "nlvr2_ood":
        _config = configs.vilt_config_nlvr2
    elif args.dataset == "nlvr2_original":
        _config = configs.vilt_config_nlvr2_original
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Adjust the batch size
    _config["batch_size"] = 32
    _config["per_gpu_batchsize"] = 16
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # ========== Initialize the datamodule for pl.Trainer ==========
    # dm = MTDataModule(_config, dist=False)
    dm = SmallMTDataModuleVILT(_config, dist=False, percentage=args.percentage)
    dm.setup("test", is_random=True)
    # train_dataloader = dm.train_dataloader()
    # val_dataloader = dm.val_dataloader()
    train_dataloader = dm.test_dataloader()

    # print("Val Dataloader Length: ", len(val_dataloader))
    print("Test Dataloader Length: ", len(train_dataloader))

    print(f"Length of the first batch: {len(next(iter(train_dataloader))['answers'])}")
    print(f"Shape of the first batch: {next(iter(train_dataloader))['image_0'][0].shape}")


    # =============== Initialize Full Precision Model ==============
    model_teacher = ViLTransformerSS(_config)
    model_student = copy.deepcopy(model_teacher)
    model_student.kd_layer = 0

    model_teacher.eval()

    # Define the modeules to train
    modules_to_train = {'layer_names': ["scale_factor",
                                        "transformer.blocks.0.mlp.fc1",
                                        "transformer.blocks.0.mlp.fc2"],
                        'kd_layer': 0}

    qat_quantizer = Int8DynActInt4WeightQATQuantizer()
    model_student = qat_quantizer.prepare(model_student, **modules_to_train)
    
    # Freeze all layers except for the specified ones
    freeze_except_layers(model_student, modules_to_train['layer_names'])

    # Initialize the KD model
    kd_model = KDLightningModule(student_model=model_student, teacher_model=model_teacher, alpha_kd=1, lr=args.learning_rate, config=_config, **modules_to_train)

    print("Model Scale Factor: ", model_student.scale_factor)
    # ========== Initialize the trainer for full precision ==========
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

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
        devices=[args.gpu],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        benchmark=True,
        deterministic=True,
        max_epochs=args.epochs,
        max_steps=50000,
        logger=False,
        accumulate_grad_batches=grad_steps,
        enable_checkpointing=False,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    # Store the initial weights before training
    fc2_weight = get_module_by_path(model_student, modules_to_train['layer_names'][-1]).weight.clone()
    

    print("Starting Full Precision Training")
    # Train the model with the quantization-aware training (QAT) quantizer
    trainer.fit(kd_model, train_dataloaders=train_dataloader)

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



    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        # strategy="ddp",
        benchmark=True,
        deterministic=False,
        max_epochs=args.epochs,
        max_steps=50000,
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