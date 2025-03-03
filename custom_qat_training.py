# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer

# General imports
import os
import torch
import torch.quantization
import pytorch_lightning as pl
from copy import deepcopy
import random
random.seed(42)

# Model Specific imports
from vilt.datamodules.multitask_datamodule import MTDataModule as MTDataModuleVILT
from meter.datamodules.multitask_datamodule import MTDataModule as MTDataModuleMeter
from vilt.modules import ViLTransformerSS
from meter.modules import METERTransformerSS

# Custom imports
import configs
from quantization_utils import get_qat_config, get_quantization_config, print_size_of_model, quantize_modules, init_trainer, get_module_by_path
from quantization_utils import  SmallMTDataModuleMETER, SmallMTDataModuleVILT
from custom_quantizer import apply_weight_quantization

import argparse
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Custom QAT Script")
parser.add_argument("-e", "--epochs",           type=int,   default=1,                  help="Number of epochs to train the model")
parser.add_argument("-l", "--learning_rate",    type=float, default=0.05,               help="Learning rate for the optimizer")
parser.add_argument("-d", "--dataset",          type=str,   default="nlvr2_original",   help="Dataset to train the model on")
parser.add_argument("-p", "--percentage",       type=float, default=0.05,               help="Percentage of the dataset to use for fine-tuning")
parser.add_argument("--gpu",                    type=int,   default=1,    help="List of GPUs to use for training")
args = parser.parse_args()
print("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐")
print("│                                                                                                     │")
print(f"│  Running with: epochs={args.epochs}, learning_rate={args.learning_rate}, dataset={args.dataset}, percentage={args.percentage}    │")
print("│                                                                                                     │")
print("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘")


# ==========================================
# ========= Set the configuration ==========
# ==========================================
# Set the general configurations

# _config = configs.vilt_config_nlvr2 if args.dataset == "nlvr2_ood" else configs.vilt_config_nlvr2_original if args.dataset == "nlvr2_original" else ModuleNotFoundError("Dataset not supported")
_config = configs.meter_config_nlvr2 if args.dataset == "nlvr2_ood" else configs.meter_config_nlvr2_original if args.dataset == "nlvr2_original" else ModuleNotFoundError("Dataset not supported")
_config["batch_size"] = 24
_config["per_gpu_batchsize"] = 24
_config["learning_rate"] = args.learning_rate

# fine_tune_percentage = 0.5
# validataion_percentage = 0.25
accumulation_steps = 1
max_epochs = args.epochs
# max_steps = 50000
validataion_percentage = 0.15
fine_tune_percentage = args.percentage
num_devices = 1

# Specify the cuda device to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(f"Batch size: {_config['batch_size']} || accumulation_steps: {accumulation_steps}")

# Set the bit-width for quantization
_config["quantization_bitwidth"] = 4

# Set the PyTorch Lightning seed
pl.seed_everything(_config["seed"])

# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "10"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "10"  # Set this to the number of CPUs you want to use

# # Set environment variables
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'

def print_frozen_layers(model):
    """
    Prints the names of the layers in a PyTorch model and whether they are frozen or not.
    
    Args:
        model (nn.Module): The PyTorch model to print the frozen status of.
    """

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Frozen: {not param.requires_grad}")


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

    # Optional: Print which layers are frozen and unfrozen for verification
    print_frozen_layers(model)


if __name__ == "__main__":
    # ==========================================
    # ========= Create the datamodules =========
    # ==========================================
    if "meter" in _config["model"]:
        # full_dm = MTDataModuleMeter(_config, dist=False)

        # val_dm = SmallMTDataModuleMETER(_config, dist=False, start_idx=3000, num_samples=50)
        # val_dm.setup("test", is_random=True)
        # val_dataloader = val_dm.test_dataloader()
        
        # fine_tune_dm = SmallMTDataModuleMETER(_config, dist=False, start_idx=10, num_samples=50)
        fine_tune_dm = SmallMTDataModuleMETER(_config, dist=False, percentage=fine_tune_percentage)
        fine_tune_dm.setup("test", is_random=True)
        fine_tune_dataloader = fine_tune_dm.test_dataloader()

    elif "vilt" in _config["model"]:
        # full_dm = MTDataModuleVILT(_config, dist=False)

        # val_dm = SmallMTDataModuleVILT(_config, dist=False, percentage=validataion_percentage)
        # val_dm.setup("test", is_random=True)
        # test_dataloval_dataloaderader = val_dm.test_dataloader()

        fine_tune_dm = SmallMTDataModuleVILT(_config, dist=False, percentage=fine_tune_percentage)
        fine_tune_dm.setup("test", is_random=True)
        fine_tune_dataloader = fine_tune_dm.test_dataloader()

    else:
        raise ValueError("Model not supported: ", _config["model"])

    # ==========================================
    # ========= Create the model ===============
    # ==========================================
    if _config["model"] == "vilt":
        model = ViLTransformerSS(_config)
        print("Initialized ViLT model")

    elif _config["model"] == "meter":
        model = METERTransformerSS(_config)
        print("Initialized METER model")

    else:
        raise ValueError("Model not supported: ", _config["model"])
    

    # Initialize the trainer
    trainer = init_trainer(_config, "gpu", [args.gpu], max_epochs, accumulation_steps=accumulation_steps)
    
    # Define the modules to quantize
    # modules_to_quantize = {'layer_names': ["transformer.blocks.0.mlp.fc1",
    #                                         "transformer.blocks.0.mlp.fc2",]
    #                     }
    modules_to_quantize = {'layer_names': ["text_transformer.encoder.layer.3.intermediate.dense",
                                            "text_transformer.encoder.layer.3.output.dense",]
                        }
    
    # Store the initial weights before training
    fc2_weight = get_module_by_path(model, modules_to_quantize['layer_names'][-1]).weight.clone()
    
    # Quantizer for int8 dynamic per token activations +
    # int4 grouped per channel weights, only for linear layers
    qat_quantizer = Int8DynActInt4WeightQATQuantizer()
    model = qat_quantizer.prepare(model, **modules_to_quantize)
    freeze_except_layers(model, modules_to_quantize['layer_names'])
    
    # Train the model with the quantization-aware training (QAT) quantizer
    trainer.fit(model, train_dataloaders=fine_tune_dataloader)

    # Store the weights after training before quantization
    fc2_weight_after_qat = get_module_by_path(model, modules_to_quantize['layer_names'][-1]).weight.clone()

    # Quantize the model
    model_quant = quantize_modules(model, modules_to_quantize['layer_names'], _config["quantization_bitwidth"])
    # model = qat_quantizer.convert(model)

    # Store the weights after quantization
    fc2_weight_after_dyn_quant = get_module_by_path(model_quant, modules_to_quantize['layer_names'][-1]).weight().int_repr().clone()

    # Print the tensor information
    print("============================================================")
    print(f"Min, Max and Mean of the ORIGINAL WEIGHTS: \n{torch.min(fc2_weight)}, {torch.max(fc2_weight)}, mean: {torch.mean(fc2_weight)}")
    print(f"Min, Max and Mean of the QAT WEIGHTS: \n{torch.min(fc2_weight_after_qat)}, {torch.max(fc2_weight_after_qat)}, mean: {torch.mean(fc2_weight_after_qat)}")
    print(f"Min, Max and Mean of the QAT WEIGHTS: \n{torch.min(fc2_weight_after_dyn_quant)}, {torch.max(fc2_weight_after_dyn_quant)}")
    print("============================================================")

    # ==========================================
    _config = configs.meter_config_nlvr2
    _config["batch_size"] = 6
    _config["per_gpu_batchsize"] = 6
    # ==========================================
    # ========= Create the datamodules =========
    # ==========================================
    if "meter" in _config["model"]:
        full_dm_ood = MTDataModuleMeter(_config, dist=False)

    elif "vilt" in _config["model"]:
        full_dm_ood = MTDataModuleVILT(_config, dist=False)

    else:
        raise ValueError("Model not supported: ", _config["model"])
    
    trainer = init_trainer(_config, "cpu", 1, max_epochs, accumulation_steps=accumulation_steps)
    trainer.test(model, full_dm_ood)

    raise ValueError("Stop here")