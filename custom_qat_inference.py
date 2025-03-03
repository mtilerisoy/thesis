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
parser.add_argument("-d", "--dataset",          type=str,   default="nlvr2_original", help="Dataset to train the model on")
parser.add_argument("-m", "--model",            type=str,   default="vilt",     help="Model to use for training")
args = parser.parse_args()

print("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐")
print("│                                                                                                     │")
print(f"│  Running with: dataset={args.dataset}, model={args.model}                                          │")
print("│                                                                                                     │")
print("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘")


# ==========================================
# ========= Set the configuration ==========
# ==========================================
# Set the general configurations
if args.model == "vilt":
    _config = configs.vilt_config_nlvr2 if args.dataset == "nlvr2_ood" else configs.vilt_config_nlvr2_original if args.dataset == "nlvr2_original" else ModuleNotFoundError("Dataset not supported")
elif args.model == "meter":
    _config = configs.meter_config_nlvr2 if args.dataset == "nlvr2_ood" else configs.meter_config_nlvr2_original if args.dataset == "nlvr2_original" else ModuleNotFoundError("Dataset not supported")
else:
    raise ValueError("Model not supported: ", args.model)

_config["batch_size"] = 64
_config["per_gpu_batchsize"] = 64
num_gpus = [1]

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
    # Load the data
    if "meter" in _config["model"]:
        full_dm = MTDataModuleMeter(_config, dist=False)


    elif "vilt" in _config["model"]:
        full_dm = MTDataModuleVILT(_config, dist=False)

    else:
        raise ValueError("Model not supported: ", _config["model"])

    # Load the model
    if _config["model"] == "vilt":
        model = ViLTransformerSS(_config)
        print("Initialized ViLT model")

    elif _config["model"] == "meter":
        model = METERTransformerSS(_config)
        print("Initialized METER model")

    else:
        raise ValueError("Model not supported: ", _config["model"])

    
    # Process each layer
    for i in range(12):

        # Copy the model to reset the quantization
        model_quant = deepcopy(model)
    

        # Define the modules to quantize
        if args.model == "vilt":
            modules_to_quantize = {'layer_names': [ # MLP Block Layers
                                                    # f"transformer.blocks.{i}.mlp.fc1",
                                                    # f"transformer.blocks.{i}.mlp.fc2",
                                                    # # Attention block layers
                                                    # f"transformer.blocks.{i}.attn.qkv",
                                                    # f"transformer.blocks.{i}.attn.proj",
                                                    # f"transformer.blocks"
                                                    # f"nlvr2_classifier"
                                                    f"transformer.blocks.{i}.attn"
                                                ]
                            }
        elif args.model == "meter":
            modules_to_quantize = {'layer_names': [ # MLP Block Layers
                                                    # f"text_transformer.encoder.layer.{i}.intermediate.dense",
                                                    # f"text_transformer.encoder.layer.{i}.output.dense",
                                                    # # Attention block layers
                                                    # f"text_transformer.encoder.layer.{i}.attention.self.query",
                                                    # f"text_transformer.encoder.layer.{i}.attention.self.key",
                                                    # f"text_transformer.encoder.layer.{i}.attention.self.value",
                                                    # f"text_transformer.encoder.layer.{i}.attention.output.dense",
                                                    f"text_transformer.encoder.layer.{i}.attention"
                                                ]
                            }
        else:
            raise ValueError("Model not supported: ", args.model)
            
        # Store the initial weights before training
        # fc2_weight = get_module_by_path(model_quant, modules_to_quantize['layer_names'][-1]+".3").weight.clone()

        # Quantizer for int8 dynamic per token activations +
        # int4 grouped per channel weights, only for linear layers
        qat_quantizer = Int8DynActInt4WeightQATQuantizer()
        model_quant = qat_quantizer.prepare(model_quant, **modules_to_quantize)

        # Quantize the model
        model_quant = qat_quantizer.convert(model_quant)

        # Store the weights after quantization
        # fc2_weight_after_dyn_quant = get_module_by_path(model_quant, modules_to_quantize['layer_names'][-1]+".3").weight.clone()

        # Print the tensor information
        # print("============================================================")
        # print(f"Min, Max and Mean of the ORIGINAL WEIGHTS: \n{torch.min(fc2_weight)}, {torch.max(fc2_weight)}, mean: {torch.mean(fc2_weight)}")
        # print(f"Min, Max and Mean of the QAT WEIGHTS: \n{torch.min(fc2_weight_after_dyn_quant)}, {torch.max(fc2_weight_after_dyn_quant)}")
        # print("============================================================")

        # Initialize the trainer
        trainer = init_trainer(_config, "gpu", num_gpus, max_epochs=1, accumulation_steps=1)

        trainer.test(model_quant, full_dm)

        