# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
from quantization_utils import get_qat_config, get_quantization_config, print_size_of_model
from quantization_utils import  SmallMTDataModuleMETER, SmallMTDataModuleVILT

# ==========================================
# ========= Set the configuration ==========
# ==========================================
# Set the general configurations
_config = configs.meter_config_nlvr2_original
_config["batch_size"] = 24
_config["per_gpu_batchsize"] = 24

# Set the bit-width for quantization
_config["quantization_bitwidth"] = 4

# Set the PyTorch Lightning seed
pl.seed_everything(_config["seed"])

# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "8"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "8"  # Set this to the number of CPUs you want to use

# Set environment variables
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

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

from torch.ao.quantization.qconfig import default_dynamic_qat_qconfig
from torch.ao.quantization.quantize import convert, propagate_qconfig_, prepare
from torch.ao.quantization.quantization_mappings import get_default_qat_module_mappings
import copy

def prepare_qat(model, qconfig_dict, mapping=None, inplace=False):
    r"""
    Prepares a copy of the model for quantization calibration or
    quantization-aware training and converts it to quantized version.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    Args:
        model: input model to be modified in-place
        mapping: dictionary that maps float modules to quantized modules to be
                 replaced.
        inplace: carry out model transformations in-place, the original module
                 is mutated
    """
    torch._C._log_api_usage_once("quantization_api.quantize.prepare_qat")
    assert model.training, "prepare_qat only works on models in training mode"
    if mapping is None:
        mapping = get_default_qat_module_mappings()

    if not inplace:
        model = copy.deepcopy(model)

    propagate_qconfig_(model, qconfig_dict=qconfig_dict)
    convert(model, mapping=mapping, inplace=True, remove_qconfig=False)
    prepare(model, observer_non_leaf_module_list=set(mapping.values()), inplace=True)
    return model

if __name__ == "__main__":

    # ==========================================
    # ========= Create the datamodules =========
    # ==========================================
    if "meter" in _config["model"]:
        full_dm = MTDataModuleMeter(_config, dist=False)
        
        # test_dm = SmallMTDataModuleMETER(_config, dist=False, percentage=0.01)
        # test_dm.setup("test", is_random=True)
        # test_dataloader = test_dm.test_dataloader()
        
        fine_tune_dm = SmallMTDataModuleMETER(_config, dist=False, percentage=0.25)
        fine_tune_dm.setup("test", is_random=True)
        fine_tune_dataloader = fine_tune_dm.test_dataloader()

    elif "vilt" in _config["model"]:
        full_dm = MTDataModuleVILT(_config, dist=False)

        # test_dm = SmallMTDataModuleVILT(_config, dist=False, num_samples=50)
        # test_dm.setup("test", is_random=True)
        # test_dataloader = test_dm.test_dataloader()

        fine_tune_dm = SmallMTDataModuleVILT(_config, dist=False, num_samples=50)
        fine_tune_dm.setup("test", is_random=True)
        fine_tune_dataloader = fine_tune_dm.test_dataloader()

    else:
        raise ValueError("Model not supported: ", _config["model"])

    print(f"Batch size: {_config['batch_size']}")
    print(f"Lenght of the dataloader: {len(fine_tune_dataloader)}")



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



    # ==========================================
    # ========= Create the trainer =============
    # ==========================================

    def init_trainer(accelerator, num_devices, max_epochs, max_steps):
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

        grad_steps = max(_config["batch_size"] // (
            _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
        ), 1)

        max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

        trainer = pl.Trainer(
                accelerator=accelerator,
                devices=num_devices,
                num_nodes=_config["num_nodes"],
                precision=_config["precision"],
                # strategy="ddp",
                benchmark=True,
                deterministic=False,
                max_epochs=max_epochs,
                max_steps=max_steps,
                callbacks=callbacks,
                logger=logger,
                accumulate_grad_batches=grad_steps,
                log_every_n_steps=10,
                fast_dev_run=_config["fast_dev_run"],
                val_check_interval=_config["val_check_interval"],
            )

        return trainer
    # ==========================================
    # ========= Evaluate the model =============
    # ==========================================
    print("Evaluating the full-precision model")
    # print("Model size before quantization:")
    # print_size_of_model(model)
    trainer = init_trainer("gpu", num_devices=2, max_epochs=50, max_steps=-1)
    trainer.test(model, full_dm)
    print("Finished evaluating the full-precision model")

    # ==========================================
    # ===== Define the modules to quantize =====
    # ==========================================
    modules_to_quantize = [
        "text_transformer.encoder.layer.2.output.dense",
        "text_transformer.encoder.layer.2.intermediate.dense",
        "text_transformer.encoder.layer.3.output.dense",
        "text_transformer.encoder.layer.3.intermediate.dense"
    ]

    

    # ==========================================
    # ========= Prepare for Dynamic ============
    # ==========================================

    dynamic_ptq_config, _ = get_quantization_config(_config["quantization_bitwidth"])

    q_config_dict = dict()

    for layer in modules_to_quantize:
        q_config_dict[layer] = dynamic_ptq_config
    
    model_dynamic = deepcopy(model)
    torch.quantization.quantize_dynamic(
        model_dynamic, q_config_dict, inplace=True
    )

    # ==========================================
    # ========= Evaluate the model =============
    # ==========================================
    print("Evaluating the dynamic-quantized model")
    print(f"Quantized layers: {modules_to_quantize}")
    # print("Model size after quantization:")
    # print_size_of_model(model)
    trainer = init_trainer("cpu", num_devices=1, max_epochs=50, max_steps=-1)
    trainer.test(model_dynamic, full_dm)



    # ==========================================
    # ========= Prepare for QAT ================
    # ==========================================
    
    qat_q_config = get_qat_config(_config["quantization_bitwidth"])

    # Create the quantization configuration dictionary
    qconfig_dict = dict()
    for layer in modules_to_quantize:
        qconfig_dict[layer] = qat_q_config

    # Prepare the model for quantization-aware training
    model_qat = prepare_qat(model, inplace=False, qconfig_dict=qconfig_dict)

    # Freeze all layers except for the quantized layers
    freeze_except_layers(model_qat, modules_to_quantize)


    # ==========================================
    # ========= Train the model ================
    # ==========================================
    trainer = init_trainer("gpu", num_devices=2, max_epochs=50, max_steps=-1)
    trainer.fit(model_qat, fine_tune_dataloader)


    # ==========================================
    # ========= Evaluate the model =============
    # ==========================================
    trainer.test(model_qat, full_dm)