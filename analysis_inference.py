import os
import copy
import torch
import pytorch_lightning as pl
from vilt.modules import ViLTransformerSS
from meter.modules import METERTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule as MTDataModuleVILT
from meter.datamodules.multitask_datamodule import MTDataModule as MTDataModuleMeter
import time
import numpy as np

import configs as config
from sensitivity_utils import print_size_of_model, SmallMTDataModuleVILT, SmallMTDataModuleMETER, get_quantization_config

# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

pl.seed_everything(0)

def get_inference_time(model, input_batch, num_iters):
    start_time = time.time()

    for i in range(num_iters):
        model(input_batch)
    
    stop_time = time.time()

    avg_time = (stop_time - start_time)/num_iters

    print(f"Average time taken for a {num_iters} passes: {avg_time}")

    return avg_time

# Set the configuration
_config = config.vilt_config_nlvr2
_config["model_"] = "vilt"

# ==========================================
# ========= Create full datamodule =========
# ==========================================
if "meter" in _config["model_"]:
    full_dm = MTDataModuleMeter(_config, dist=False)
    full_dm.setup("test")
    full_dataloader = full_dm.test_dataloader()

    test_dm = SmallMTDataModuleMETER(_config, dist=False, num_samples=1)
    test_dm.setup("test")
    test_dataloader = test_dm.test_dataloader()

    calibrate_dm = SmallMTDataModuleMETER(_config, dist=False, num_samples=600)
else:
    full_dm = MTDataModuleVILT(_config, dist=False)
    # full_dm.setup("test")
    # full_dataloader = full_dm.test_dataloader()

    test_dm = SmallMTDataModuleVILT(_config, dist=False, num_samples=1)
    test_dm.setup("test")
    test_dataloader = test_dm.test_dataloader()

    calibrate_dm = SmallMTDataModuleVILT(_config, dist=False, num_samples=600)


print(f"Length of the test dataloader: {len(test_dataloader.dataset)}")
# print(f"Length of the full dataloader: {len(full_dataloader.dataset)}")

if _config["model_"] == "vilt":
    model = ViLTransformerSS(_config)
    print("Initialized ViLT model")

elif _config["model_"] == "meter":
    model = METERTransformerSS(_config)
    print("Initialized METER model")

else:
    raise ValueError("Model not supported")


num_bit = 8
quant_config, embedding_config = get_quantization_config(num_bit)
quant_dict = {torch.nn.Linear: quant_config, torch.nn.Dropout: quant_config, torch.nn.GELU: quant_config, torch.nn.Embedding: embedding_config, torch.nn.LayerNorm: embedding_config}

model_dynamic = copy.deepcopy(model)
torch.quantization.quantize_dynamic(
        model_dynamic, quant_dict, inplace=True
    )

print("Size after quantization:")
print_size_of_model(model_dynamic)
print(model_dynamic)

input_batch = next(iter(test_dataloader))
num_iters = 1000


get_inference_time(model, input_batch, num_iters)
get_inference_time(model_dynamic, input_batch, num_iters)

# print(f"Average time taken for a {num_iters} passes for fully quantized model: {np.mean(time.time() - start_time)/num_iters}")

