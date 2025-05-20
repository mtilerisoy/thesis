import os
# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "9"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "9"  # Set this to the number of CPUs you want to use

import copy
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule

import time
import torch
import quantization_utils
from torch.utils.data import Subset


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # ========== Initialize the datamodule for pl.Trainer ==========
    dm = MTDataModule(_config, dist=False)
    # dm = SmallMTDataModuleVILT(_config, dist=False, num_samples=10, start_idx=0)

    # =============== Initialize Full Precision Model ==============
    model = ViLTransformerSS(_config)

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

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    # =============== Testing Quantized Model ===============
    trainer_cpu = pl.Trainer(
        accelerator="cpu",
        devices=1,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        # strategy="ddp",
        benchmark=True,
        deterministic=False,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        # accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    print("========== Testing Quantized Model - Mixed Precision ==========")
    # bit8_linear, _ = quantization_utils.get_quantization_config(8)
    # bit4_linear, _ = quantization_utils.get_quantization_config(4)
    bit2_linear, bit2_embedding = quantization_utils.get_quantization_config(2)

    torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Embedding: bit2_embedding, "nlvr2_classifier": bit2_linear, "pooler": bit2_linear, "transformer": bit2_linear},
    dtype=torch.quint8, inplace=True
)

    start_time_int4 = time.time()
    trainer_cpu.test(model, datamodule=dm)
    end_time_int4 = time.time()
    print("Time taken for DYNAMIC INT4 Inference: ", end_time_int4 - start_time_int4)

    # # ========== Testing Quantized Model ==========
    # print("========== Testing Quantized Model - INT8 ==========")
    # model_int8 = dq.quantize_model_dynamic(model, 8)

    # start_time_int8 = time.time()
    # trainer_cpu.test(model_int8, datamodule=dm)
    # end_time_int8 = time.time()
    # print("Time taken for DYNAMIC INT8 Inference: ", end_time_int8 - start_time_int8)
    




    # # ========== Testing Quantized Model ==========
    # print("========== INT8 DEFAULT !!!! QUANT ==========")
    # _model = copy.deepcopy(model)
    # torch.quantization.quantize_dynamic(
    #         _model, {torch.nn.Embedding}, dtype=torch.quint8, inplace=True
    #     )
    # torch.quantization.quantize_dynamic(
    #         _model, {torch.nn.Linear, torch.nn.LayerNorm}, dtype=torch.qint8, inplace=True
    #     )

    # start_time_int8 = time.time()
    # trainer_cpu.test(_model, datamodule=dm)
    # end_time_int8 = time.time()
    # print("Time taken for DYNAMIC INT8 Inference: ", end_time_int8 - start_time_int8)