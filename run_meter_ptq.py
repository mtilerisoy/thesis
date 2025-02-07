import os
# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "8"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "8"  # Set this to the number of CPUs you want to use

import copy
import pytorch_lightning as pl
import os
os.environ["NCCL_DEBUG"] = "INFO"

from meter.config import ex
from meter.modules import METERTransformerSS
from meter.datamodules.multitask_datamodule import MTDataModule

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

import time
import torch
from torch.utils.data import Subset
import dynamic_quantization as dq
from torch.utils.data import Subset

class SmallMTDataModuleVILT(MTDataModule):
    def __init__(self, _config, dist=False, num_samples=10, start_idx=100):
        super().__init__(_config, dist)
        self.num_samples = num_samples
        self.start_idx = start_idx

    def setup(self, stage):
        super().setup(stage)
        
        # Limit the number of samples in the datasets
        self.train_dataset = Subset(self.train_dataset, range(self.start_idx, self.start_idx+self.num_samples))
        self.val_dataset = Subset(self.val_dataset, range(self.start_idx, self.start_idx+self.num_samples))
        self.test_dataset = Subset(self.test_dataset, range(self.start_idx, self.start_idx+self.num_samples))

def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=False)
    # dm = SmallMTDataModuleVILT(_config, dist=False, num_samples=10, start_idx=100)
    
    model = METERTransformerSS(_config)

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

    # trainer = pl.Trainer(
    #     accelerator=_config["accelerator"],
    #     devices=_config["num_gpus"],
    #     num_nodes=_config["num_nodes"],
    #     precision=_config["precision"],
    #     strategy="ddp",
    #     benchmark=True,
    #     deterministic=True,
    #     max_epochs=_config["max_epoch"] if max_steps is None else 1000,
    #     max_steps=max_steps,
    #     callbacks=callbacks,
    #     logger=logger,
    #     # prepare_data_per_node=False,
    #     # replace_sampler_ddp=False,
    #     accumulate_grad_batches=grad_steps,
    #     log_every_n_steps=10,
    #     # flush_logs_every_n_steps=10,
    #     # resume_from_checkpoint=_config["resume_from"],
    #     # weights_summary="top",
    #     fast_dev_run=_config["fast_dev_run"],
    #     val_check_interval=_config["val_check_interval"],
    # )

    # if not _config["test_only"]:
    #     print(f"Training model")
    #     trainer.fit(model, datamodule=dm)
    # else:
    #     print(f"Testing model")
    #     trainer.test(model, datamodule=dm)

    # =============== Testing QUANTIZED Model ===============
    trainer_cpu = pl.Trainer(
        accelerator="cpu",
        devices=1,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        strategy="ddp",
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

    # # =============== Testing HALF Model ===============
    # trainer_half = pl.Trainer(
    #     accelerator="cpu",
    #     devices=1,
    #     num_nodes=_config["num_nodes"],
    #     precision=16,
    #     strategy="ddp",
    #     benchmark=True,
    #     deterministic=False,
    #     max_epochs=_config["max_epoch"] if max_steps is None else 1000,
    #     max_steps=max_steps,
    #     callbacks=callbacks,
    #     logger=logger,
    #     # accumulate_grad_batches=grad_steps,
    #     log_every_n_steps=10,
    #     fast_dev_run=_config["fast_dev_run"],
    #     val_check_interval=_config["val_check_interval"],
    # )


    # # ========== Testing Full Precision Model ==========
    # print("========== Testing Full Precision Model - FP32 ==========")
    # start_time_fp32 = time.time()
    # trainer_cpu.test(model, datamodule=dm)
    # end_time_fp32 = time.time()
    # print("Time taken for FP32 Inference: ", end_time_fp32 - start_time_fp32)


    # # ========== Testing Half Precision Model ==========
    # print("========== Testing Half Precision Model - FP16 ==========")
    # model_half = copy.deepcopy(model)
    # model_half = model_half.half()

    # start_time_fp16 = time.time()
    # trainer_half.test(model_half, datamodule=dm)
    # end_time_fp16 = time.time()


    # ========== Testing Quantized Model ==========
    print("========== Testing Quantized Model - INT8 ==========")
    model_int8 = dq.quantize_model_dynamic(model, 8)

    start_time_int8 = time.time()
    trainer_cpu.test(model_int8, datamodule=dm)
    end_time_int8 = time.time()
    print("Time taken for STATIC INT8 Inference: ", end_time_int8 - start_time_int8)


    # ========== Testing Quantized Model ==========
    print("========== Testing Quantized Model - INT4 ==========")
    model_int4 = dq.quantize_model_dynamic(model, 4)

    start_time_int4 = time.time()
    trainer_cpu.test(model_int4, datamodule=dm)
    end_time_int4 = time.time()
    print("Time taken for DYNAMIC INT4 Inference: ", end_time_int4 - start_time_int4)


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
    # print("Time taken for DYNAMIC INT8 DEfAULT Inference: ", end_time_int8 - start_time_int8)

    
    # print("Time taken for FP16 Inference: ", end_time_fp16 - start_time_fp16)
    
    