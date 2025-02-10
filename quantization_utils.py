import random
random.seed(42)
from torch.utils.data import Subset
from vilt.datamodules.multitask_datamodule import MTDataModule as MTDataModuleVILT
from meter.datamodules.multitask_datamodule import MTDataModule as MTDataModuleMeter

class SmallMTDataModuleVILT(MTDataModuleVILT):
    def __init__(self, _config, dist=False, num_samples=5, start_idx=100):
        super().__init__(_config, dist)
        self.num_samples = num_samples
        self.start_idx = start_idx

    def setup(self, stage, is_random=False):
        super().setup(stage)
        
        # Limit the number of samples in the datasets
        if is_random:
            self.train_dataset = self._get_random_subset(self.train_dataset, self.num_samples)
            self.val_dataset = self._get_random_subset(self.val_dataset, self.num_samples)
            self.test_dataset = self._get_random_subset(self.test_dataset, self.num_samples)
        else:    
            self.train_dataset = Subset(self.train_dataset, range(self.start_idx, self.start_idx+self.num_samples))
            self.val_dataset = Subset(self.val_dataset, range(self.start_idx, self.start_idx+self.num_samples))
            self.test_dataset = Subset(self.test_dataset, range(self.start_idx, self.start_idx+self.num_samples))
        
    def _get_random_subset(self, dataset, num_samples, percentage):
        if percentage:
            indices = random.sample(range(len(dataset)), int(len(dataset)*percentage))
        else:
            indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, indices)

class SmallMTDataModuleMETER(MTDataModuleMeter):
    def __init__(self, _config, dist=False, num_samples=10, start_idx=100, percentage=None):
        super().__init__(_config, dist)
        self.num_samples = num_samples
        self.start_idx = start_idx
        self.percentage = percentage

    def setup(self, stage, is_random=False):
        super().setup(stage)
        
        # Limit the number of samples in the datasets
        if is_random:
            self.train_dataset = self._get_random_subset(self.train_dataset, self.num_samples, self.percentage)
            self.val_dataset = self._get_random_subset(self.val_dataset, self.num_samples, self.percentage)
            self.test_dataset = self._get_random_subset(self.test_dataset, self.num_samples, self.percentage)
        else:    
            self.train_dataset = Subset(self.train_dataset, range(self.start_idx, self.start_idx+self.num_samples))
            self.val_dataset = Subset(self.val_dataset, range(self.start_idx, self.start_idx+self.num_samples))
            self.test_dataset = Subset(self.test_dataset, range(self.start_idx, self.start_idx+self.num_samples))
        
    
    def _get_random_subset(self, dataset, num_samples, percentage):
        if percentage:
            indices = random.sample(range(len(dataset)), int(len(dataset)*percentage))
        else:
            indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, indices)

def print_size_of_model(model):
    """
    Function to print the size of the model.

    Args:
        model (torch.nn.Module): The model to get the size
    
    Returns:
        None
    """
    torch.save(model.state_dict(), "temp.p")
    print('Size of the model (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

# def generate_command(config):
#     command = ["python", "run_ptq.py", "with"]
#     for key, value in config.items():
#         if isinstance(value, dict):
#             for sub_key, sub_value in value.items():
#                 command.append(f"{key}.{sub_key}={sub_value}")
#         else:
#             command.append(f"{key}={value}")
#     return command

import os
import pytorch_lightning as pl

def init_trainer(_config, accelerator, num_devices=1):
    """
    Function to initialize the trainer for CPU inference. Usually used for quantization.

    Args:
        _config (dict): Configuration dictionary from sacred experiments.
    """
    # ========== Initialize the trainer ==========
    pl.seed_everything(_config["seed"])

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

    trainer = pl.Trainer(
        accelerator=accelerator,
        # devices=_config["num_gpus"],
        devices=num_devices,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        # strategy=strategy,
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        # replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        # flush_logs_every_n_steps=10,
        # resume_from_checkpoint=_config["resume_from"],
        # weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    return trainer


import torch
from torch.quantization import PlaceholderObserver, MinMaxObserver, QConfig, PerChannelMinMaxObserver

def get_quantization_config(precision):

    if precision == 8:
        quantization_config = QConfig(
            activation=PlaceholderObserver.with_args(
                                                dtype=torch.quint8,
                                                quant_min=0,
                                                quant_max=255,
                                                is_dynamic=True,
                                            ),
            weight=MinMaxObserver.with_args(
                                            dtype=torch.qint8,
                                            qscheme=torch.per_tensor_symmetric,
                                            quant_min=-128,
                                            quant_max=127,
                                        ),
        )

        embedding_layer_qconfig = QConfig(
            activation=PlaceholderObserver,
            weight=PerChannelMinMaxObserver.with_args(
                                                dtype=torch.quint8,
                                                qscheme=torch.per_channel_affine_float_qparams,
                                                ch_axis=0,
                                                quant_min=0,
                                                quant_max=255,
                                            ),
        )

    elif precision == 4:
        quantization_config = QConfig(
            activation=PlaceholderObserver.with_args(
                                                dtype=torch.quint8,
                                                quant_min=0,
                                                quant_max=15,
                                                is_dynamic=True,
                                            ),
            weight=MinMaxObserver.with_args(
                                            dtype=torch.qint8,
                                            qscheme=torch.per_tensor_symmetric,
                                            quant_min=-8,
                                            quant_max=7,
                                        ),
        )

        embedding_layer_qconfig = QConfig(
            activation=PlaceholderObserver,
            weight=PerChannelMinMaxObserver.with_args(
                                                dtype=torch.quint8,
                                                qscheme=torch.per_channel_affine_float_qparams,
                                                ch_axis=0,
                                                quant_min=0,
                                                quant_max=15,
                                            ),
        )
    
    elif precision == 2:
        quantization_config = QConfig(
            activation=PlaceholderObserver.with_args(
                                                dtype=torch.quint8,
                                                quant_min=0,
                                                quant_max=3,
                                                is_dynamic=True,
                                            ),
            weight=MinMaxObserver.with_args(
                                            dtype=torch.qint8,
                                            qscheme=torch.per_tensor_symmetric,
                                            quant_min=-2,
                                            quant_max=1,
                                        ),
        )

        embedding_layer_qconfig = QConfig(
            activation=PlaceholderObserver,
            weight=PerChannelMinMaxObserver.with_args(
                                                dtype=torch.quint8,
                                                qscheme=torch.per_channel_affine_float_qparams,
                                                ch_axis=0,
                                                quant_min=0,
                                                quant_max=3,
                                            ),
        )
    
    elif precision == 1:
        quantization_config = QConfig(
            activation=PlaceholderObserver.with_args(
                                                dtype=torch.quint8,
                                                quant_min=0,
                                                quant_max=1,
                                                is_dynamic=True,
                                            ),
            weight=MinMaxObserver.with_args(
                                            dtype=torch.qint8,
                                            qscheme=torch.per_tensor_symmetric,
                                            quant_min=-1,
                                            quant_max=0,
                                        ),
        )

        embedding_layer_qconfig = QConfig(
            activation=PlaceholderObserver,
            weight=PerChannelMinMaxObserver.with_args(
                                                dtype=torch.quint8,
                                                qscheme=torch.per_channel_affine_float_qparams,
                                                ch_axis=0,
                                                quant_min=0,
                                                quant_max=1,
                                            ),
        )

    else:
        raise ValueError("Precision not supported")

    return quantization_config, embedding_layer_qconfig


from torch.quantization import FakeQuantize, MovingAverageMinMaxObserver
def get_qat_config(precision):

    if precision == 8:
        quantization_config = QConfig(
            activation=PlaceholderObserver.with_args(
                                                dtype=torch.quint8,
                                                quant_min=0,
                                                quant_max=255,
                                                is_dynamic=True,
                                            ),
            weight=FakeQuantize.with_args(
                                        observer=MovingAverageMinMaxObserver,
                                        quant_min=-128,
                                        quant_max=127,
                                        dtype=torch.qint8,
                                        qscheme=torch.per_tensor_symmetric,
                                        reduce_range=False,
                                    ),
        )

    elif precision == 4:
        quantization_config = QConfig(
            activation=PlaceholderObserver.with_args(
                                                dtype=torch.quint8,
                                                quant_min=0,
                                                quant_max=15,
                                                is_dynamic=True,
                                            ),
            weight=FakeQuantize.with_args(
                                        observer=MovingAverageMinMaxObserver,
                                        quant_min=-8,
                                        quant_max=7,
                                        dtype=torch.qint8,
                                        qscheme=torch.per_tensor_symmetric,
                                        reduce_range=False,
                                    ),
        )
    
    elif precision == 2:
        quantization_config = QConfig(
            activation=PlaceholderObserver.with_args(
                                                dtype=torch.quint8,
                                                quant_min=0,
                                                quant_max=3,
                                                is_dynamic=True,
                                            ),
            weight=FakeQuantize.with_args(
                                        observer=MovingAverageMinMaxObserver,
                                        quant_min=-2,
                                        quant_max=1,
                                        dtype=torch.qint8,
                                        qscheme=torch.per_tensor_symmetric,
                                        reduce_range=False,
                                    ),
        )

    else:
        raise ValueError("Precision not supported")

    return quantization_config

