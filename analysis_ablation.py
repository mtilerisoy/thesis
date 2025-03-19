import os
# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "8"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "8"  # Set this to the number of CPUs you want to use

from quantization_utils import init_trainer, get_quantization_config, print_size_of_model
import torch
from datetime import datetime
from copy import deepcopy
import pytorch_lightning as pl

    
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run METER PTQ")
    parser.add_argument('--model', type=str, default="meter", help='Model to use: vilt, meter')
    parser.add_argument('--task', type=str, default="nlvr2_ood", help='Task to run: nlvr2, vqa')
    parser.add_argument('--precision', type=int, default=2, help='Precision to use: 4, 8')
    parser.add_argument('--layer2quantize', type=str, default="text_transformer.encoder", help='Layer to quantize: transformer.patch_embed')
    args = parser.parse_args()

    print("┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                                                                                                     │")
    print(f"│  Running with: model={args.model}, task={args.task}, precision={args.precision}                                   \n│")                              
    print(f"│                                                                                                   \n│")
    print(f"│  quantized module is: {args.layer2quantize}                                                              \n│")
    print("│                                                                                                     │")
    print("└─────────────────────────────────────────────────────────────────────────────────────────────────────┘")

    if "vilt" in args.model:
        from vilt.datamodules.multitask_datamodule import MTDataModule
        from vilt.modules import ViLTransformerSS as MODEL
        from quantization_utils import SmallMTDataModuleMETER as SmallMTDataModule

        if "nlvr" in args.task:
            from configs import vilt_config_nlvr2 as CONFIG
        elif "vqa" in args.task:
            from configs import vilt_config_vqav2 as CONFIG
        else:
            raise ValueError("Task not supported")

    elif "meter" in args.model:
        from meter.datamodules.multitask_datamodule import MTDataModule
        from meter.modules import METERTransformerSS as MODEL
        from quantization_utils import SmallMTDataModuleMETER as SmallMTDataModule

        if "nlvr2_original" in args.task:
            from configs import meter_config_nlvr2_original as CONFIG
        elif "nlvr2_ood" in args.task:
            from configs import meter_config_nlvr2 as CONFIG
        elif "vqa" in args.task:
            from configs import meter_config_vqav2 as CONFIG
        else:
            raise ValueError("Task not supported")
    else:
        raise ValueError("Model not supported")
    

    # Initialize the datamodules
    dm = MTDataModule(CONFIG, dist=False)
    # test_dm = SmallMTDataModule(CONFIG, dist=False, num_samples=250, start_idx=100)
    print("Datamodules initialized")

    # Initialize the model
    model = MODEL(CONFIG)
    print("Model initialized")

    # Initialize the trainer
    # ========== Initialize the trainer for full precision ==========
    CONFIG["exp_name"] = f"2-bit-quant-{args.model}"
    CONFIG["log_dir"] = "experiments/quantization"
    exp_name = f'{CONFIG["exp_name"]}'
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    logger = pl.loggers.TensorBoardLogger(
        CONFIG["log_dir"],
        name=f'{exp_name}_{CONFIG["load_path"].split("/")[-1][:-5]}',
        default_hp_metric=False
    )

    num_gpus = (
        CONFIG["num_gpus"]
        if isinstance(CONFIG["num_gpus"], int)
        else len(CONFIG["num_gpus"])
    )

    grad_steps = CONFIG["batch_size"] // (
        CONFIG["per_gpu_batchsize"] * num_gpus * CONFIG["num_nodes"]
    )

    print("Gradient Accumulation Steps: ", grad_steps)

    # =============== Testing Quantized Model ===============
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        num_nodes=CONFIG["num_nodes"],
        precision=CONFIG["precision"],
        benchmark=True,
        deterministic=True,
        max_steps=1000,
        logger=logger,
        # callbacks=lr_callback,
        log_every_n_steps=1,
        accumulate_grad_batches=grad_steps,
        enable_checkpointing=False,
        fast_dev_run=CONFIG["fast_dev_run"],
        val_check_interval=CONFIG["val_check_interval"],
    )
    print("Trainer initialized")
    
    print(f"Initializing the quantizers using precision: {args.precision}")
    quantization_config, embedding_layer_qconfig = get_quantization_config(precision=args.precision)
    print("Quantization config initialized")

    # ========== Quantization ==========
    # Get the names of the modules
    names, _ = zip(*list(model.named_modules()))
    
    # Quantize the model
    layer_to_quantize = args.layer2quantize

    # Check if the layer to quantize is in the model
    assert layer_to_quantize in names, f"Layer {layer_to_quantize} not found in the model"

    if "embeddings" in layer_to_quantize:
        quantization_config = embedding_layer_qconfig

    model_dynamic = deepcopy(model)
    
    torch.quantization.quantize_dynamic(
        model_dynamic, {layer_to_quantize: quantization_config}, inplace=True
    )

    # Print the size of the model
    param_size = 0
    for param in model_dynamic.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model_dynamic.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


    # Accuracy Testing
    trainer.test(model_dynamic, datamodule=dm)

    print(f"Model Used: {args.model}")
    print(f"Task Used: {args.task}")
    print(f"Precision: {args.precision}")
    print(f"Quantized Block: {layer_to_quantize}")
    print('model size: {:.3f}MB'.format(size_all_mb))
    print(f"Completed at: {datetime.now().strftime('%Y%m%d_%H%M')}")
    print("===================================")

    # modules_to_quantize = [
    #     "text_transformer.encoder.layer.2.output.dense",
    #     "text_transformer.encoder.layer.2.intermediate.dense"
    #     # "text_transformer.encoder.layer.3.output.dense",
    #     # "text_transformer.encoder.layer.3.intermediate.dense"
    # ]

    # # Initialize the dictionary of the quantization configuration
    # q_config_dict = dict()

    # # Assign the quantization configuration to the layers
    # for layer in modules_to_quantize:
    #     q_config_dict[layer] = quantization_config

    # # Quantize the model dynamically
    # model_dynamic = deepcopy(model)
    # torch.quantization.quantize_dynamic(
    #     model_dynamic, q_config_dict, inplace=True
    # )

    # # Accuracy Testing
    # trainer.test(model_dynamic, datamodule=dm)

    # print(f"Model Used: {args.model}")
    # print(f"Task Used: {args.task}")
    # print_size_of_model(model_dynamic)
    # print(f"Precision: {args.precision}")
    # print(f"Quantized Module: {modules_to_quantize}")
    # print(f"Completed at: {datetime.now().strftime('%Y%m%d_%H%M')}")
    # print(f"Used quantization config: {quantization_config}")
    # print("===================================")

    # for i in range(12):
    #     # i += 6
    #     print(f"Block {i}")

    #     layer_to_quantize = "text_transformer.encoder.layer." + str(i)

    #     # Check if the layer to quantize is in the model
    #     assert layer_to_quantize in names, f"Layer {layer_to_quantize} not found in the model"

    #     if "embeddings" in layer_to_quantize:
    #         quantization_config = embedding_layer_qconfig

    #     model_dynamic = deepcopy(model)
        
    #     torch.quantization.quantize_dynamic(
    #         model_dynamic, {layer_to_quantize: quantization_config}, inplace=True
    #     )

    #     # Accuracy Testing
    #     trainer.test(model_dynamic, datamodule=dm)

    #     print(f"Model Used: {args.model}")
    #     print(f"Task Used: {args.task}")
    #     print_size_of_model(model_dynamic)
    #     print(f"Precision: {args.precision}")
    #     print(f"Quantized Block: {layer_to_quantize}")
    #     print(f"Completed at: {datetime.now().strftime('%Y%m%d_%H%M')}")
    #     print(".attn sub-module is quantized")
    #     print("===================================")









    #     # layer_to_quantize = "text_transformer.encoder.layer." + str(i) + ".attention"

    #     # # Check if the layer to quantize is in the model
    #     # assert layer_to_quantize in names, f"Layer {layer_to_quantize} not found in the model"

    #     # if "embeddings" in layer_to_quantize:
    #     #     quantization_config = embedding_layer_qconfig

    #     # model_dynamic = deepcopy(model)
        
    #     # torch.quantization.quantize_dynamic(
    #     #     model_dynamic, {layer_to_quantize: quantization_config}, inplace=True
    #     # )

    #     # # Accuracy Testing
    #     # trainer.test(model_dynamic, datamodule=dm)

    #     # print(f"Model Used: {args.model}")
    #     # print(f"Task Used: {args.task}")
    #     # print_size_of_model(model_dynamic)
    #     # print(f"Precision: {args.precision}")
    #     # print(f"Quantized Block: {layer_to_quantize}")
    #     # print(f"Completed at: {datetime.now().strftime('%Y%m%d_%H%M')}")
    #     # print(".attn sub-module is quantized")
    #     # print("===================================")



    #     # layer_to_quantize = "text_transformer.encoder.layer." + str(i) + ".intermediate"

    #     # # Check if the layer to quantize is in the model
    #     # assert layer_to_quantize in names, f"Layer {layer_to_quantize} not found in the model"

    #     # if "embeddings" in layer_to_quantize:
    #     #     quantization_config = embedding_layer_qconfig

    #     # model_dynamic = deepcopy(model)
        
    #     # torch.quantization.quantize_dynamic(
    #     #     model_dynamic, {layer_to_quantize: quantization_config}, inplace=True
    #     # )

        
    #     # layer_to_quantize = "text_transformer.encoder.layer." + str(i) + ".output"

    #     # # Check if the layer to quantize is in the model
    #     # assert layer_to_quantize in names, f"Layer {layer_to_quantize} not found in the model"

    #     # if "embeddings" in layer_to_quantize:
    #     #     quantization_config = embedding_layer_qconfig
        
    #     # torch.quantization.quantize_dynamic(
    #     #     model_dynamic, {layer_to_quantize: quantization_config}, inplace=True
    #     # )

    #     # # Accuracy Testing
    #     # trainer.test(model_dynamic, datamodule=dm)

    #     # print(f"Model Used: {args.model}")
    #     # print(f"Task Used: {args.task}")
    #     # print_size_of_model(model_dynamic)
    #     # print(f"Precision: {args.precision}")
    #     # print(f"Quantized Block: {layer_to_quantize}")
    #     # print(f"Completed at: {datetime.now().strftime('%Y%m%d_%H%M')}")
    #     # print(".mlp sub-module is quantized")
    #     # print("===================================")

