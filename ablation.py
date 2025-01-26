import os
# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "8"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "8"  # Set this to the number of CPUs you want to use

from sensitivity_utils import init_trainer, get_quantization_config, print_size_of_model
import torch
from datetime import datetime
    
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run METER PTQ")
    parser.add_argument('model', type=str, help='Model to use: vilt, meter')
    parser.add_argument('task', type=str, help='Task to run: nlvr2, vqa')
    parser.add_argument('precision', type=int, help='Precision to use: 4, 8')
    # parser.add_argument('layer2quantize', type=str, help='Layer to quantize: transformer.patch_embed')
    args = parser.parse_args()

    if "vilt" in args.model:
        from vilt.datamodules.multitask_datamodule import MTDataModule
        from vilt.modules import ViLTransformerSS as MODEL
        from sensitivity_utils import SmallMTDataModuleMETER as SmallMTDataModule

        if "nlvr" in args.task:
            from configs import vilt_config_nlvr2 as CONFIG
        elif "vqa" in args.task:
            from configs import vilt_config_vqav2 as CONFIG
        else:
            raise ValueError("Task not supported")

    elif "meter" in args.model:
        from meter.datamodules.multitask_datamodule import MTDataModule
        from meter.modules import METERTransformerSS as MODEL
        from sensitivity_utils import SmallMTDataModuleMETER as SmallMTDataModule

        if "nlvr" in args.task:
            from configs import meter_config_nlvr2 as CONFIG
        elif "vqa" in args.task:
            from configs import meter_config_vqav2 as CONFIG
        else:
            raise ValueError("Task not supported")
    else:
        raise ValueError("Model not supported")
    

    # Initialize the datamodules
    dm = MTDataModule(CONFIG, dist=False)
    test_dm = SmallMTDataModule(CONFIG, dist=False, num_samples=250, start_idx=100)
    print("Datamodules initialized")

    # Initialize the model
    model = MODEL(CONFIG)
    print("Model initialized")

    # Initialize the trainer
    trainer = init_trainer(CONFIG)
    print("Trainer initialized")
    
    print(f"Initializing the quantizers using precision: {args.precision}")
    quantization_config, embedding_layer_qconfig = get_quantization_config(precision=args.precision)
    print("Quantization config initialized")

    # ========== Quantization ==========
    # Get the names of the modules
    names, _ = zip(*list(model.named_modules()))
    
    # Quantize the model
    # layer_to_quantize = args.layer2quantize

    for i in range(9):
        i += 4
        print(f"Block {i}")

        layer_to_quantize = "transformer.blocks." + str(i)

        # Check if the layer to quantize is in the model
        assert layer_to_quantize in names, f"Layer {layer_to_quantize} not found in the model"

        if "embeddings" in layer_to_quantize:
            quantization_config = embedding_layer_qconfig
        
        torch.quantization.quantize_dynamic(
            model, {layer_to_quantize: quantization_config}, inplace=True
        )

        # Accuracy Testing
        trainer.test(model, datamodule=dm)

        print(f"Model Used: {args.model}")
        print(f"Task Used: {args.task}")
        print_size_of_model(model)
        print(f"Precision: {args.precision}")
        print(f"Quantized Block: {layer_to_quantize}")
        print(f"Completed at: {datetime.now().strftime('%Y%m%d_%H%M')}")

