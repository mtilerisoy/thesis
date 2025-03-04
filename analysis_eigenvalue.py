import os
# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "10"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "10"  # Set this to the number of CPUs you want to use

from loguru import logger
import sys

# Configure loguru for terminal output only
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

# Initialize distributed backend
import torch.distributed as dist
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group(backend='gloo', init_method='env://', world_size=1, rank=0)
logger.info(f"Distributed backend initialized: {dist.is_initialized()}")


import torch
import random
random.seed(42)

from vilt.modules import ViLTransformerSS
from meter.modules import METERTransformerSS

from quantization_utils import SmallMTDataModuleMETER, SmallMTDataModuleVILT

# Set the configuration
import pytorch_lightning as pl
import configs
_config = configs.meter_config_nlvr2_original
_config["batch_size"] = 1
_config["per_gpu_batchsize"] = 1
pl.seed_everything(_config["seed"])

# Set the GPU device
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(gpu_id)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())




# Hessian analysis
def compute_gradients(pl_module, batch, layer):
    pl_module.zero_grad()

    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(device).long()  # Move labels to GPU
    loss = torch.nn.functional.cross_entropy(nlvr2_logits, nlvr2_labels)

    grad_params = torch.autograd.grad(loss, layer.parameters(), create_graph=True)
    return grad_params

def hvp(layer, grad_params, v):
    # Flatten gradients and vector v
    grads = torch.cat([g.contiguous().view(-1) for g in grad_params])
    v = torch.cat([vi.contiguous().view(-1) for vi in v])
    
    # Compute g^T * v
    gTv = torch.dot(grads, v)
    
    # Compute Hv = ∇(g^T v)
    Hv = torch.autograd.grad(gTv, layer.parameters(), retain_graph=True)
    Hv = [h.detach() for h in Hv]  # Detach to stop gradient tracking
    return Hv

def compute_top_eigenvalue(model, layer, input, num_iterations=50):
    grad_params = compute_gradients(model, input, layer)
    
    # Initialize random vector v with same shape as parameters
    params = list(layer.parameters())
    v = [torch.randn_like(p).to(device) for p in params]  # Move v to GPU
    
    # Normalize v
    v_flat = torch.cat([vi.view(-1) for vi in v])
    v_norm = torch.norm(v_flat)
    v = [vi / v_norm for vi in v]
    
    for i in range(num_iterations):
        Hv = hvp(layer, grad_params, v)
        Hv_flat = torch.cat([hvi.view(-1) for hvi in Hv])
        
        # Update v and eigenvalue estimate
        v_norm = torch.norm(Hv_flat)
        v = [hvi / v_norm for hvi in Hv]
        eigenvalue = v_norm.item()

        logger.debug(f"Iteration: {i}, Eigenvalue: {eigenvalue}")
    
    return eigenvalue

def compute_layer_eigenvalues(model, input, num_iterations=10):
    eigenvalues = {}

    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Linear)):
            if "encoder" not in name or "intermediate" not in name or "output" in name or "attention" in name:
                continue
            logger.info(f"Computing eigenvalue for layer: {name}")
            eigenvalue = compute_top_eigenvalue(model, layer, input, num_iterations)

            eigenvalues[name] = eigenvalue   

            logger.info("=" * 45)
            logger.info(f"Computed eigenvalue for layer {name}: {eigenvalue}")
            logger.info("All eigenvalues computed so far:")
            logger.info(f"{eigenvalues}")
            logger.info("=" * 45)

    return eigenvalues

def compute_averaged_eigenvalues(model, dataloader, num_batches, num_iterations=50):
    eigenvalues = {}
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Linear)):
            layer_eigenvalues = []
            if "encoder" not in name or "intermediate" in name or "output" not in name or "attention" in name:
                continue
            
            logger.info(f"Computing eigenvalues for layer: {name}")
            
            for i, input_batch in enumerate(dataloader):
                try:
                    if model.device == "cuda":
                        # Move input data to GPU
                        for key in input_batch:
                            if isinstance(input_batch[key], torch.Tensor):
                                input_batch[key] = input_batch[key].to(device)
                        
                        input_batch["image_0"][0] = input_batch["image_0"][0].to(device)
                        input_batch["image_1"][0] = input_batch["image_1"][0].to(device)

                    logger.debug(f"Processing batch {i+1}/{num_batches}")

                    batch_eigenvalues = compute_top_eigenvalue(
                        model, layer, input_batch, num_iterations=num_iterations
                    )
                    
                    layer_eigenvalues.append(batch_eigenvalues)
                    model.zero_grad()
                except Exception as e:
                    logger.error(f"Error processing batch {i+1}: {str(e)}")
                    continue
            
            # Average eigenvalues over batches
            layer_eigenvalues = torch.tensor(layer_eigenvalues).mean(dim=0).tolist()
            eigenvalues[name] = layer_eigenvalues
            logger.info(f"Layer {name}: Averaged eigenvalues = {layer_eigenvalues}")
    return eigenvalues

if __name__ == '__main__':
    try:
        # ==========================================
        # ========= Create full datamodule =========
        # ==========================================
        if "meter" in _config["model"]:
            infer_dm = SmallMTDataModuleMETER(_config, dist=False, num_samples=5, start_idx=1203)
            infer_dm.setup("test")
            infer_dataloader = infer_dm.test_dataloader()
            logger.info("METER datamodule initialized")

        elif "vilt" in _config["model"]:
            infer_dm = SmallMTDataModuleVILT(_config, dist=False, num_samples=100, start_idx=0)
            infer_dm.setup("test", is_random=True)
            infer_dataloader = infer_dm.test_dataloader()
            logger.info("ViLT datamodule initialized")

        else:
            logger.error(f"Model not supported: {_config['model']}")
            raise ValueError("Model not supported: " + _config["model"])

        logger.info(f"Batch size: {_config['batch_size']}")

        # ==========================================
        # ========= Initialize the model ===========
        # ==========================================
        if _config["model"] == "vilt":
            model = ViLTransformerSS(_config)
            logger.info("Initialized ViLT model")

        elif _config["model"] == "meter":
            model = METERTransformerSS(_config)
            logger.info("Initialized METER model")

        else:
            logger.error(f"Model not supported: {_config['model']}")
            raise ValueError("Model not supported: " + _config["model"])

        # Move model to device
        model.to(device)
        model.eval()
        logger.info(f"Model moved to device: {device}")

        # ==========================================
        # ======= Initialize the dataloader ========
        # ==========================================
        input_batch = next(iter(infer_dataloader))
        num_batches = len(infer_dataloader)
        
        logger.debug(f"Input batch keys: {input_batch.keys()}")
        logger.info(f"Number of batches: {num_batches}")
        logger.info(f"Samples in a batch: {len(input_batch['answers'])}")

        # ==========================================
        # ========= Compute eigenvalues ============
        # ==========================================
        eigenvalues = compute_averaged_eigenvalues(model, infer_dataloader, num_batches, num_iterations=5)
        
        # Save the eigenvalues to a txt file
        output_file = "eigenvalues_meter.txt"
        with open(output_file, "w") as f:
            f.write(str(eigenvalues))
        logger.success(f"Successfully saved eigenvalues to {output_file}")

    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        raise