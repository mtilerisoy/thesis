import os
# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "10"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "10"  # Set this to the number of CPUs you want to use


import torch
import random
random.seed(42)
from torch.utils.data import Subset
from vilt.datamodules.multitask_datamodule import MTDataModule as MTDataModuleVILT
from meter.datamodules.multitask_datamodule import MTDataModule as MTDataModuleMeter

from vilt.modules import ViLTransformerSS
from meter.modules import METERTransformerSS

# Set the configuration
import pytorch_lightning as pl
import configs
_config = configs.meter_config_nlvr2_original
_config["batch_size"] = 1
_config["per_gpu_batchsize"] = 1
pl.seed_everything(_config["seed"])

# # Set the GPU device
# gpu_id = 0
# device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(gpu_id)

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
device = "cpu"



class SmallMTDataModuleVILT(MTDataModuleVILT):
    def __init__(self, _config, dist=False, num_samples=5, start_idx=100):
        super().__init__(_config, dist)
        self.num_samples = num_samples
        self.start_idx = start_idx

    def setup(self, stage, is_random):
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
        
    def _get_random_subset(self, dataset, num_samples):
        indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, indices)

class SmallMTDataModuleMETER(MTDataModuleMeter):
    def __init__(self, _config, dist=False, num_samples=10, start_idx=100):
        super().__init__(_config, dist)
        self.num_samples = num_samples
        self.start_idx = start_idx

    def setup(self, stage, is_random):
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
        
    
    def _get_random_subset(self, dataset, num_samples):
        indices = random.sample(range(len(dataset)), num_samples)
        return Subset(dataset, indices)

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
        # if i % 5 == 0:
        #     print(f"Iteration: {i}")
        
        Hv = hvp(layer, grad_params, v)
        Hv_flat = torch.cat([hvi.view(-1) for hvi in Hv])
        
        # Update v and eigenvalue estimate
        v_norm = torch.norm(Hv_flat)
        v = [hvi / v_norm for hvi in Hv]
        eigenvalue = v_norm.item()
    
    return eigenvalue

def compute_layer_eigenvalues(model, input, num_iterations=10):
    eigenvalues = {}

    for name, layer in model.named_modules():
        
        if isinstance(layer, (torch.nn.Linear)):
            # print(f"Layer: {name}")
            if "encoder" not in name or "intermediate" not in name or "output" in name or "attention" in name:
                continue
            print(f"Computing eigenvalue for layer: {name}")
            eigenvalue = compute_top_eigenvalue(model, layer, input, num_iterations)

            eigenvalues[name] = eigenvalue   

            print("==============================================")
            print(f"Computed eigenvalue for layer {name} : {eigenvalue}")
            print("All eigenvalues computed so far:")
            print(f"{eigenvalues}")
            print("==============================================")

    return eigenvalues

def compute_averaged_eigenvalues(model, dataloader, num_batches, num_iterations=50):
    eigenvalues = {}
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Linear)):
            layer_eigenvalues = []
            if "encoder" not in name or "intermediate" in name or "output" not in name or "attention" in name:
                continue
            
            print(f"Computing eigenvalues for layer: {name}")
            
            for i, input_batch in enumerate(dataloader):
                if i % 10 == 0:
                    print(f"Batch {i+1}/{num_batches}")
                # if i >= num_batches:
                #     break

                batch_eigenvalues = compute_top_eigenvalue(
                    model, layer, input_batch, num_iterations=num_iterations
                )
                
                layer_eigenvalues.append(batch_eigenvalues)
            
            # Average eigenvalues over batches
            layer_eigenvalues = torch.tensor(layer_eigenvalues).mean(dim=0).tolist()
            eigenvalues[name] = layer_eigenvalues
            print(f"Layer {name}: Averaged eigenvalues = {layer_eigenvalues}")
    return eigenvalues

if __name__ == '__main__':
    # ==========================================
    # ========= Create full datamodule =========
    # ==========================================
    if "meter" in _config["model"]:
        # full_dm = MTDataModuleMeter(_config, dist=False)
        
        # calibrarte_dm = SmallMTDataModuleMETER(_config, dist=False, num_samples=5, start_idx=100)
        
        infer_dm = SmallMTDataModuleMETER(_config, dist=False, num_samples=100, start_idx=0)
        infer_dm.setup("test", is_random=True)
        infer_dataloader = infer_dm.test_dataloader()

    elif "vilt" in _config["model"]:
        # full_dm = MTDataModuleVILT(_config, dist=False)

        # calibrarte_dm = SmallMTDataModuleVILT(_config, dist=False, num_samples=5)
        
        infer_dm = SmallMTDataModuleVILT(_config, dist=False, num_samples=100, start_idx=0)
        infer_dm.setup("test", is_random=True)
        infer_dataloader = infer_dm.test_dataloader()

    else:
        raise ValueError("Model not supported: ", _config["model"])

    print(f"Batch size: {_config['batch_size']}")

    # ==========================================
    # ========= Initialize the model ===========
    # ==========================================
    if _config["model"] == "vilt":
        model = ViLTransformerSS(_config)
        print("Initialized ViLT model")

    elif _config["model"] == "meter":
        model = METERTransformerSS(_config)
        print("Initialized METER model")

    else:
        raise ValueError("Model not supported: ", _config["model"])


    # Move model to GPU
    model.to(device)

    # ==========================================
    # ======= Initialize the dataloader ========
    # ==========================================
    input_batch = next(iter(infer_dataloader))
    num_batches = len(infer_dataloader)
    
    print(input_batch.keys())
    print(f"Number of batches: {num_batches}")
    print(f"Samples in a batch: {len(input_batch['answers'])}")

    # # Move input data to GPU
    # for key in input_batch:
    #     if isinstance(input_batch[key], torch.Tensor):
    #         input_batch[key] = input_batch[key].to(device)
    
    # input_batch["image_0"][0] = input_batch["image_0"][0].to(device)
    # input_batch["image_1"][0] = input_batch["image_1"][0].to(device)


    # ==========================================
    # ========= Compute eigenvalues ============
    # ==========================================
    # for i in range(20):
    #     infer_dm = SmallMTDataModuleMETER(_config, dist=False, num_samples=1, start_idx=0)
    #     infer_dm.setup("test", is_random=True)
    #     infer_dataloader = infer_dm.test_dataloader()
    #     eigenvalues = compute_layer_eigenvalues(model, input_batch, num_iterations=50)

    #     # Save the eigenvalues to a txt file
    #     with open(f"eigenvalues_meter_{i}.txt", "w") as f:
    #         f.write(str(eigenvalues))
    eigenvalues = compute_averaged_eigenvalues(model, infer_dataloader, num_batches, num_iterations=25)
    # Save the eigenvalues to a txt file
    with open(f"eigenvalues_meter.txt", "w") as f:
        f.write(str(eigenvalues))