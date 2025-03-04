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

# Limit the number of CPUs
os.environ["OMP_NUM_THREADS"] = "10"  # Set this to the number of CPUs you want to use
os.environ["MKL_NUM_THREADS"] = "10"  # Set this to the number of CPUs you want to use


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


import torch
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }
    return ret

def set_schedule(pl_module, trainer, dataloader):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier", "mlm_score", "itm_score", "snli_classifier"]
    cross_modal_names = ['cross_modal']
    lr_mult_head = pl_module.hparams.config["lr_mult_head"]
    lr_mult_cross_modal = pl_module.hparams.config["lr_mult_cross_modal"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = "sgd" #adam" # pl_module.hparams.config["optim_type"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_head,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                and not any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_head,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult_cross_modal,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
                and any(ht in n for ht in cross_modal_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult_cross_modal,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if trainer.max_steps == -1:
        max_steps = (
            len(dataloader)
            * trainer.max_epochs
            // trainer.accumulate_grad_batches
        )
    else:
        max_steps = trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )

def train_nlvr2_loop(pl_module, trainer, train_dataloader, num_epochs, accumulation_steps):
    """
    Basic training loop for NLVR2 task.

    Args:
        pl_module: The PyTorch Lightning module or a similar model with necessary attributes
                   like `infer`, `nlvr2_classifier`, `device`, and `hparams`.
        train_dataloader: PyTorch DataLoader for the training dataset.
        num_epochs: Number of training epochs.
    """

    # Initialize optimizer and scheduler
    optimizers, schedulers = set_schedule(pl_module, trainer, train_dataloader)
    optimizer = optimizers[0]  # Assuming only one optimizer is returned
    scheduler = schedulers[0]['scheduler'] # Assuming only one scheduler is returned

    # pl_module.train() # Set the model to training mode

    global_step = 0 # Track global steps for scheduler

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            batch["image_0"][0] = batch["image_0"][0].to(pl_module.device)
            batch["image_1"][0] = batch["image_1"][0].to(pl_module.device)


            # Compute loss and logits
            output = compute_nlvr2(pl_module, batch)
            loss = output["nlvr2_loss"]
            logits = output["nlvr2_logits"]
            labels = output["nlvr2_labels"]

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # # Backward pass and optimization
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # scheduler.step() # Update learning rate scheduler

            # epoch_loss += loss.item()
            # global_step += 1

            #################################
            # Custom small batch accumulation
            #################################

            # Scale loss to account for accumulation
            loss = loss / accumulation_steps
            loss.backward()

            # Perform an update only after accumulating the desired number of batches
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                optimizer.step()
                scheduler.step()  # Update learning rate scheduler
                optimizer.zero_grad()

            # Scale back for logging
            epoch_loss += loss.item() * accumulation_steps
            global_step += 1

            ##########################################
            # Custom small batch accumulation finished
            ##########################################

            # Print batch loss and accuracy (optional, print every few batches)
            if (batch_idx / accumulation_steps) % 100 == 0:
                batch_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item() * accumulation_steps:.4f}, Accuracy: {batch_accuracy:.4f}")

        # Calculate average epoch loss and accuracy
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_epoch_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        print(f"Epoch [{epoch+1}/{num_epochs}] Summary: Avg Loss: {avg_epoch_loss:.4f}, Avg Accuracy: {avg_epoch_accuracy:.4f}")

    print("Training finished!")