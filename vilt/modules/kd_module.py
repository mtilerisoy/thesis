import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from vilt.modules import vilt_utils_kd as vilt_utils

class KDLightningModule(pl.LightningModule):
    def __init__(self, student_model, teacher_model, alpha_kd=0.5, lr=2e-5, config=None, T=1.0, **kwargs):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.alpha_kd = alpha_kd
        self.lr = lr
        self.T = T
        self.kd_layer = kwargs.get("kd_layer", -1)

        print(f"Applying KD to Layer: {self.kd_layer}")

        self.current_tasks = list()        
        self.save_hyperparameters()
        vilt_utils.set_metrics(self)

        # Set teacher model to eval mode and disable gradient updates
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Placeholders for fusion layer outputs
        self.student_fusion_feats = None
        self.teacher_fusion_feats = None

        # Register forward hooks for both student and teacher
        self._register_hooks()

    def _register_hooks(self):
        """ Registers hooks to capture the fusion block outputs. """
        def student_hook(module, inp, out):
            self.student_fusion_feats = out[0] #[0][:, 0]

        def teacher_hook(module, inp, out):
            self.teacher_fusion_feats = out[0] #[0][:, 0]

        # Register hook on the last transformer block
        self.student_model.transformer.blocks[self.kd_layer].register_forward_hook(student_hook)
        self.teacher_model.transformer.blocks[self.kd_layer].register_forward_hook(teacher_hook)

    def compute_nlvr2_loss(self, batch):
        """ Compute NLVR2 classification loss """
        infer1 = self.student_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
        infer2 = self.student_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)

        cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
        nlvr2_logits = self.student_model.nlvr2_classifier(cls_feats)

        nlvr2_labels = batch["answers"]
        nlvr2_labels = torch.tensor(nlvr2_labels).to(self.device).long()
        
        nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

        ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
        }


        phase = "train" if self.student_model.training else "val"

        if phase == "train":
            loss = getattr(self.student_model, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
            acc = getattr(self.student_model, f"{phase}_nlvr2_accuracy")(
                ret["nlvr2_logits"], ret["nlvr2_labels"]
            )
            self.log(f"nlvr2/{phase}/loss", loss)
            self.log(f"nlvr2/{phase}/accuracy", acc)
        else:
            dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
            test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

            if dev_batches:
                dev_loss = getattr(self.student_model, f"dev_nlvr2_loss")(
                    F.cross_entropy(
                        ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                    )
                )
                dev_acc = getattr(self.student_model, f"dev_nlvr2_accuracy")(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
                self.log(f"nlvr2/dev/loss", dev_loss)
                self.log(f"nlvr2/dev/accuracy", dev_acc)
            if test_batches:
                test_loss = getattr(self.student_model, f"test_nlvr2_loss")(
                    F.cross_entropy(
                        ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                    )
                )
                test_acc = getattr(self.student_model, f"test_nlvr2_accuracy")(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
                self.log(f"nlvr2/test/loss", test_loss)
                self.log(f"nlvr2/test/accuracy", test_acc)

        return ret

    def compute_kd_loss(self, batch):
        """ Compute KD loss by matching fusion layer outputs """
        with torch.no_grad():
            self.teacher_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
            self.teacher_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)

        self.student_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
        self.student_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)

        if self.student_fusion_feats is None or self.teacher_fusion_feats is None:
            raise RuntimeError("Fusion layer hooks did not capture outputs!")

        # Detach the teacher features to avoid backpropagating through it
        teacher_feats = self.teacher_fusion_feats.detach()

        # Normalize the features before computing KD loss
        teacher_feats = F.normalize(teacher_feats, dim=-1)
        student_feats = F.normalize(self.student_fusion_feats, dim=-1)

        # Compute Mean Squared Error (MSE) loss
        kd_loss = F.mse_loss(self.student_fusion_feats, teacher_feats)
        # kd_loss = F.cosine_similarity(self.student_fusion_feats, teacher_feats, dim=-1).mean()

        return kd_loss
    
    def forward(self, batch):
        ret = dict()
        ret.update(self.compute_nlvr2_loss(batch))

    def training_step(self, batch, batch_idx):
        """ Compute total loss and return for backpropagation """
        ret = self.compute_nlvr2_loss(batch)
        
        nlvr2_loss = ret["nlvr2_loss"]
        kd_loss = self.compute_kd_loss(batch)

        total_loss = (1-self.alpha_kd) * nlvr2_loss + self.alpha_kd * kd_loss

        self.log("train_nlvr2_loss", nlvr2_loss, prog_bar=True)
        self.log("train_kd_loss", kd_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)

        return total_loss
    

    def on_train_epoch_end(self):
        vilt_utils.epoch_wrapup(self)
        self.log("Student scale_factor", self.student_model.scale_factor)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

        return output

    def on_test_epoch_end(self):
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        """ Define optimizer and learning rate """
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.lr)
        return optimizer
