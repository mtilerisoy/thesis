import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from vilt.modules import vilt_utils_kd as vilt_utils

class KDLightningModule(pl.LightningModule):
    def __init__(self, student_model, teacher_model, alpha_kd=0.5, lr=2e-5, config=None, **kwargs):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.alpha_kd = alpha_kd
        self.lr = lr
        self.kd_layer = kwargs.get("kd_layer", None)

        print(f"Applying KD to Layer: {self.kd_layer}")

        self.current_tasks = list()        
        self.save_hyperparameters()
        vilt_utils.set_metrics(self)

        # Set teacher model to eval mode and disable gradient updates
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Placeholders for fusion layer outputs
        self.student_fusion_feats_mlp = None
        self.teacher_fusion_feats_mlp = None
        self.student_fusion_feats_pooler = None
        self.teacher_fusion_feats_pooler = None
        self.student_fusion_feats_attn = None
        self.teacher_fusion_feats_attn = None

        # Register forward hooks for both student and teacher
        self._register_hooks()

    def _register_hooks(self):
        """ Registers hooks to capture the fusion block outputs. """
        def student_hook_mlp(module, inp, out):
            # MLP layer CLS token
            self.student_fusion_feats_mlp = out[0][:, 0]

        def student_hook_pooler(module, inp, out):
            # Pooler layer CLS token
            self.student_fusion_feats_pooler = out[:, 0]
        
        def student_hook_attn(module, inp, out):
            # Pooler layer CLS token
            self.student_fusion_feats_attn = out[:, 0]

        def teacher_hook_mlp(module, inp, out):
            # MLP layer CLS token
            self.teacher_fusion_feats_mlp = out[0][:, 0]
        
        def teacher_hook_pooler(module, inp, out):
            # Pooler layer CLS token
            self.teacher_fusion_feats_pooler = out[:, 0]
        
        def teacher_hook_attn(module, inp, out):
            # Pooler layer CLS token
            self.teacher_fusion_feats_attn = out[:, 0]

        # Register hook on the last transformer block
        # self.student_model.transformer.blocks[self.kd_layer].register_forward_hook(student_hook_mlp)
        # self.teacher_model.transformer.blocks[self.kd_layer].register_forward_hook(teacher_hook_mlp)

        # self.student_model.transformer.blocks[self.kd_layer].attn.register_forward_hook(student_hook_mlp)
        # self.teacher_model.transformer.blocks[self.kd_layer].attn.register_forward_hook(teacher_hook_mlp)

        # Pooler Layers
        self.student_model.pooler.dense.register_forward_hook(student_hook_pooler)
        self.teacher_model.pooler.dense.register_forward_hook(teacher_hook_pooler)

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

    def compute_vqa_loss(self, batch):
        infer = self.student_model.infer(batch, mask_text=False, mask_image=False)
        vqa_logits = self.student_model.vqa_classifier(infer["cls_feats"])

        vqa_targets = torch.zeros(
            len(vqa_logits), self.student_model.hparams.config["vqav2_label_size"]
        ).to(self.student_model.device)

        vqa_labels = batch["vqa_labels"]
        vqa_scores = batch["vqa_scores"]

        for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
            for l, s in zip(_label, _score):
                vqa_targets[i, l] = s

        vqa_loss = (
            F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
            * vqa_targets.shape[1]
        )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

        ret = {
            "vqa_loss": vqa_loss,
            "vqa_logits": vqa_logits,
            "vqa_targets": vqa_targets,
            "vqa_labels": vqa_labels,
            "vqa_scores": vqa_scores,
        }

        phase = "train" if self.student_model.training else "val"
        loss = getattr(self.student_model, f"{phase}_vqa_loss")(ret["vqa_loss"])
        score = getattr(self.student_model, f"{phase}_vqa_score")(
            ret["vqa_logits"], ret["vqa_targets"]
        )
        self.log(f"vqa/{phase}/loss", loss)
        self.log(f"vqa/{phase}/score", score)

        return ret

    def compute_kd_loss(self, batch):
        """ Compute KD loss by matching fusion layer outputs """
        with torch.no_grad():
            self.teacher_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
            self.teacher_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)

        self.student_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
        self.student_model.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)

        # # Detach the teacher features to avoid backpropagating through it
        # teacher_feats_mlp = self.teacher_fusion_feats_mlp.detach()
        teacher_feats_pooler = self.teacher_fusion_feats_pooler.detach()

        # Compute Cosine Similarity loss either for MLP or Pooler layer
        # kd_loss = -torch.mean(F.cosine_similarity(self.student_fusion_feats_mlp, teacher_feats_mlp, dim=-1))
        kd_loss = -torch.mean(F.cosine_similarity(self.student_fusion_feats_pooler, teacher_feats_pooler, dim=-1))
        
        # cls_loss = -torch.mean(F.cosine_similarity(self.student_fusion_feats_mlp[:, 0], teacher_feats_mlp[:, 0], dim=-1))
        cls_loss = -torch.mean(F.cosine_similarity(self.student_fusion_feats_pooler, teacher_feats_pooler, dim=-1))

        return kd_loss, cls_loss
    
    def forward(self, batch):
        ret = dict()
        ret.update(self.compute_nlvr2_loss(batch))

    def training_step(self, batch, batch_idx):
        """ Compute total loss and return for backpropagation """
        ret = self.compute_nlvr2_loss(batch)
        task_loss = ret["nlvr2_loss"]
        
        kd_loss, cls_loss = self.compute_kd_loss(batch)

        total_loss =  task_loss + self.alpha_kd * kd_loss

        self.log("train_task_loss", task_loss, prog_bar=True)
        self.log("train_kd_loss", kd_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)
        self.log("CLS_loss", cls_loss, prog_bar=True)

        return total_loss
    

    def on_train_epoch_end(self):
        vilt_utils.epoch_wrapup(self)

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
