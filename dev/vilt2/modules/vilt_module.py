import torch
import torch.nn as nn
import vilt2.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt2.modules import heads, objectives#, vilt_utils

# import config
import torch.nn.functional as F

class Accuracy():
    def __init__(self, device):
        self.device = device
        self.correct = torch.tensor(0.0)
        self.total = torch.tensor(0.0)
    
    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.device),
            target.detach().to(self.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total

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
    nlvr2_logits = pl_module.dequant(nlvr2_logits)
    # nlvr2_labels = pl_module.dequant(nlvr2_labels)
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    accuracy = Accuracy(pl_module.device)
    accuracy.update(nlvr2_logits, nlvr2_labels)
    acc = accuracy.compute()

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
        "nlvr2_acc": acc,
    }

    return ret


class ViLTransformerSS(nn.Module):
    def __init__(self, config, quantized=False, device="cpu"):
        super().__init__()

        self.outputs = []
        self.config = config
        self.device = device

        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.hidden_size * config.mlp_ratio,
            max_position_embeddings=config.max_text_len,
            hidden_dropout_prob=config.drop_rate,
            attention_probs_dropout_prob=config.drop_rate,
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)

        # if self.hparams.config["load_path"] == "":
        #     self.transformer = getattr(vit, self.hparams.config["vit"])(
        #         pretrained=True, config=self.hparams.config
        #     )
        # else:
        self.transformer = getattr(vit, config.vit)(
            pretrained=False, config=config
        )

        self.pooler = heads.Pooler(config.hidden_size)
        self.pooler.apply(objectives.init_weights)

        # ===================== Downstream ===================== #

        hs = config.hidden_size

        self.nlvr2_classifier = nn.Sequential(
            nn.Linear(hs * 2, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, 2),
        )
        self.nlvr2_classifier.apply(objectives.init_weights)
        emb_data = self.token_type_embeddings.weight.data
        self.token_type_embeddings = nn.Embedding(3, hs)
        self.token_type_embeddings.apply(objectives.init_weights)
        self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
        self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
        self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        # vilt_utils.set_metrics(self)
        self.current_tasks = list()
        self.current_tasks.append("nlvr2")

        # ===================== load downstream (test_only) ======================

        if config.load_path != "" and config.test_only:
            ckpt = torch.load(config.load_path, map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        # text_ids = self.quant(text_ids)
        # text_masks = self.quant(text_masks)

        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.config.max_image_len,
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            self.dequant(text_embeds) + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.quant(x)
        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        # cls_feats = self.dequant(cls_feats)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        # if len(self.current_tasks) == 0:
        #     ret.update(self.infer(batch))
        #     return ret
        
        # NLVR2 Task
        ret.update(compute_nlvr2(self, batch))

        return ret
