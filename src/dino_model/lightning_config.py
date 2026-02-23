import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from dino_segmenter import DinoSegmenter
from losses import FocalLossFromLogits, DiceLossFromLogits


hyperparameters = {
    "lr": 5e-6,
    "gamma": 2.0,
    "alpha": 0.75,
    "smooth": 1e-5,
    "batch_size": 8,
    "focal_weight": 20.0,
    "hw": 256,
    "skip_zero_edit": True,
    "segmenter_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "segmenter_layers": 2,
    "segmenter_mlp_mult": 4,
}


class DinoLightning(LightningModule):
    def __init__(self, segmenter: DinoSegmenter, hyperparams: Dict):
        super().__init__()
        self.segmenter = segmenter
        self.lr = hyperparams["lr"]
        self.FocalLoss = FocalLossFromLogits(
            gamma=hyperparams["gamma"], alpha=hyperparams["alpha"]
        )
        self.DiceLoss = DiceLossFromLogits(smooth=hyperparams["smooth"])
        self.focal_weight = hyperparams["focal_weight"]
        self.save_hyperparameters(ignore=["segmenter"])
        self._log_cache = None
        self.val_samples: List = []
        self.val_sample_count = 0
        self.max_val_samples = 12

    def forward(self, original_inputs, edited_inputs):
        orig_inputs = self._move_inputs_to_device(original_inputs)
        edit_inputs = self._move_inputs_to_device(edited_inputs)
        return self.segmenter(original_inputs=orig_inputs, edited_inputs=edit_inputs)

    def _prepare_inputs(self, tensor_batch):
        imgs = self._tensor_to_images(tensor_batch)
        inputs = self.segmenter.processor(images=imgs, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _move_inputs_to_device(self, inputs: Dict[str, torch.Tensor]):
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _tensor_to_images(self, tensor_batch):
        tensor_batch = tensor_batch.detach().float().cpu()
        tensor_batch = ((tensor_batch * 0.5) + 0.5).clamp(0, 1)
        tensor_batch = (tensor_batch * 255).byte()
        imgs = tensor_batch.permute(0, 2, 3, 1).numpy()
        return [img for img in imgs]

    def _mask_logits(self, mask_pred):
        eps = 1e-6
        mask_pred = mask_pred.clamp(eps, 1 - eps)
        return torch.logit(mask_pred)

    def _resize_pred(self, pred, target_shape):
        return F.interpolate(pred, size=target_shape, mode="bilinear", align_corners=False)

    def _compute_loss(self, batch):
        y = batch["mask"]
        yhat = self(batch["original_inputs"], batch["edited_inputs"])
        yhat = self._resize_pred(yhat, y.shape[-2:])
        logits = self._mask_logits(yhat)
        if y.shape[1] == 1 and logits.shape[1] > 1:
            y = y.repeat(1, logits.shape[1], 1, 1)
        focal_loss = self.FocalLoss(y, logits)
        dice_loss = self.DiceLoss(y, logits)
        loss = focal_loss * self.focal_weight + dice_loss
        return focal_loss, dice_loss, loss, logits

    def training_step(self, batch, batch_idx):
        focal_loss, dice_loss, loss, logits = self._compute_loss(batch)
        if batch_idx == 0:
            self._log_cache = (
                batch["original"][0:1].detach(),
                batch["edited"][0:1].detach(),
                batch["mask"][0:1].detach(),
                logits[0:1].detach(),
            )
        self.log("Dice", dice_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("Focal", focal_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("Total", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self._log_cache is None:
            return
        original, edited, mask, logits = self._log_cache
        self.segmenter.eval()
        with torch.no_grad():
            y_hat = torch.sigmoid(logits)
        self.segmenter.train()

        import wandb

        def _to_wandb_img(t):
            img = t.detach().cpu().clamp(0, 1)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            return wandb.Image((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        self.logger.experiment.log(
            {
                "orig": _to_wandb_img(original[0] * 0.5 + 0.5),
                "mask_gt": _to_wandb_img(mask[0]),
                "mask_pred": _to_wandb_img(y_hat[0, :1]),
            }
        )

    def on_validation_epoch_start(self):
        self.val_samples = []
        self.val_sample_count = 0

    def validation_step(self, batch, batch_idx):
        focal_loss, dice_loss, loss, logits = self._compute_loss(batch)
        B = batch["original"].shape[0]
        self.log("val_Dice", dice_loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Focal", focal_loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Total", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)

        # Composite vs edited cosine using the shared DINO backbone
        original, edited, mask = batch["original"], batch["edited"], batch["mask"]
        hard_mask = (mask > 0.5).float()
        composite = original * (1 - hard_mask) + edited * hard_mask
        comp_inputs = self._prepare_inputs(composite)
        edit_inputs = self._prepare_inputs(edited)
        with torch.no_grad():
            v = self.segmenter.backbone(**comp_inputs).last_hidden_state
            w = self.segmenter.backbone(**edit_inputs).last_hidden_state
        v = v / v.norm(dim=-1, keepdim=True)
        w = w / w.norm(dim=-1, keepdim=True)
        cosine_sim = (v * w).sum(dim=-1).mean(dim=-1)
        self.log(
            "val_Dino_Cosine", cosine_sim.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=B
        )
        self.log(
            "val_Dino_PassRate",
            (cosine_sim > 0.93).float().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )

        for i in range(B):
            self.val_sample_count += 1
            entry = (
                batch["original"][i : i + 1].detach(),
                batch["edited"][i : i + 1].detach(),
                batch["mask"][i : i + 1].detach(),
                logits[i : i + 1].detach(),
            )
            if len(self.val_samples) < self.max_val_samples:
                self.val_samples.append(entry)
            else:
                j = random.randint(0, self.val_sample_count - 1)
                if j < self.max_val_samples:
                    self.val_samples[j] = entry
        return loss

    def on_validation_epoch_end(self):
        if not self.val_samples:
            return

        import wandb

        def _to_wandb_img(t):
            img = t.detach().cpu().clamp(0, 1)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            return wandb.Image((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        rows = []
        for original, edited, mask, logits in self.val_samples:
            with torch.no_grad():
                pred = torch.sigmoid(logits)
            rows.append(
                [
                    _to_wandb_img(original[0] * 0.5 + 0.5),
                    _to_wandb_img(mask[0]),
                    _to_wandb_img(pred[0, :1]),
                ]
            )
        self.logger.experiment.log(
            {
                "val_samples": wandb.Table(
                    columns=["original", "mask_gt", "mask_pred"],
                    data=rows,
                )
            }
        )
        self.val_samples = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
