import os, shutil, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, random_split

from unet import UNet
from unet_dataset import UNetDataset
from losses import FocalLossFromLogits, DiceLossFromLogits

hyperparameters = {
    'lr': 2e-5,
    'filters': [32, 64, 128, 256],
    'gamma': 2,
    'alpha': 0.75,
    'smooth': 1e-5,
    'batch_size': 8,
    'focal_weight': 20,
    'hw': 256,
    'delta_mode': 'dino'
}

class UNetLightning(LightningModule):
    def __init__(self, unet, hyperparams):
        super().__init__()
        self.unet = unet
        self.dino = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to('cuda')
        self.lr = hyperparams['lr']
        self.FocalLoss = FocalLossFromLogits(gamma=hyperparams['gamma'], alpha=hyperparams['alpha'])
        self.DiceLoss = DiceLossFromLogits(smooth=hyperparams['smooth'])
        self.focal_weight = hyperparams['focal_weight']
        self.cache = None
        self.val_samples = []
        self.val_sample_count = 0
        self.max_val_samples = 20

    def forward(self, original, edited, delta):
        return self.unet(torch.cat([original, edited, delta], dim=1))

    def _compute_loss(self, batch):
        y = batch['mask']
        yhat = self(batch['original'], batch['edited'], batch['delta'])
        focal_loss = self.FocalLoss(y, yhat)
        dice_loss = self.DiceLoss(y, yhat)
        loss = focal_loss * self.focal_weight + dice_loss
        return focal_loss, dice_loss, loss

    def training_step(self, batch, batch_idx):
        y = batch['mask']
        yhat = self(batch['original'], batch['edited'], batch['delta'])
        if batch_idx == 0:
            self.cache = (batch['original'][0:1].detach(),
                          batch['edited'][0:1].detach(),
                          batch['mask'][0:1].detach(),
                          batch['delta'][0:1].detach())

        focal_loss = self.FocalLoss(y, yhat)
        dice_loss = self.DiceLoss(y, yhat)
        self.log('Dice', dice_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('Focal', focal_loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        loss = focal_loss * self.focal_weight + dice_loss
        self.log('Total', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        original, edited, mask, delta = self.cache

        self.unet.eval()
        with torch.no_grad():
            y_hat = torch.sigmoid(self(original, edited, delta))
        self.unet.train()

        import wandb
        def _to_wandb_img(t):
            img = t.detach().cpu().clamp(0, 1)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            return wandb.Image((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        self.logger.experiment.log({
            "orig": _to_wandb_img(original[0] * 0.5 + 0.5),
            "mask_gt": _to_wandb_img(mask[0]),
            "mask_pred": _to_wandb_img(y_hat[0]),
        })

    def on_validation_epoch_start(self):
        self.val_samples = []
        self.val_sample_count = 0

    def validation_step(self, batch, batch_idx):
        focal_loss, dice_loss, loss = self._compute_loss(batch)
        B = batch['original'].shape[0]
        self.log('val_Dice', dice_loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_Focal', focal_loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_Total', loss.detach(), on_step=False, on_epoch=True, prog_bar=True)

        # DINO cosine: composite vs edited
        original, edited, mask = batch['original'], batch['edited'], batch['mask']
        hard_mask = mask > 0.5
        composite = original * (~hard_mask).float() + edited * hard_mask.float()
        with torch.no_grad():
            v = self.dino(F.interpolate(composite * 0.5 + 0.5, size=(224, 224), mode='bilinear')).last_hidden_state
            w = self.dino(F.interpolate(edited * 0.5 + 0.5, size=(224, 224), mode='bilinear')).last_hidden_state
        v = v / v.norm(dim=-1, keepdim=True)
        w = w / w.norm(dim=-1, keepdim=True)
        cosine_sim = (v * w).sum(dim=-1).mean(dim=-1)  # per-sample mean over tokens
        self.log('val_Dino_Cosine', cosine_sim.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log('val_Dino_PassRate', (cosine_sim > 0.93).float().mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        # Reservoir sampling: collect up to max_val_samples random items
        B = batch['original'].shape[0]
        for i in range(B):
            self.val_sample_count += 1
            entry = (batch['original'][i:i+1].detach(),
                     batch['edited'][i:i+1].detach(),
                     batch['mask'][i:i+1].detach(),
                     batch['delta'][i:i+1].detach())
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

        self.unet.eval()
        rows = []
        with torch.no_grad():
            for original, edited, mask, delta in self.val_samples:
                y_hat = torch.sigmoid(self(original, edited, delta))
                rows.append([
                    _to_wandb_img(original[0] * 0.5 + 0.5),
                    _to_wandb_img(mask[0]),
                    _to_wandb_img(y_hat[0]),
                ])
        self.unet.train()

        self.logger.experiment.log({
            "val_samples": wandb.Table(
                columns=["original", "mask_gt", "mask_pred"],
                data=rows,
            )
        })
        self.val_samples = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    import dotenv
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

    logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_USER"),
        name="unet-local-test",
    )

    data_path = os.environ['SAVE_PATH']
    shutil.unpack_archive(r'C:\Users\lex\Downloads\data.zip', data_path + '/data_sample/')
    full_dataset = UNetDataset(path=data_path + '/',
                               hw=hyperparameters['hw'],
                               mode=hyperparameters['delta_mode'],
                               n=50)
    val_size = max(1, len(full_dataset) // 10)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    dataloader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)
    in_channels = 7 if hyperparameters['delta_mode'] == 'dino' else 9
    unet = UNet(hyperparameters['filters'], in_channels=in_channels, n_heads=8)
    model = UNetLightning(unet, hyperparameters)
    trainer = Trainer(max_epochs=50, logger=logger)
    trainer.fit(model, dataloader, val_dataloaders=val_dataloader)
    torch.save(model.unet.state_dict(), "unet.pt")
    wandb.finish()
