import os, shutil
import torch
import torch.nn.functional as F
from transformers import AutoModel
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

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

    def forward(self, original, edited, delta):
        return self.unet(torch.cat([original, edited, delta], dim=1))

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

        # Recall that mask is what we stitch from edited onto original
        hard_mask = mask > 0.5
        not_mask = hard_mask.logical_not()
        composite = original * not_mask.float() + edited * hard_mask.float()
        with torch.no_grad():
            v = self.dino(F.interpolate(composite * 0.5 + 0.5, size=(224, 224), mode='bilinear')).last_hidden_state[0]
            w = self.dino(F.interpolate(edited * 0.5 + 0.5, size=(224, 224), mode='bilinear')).last_hidden_state[0]
        v = v / v.norm(dim=-1, keepdim=True)
        w = w / w.norm(dim=-1, keepdim=True)
        cosine_sim = (v * w).sum(dim=-1)
        self.log('Dino_Cosine', cosine_sim.mean(), on_step=False, on_epoch=True, prog_bar=True)

        self.unet.eval()
        with torch.no_grad():
            y_hat = torch.sigmoid(self(original, edited, delta))
        self.unet.train()

        self.logger.experiment.add_image('orig', original[0]*0.5 + 0.5, self.current_epoch)
        self.logger.experiment.add_image('mask_gt', mask[0], self.current_epoch)
        self.logger.experiment.add_image('mask_pred', y_hat[0], self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    import subprocess
    import time
    import dotenv

    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

    logger = TensorBoardLogger("unet_logs", name="unet_segmenter")
    tb_process = subprocess.Popen(['tensorboard', '--logdir', 'unet_logs', '--port', '6006'])
    time.sleep(10)
    print("Tensorboard running on http://localhost:6006")

    try:
        data_path = os.environ['SAVE_PATH']
        shutil.unpack_archive(r'C:\Users\lex\Downloads\data.zip', data_path + '/data_sample/')
        dataset = UNetDataset(path=data_path + '/',
                              hw=hyperparameters['hw'],
                              mode=hyperparameters['delta_mode'],
                              n=50)
        dataloader = DataLoader(dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
        in_channels = 7 if hyperparameters['delta_mode'] == 'dino' else 9
        unet = UNet(hyperparameters['filters'], in_channels=in_channels, n_heads=8)
        model = UNetLightning(unet, hyperparameters)
        trainer = Trainer(max_epochs=50, logger=logger)
        trainer.fit(model, dataloader)
        torch.save(model.unet.state_dict(), "unet.pt")
    finally:
        tb_process.terminate()
        print("Terminated Tensorboard Process")
