import os

import modal

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("dino-segmenter-train")

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
dataset_volume = modal.Volume.from_name("picobanana-dataset", create_if_missing=False)
checkpoint_volume = modal.Volume.from_name("picobanana-checkpoints", create_if_missing=True)
DATASET_MOUNT = "/vol/data"
CHECKPOINT_MOUNT = "/vol/checkpoints"

# ---------------------------------------------------------------------------
# Secret for API keys (reads from src/.env)
# ---------------------------------------------------------------------------
LOCAL_SRC = os.path.dirname(os.path.abspath(__file__))
api_secret = modal.Secret.from_dotenv(os.path.join(LOCAL_SRC, "..", ".env"))


# ---------------------------------------------------------------------------
# Image: install deps + bake DINO weights into the layer
# ---------------------------------------------------------------------------
def download_models():
    from transformers import AutoModel, AutoImageProcessor

    AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install_from_requirements(os.path.join(LOCAL_SRC, "requirements.txt"))
    .run_function(download_models, gpu="any", secrets=[api_secret])
)

# ---------------------------------------------------------------------------
# Add local model source files to image
# ---------------------------------------------------------------------------
image = (
    image.add_local_file(
        os.path.join(LOCAL_SRC, "dino_segmenter.py"), "/root/dino_model/dino_segmenter.py"
    )
    .add_local_file(
        os.path.join(LOCAL_SRC, "dino_dataset.py"), "/root/dino_model/dino_dataset.py"
    )
    .add_local_file(os.path.join(LOCAL_SRC, "losses.py"), "/root/dino_model/losses.py")
    .add_local_file(
        os.path.join(LOCAL_SRC, "lightning_config.py"), "/root/dino_model/lightning_config.py"
    )
    .add_local_file(os.path.join(LOCAL_SRC, "config.json"), "/root/dino_model/config.json")
    .add_local_file(
        os.path.join(LOCAL_SRC, "config_base.json"), "/root/dino_model/config_base.json"
    )
    .add_local_file(
        os.path.join(LOCAL_SRC, "config_small.json"), "/root/dino_model/config_small.json"
    )
)


# ---------------------------------------------------------------------------
# GPU function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",
    secrets=[api_secret],
    volumes={
        DATASET_MOUNT: dataset_volume,
        CHECKPOINT_MOUNT: checkpoint_volume,
    },
    timeout=86400,
)
def train(
    max_epochs: int = None,
    batch_size: int = None,
    num_workers: int = 4,
    lr: float = None,
    n: int = None,
    hw: int = None,
    run_name: str = None,
):
    import sys

    sys.path.insert(0, "/root/dino_model")

    os.environ["SAVE_PATH"] = DATASET_MOUNT

    import json
    import torch
    import wandb
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, random_split

    from dino_dataset import DinoDataset
    from dino_segmenter import DinoSegmenter
    from lightning_config import DinoLightning, hyperparameters

    with open("/root/dino_model/config.json", "r") as f:
        config = json.load(f)

    max_epochs = max_epochs if max_epochs is not None else config["max_epochs"]
    batch_size = batch_size if batch_size is not None else config["batch_size"]
    lr = lr if lr is not None else config["lr"]
    lr *= batch_size / 8
    hw = hw if hw is not None else config["hw"]

    hyperparameters.update(config)
    hyperparameters["batch_size"] = batch_size
    hyperparameters["lr"] = lr
    hyperparameters["hw"] = hw

    print(
        f"Training with: max_epochs={max_epochs}, batch_size={batch_size}, lr={lr}, hw={hw}, n={n}, "
        f"segmenter_layers={hyperparameters['segmenter_layers']}"
    )
    print(f"Full config: {hyperparameters}")

    full_dataset = DinoDataset(
        path=DATASET_MOUNT + "/",
        hw=hyperparameters["hw"],
        n=n,
        skip_zero_edit=hyperparameters["skip_zero_edit"],
        processor_name=hyperparameters["segmenter_model_name"],
    )
    val_size = max(1, len(full_dataset) // 10)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Split: {train_size} train, {val_size} val")

    dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    segmenter = DinoSegmenter(
        layers=hyperparameters["segmenter_layers"],
        mlp_mult=hyperparameters["segmenter_mlp_mult"],
        model_name=hyperparameters["segmenter_model_name"],
    )
    model = DinoLightning(segmenter, hyperparameters)

    if run_name is None:
        run_name = (
            f"dino-seg-{hw}px-bs{batch_size}-lr{lr:.2e}-L{hyperparameters['segmenter_layers']}"
        )

    wandb_logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT", "character-consistency"),
        entity=os.environ.get("WANDB_USER"),
        name=run_name,
        log_model=False,
        save_dir=CHECKPOINT_MOUNT,
    )
    wandb_logger.experiment.config.update(hyperparameters)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_MOUNT, "checkpoints", run_name),
        filename="dino-seg-{epoch:02d}-{Total_epoch:.4f}",
        save_top_k=3,
        monitor="Total_epoch",
        mode="min",
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=1,
    )
    trainer.fit(model, dataloader, val_dataloaders=val_dataloader)

    torch.save(
        model.segmenter.state_dict(),
        os.path.join(CHECKPOINT_MOUNT, "dino_segmenter_final.pt"),
    )

    checkpoint_volume.commit()
    wandb.finish()
    print("Training complete.")


@app.local_entrypoint()
def main(
    max_epochs: int = None,
    batch_size: int = None,
    num_workers: int = 4,
    lr: float = None,
    n: int = None,
    hw: int = None,
    run_name: str = None,
):
    import json

    config_path = os.path.join(LOCAL_SRC, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    max_epochs = max_epochs if max_epochs is not None else config["max_epochs"]
    batch_size = batch_size if batch_size is not None else config["batch_size"]
    lr = lr if lr is not None else config["lr"]
    hw = hw if hw is not None else config["hw"]

    print(
        f"Launching training on Modal (max_epochs={max_epochs}, batch_size={batch_size}, lr={lr}, hw={hw}, n={n})..."
    )
    train.remote(
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        n=n,
        hw=hw,
        num_workers=num_workers,
        run_name=run_name,
    )
    print("Done. Checkpoints saved to Modal Volume 'picobanana-checkpoints'.")
