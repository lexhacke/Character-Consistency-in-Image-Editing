"""
Modal training script for SAM3ChangeDetector.

Usage:
    # Full training with defaults from config.json
    modal run src/sam3_model/train_modal.py

    # Quick overfit sanity check (1 batch, 100 epochs)
    modal run src/sam3_model/train_modal.py --overfit-batches 1 --max-epochs 100

    # Custom run
    modal run src/sam3_model/train_modal.py --batch-size 2 --lr 5e-6 --max-epochs 30

Data:
    Expects clean_data/ to be uploaded to the 'picobanana-dataset' Modal volume.
    Upload once with:
        modal volume put picobanana-dataset clean_data /clean_data
"""

import os

import modal

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("sam3-change-detector-train")

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
dataset_volume = modal.Volume.from_name("picobanana-dataset", create_if_missing=False)
checkpoint_volume = modal.Volume.from_name("picobanana-checkpoints", create_if_missing=True)
DATASET_MOUNT = "/vol/data"
CHECKPOINT_MOUNT = "/vol/checkpoints"

# ---------------------------------------------------------------------------
# Secret
# ---------------------------------------------------------------------------
LOCAL_SRC = os.path.dirname(os.path.abspath(__file__))
api_secret = modal.Secret.from_dotenv(os.path.join(LOCAL_SRC, "..", ".env"))


# ---------------------------------------------------------------------------
# Image: install deps, bake SAM3 weights + BPE vocab into the layer
# ---------------------------------------------------------------------------
def download_sam3():
    """Pre-download SAM3 weights from HuggingFace and the CLIP BPE vocab."""
    import pathlib
    import requests
    import sam3
    from sam3 import build_sam3_image_model

    # Download SAM3 weights (caches to ~/.cache/huggingface)
    try:
        model = build_sam3_image_model(load_from_HF=True)
        del model
    except FileNotFoundError:
        # BPE vocab missing — download it then retry
        bpe_path = (
            pathlib.Path(sam3.__file__).parent.parent
            / "assets/bpe_simple_vocab_16e6.txt.gz"
        )
        bpe_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(
            "https://github.com/openai/CLIP/raw/refs/heads/main/clip/bpe_simple_vocab_16e6.txt.gz",
            stream=True,
        )
        r.raise_for_status()
        with open(bpe_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        model = build_sam3_image_model(load_from_HF=True)
        del model
    print("SAM3 weights and BPE vocab downloaded successfully.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install_from_requirements(os.path.join(LOCAL_SRC, "requirements.txt"))
    .pip_install("wandb", "python-dotenv")
    .run_function(download_sam3, gpu="any", secrets=[api_secret])
)

# ---------------------------------------------------------------------------
# Add local source files into image
# ---------------------------------------------------------------------------
image = (
    image
    .add_local_file(
        os.path.join(LOCAL_SRC, "sam3_wrapper.py"), "/root/sam3_model/sam3_wrapper.py"
    )
    .add_local_file(
        os.path.join(LOCAL_SRC, "sam3_dataset.py"), "/root/sam3_model/sam3_dataset.py"
    )
    .add_local_file(
        os.path.join(LOCAL_SRC, "sam3_builder.py"), "/root/sam3_model/sam3_builder.py"
    )
    .add_local_file(
        os.path.join(LOCAL_SRC, "train_change_detector.py"),
        "/root/sam3_model/train_change_detector.py",
    )
    .add_local_file(
        os.path.join(LOCAL_SRC, "config.json"), "/root/sam3_model/config.json"
    )
)


# ---------------------------------------------------------------------------
# GPU function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",       # 24 GB VRAM — backbone frozen so ~95M params in fp16
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
    lr: float = None,
    min_sim: float = None,
    num_workers: int = None,
    save_freq: int = None,
    overfit_batches: int = None,
    run_name: str = None,
):
    import sys
    import json
    import math
    import os

    sys.path.insert(0, "/root/sam3_model")

    import torch
    import torch.nn as nn
    import wandb
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Subset

    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks, CORE_LOSS_KEY
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2

    from sam3_wrapper import SAM3ChangeDetector
    from sam3_dataset import SAM3ChangeDetectionDataset
    from train_change_detector import build_change_detector, build_loss, make_collate_fn

    # ---- config ----
    with open("/root/sam3_model/config.json") as f:
        config = json.load(f)

    max_epochs     = max_epochs     if max_epochs     is not None else config["max_epochs"]
    batch_size     = batch_size     if batch_size     is not None else config["batch_size"]
    lr             = lr             if lr             is not None else config["lr"]
    min_sim        = min_sim        if min_sim        is not None else config["min_sim"]
    num_workers    = num_workers    if num_workers    is not None else config["num_workers"]
    save_freq      = save_freq      if save_freq      is not None else config["save_freq"]
    overfit_batches = overfit_batches if overfit_batches is not None else config["overfit_batches"]
    data_root      = os.path.join(DATASET_MOUNT, config["data_subdir"])

    hparams = {
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "min_sim": min_sim,
        "overfit_batches": overfit_batches,
    }
    print(f"Config: {hparams}")

    # ---- datasets ----
    train_ds = SAM3ChangeDetectionDataset(
        data_root=data_root,
        resolution=config["resolution"],
        min_sim=min_sim,
        split="train",
    )
    val_ds = SAM3ChangeDetectionDataset(
        data_root=data_root,
        resolution=config["resolution"],
        min_sim=min_sim,
        split="val",
    )

    if overfit_batches:
        n = min(batch_size * overfit_batches, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))

    collate = make_collate_fn()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=not overfit_batches,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ---- model ----
    device = "cuda"
    model = build_change_detector(freeze_backbone=True, device=device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M / "
        f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M total"
    )

    proj_ids = {id(p) for p in model.orig_proj.parameters()}
    other_params = [p for p in trainable if id(p) not in proj_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.orig_proj.parameters()), "lr": lr * 10},
            {"params": other_params,                        "lr": lr},
        ],
        weight_decay=1e-4,
    )

    total_steps = max_epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    loss_fn = build_loss(device=device)

    # ---- wandb ----
    if run_name is None:
        run_name = f"sam3-change-bs{batch_size}-lr{lr:.1e}-ep{max_epochs}"

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "character-consistency"),
        entity=os.environ.get("WANDB_USER"),
        name=run_name,
        config=hparams,
        dir=CHECKPOINT_MOUNT,
    )

    # ---- output dir ----
    output_dir = os.path.join(CHECKPOINT_MOUNT, "checkpoints", run_name)
    os.makedirs(output_dir, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0

    # ---- training loop ----
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = copy_data_to_device(batch, device)

            with autocast():
                outputs = model(batch)
                loss_dict = loss_fn(outputs, batch.find_targets)
                loss = loss_dict[CORE_LOSS_KEY]

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()

            log_data = {k: v.item() for k, v in loss_dict.items()}
            log_data["lr"] = optimizer.param_groups[1]["lr"]
            wandb.log(log_data, step=global_step)

            if batch_idx % 50 == 0:
                parts = {k: f"{v:.4f}" for k, v in log_data.items() if k != CORE_LOSS_KEY}
                print(
                    f"  Ep{epoch} [{batch_idx}/{len(train_loader)}] "
                    f"loss={loss.item():.4f} {parts}"
                )

        avg_train = epoch_loss / len(train_loader)

        if not overfit_batches:
            val_loss = _validate(model, val_loader, loss_fn, device)
            print(f"Epoch {epoch}: train={avg_train:.4f}  val={val_loss:.4f}")
            wandb.log({"val_loss": val_loss, "train_loss_epoch": avg_train}, step=global_step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save(model, optimizer, epoch, os.path.join(output_dir, "best.pt"))
        else:
            print(f"Epoch {epoch}: train_loss={avg_train:.4f}")
            wandb.log({"train_loss_epoch": avg_train}, step=global_step)

        if (epoch + 1) % save_freq == 0:
            _save(model, optimizer, epoch, os.path.join(output_dir, f"epoch_{epoch:04d}.pt"))

    # ---- final save ----
    _save(model, optimizer, max_epochs - 1, os.path.join(output_dir, "final.pt"))
    checkpoint_volume.commit()
    wandb.finish()
    print("Training complete.")


@torch.no_grad()
def _validate(model, loader, loss_fn, device):
    import torch
    model.eval()
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    total = 0.0
    for batch in loader:
        batch = copy_data_to_device(batch, device)
        outputs = model(batch)
        loss_dict = loss_fn(outputs, batch.find_targets)
        total += loss_dict[CORE_LOSS_KEY].item()
    model.train()
    return total / max(len(loader), 1)


def _save(model, optimizer, epoch, path):
    import torch
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch":     epoch,
    }, path)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    max_epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    min_sim: float = None,
    num_workers: int = 4,
    save_freq: int = None,
    overfit_batches: int = None,
    run_name: str = None,
):
    import json

    config_path = os.path.join(LOCAL_SRC, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    max_epochs  = max_epochs  if max_epochs  is not None else config["max_epochs"]
    batch_size  = batch_size  if batch_size  is not None else config["batch_size"]
    lr          = lr          if lr          is not None else config["lr"]
    overfit_batches = overfit_batches if overfit_batches is not None else config["overfit_batches"]

    print(
        f"Launching SAM3 training on Modal "
        f"(max_epochs={max_epochs}, batch_size={batch_size}, lr={lr}, "
        f"overfit_batches={overfit_batches})…"
    )
    train.remote(
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        min_sim=min_sim,
        num_workers=num_workers,
        save_freq=save_freq,
        overfit_batches=overfit_batches,
        run_name=run_name,
    )
    print("Done. Checkpoints saved to Modal Volume 'picobanana-checkpoints'.")
