"""
Standalone training script for SAM3ChangeDetector.

Usage:
    # Full training
    python src/model/train_change_detector.py \
        --data_root clean_data \
        --output_dir runs/change_detector

    # Overfit-one-batch sanity check (Phase 3.5 from the architecture doc)
    python src/model/train_change_detector.py \
        --data_root clean_data \
        --output_dir runs/overfit_test \
        --batch_size 4 \
        --overfit_batches 1 \
        --max_epochs 100

Run from the project root so that both `sam3` and `src` are importable.
"""

import argparse
import math
import sys
from pathlib import Path

# Ensure both the project root and the SAM3 submodule are on the path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "sam3"))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.model_misc import SAM3Output
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2

from src.sam3_model.sam3_wrapper import SAM3ChangeDetector
from src.sam3_model.sam3_dataset import SAM3ChangeDetectionDataset


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_change_detector(freeze_backbone: bool = True, device: str = "cuda") -> SAM3ChangeDetector:
    """
    Build SAM3ChangeDetector:
      1. Build a standard Sam3Image (downloads weights from HuggingFace if needed).
      2. Upgrade in-place to SAM3ChangeDetector, freeze backbone, add orig_proj.
    """
    print("Building SAM3 base model (downloads from HF if not cached)…")
    base = build_sam3_image_model(
        device=device,
        eval_mode=False,      # instantiates the Hungarian matcher
        load_from_HF=True,
    )
    model = SAM3ChangeDetector.from_sam3_image_model(base, freeze_backbone=freeze_backbone)
    return model


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def build_loss(device: str = "cuda") -> Sam3LossWrapper:
    """
    Focal + dice mask loss, GIOU box loss, and presence BCE — same as SAM3's
    standard fine-tuning objective, but with mask loss enabled from step 0.

    normalization="local" avoids the distributed all_reduce call so this works
    on a single GPU without any DDP setup.
    """
    return Sam3LossWrapper(
        normalization="local",
        matcher=BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        ),
        loss_fns_find=[
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(
                weak_loss=False,
                pos_weight=10.0,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pos_focal=True,
                pad_n_queries=200,
                pad_scale_pos=1.0,
                weight_dict={"loss_ce": 2.0, "presence_loss": 1.0},
            ),
            Masks(
                weight_dict={"loss_mask": 5.0, "loss_dice": 5.0},
                focal_alpha=0.25,
                focal_gamma=2,
                num_sample_points=12544,    # 112×112 random point samples
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
            ),
        ],
        loss_fn_semantic_seg=None,
        scale_by_find_batch_size=False,
    ).to(device)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def make_collate_fn():
    """Thin wrapper: collate_fn_api returns a dict; we unwrap the BatchedDatapoint."""
    def collate(batch):
        return collate_fn_api(batch, dict_key="find", with_seg_masks=True)["find"]
    return collate


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- datasets ---
    train_ds = SAM3ChangeDetectionDataset(
        data_root=args.data_root,
        resolution=1008,
        min_sim=args.min_sim,
        split="train",
    )
    val_ds = SAM3ChangeDetectionDataset(
        data_root=args.data_root,
        resolution=1008,
        min_sim=args.min_sim,
        split="val",
    )

    if args.overfit_batches:
        n = min(args.batch_size * args.overfit_batches, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))

    collate = make_collate_fn()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not args.overfit_batches,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
    )

    # --- model ---
    model = build_change_detector(freeze_backbone=True, device=device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable: {sum(p.numel() for p in trainable)/1e6:.1f}M / "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M total params")

    # Slightly higher LR for orig_proj (it starts as identity and needs to learn fast)
    proj_ids = {id(p) for p in model.orig_proj.parameters()}
    other_params = [p for p in trainable if id(p) not in proj_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.orig_proj.parameters()), "lr": args.lr * 10},
            {"params": other_params,                        "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    total_steps = args.max_epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=(device == "cuda"))

    # --- loss ---
    loss_fn = build_loss(device=device)

    # --- output dir ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = copy_data_to_device(batch, device)

            with autocast(enabled=(device == "cuda")):
                # model() returns SAM3Output (list of stages, each a list of steps)
                outputs: SAM3Output = model(batch)
                # loss_fn takes (SAM3Output, list[BatchedFindTarget])
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

            if batch_idx % 50 == 0:
                lr = optimizer.param_groups[1]["lr"]
                parts = {k: f"{v.item():.4f}" for k, v in loss_dict.items()
                         if k != CORE_LOSS_KEY}
                print(f"  Ep{epoch} [{batch_idx}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} lr={lr:.2e} {parts}")

        avg_train = epoch_loss / len(train_loader)

        if not args.overfit_batches:
            val_loss = _validate(model, val_loader, loss_fn, device)
            print(f"Epoch {epoch}: train={avg_train:.4f}  val={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save(model, optimizer, epoch, output_dir / "best.pt")
        else:
            print(f"Epoch {epoch}: train_loss={avg_train:.4f}")

        if (epoch + 1) % args.save_freq == 0:
            _save(model, optimizer, epoch, output_dir / f"epoch_{epoch:04d}.pt")


@torch.no_grad()
def _validate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = copy_data_to_device(batch, device)
        outputs = model(batch)
        loss_dict = loss_fn(outputs, batch.find_targets)
        total += loss_dict[CORE_LOSS_KEY].item()
    model.train()
    return total / max(len(loader), 1)


def _save(model, optimizer, epoch, path):
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch":     epoch,
    }, path)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Train SAM3ChangeDetector")
    p.add_argument("--data_root",       type=str,   default="clean_data",
                   help="Root directory of the clean_data/ dataset")
    p.add_argument("--output_dir",      type=str,   default="runs/change_detector")
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-5,
                   help="Base LR for transformer/decoder; orig_proj uses 10×")
    p.add_argument("--max_epochs",      type=int,   default=20)
    p.add_argument("--min_sim",         type=float, default=0.94,
                   help="Minimum DINO similarity score to include a training sample")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--save_freq",       type=int,   default=5,
                   help="Save a checkpoint every N epochs")
    p.add_argument("--overfit_batches", type=int,   default=0,
                   help="If >0, overfit on this many batches (sanity check)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
