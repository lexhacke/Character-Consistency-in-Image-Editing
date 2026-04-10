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
    Expects clean_data/ to be uploaded to the 'sam3-clean-data' Modal volume.
    Upload once with:
        modal volume put sam3-clean-data C:/Users/rhackett/Character-Consistency-in-Image-Editing/clean_data /clean_data
"""

import os

import modal

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("sam3-train")

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
dataset_volume = modal.Volume.from_name("clean-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("picobanana-checkpoints", create_if_missing=True)
DATASET_MOUNT = "/vol/data"
CHECKPOINT_MOUNT = "/vol/checkpoints"

# ---------------------------------------------------------------------------
# Secret
# ---------------------------------------------------------------------------
LOCAL_SRC = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(LOCAL_SRC, "..", ".env")
api_secret = modal.Secret.from_dotenv(_ENV_PATH) if os.path.exists(_ENV_PATH) else modal.Secret.from_dotenv()


# ---------------------------------------------------------------------------
# Image: install deps, bake SAM3 weights + BPE vocab into the layer
# ---------------------------------------------------------------------------
def download_sam3():
    """Pre-download SAM3 weights and CLIP BPE vocab into the image layer."""
    import pathlib
    import requests
    import sam3

    # BPE vocab (may be missing from the PyPI package)
    bpe_dir = pathlib.Path(sam3.__file__).parent.parent / "assets"
    bpe_dir.mkdir(parents=True, exist_ok=True)
    bpe_path = bpe_dir / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        resp = requests.get(
            "https://github.com/openai/CLIP/raw/refs/heads/main/clip/bpe_simple_vocab_16e6.txt.gz",
            stream=True,
        )
        resp.raise_for_status()
        with open(bpe_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    # Authenticate with HuggingFace (required for gated facebook/sam3 repo)
    import os
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # SAM3 checkpoint
    from sam3.model_builder import build_sam3_image_model
    build_sam3_image_model()
    print("SAM3 weights and BPE vocab downloaded successfully.")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install_from_requirements(os.path.join(LOCAL_SRC, "requirements.txt"))
    .run_function(download_sam3, gpu="any", secrets=[api_secret, modal.Secret.from_name("huggingface-secret")])
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
        os.path.join(LOCAL_SRC, "config.json"), "/root/sam3_model/config.json"
    )
)


# ---------------------------------------------------------------------------
# GPU function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",       # 24 GB VRAM — backbone frozen so ~95M params in fp16
    secrets=[api_secret, modal.Secret.from_name("huggingface-secret")],
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
    val_fraction: float = None,
    num_workers: int = None,
    save_freq: int = None,
    num_images_logged: int = None,
    overfit_batches: int = None,
    run_name: str = None,
):
    import sys
    import json
    import math
    import os

    sys.path.insert(0, "/root/sam3_model")

    # HF auth (needed if model cache misses)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import wandb
    import numpy as np
    from torch.cuda.amp import GradScaler
    from torch.utils.data import DataLoader, Subset

    from sam3.model.utils.misc import copy_data_to_device
    from sam3.model_builder import build_sam3_image_model
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks, CORE_LOSS_KEY
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

    from sam3_wrapper import SAM3ChangeDetector
    from sam3_dataset import SAM3ChangeDetectionDataset

    def build_change_detector(freeze_backbone=True, device="cuda"):
        base = build_sam3_image_model(device=device, eval_mode=False, load_from_HF=True)
        return SAM3ChangeDetector.from_sam3_image_model(base, freeze_backbone=freeze_backbone)

    def build_loss(device="cuda"):
        return Sam3LossWrapper(
            normalization="local",
            matcher=BinaryHungarianMatcherV2(
                focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                alpha=0.25, gamma=2, stable=False,
            ),
            o2m_matcher=BinaryOneToManyMatcher(
                alpha=0.3, threshold=0.4, topk=4,
            ),
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            loss_fns_find=[
                Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
                IABCEMdetr(
                    weak_loss=False, pos_weight=5.0, alpha=0.25, gamma=2,
                    use_presence=True, pos_focal=False, pad_n_queries=200,
                    pad_scale_pos=1.0, weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                ),
                Masks(
                    weight_dict={"loss_mask": 5.0, "loss_dice": 5.0},
                    focal_alpha=0.25, focal_gamma=2, num_sample_points=12544,
                    oversample_ratio=3.0, importance_sample_ratio=0.75,
                ),
            ],
            loss_fn_semantic_seg=None,
            scale_by_find_batch_size=True,
        ).to(device)

    def make_collate_fn():
        def collate(batch):
            return collate_fn_api(batch, dict_key="find", with_seg_masks=True)["find"]
        return collate

    def tensor_to_wandb_image(tensor, is_mask=False):
        tensor = tensor.detach().cpu()
        if is_mask:
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            arr = (tensor.float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return wandb.Image(arr)

        img = ((tensor.float().clamp(-1, 1) + 1.0) * 127.5).round().byte()
        arr = img.permute(1, 2, 0).numpy()
        return wandb.Image(arr)

    def tensor_to_wandb_image_with_caption(tensor, caption, is_mask=False):
        tensor = tensor.detach().cpu()
        if is_mask:
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            arr = (tensor.float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return wandb.Image(arr, caption=caption)

        img = ((tensor.float().clamp(-1, 1) + 1.0) * 127.5).round().byte()
        arr = img.permute(1, 2, 0).numpy()
        return wandb.Image(arr, caption=caption)

    def _get_gt_masks_per_sample(stage):
        """Unpack stage.segments (flat packed list) into per-sample lists using num_boxes."""
        num_boxes = stage.num_boxes.tolist()
        gt_masks = []
        offset = 0
        for n in num_boxes:
            gt_masks.append([stage.segments[offset + j] for j in range(n)])
            offset += n
        return gt_masks

    def _blank_mask(hw):
        """Return a zeros mask tensor for samples with no GT mask."""
        return torch.zeros(hw, dtype=torch.float32)

    def get_top2_pred_masks(outputs, target_hw):
        """Return top-2 predicted masks per sample: [B, 2, H, W]."""
        out = outputs[0]
        logits = out["pred_logits"].squeeze(-1)  # [B, 200]
        top2_idx = logits.topk(2, dim=1).indices  # [B, 2]
        B = top2_idx.shape[0]
        batch_idx = torch.arange(B, device=top2_idx.device).unsqueeze(1).expand_as(top2_idx)
        pred_masks = out["pred_masks"][batch_idx, top2_idx]  # [B, 2, mask_H, mask_W]
        pred_masks = torch.sigmoid(pred_masks)
        if pred_masks.shape[-2:] != target_hw:
            pred_masks = F.interpolate(
                pred_masks.flatten(0, 1).unsqueeze(1),  # [B*2, 1, mH, mW]
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).view(B, 2, *target_hw)
        return pred_masks

    def log_visuals(batch, outputs, phase, epoch, step, max_items=4):
        stage = batch.find_targets[0]
        prompts = batch.find_text_batch
        gt_masks = _get_gt_masks_per_sample(stage)
        batch_size = min(len(prompts), len(gt_masks), max_items)
        if batch_size == 0:
            return

        target_hw = stage.segments[0].shape[-2:] if len(stage.segments) > 0 else (1008, 1008)
        pred_masks = get_top2_pred_masks(outputs, target_hw=target_hw)

        columns = [
            "original", "edited",
            "sub_mask_gt", "union_mask_gt",
            "sub_mask_pred", "union_mask_pred",
            "prompt",
        ]
        rows = []
        media = {
            f"{phase}_original": [],
            f"{phase}_edited": [],
            f"{phase}_sub_mask_gt": [],
            f"{phase}_union_mask_gt": [],
            f"{phase}_sub_mask_pred": [],
            f"{phase}_union_mask_pred": [],
        }

        for i in range(min(batch_size, pred_masks.shape[0])):
            prompt = prompts[i]
            original = batch.img_batch[2 * i + 1]
            edited = batch.img_batch[2 * i]
            sample_gt = gt_masks[i]  # list of 0-2 masks

            # GT masks: first is subtraction, second is union (matches dataset ordering)
            sub_gt = sample_gt[0] if len(sample_gt) >= 1 else _blank_mask(target_hw)
            union_gt = sample_gt[1] if len(sample_gt) >= 2 else _blank_mask(target_hw)

            sub_pred = pred_masks[i, 0]
            union_pred = pred_masks[i, 1]

            rows.append([
                tensor_to_wandb_image(original, is_mask=False),
                tensor_to_wandb_image(edited, is_mask=False),
                tensor_to_wandb_image(sub_gt, is_mask=True),
                tensor_to_wandb_image(union_gt, is_mask=True),
                tensor_to_wandb_image(sub_pred, is_mask=True),
                tensor_to_wandb_image(union_pred, is_mask=True),
                prompt,
            ])
            media[f"{phase}_original"].append(
                tensor_to_wandb_image_with_caption(original, caption=prompt, is_mask=False))
            media[f"{phase}_edited"].append(
                tensor_to_wandb_image_with_caption(edited, caption=prompt, is_mask=False))
            media[f"{phase}_sub_mask_gt"].append(
                tensor_to_wandb_image_with_caption(sub_gt, caption=prompt, is_mask=True))
            media[f"{phase}_union_mask_gt"].append(
                tensor_to_wandb_image_with_caption(union_gt, caption=prompt, is_mask=True))
            media[f"{phase}_sub_mask_pred"].append(
                tensor_to_wandb_image_with_caption(sub_pred, caption=prompt, is_mask=True))
            media[f"{phase}_union_mask_pred"].append(
                tensor_to_wandb_image_with_caption(union_pred, caption=prompt, is_mask=True))

        log_data = {
            f"{phase}_samples": wandb.Table(columns=columns, data=rows),
            f"{phase}_samples_epoch": epoch,
        }
        log_data.update(media)
        wandb.log(log_data, step=step)

    # ---- config ----
    with open("/root/sam3_model/config.json") as f:
        config = json.load(f)

    max_epochs     = max_epochs     if max_epochs     is not None else config["max_epochs"]
    batch_size     = batch_size     if batch_size     is not None else config["batch_size"]
    lr             = lr             if lr             is not None else config["lr"]
    min_sim        = min_sim        if min_sim        is not None else config["min_sim"]
    val_fraction   = val_fraction   if val_fraction   is not None else config.get("val_fraction", 0.05)
    num_workers    = num_workers    if num_workers    is not None else config["num_workers"]
    save_freq      = save_freq      if save_freq      is not None else config["save_freq"]
    num_images_logged = num_images_logged if num_images_logged is not None else config.get("num_images_logged", 4)
    overfit_batches = overfit_batches if overfit_batches is not None else config["overfit_batches"]
    data_root      = os.path.join(DATASET_MOUNT, config["data_subdir"])

    hparams = {
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "min_sim": min_sim,
        "val_fraction": val_fraction,
        "num_images_logged": num_images_logged,
        "overfit_batches": overfit_batches,
    }
    print(f"Config: {hparams}")

    # ---- datasets ----
    train_ds = SAM3ChangeDetectionDataset(
        data_root=data_root,
        resolution=config["resolution"],
        min_sim=min_sim,
        val_fraction=val_fraction,
        split="train",
    )
    val_ds = SAM3ChangeDetectionDataset(
        data_root=data_root,
        resolution=config["resolution"],
        min_sim=min_sim,
        val_fraction=val_fraction,
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
        logged_train_visuals = False

        for batch_idx, batch in enumerate(train_loader):
            batch = copy_data_to_device(batch, device)

            with torch.amp.autocast('cuda'):
                outputs = model(batch)
                targets = [model.back_convert(t) for t in batch.find_targets]
                loss_dict = loss_fn(outputs, targets)
                loss = loss_dict[CORE_LOSS_KEY]

            if not logged_train_visuals:
                log_visuals(
                    batch,
                    outputs,
                    phase="train",
                    epoch=epoch,
                    step=global_step,
                    max_items=4,
                )
                logged_train_visuals = True

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
            val_loss = _validate(
                model,
                val_loader,
                loss_fn,
                device,
                epoch=epoch,
                step=global_step,
                num_images_logged=num_images_logged,
            )
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


def _validate(model, loader, loss_fn, device, epoch=None, step=None, num_images_logged=4):
    import torch
    import torch.nn.functional as F
    import wandb
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.loss.loss_fns import CORE_LOSS_KEY
    model.eval()
    total = 0.0
    val_rows = []
    columns = [
        "original", "edited",
        "sub_mask_gt", "union_mask_gt",
        "sub_mask_pred", "union_mask_pred",
        "prompt",
    ]
    media = {
        "val_original": [], "val_edited": [],
        "val_sub_mask_gt": [], "val_union_mask_gt": [],
        "val_sub_mask_pred": [], "val_union_mask_pred": [],
    }

    def _to_img(tensor, is_mask=False):
        tensor = tensor.detach().cpu()
        if is_mask:
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            arr = (tensor.float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
            return wandb.Image(arr)
        img = ((tensor.float().clamp(-1, 1) + 1.0) * 127.5).round().byte()
        return wandb.Image(img.permute(1, 2, 0).numpy())

    with torch.no_grad():
        for batch in loader:
            batch = copy_data_to_device(batch, device)
            outputs = model(batch)
            if step is not None and epoch is not None and len(val_rows) < max(0, num_images_logged):
                stage = batch.find_targets[0]
                prompts = batch.find_text_batch
                remaining = max(0, num_images_logged - len(val_rows))
                batch_size = min(len(prompts), len(stage.num_boxes), remaining)

                # Unpack GT masks per sample
                num_boxes = stage.num_boxes.tolist()
                gt_per_sample = []
                seg_offset = 0
                for n in num_boxes:
                    gt_per_sample.append([stage.segments[seg_offset + j] for j in range(n)])
                    seg_offset += n

                # Top-2 predictions
                target_hw = stage.segments[0].shape[-2:] if len(stage.segments) > 0 else (1008, 1008)
                out = outputs[0]
                logits = out["pred_logits"].squeeze(-1)
                top2_idx = logits.topk(2, dim=1).indices
                B = top2_idx.shape[0]
                bi = torch.arange(B, device=top2_idx.device).unsqueeze(1).expand_as(top2_idx)
                pred_masks = torch.sigmoid(out["pred_masks"][bi, top2_idx])
                if pred_masks.shape[-2:] != target_hw:
                    pred_masks = F.interpolate(
                        pred_masks.flatten(0, 1).unsqueeze(1),
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1).view(B, 2, *target_hw)

                blank = torch.zeros(target_hw, dtype=torch.float32)
                for i in range(batch_size):
                    original = batch.img_batch[2 * i + 1]
                    edited = batch.img_batch[2 * i]
                    prompt = prompts[i]
                    sample_gt = gt_per_sample[i]

                    sub_gt = sample_gt[0] if len(sample_gt) >= 1 else blank
                    union_gt = sample_gt[1] if len(sample_gt) >= 2 else blank
                    sub_pred = pred_masks[i, 0]
                    union_pred = pred_masks[i, 1]

                    val_rows.append([
                        _to_img(original), _to_img(edited),
                        _to_img(sub_gt, is_mask=True), _to_img(union_gt, is_mask=True),
                        _to_img(sub_pred, is_mask=True), _to_img(union_pred, is_mask=True),
                        prompt,
                    ])
                    media["val_original"].append(_to_img(original))
                    media["val_edited"].append(_to_img(edited))
                    media["val_sub_mask_gt"].append(_to_img(sub_gt, is_mask=True))
                    media["val_union_mask_gt"].append(_to_img(union_gt, is_mask=True))
                    media["val_sub_mask_pred"].append(_to_img(sub_pred, is_mask=True))
                    media["val_union_mask_pred"].append(_to_img(union_pred, is_mask=True))

            targets = [model.back_convert(t) for t in batch.find_targets]
            loss_dict = loss_fn(outputs, targets)
            total += loss_dict[CORE_LOSS_KEY].item()

    if val_rows and step is not None and epoch is not None:
        log_data = {
            "val_samples": wandb.Table(columns=columns, data=val_rows),
            "val_samples_epoch": epoch,
        }
        log_data.update(media)
        wandb.log(log_data, step=step)

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
    val_fraction: float = None,
    num_workers: int = 4,
    save_freq: int = None,
    num_images_logged: int = None,
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
    val_fraction = val_fraction if val_fraction is not None else config.get("val_fraction", 0.05)
    num_images_logged = num_images_logged if num_images_logged is not None else config.get("num_images_logged", 4)
    overfit_batches = overfit_batches if overfit_batches is not None else config["overfit_batches"]

    print(
        f"Launching SAM3 training on Modal "
        f"(max_epochs={max_epochs}, batch_size={batch_size}, lr={lr}, "
        f"val_fraction={val_fraction}, overfit_batches={overfit_batches}, "
        f"num_images_logged={num_images_logged})…"
    )
    train.remote(
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        min_sim=min_sim,
        val_fraction=val_fraction,
        num_workers=num_workers,
        save_freq=save_freq,
        num_images_logged=num_images_logged,
        overfit_batches=overfit_batches,
        run_name=run_name,
    )
    print("Done. Checkpoints saved to Modal Volume 'picobanana-checkpoints'.")
