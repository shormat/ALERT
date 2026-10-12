"""Segmentation training entry point."""
from __future__ import annotations

import os
import socket
from pathlib import Path

import cv2
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from alert.segmentation.datasets.bdd100k import BDD100KSegDataset
from alert.segmentation.factory import build_segmentor
from alert.utils.device import resolve_device

_REPO = Path(__file__).resolve().parents[2]

_CLASS_LABELS = {0: "direct", 1: "alternative", 2: "background"}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class _Collate:
    def __init__(self, size: tuple[int, int] | None):
        self.size = size

    def __call__(self, batch):
        imgs, masks = [], []
        for frame, mask in batch:
            if self.size is not None:
                h, w = self.size
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
            t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
            imgs.append(t)
            masks.append(torch.from_numpy(mask.astype("int64")))
        return torch.stack(imgs), torch.stack(masks)


def _setup_ddp() -> tuple[int, int, int, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        return 0, 0, 1, False
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), local_rank, dist.get_world_size(), True


def train_segmentation(config: dict, args) -> Path:
    rank, local_rank, world_size, is_ddp = _setup_ddp()
    is_main = rank == 0

    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        config = {**config, "device": f"cuda:{local_rank}"}
    else:
        device = torch.device(resolve_device(config))

    segmentor = build_segmentor(config)
    model = segmentor.model

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    epochs      = getattr(args, "epochs", 20)
    total_batch = getattr(args, "batch", 8)
    per_gpu_batch = max(1, total_batch // world_size)
    lr      = getattr(args, "lr", 1e-4)
    workers = getattr(args, "workers", 4)

    seg_cfg = config.get("segmentation", {})
    project   = seg_cfg.get("project", "checkpoints/segmentation")
    if not Path(project).is_absolute():
        project = str(_REPO / project)
    model_key = seg_cfg.get("model", "segmentor")
    name      = seg_cfg.get("name", f"{model_key}_bdd100k")
    save_dir  = Path(project) / name
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    img_size = seg_cfg.get("img_size")
    size_hw: tuple[int, int] | None = (int(img_size[0]), int(img_size[1])) if img_size else None
    amp_arg = getattr(args, "amp", None)
    amp_cfg = bool(seg_cfg.get("amp", True)) if amp_arg is None else bool(amp_arg)
    use_amp = device.type == "cuda" and amp_cfg

    if is_main:
        wandb.init(
            project="alert-segmentation",
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=f"{model_key}-ep{epochs}-bs{total_batch}",
            tags=[socket.gethostname(), str(model_key), f"gpus{world_size}"],
            config={
                "segmentation": seg_cfg,
                "epochs": epochs,
                "batch_total": total_batch,
                "batch_per_gpu": per_gpu_batch,
                "lr": lr,
                "amp": use_amp,
                "world_size": world_size,
            },
        )

    train_ds = BDD100KSegDataset(split="train")
    val_ds   = BDD100KSegDataset(split="val")
    collate  = _Collate(size_hw)
    pin = device.type == "cuda"

    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(
            train_ds, batch_size=per_gpu_batch, sampler=train_sampler,
            num_workers=workers, pin_memory=pin, drop_last=True, collate_fn=collate,
        )
        val_loader = DataLoader(
            val_ds, batch_size=per_gpu_batch, sampler=val_sampler,
            num_workers=workers, pin_memory=pin, drop_last=False, collate_fn=collate,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=per_gpu_batch, shuffle=True,
            num_workers=workers, pin_memory=pin, drop_last=True, collate_fn=collate,
        )
        val_loader = DataLoader(
            val_ds, batch_size=per_gpu_batch, shuffle=False,
            num_workers=workers, pin_memory=pin, drop_last=False, collate_fn=collate,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_classes   = int(seg_cfg.get("num_classes", 3))
    best_val_loss = float("inf")
    best_path     = save_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss      = 0.0
        n_train_batches = len(train_loader)
        for i, (img_t, msk_t) in enumerate(train_loader):
            img_t = img_t.to(device, non_blocking=pin)
            msk_t = msk_t.to(device, non_blocking=pin)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(img_t)
                if logits.shape[-2:] != msk_t.shape[-2:]:
                    logits = F.interpolate(logits, size=msk_t.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, msk_t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            if is_main and (i + 1) % 100 == 0:
                print(f"  epoch {epoch}/{epochs}  step {i+1}/{n_train_batches}  loss={loss.item():.4f}")

        model.eval()
        val_loss      = 0.0
        n_val_batches = len(val_loader)
        inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
        union = torch.zeros(num_classes, dtype=torch.float64, device=device)
        sample_imgs: list[wandb.Image] = []

        with torch.no_grad():
            for batch_idx, (img_t, msk_t) in enumerate(val_loader):
                img_t = img_t.to(device, non_blocking=pin)
                msk_t = msk_t.to(device, non_blocking=pin)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(img_t)
                    if logits.shape[-2:] != msk_t.shape[-2:]:
                        logits = F.interpolate(logits, size=msk_t.shape[-2:], mode="bilinear", align_corners=False)
                    loss = criterion(logits, msk_t)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                for c in range(num_classes):
                    p_c = preds == c
                    g_c = msk_t == c
                    inter[c] += (p_c & g_c).sum()
                    union[c] += (p_c | g_c).sum()

                if is_main and batch_idx == 0 and not sample_imgs:
                    n_show  = min(4, img_t.shape[0])
                    img_np  = (img_t[:n_show].clamp(0, 1) * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
                    gt_np   = msk_t[:n_show].cpu().numpy()
                    pr_np   = preds[:n_show].cpu().numpy()
                    for j in range(n_show):
                        sample_imgs.append(wandb.Image(
                            img_np[j],
                            masks={
                                "ground_truth": {"mask_data": gt_np[j], "class_labels": _CLASS_LABELS},
                                "prediction":   {"mask_data": pr_np[j], "class_labels": _CLASS_LABELS},
                            },
                        ))

        if is_ddp:
            dist.all_reduce(inter, op=dist.ReduceOp.SUM)
            dist.all_reduce(union, op=dist.ReduceOp.SUM)
            tl_t = torch.tensor([train_loss, float(n_train_batches)], device=device)
            vl_t = torch.tensor([val_loss,   float(n_val_batches)],   device=device)
            dist.all_reduce(tl_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(vl_t, op=dist.ReduceOp.SUM)
            avg_train = tl_t[0].item() / max(tl_t[1].item(), 1)
            avg_val   = vl_t[0].item() / max(vl_t[1].item(), 1)
        else:
            avg_train = train_loss / max(n_train_batches, 1)
            avg_val   = val_loss   / max(n_val_batches, 1)

        iou       = torch.where(union > 0, inter / union.clamp(min=1), torch.full_like(inter, float("nan")))
        iou_cpu   = iou.cpu().tolist()
        valid_iou = [v for v in iou_cpu if v == v]
        miou      = sum(valid_iou) / len(valid_iou) if valid_iou else float("nan")

        if is_main:
            print(
                f"Epoch {epoch}/{epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
                f"mIoU={miou:.4f}  per-class={['nan' if v != v else f'{v:.3f}' for v in iou_cpu]}"
            )

            is_best = avg_val < best_val_loss
            if is_best:
                best_val_loss = avg_val
                state = model.module.state_dict() if is_ddp else model.state_dict()
                torch.save(state, best_path)
                print(f"  -> saved best checkpoint ({best_path})")

            log_payload: dict = {
                "epoch": epoch,
                "train/loss": avg_train,
                "val/loss": avg_val,
                "val/best_loss": best_val_loss,
                "val/mIoU": miou,
                "val/samples": sample_imgs,
            }
            for c, v in enumerate(iou_cpu):
                log_payload[f"val/IoU_{_CLASS_LABELS.get(c, str(c))}"] = v
            wandb.log(log_payload)

        if is_ddp:
            dist.barrier()

    if is_main:
        if best_path.exists():
            artifact = wandb.Artifact(
                name=f"{model_key}_bdd100k", type="model",
                metadata={"val_loss": best_val_loss, "epochs": epochs},
            )
            artifact.add_file(str(best_path))
            wandb.log_artifact(artifact)
        wandb.finish()

    if is_ddp:
        dist.destroy_process_group()

    return best_path
