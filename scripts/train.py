"""
Training Script for HCN-TA.

Usage:
    python scripts/train.py --config configs/asvspoof2019.yaml --data_dir data/ASVspoof2019/LA
    python scripts/train.py --config configs/for_dataset.yaml --data_dir data/FoR --dataset for
"""

import os, sys, argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import HCNTA, MarginLoss
from datasets.asvspoof2019 import ASVspoof2019Builder
from datasets.for_dataset import FoRBuilder
from utils.metrics import compute_metrics, compute_accuracy
from utils.logger import TrainingLogger


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, labels_all, preds_all = 0.0, [], []
    for imgs, lbls in tqdm(loader, desc="Train", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        v = model(imgs)
        loss = criterion(v, lbls)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        norms = torch.sqrt((v**2).sum(-1) + 1e-8)
        labels_all.extend(lbls.cpu().numpy())
        preds_all.extend(norms.argmax(1).cpu().detach().numpy())
    return loss_sum / len(loader.dataset), compute_accuracy(labels_all, preds_all)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, labels_all, preds_all, scores_all = 0.0, [], [], []
    for imgs, lbls in tqdm(loader, desc="Eval", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        v = model(imgs)
        loss_sum += criterion(v, lbls).item() * imgs.size(0)
        norms = torch.sqrt((v**2).sum(-1) + 1e-8)
        labels_all.extend(lbls.cpu().numpy())
        preds_all.extend(norms.argmax(1).cpu().numpy())
        scores_all.extend(norms[:, 1].cpu().numpy())
    m = compute_metrics(labels_all, preds_all, scores_all)
    m["loss"] = loss_sum / len(loader.dataset)
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="experiments/default")
    p.add_argument("--dataset", default="asvspoof2019", choices=["asvspoof2019", "for"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}\n  HCN-TA Training | {args.dataset} | {device}\n{'='*60}\n")

    model = HCNTA(num_classes=2, pretrained_backbone=True, locality_awareness=True).to(device)
    total, trainable = model.get_num_params()
    print(f"Parameters: {total:,} total, {trainable:,} trainable\n")

    criterion = MarginLoss(m_plus=0.9, m_minus=0.1, lambda_val=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.dataset == "asvspoof2019":
        builder = ASVspoof2019Builder(args.data_dir)
        train_ds, val_ds = builder.get_train(), builder.get_dev()
    else:
        builder = FoRBuilder(args.data_dir)
        ds = builder.build()
        train_ds, val_ds = ds["train"], ds["val"]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logger = TrainingLogger(args.output_dir, experiment_name="hcn_ta")
    logger.log(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    best_acc, patience, patience_limit = 0.0, 0, 15
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        logger.log_epoch(epoch, tr_loss, val_m["loss"], val_m["accuracy"], val_m.get("eer"))

        is_best = val_m["accuracy"] > best_acc
        if is_best: best_acc = val_m["accuracy"]; patience = 0
        else: patience += 1
        logger.save_checkpoint(model, optimizer, epoch, val_m["accuracy"], is_best)
        if patience >= patience_limit:
            logger.log(f"Early stopping at epoch {epoch}"); break

    logger.log(f"\nBest accuracy: {best_acc:.2f}%"); logger.close()

if __name__ == "__main__":
    main()
