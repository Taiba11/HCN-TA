"""
Evaluation Script for HCN-TA (including cross-dataset on ASVspoof 2021).

Usage:
    python scripts/evaluate.py --checkpoint experiments/best_model.pth --data_dir data/ASVspoof2019/LA
    python scripts/evaluate.py --checkpoint experiments/best_model.pth --data_dir data/ASVspoof2021/LA --dataset asvspoof2021
"""

import os, sys, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import HCNTA, MarginLoss
from datasets.asvspoof2019 import ASVspoof2019Builder
from datasets.for_dataset import FoRBuilder
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix


@torch.no_grad()
def evaluate_dataset(model, loader, device):
    model.eval()
    labels_all, preds_all, scores_all = [], [], []
    for imgs, lbls in tqdm(loader, desc="Evaluating"):
        imgs = imgs.to(device)
        v = model(imgs)
        norms = torch.sqrt((v ** 2).sum(-1) + 1e-8)
        labels_all.extend(lbls.numpy())
        preds_all.extend(norms.argmax(1).cpu().numpy())
        scores_all.extend(norms[:, 1].cpu().numpy())
    return compute_metrics(labels_all, preds_all, scores_all)


def main():
    p = argparse.ArgumentParser(description="Evaluate HCN-TA")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dataset", default="asvspoof2019",
                    choices=["asvspoof2019", "for", "asvspoof2021"])
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = HCNTA(num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    if args.dataset in ("asvspoof2019", "asvspoof2021"):
        builder = ASVspoof2019Builder(args.data_dir)
        test_ds = builder.get_eval()
    else:
        builder = FoRBuilder(args.data_dir)
        test_ds = builder.build()["test"]

    loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)
    metrics = evaluate_dataset(model, loader, device)

    print(f"\n{'='*60}")
    print(f"  HCN-TA Evaluation | {args.dataset}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1']:.2f}%")
    print(f"  EER:       {metrics.get('eer', 0):.4f}%")
    print(f"{'='*60}\n")

    with open(os.path.join(args.output_dir, f"eval_{args.dataset}.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    if "confusion_matrix" in metrics:
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            save_path=os.path.join(args.output_dir, "confusion_matrix.png"),
        )


if __name__ == "__main__":
    main()
