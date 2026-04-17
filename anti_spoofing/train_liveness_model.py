from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRAME_MANIFEST = ROOT / "data" / "anti_spoof_frame_manifest.csv"
DEFAULT_MODELS_DIR = ROOT / "data" / "anti_spoof_models"
DISPLAY_ATTACK_TYPES = {"screen_replay", "display_spoof"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model anti-spoofing real/fake.")
    parser.add_argument("--frame-manifest", type=Path, default=DEFAULT_FRAME_MANIFEST)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--display-attack-boost", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


class FrameDataset(Dataset):
    def __init__(self, rows: list[dict], transform=None) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label_id"])
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": row["relative_video_path"],
            "label_name": row["label"],
            "row": row,
        }


def create_transforms(image_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.03),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def build_model(pretrained: bool) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = ResNet18_Weights.DEFAULT
        except Exception:
            weights = None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        print(f"[Model] Gagal memuat pretrained weights ({exc}), fallback ke random init.")
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    train_rows = [row for row in rows if row["model_split"] == "train"]
    val_rows = [row for row in rows if row["model_split"] == "val"]
    test_rows = [row for row in rows if row["model_split"] == "test"]
    return train_rows, val_rows, test_rows


def make_train_sampler(rows: list[dict], display_attack_boost: float) -> WeightedRandomSampler:
    label_counts = Counter(int(row["label_id"]) for row in rows)
    sample_weights = []
    for row in rows:
        weight = 1.0 / label_counts[int(row["label_id"])]
        if row.get("attack_type") in DISPLAY_ATTACK_TYPES:
            weight *= display_attack_boost
        sample_weights.append(weight)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def class_weight_tensor(rows: list[dict], device: torch.device) -> torch.Tensor:
    label_counts = Counter(int(row["label_id"]) for row in rows)
    total = sum(label_counts.values())
    weights = []
    for label_id in [0, 1]:
        count = label_counts.get(label_id, 1)
        weights.append(total / (2.0 * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def metrics_from_arrays(labels: np.ndarray, probs_real: np.ndarray) -> dict:
    pred_labels = (probs_real >= 0.5).astype(np.int64)
    accuracy = float((pred_labels == labels).mean()) if len(labels) else 0.0

    tp = int(np.sum((pred_labels == 1) & (labels == 1)))
    tn = int(np.sum((pred_labels == 0) & (labels == 0)))
    fp = int(np.sum((pred_labels == 1) & (labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (labels == 1)))

    recall_real = tp / max(tp + fn, 1)
    recall_fake = tn / max(tn + fp, 1)
    precision_real = tp / max(tp + fp, 1)
    f1_real = 0.0 if precision_real + recall_real == 0 else 2 * precision_real * recall_real / (precision_real + recall_real)
    balanced_accuracy = (recall_real + recall_fake) / 2.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy),
        "precision_real": float(precision_real),
        "recall_real": float(recall_real),
        "recall_fake": float(recall_fake),
        "f1_real": float(f1_real),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def aggregate_video_metrics(labels: list[int], probs_real: list[float], video_ids: list[str]) -> dict:
    grouped_probs: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}
    for label, prob, video_id in zip(labels, probs_real, video_ids):
        grouped_probs[video_id].append(prob)
        grouped_labels[video_id] = label

    video_labels = np.array([grouped_labels[video_id] for video_id in grouped_probs], dtype=np.int64)
    video_probs = np.array([float(np.mean(grouped_probs[video_id])) for video_id in grouped_probs], dtype=np.float32)
    metrics = metrics_from_arrays(video_labels, video_probs)
    metrics["num_videos"] = int(len(video_labels))
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> dict:
    model.eval()
    losses = []
    all_labels: list[int] = []
    all_probs_real: list[float] = []
    all_video_ids: list[str] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        losses.append(float(loss.item()))
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs_real.extend(probs.cpu().numpy().tolist())
        all_video_ids.extend(batch["video_id"])

    labels_np = np.array(all_labels, dtype=np.int64)
    probs_np = np.array(all_probs_real, dtype=np.float32)

    frame_metrics = metrics_from_arrays(labels_np, probs_np)
    video_metrics = aggregate_video_metrics(all_labels, all_probs_real, all_video_ids)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "frame": frame_metrics,
        "video": video_metrics,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.autocast(device_type="cuda", enabled=True) if device.type == "cuda" else nullcontext()
        with amp_context:
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    rows = read_rows(args.frame_manifest)
    train_rows, val_rows, test_rows = split_rows(rows)
    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError("Split train/val/test tidak lengkap. Jalankan prepare_video_frames.py dulu.")

    run_name = args.run_name.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.models_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = create_transforms(args.image_size)
    train_ds = FrameDataset(train_rows, transform=train_tf)
    val_ds = FrameDataset(val_rows, transform=eval_tf)
    test_ds = FrameDataset(test_rows, transform=eval_tf)

    sampler = make_train_sampler(train_rows, display_attack_boost=args.display_attack_boost)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(train_rows, device=device))
    scaler = GradScaler(enabled=device.type == "cuda")

    config = {
        "run_name": run_name,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "pretrained": bool(args.pretrained),
        "display_attack_boost": args.display_attack_boost,
        "seed": args.seed,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "train_label_counts": dict(Counter(row["label"] for row in train_rows)),
        "val_label_counts": dict(Counter(row["label"] for row in val_rows)),
        "test_label_counts": dict(Counter(row["label"] for row in test_rows)),
    }
    save_json(run_dir / "config.json", config)

    history = []
    best_score = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_payload)

        score = val_metrics["video"]["balanced_accuracy"]
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_frame_acc={val_metrics['frame']['accuracy']:.4f} "
            f"val_video_bal_acc={score:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
            run_dir / "last_model.pt",
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                run_dir / "best_model.pt",
            )

    best_checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    metrics_payload = {
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_video_balanced_accuracy": best_score,
        "test": test_metrics,
    }
    save_json(run_dir / "metrics.json", metrics_payload)

    print("\n[Test Metrics]")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nArtifact terbaik: {run_dir / 'best_model.pt'}")
    print(f"Metrics: {run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
