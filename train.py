# ======== 
# train root : --root /home/work/KSW/sound_classification/train
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np  # Mixup을 위해 추가

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
import yaml

from soundclf.config import load_config
from soundclf.data import AudioDataset
from soundclf.model import AudioCNN
from soundclf.utils import (
    get_class_names,
    get_class_to_binary,
    read_index_csv,
    resolve_device,
    seed_everything,
    stratified_split,
    write_index_csv,
)


def compute_class_weights(rows, num_classes):
    counts = [0] * num_classes
    for row in rows:
        counts[int(row["class_index"])] += 1
    total = sum(counts)
    weights = []
    for count in counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


# [필수] None(무음) 데이터 걸러내기
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# === Mixup 함수 ===
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def run_epoch(model, loader, criterion, optimizer, device, log_interval, mixup_alpha=0.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="Train" if is_train else "Val", leave=False)

    for step, batch in enumerate(pbar, start=1):
        if batch is None:
            continue

        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            if is_train and mixup_alpha > 0:
                features_mixed, y_a, y_b, lam = mixup_data(features, labels, mixup_alpha, device)
                logits = model(features_mixed)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(features)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

        if total_count > 0:
            avg_loss = total_loss / total_count
            avg_acc = total_correct / total_count
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.3f}"})

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train ITFA-based multi-task audio classifier with Mixup.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--root", default=None, help="Dataset root override")
    parser.add_argument("--index-csv", default=None, help="Index CSV override")
    parser.add_argument("--train-csv", default=None, help="Train CSV override")
    parser.add_argument("--val-csv", default=None, help="Val CSV override")
    parser.add_argument("--regenerate-splits", action="store_true", help="Force split regen")
    parser.add_argument("--run-name", default=None, help="Output run name")
    parser.add_argument("--device", default=None, help="Device override")
    
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for multi-class loss")
    parser.add_argument("--beta", type=float, default=0.4, help="Weight for binary loss")
    
    # [Mixup 옵션 추가] 기본값 0.4 (논문 권장값)
    parser.add_argument("--mixup-alpha", type=float, default=0.4, 
                       help="Mixup alpha value (default: 0.4). Set 0 to disable.")
    
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--use-sampler", action="store_true",
                    help="Use WeightedRandomSampler for class balancing")
    parser.add_argument("--use-class-weights", action="store_true",
                    help="Use class-weighted CrossEntropyLoss")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    root = Path(args.root or data_cfg["root"]).resolve()
    index_csv = Path(args.index_csv or data_cfg["index_csv"])
    train_csv = Path(args.train_csv or data_cfg["train_csv"])
    val_csv = Path(args.val_csv or data_cfg["val_csv"])

    if not index_csv.exists():
        raise SystemExit(f"Missing index CSV at {index_csv}")
    index_rows = read_index_csv(index_csv)
    if not index_rows:
        raise SystemExit(f"Index CSV is empty")

    seed_everything(train_cfg["seed"])

    if args.regenerate_splits or not train_csv.exists() or not val_csv.exists():
        train_rows, val_rows = stratified_split(index_rows, "class_name", train_cfg["val_split"], train_cfg["seed"])
        write_index_csv(train_rows, train_csv)
        write_index_csv(val_rows, val_csv)
        print(f"Wrote splits: {train_csv}, {val_csv}")

    device = resolve_device(args.device or train_cfg.get("device", "auto"))
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    class_names = get_class_names(config)
    class_to_binary = get_class_to_binary(config)
    silence_threshold = data_cfg.get("silence_threshold", 1e-4)
    
    # Train Dataset: 기존 증강(Time Stretch/Pitch Shift) + Mixup 준비
    train_ds = AudioDataset(
        csv_path=train_csv, root=root, sample_rate=data_cfg["sample_rate"],
        clip_seconds=data_cfg["clip_seconds"], n_mels=model_cfg["n_mels"],
        n_fft=model_cfg["n_fft"], hop_length=model_cfg["hop_length"], win_length = model_cfg["win_length"],
        training=True, segment_long=True, fixed_length=True,
        silence_threshold=silence_threshold,
        paper_augment=True, # 기존 증강 유지
    )
    train_ds.debug_save_audio = True
    train_ds.debug_dir = "debug_audio"

    # Val Dataset: 순정 상태
    val_ds = AudioDataset(
        csv_path=val_csv, root=root, sample_rate=data_cfg["sample_rate"],
        clip_seconds=data_cfg["clip_seconds"], n_mels=model_cfg["n_mels"],
        n_fft=model_cfg["n_fft"], hop_length=model_cfg["hop_length"], win_length = model_cfg["win_length"],
        training=False, segment_long=True, fixed_length=True,
        silence_threshold=silence_threshold,
        paper_augment=False, 
    )

    print(f"\nDataset Statistics:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples: {len(val_ds)}")

    # =========================
    # [New] Class-balanced sampler (WeightedRandomSampler)
    # =========================
    from collections import Counter
    train_labels = []
    for i in range(len(train_ds)):
        original_idx = i // 5 if getattr(train_ds, "paper_augment", False) else i
        train_labels.append(int(train_ds.samples[original_idx]['class_index']))

    class_counts = Counter(train_labels)
    sample_weights = torch.DoubleTensor([1.0 / class_counts[y] for y in train_labels])

    # 클래스별 count
    from collections import Counter

    sampler = None
    shuffle = True

    if args.use_sampler:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # sampler 쓰면 shuffle은 False

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=sampler,
        shuffle=shuffle,
        num_workers=data_cfg["num_workers"],
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"], pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    model = AudioCNN(n_mels=model_cfg["n_mels"], num_classes=len(class_names)).to(device)
    
    rows = train_ds.samples
    class_weights = compute_class_weights(rows, len(class_names)).to(device)

    if args.use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    
    print(f"\nTraining Config:")
    print(f"  Mixup Alpha: {args.mixup_alpha}")
    print(f"  Data Augmentation: True (Time Stretch / Pitch Shift)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"], eta_min=train_cfg["lr"] * 0.01)

    current_time_str = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"mixup_run_{current_time_str}"
    output_dir = Path("outputs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config 저장
    config_copy = config.copy()
    config_copy["training_override"] = {"alpha": args.alpha, "beta": args.beta, "mixup_alpha": args.mixup_alpha}
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_copy, handle, sort_keys=False)

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,val_loss,val_acc,lr\n")

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = output_dir / f"best_{current_time_str}.pt"
    last_model_path = output_dir / f"last_{current_time_str}.pt"

    print(f"\nStarting training with Mixup...")
    
    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"\n[Epoch {epoch}/{train_cfg['epochs']}]")
        
        # Train (Mixup O)
        train_stats = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train_cfg["log_interval"],
            mixup_alpha=args.mixup_alpha
        )
        
        # Val (Mixup X)
        val_stats = run_epoch(
            model, val_loader, criterion, None, device, 0,
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save Metrics
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{epoch},{train_stats[0]:.6f},{val_stats[0]:.6f},{val_stats[1]:.6f},{current_lr:.8f}\n")


        print(f"  Train Loss: {train_stats[0]:.4f} | Acc: {train_stats[1]:.4f}")
        print(f"  Val   Loss: {val_stats[0]:.4f} | Acc: {val_stats[1]:.4f}")

        if val_stats[0] < best_val_loss:
            best_val_loss = val_stats[0]
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "class_names": class_names, "class_to_binary": class_to_binary,
            }, best_model_path)
            print(f"  ✓ Best model saved! (val_loss: {val_stats[0]:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print("  Early stopping triggered.")
            break

    torch.save({"model_state": model.state_dict(), "class_names": class_names}, last_model_path)
    print(f"\nTraining completed! Saved to {output_dir}")

if __name__ == "__main__":
    main()