"""
🆕 IMPROVED EVALUATION with Test-Time Augmentation (TTA)

Key improvements:
1. Test-Time Augmentation for robust predictions
2. Weighted voting for long audio segments
3. Better confidence estimation
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
 
import torch
import torch.nn.functional as F

from soundclf.config import load_config
from soundclf.data import AudioPreprocessor, load_audio
from soundclf.model import AudioCNN
from soundclf.utils import apply_label_alias, get_class_names, get_class_to_binary, resolve_device


def save_classification_report(y_true, y_pred, class_names, output_base_path: Path):
    """Save classification report in both txt and json formats"""
    report_txt = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    txt_path = output_base_path.with_suffix(".report.txt")
    json_path = output_base_path.with_suffix(".report.json")

    txt_path.write_text(report_txt, encoding="utf-8")
    json_path.write_text(json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Classification report saved to {txt_path}")
    print(f"Classification report (json) saved to {json_path}")


# ========================================
# 🆕 IMPROVEMENT 1: Test-Time Augmentation
# ========================================
def apply_tta_augmentation(waveform, sample_rate, aug_type):
    """
    Apply light augmentation for TTA
    
    Args:
        waveform: input audio
        sample_rate: sampling rate
        aug_type: 0=original, 1=gain_up, 2=gain_down, 3=shift_left, 4=shift_right
    
    Returns:
        augmented waveform
    """
    if aug_type == 0:
        return waveform
    
    elif aug_type == 1:  # Gain up
        return waveform * 1.1
    
    elif aug_type == 2:  # Gain down
        return waveform * 0.9
    
    elif aug_type == 3:  # Shift left (slight time shift)
        shift_amount = int(sample_rate * 0.05)  # 50ms
        return torch.roll(waveform, shifts=-shift_amount, dims=-1)
    
    elif aug_type == 4:  # Shift right
        shift_amount = int(sample_rate * 0.05)
        return torch.roll(waveform, shifts=shift_amount, dims=-1)
    
    return waveform


def predict_with_tta(model, preprocessor, device, waveform, sample_rate, n_tta=5):
    """
    Predict with Test-Time Augmentation
    
    Args:
        model: trained model
        preprocessor: audio preprocessor
        device: torch device
        waveform: input audio waveform
        sample_rate: sampling rate
        n_tta: number of TTA iterations (default: 5)
    
    Returns:
        averaged probabilities across all TTA iterations
    """
    all_probs = []
    
    with torch.no_grad():
        for aug_idx in range(n_tta):
            # Apply augmentation
            aug_waveform = apply_tta_augmentation(waveform, sample_rate, aug_idx)
            
            # Preprocess
            features = preprocessor(aug_waveform, sample_rate).unsqueeze(0).to(device)
            
            # Predict
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
    
    # Average predictions
    avg_probs = torch.stack(all_probs).mean(dim=0).squeeze(0)
    return avg_probs


# ========================================
# 🆕 IMPROVEMENT 2: Weighted Voting for Long Audio
# ========================================
def gaussian_weights(n_windows):
    """
    Generate gaussian weights that give more importance to center windows
    
    Args:
        n_windows: number of windows
    
    Returns:
        normalized weights
    """
    if n_windows == 1:
        return torch.tensor([1.0])
    
    # Create gaussian distribution centered at middle
    x = torch.linspace(-2, 2, n_windows)
    weights = torch.exp(-x**2)
    weights = weights / weights.sum()
    
    return weights


def predict_long_audio_with_tta(model, preprocessor, device, audio_path, n_tta=5, use_weighted_voting=True):
    """
    🆕 IMPROVED: Predict long audio with TTA + Weighted Voting
    
    BEFORE: Simple average of window predictions
    AFTER:  TTA on each window + Gaussian weighted voting
    """
    waveform, sample_rate = load_audio(audio_path)

    if waveform is None or waveform.numel() == 0:
        return None, None
    if waveform.abs().max() < 1e-4:
        return None, None

    clip_samples = int(sample_rate * preprocessor.clip_samples / sample_rate) if hasattr(preprocessor, 'clip_samples') else int(sample_rate * 10)

    total_len = waveform.size(-1)
    
    # Create windows
    if total_len <= clip_samples:
        windows = [0]
    else:
        stride = clip_samples // 2
        windows = list(range(0, total_len - clip_samples + 1, stride))
        if len(windows) < 3:
            windows = [0, (total_len - clip_samples) // 2, total_len - clip_samples]
        elif len(windows) > 10:
            step = len(windows) // 10
            windows = windows[::step][:10]

    all_window_probs = []

    # Process each window with TTA
    for start in windows:
        end = min(start + clip_samples, total_len)
        chunk = waveform[..., start:end]

        if chunk.size(-1) < clip_samples:
            pad = clip_samples - chunk.size(-1)
            chunk = F.pad(chunk, (0, pad))

        try:
            # 🆕 Apply TTA to this window
            window_probs = predict_with_tta(model, preprocessor, device, chunk, sample_rate, n_tta)
            all_window_probs.append(window_probs)

        except Exception as e:
            print(f"  Window {start} error: {e}")
            continue

    if not all_window_probs:
        return None, None

    # 🆕 Weighted voting
    if use_weighted_voting and len(all_window_probs) > 1:
        weights = gaussian_weights(len(all_window_probs)).to(all_window_probs[0].device)
        avg_probs = sum(p * w for p, w in zip(all_window_probs, weights))
    else:
        avg_probs = torch.stack(all_window_probs).mean(dim=0)

    pred_idx = int(avg_probs.argmax().item())
    pred_prob = float(avg_probs[pred_idx].item())

    return pred_idx, pred_prob


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(24, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, 
           yticklabels=class_names,
           title='Confusion Matrix (with TTA)',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    fontsize=10,
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion Matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluation with TTA")
    parser.add_argument("--config", default="soundclf/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--root", default="/home/work/KSW/sound_classification/20260123_test1", 
                       help="Root folder containing RMS folders")
    parser.add_argument("--output", default=None, help="Optional CSV to store predictions")
    parser.add_argument("--device", default=None, help="Device override")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    parser.add_argument("--pooling", default="gap", choices=["gap", "ssrp_t"], help="Pooling type for model")
    
    # ========================================
    # 🆕 NEW ARGUMENTS
    # ========================================
    parser.add_argument("--n-tta", type=int, default=5, 
                       help="Number of TTA iterations (default: 5, set 1 to disable)")
    parser.add_argument("--disable-weighted-voting", action="store_true",
                       help="Disable gaussian weighted voting for long audio")
    
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    class_names = get_class_names(config)
    class_to_binary = get_class_to_binary(config)
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    device = resolve_device(args.device or config["train"].get("device", "auto"))

    # Load Model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = AudioCNN(n_mels=model_cfg["n_mels"], num_classes=len(class_names), pooling=args.pooling).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    
    if "class_names" in checkpoint:
        ckpt_classes = checkpoint["class_names"]
        if ckpt_classes == class_names:
            print(f"Restored {len(class_names)} classes from checkpoint")
        else:
            print(f"Warning: Checkpoint has {len(ckpt_classes)} classes, config has {len(class_names)}")

    # Preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=data_cfg["sample_rate"],
        clip_seconds=data_cfg["clip_seconds"],
        n_mels=model_cfg["n_mels"],
        n_fft=model_cfg["n_fft"],
        hop_length=model_cfg["hop_length"],
        win_length=model_cfg["win_length"],
        training=False,
        fixed_length=True,
    )

    print(f"\n✨ Evaluation Settings:")
    print(f"  TTA iterations: {args.n_tta}")
    print(f"  Weighted voting: {'Enabled' if not args.disable_weighted_voting else 'Disabled'}")

    root = Path(args.root)
    parent_folders = sorted([p for p in root.glob("RMS*_split") if p.is_dir()])
    
    if not parent_folders:
        print(f"Warning: No *_split folders found in {root}")
        return
    
    print(f"\nFound {len(parent_folders)} split folders.")

    # Global statistics
    total_global = 0
    correct_global = 0
    total_binary_global = 0
    correct_binary_global = 0
    binary_fp_global = 0
    binary_tn_global = 0
    rows = []

    all_preds = []
    all_targets = []

    print(f"\n{'='*80}")
    print(f"{'Folder Name':<25} | {'Samples':<8} | {'Top-1 Acc':<10} | {'Binary Acc':<10}")
    print(f"{'-'*80}")

    for parent_dir in parent_folders:
        json_path = parent_dir / "timestamp_labels.json"
        
        if not json_path.exists():
            print(f"Skipping {parent_dir.name}: No timestamp_labels.json")
            continue

        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        folder_total = 0
        folder_correct = 0
        folder_binary_total = 0
        folder_binary_correct = 0
        folder_binary_fp = 0
        folder_binary_tn = 0

        for key, entry in data.items():
            raw_label = entry.get("label", None)
            if raw_label is None:
                continue

            label = apply_label_alias(raw_label, config)
            if label not in class_to_index:
                if args.debug:
                    print(f"  Unknown label: {label}")
                continue

            audio_path = parent_dir / f"audio_{key}.wav"
            if not audio_path.exists():
                continue

            try:
                # 🆕 Use improved prediction with TTA
                pred_idx, pred_prob = predict_long_audio_with_tta(
                    model, preprocessor, device, audio_path, 
                    n_tta=args.n_tta,
                    use_weighted_voting=not args.disable_weighted_voting
                )
                
                if pred_idx is None:
                    if args.debug:
                        print(f"  Skipped {audio_path.name}: prediction failed")
                    continue

                gt_idx = class_to_index[label]
                gt_binary = class_to_binary[gt_idx]
                binary_pred = class_to_binary[pred_idx]

                all_preds.append(pred_idx)
                all_targets.append(gt_idx)

                folder_total += 1
                if pred_idx == gt_idx:
                    folder_correct += 1

                folder_binary_total += 1
                if binary_pred == gt_binary:
                    folder_binary_correct += 1
                if gt_binary == 0 and binary_pred == 1:
                    folder_binary_fp += 1
                elif gt_binary == 0 and binary_pred == 0:
                    folder_binary_tn += 1

                total_global += 1
                if pred_idx == gt_idx:
                    correct_global += 1
                total_binary_global += 1
                if binary_pred == gt_binary:
                    correct_binary_global += 1
                if gt_binary == 0 and binary_pred == 1:
                    binary_fp_global += 1
                elif gt_binary == 0 and binary_pred == 0:
                    binary_tn_global += 1

                if args.output:
                    rows.append({
                        "path": str(audio_path),
                        "folder": parent_dir.name,
                        "gt_label": raw_label,
                        "gt_label_mapped": label,
                        "pred_label": class_names[pred_idx],
                        "pred_prob": f"{pred_prob:.6f}",
                        "binary_gt": "target" if gt_binary == 1 else "non_target",
                        "binary_pred": "target" if binary_pred == 1 else "non_target",
                    })

            except Exception as e:
                if args.debug:
                    print(f"Error processing {audio_path.name}: {e}")
                continue

        acc = folder_correct / folder_total if folder_total > 0 else 0.0
        bin_acc = folder_binary_correct / folder_binary_total if folder_binary_total > 0 else 0.0
        bin_fpr = (folder_binary_fp / (folder_binary_fp + folder_binary_tn)) if (folder_binary_fp + folder_binary_tn) > 0 else 0.0
        print(f"{parent_dir.name:<25} | {folder_total:<8} | {acc:.4f}     | {bin_acc:.4f} | FPR: {bin_fpr:.4f}")

    print(f"{'='*80}")

    if total_global == 0:
        print("No samples evaluated.")
        return

    global_acc = correct_global / total_global
    global_bin_acc = correct_binary_global / total_binary_global if total_binary_global > 0 else 0.0
    global_bin_fpr = (binary_fp_global / (binary_fp_global + binary_tn_global)) if (binary_fp_global + binary_tn_global) > 0 else 0.0

    print(f"\n>>> [IMPROVED Result with TTA]")
    print(f"Evaluated {total_global} samples across {len(parent_folders)} folders")
    print(f"Top-1 Accuracy  : {global_acc:.4f} ({correct_global}/{total_global})")
    print(f"Binary Accuracy : {global_bin_acc:.4f} ({correct_binary_global}/{total_binary_global})")
    print(f"Binary FPR      : {global_bin_fpr:.4f} (FP={binary_fp_global}, TN={binary_tn_global})")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("path,folder,gt_label,gt_label_mapped,pred_label,pred_prob,binary_gt,binary_pred\n")
            for row in rows:
                handle.write(
                    f"{row['path']},{row['folder']},{row['gt_label']},{row['gt_label_mapped']},"
                    f"{row['pred_label']},{row['pred_prob']},"
                    f"{row['binary_gt']},{row['binary_pred']}\n"
                )
        print(f"\nCSV saved to {output_path}")

        # Confusion Matrix
        cm_path = output_path.with_name("confusion_matrix_tta.png")
        plot_confusion_matrix(all_targets, all_preds, class_names, cm_path)
        save_classification_report(all_targets, all_preds, class_names, output_path)


if __name__ == "__main__":
    main()
