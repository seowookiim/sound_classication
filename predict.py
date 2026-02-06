import argparse
import csv
from pathlib import Path

import torch

from soundclf.config import load_config
from soundclf.data import AudioPreprocessor, load_audio
from soundclf.model import AudioCNN
from soundclf.utils import get_class_names, get_class_to_binary, resolve_device


def predict_file(model, preprocessor, device, audio_path, class_names, class_to_binary, threshold):
    """
    단일 파일 예측
    
    Returns:
        pred_idx: 예측된 class index
        pred_prob: 예측된 class의 확률
        target_prob: target 확률 (binary head 사용)
        binary_label: "target" or "non_target"
    """
    waveform, sample_rate = load_audio(audio_path)
    features = preprocessor(waveform, sample_rate).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # ITFA 모델은 두 개의 output 반환
        logits_multi, logits_binary = model(features)
        
        # Multi-class prediction
        probs_multi = torch.softmax(logits_multi, dim=1).squeeze(0)
        pred_idx = int(probs_multi.argmax().item())
        pred_prob = float(probs_multi[pred_idx].item())
        
        # Binary prediction (직접 binary head 사용)
        probs_binary = torch.softmax(logits_binary, dim=1).squeeze(0)
        target_prob = float(probs_binary[1].item())  # target class 확률
        binary_label = "target" if target_prob >= threshold else "non_target"
    
    return pred_idx, pred_prob, target_prob, binary_label


def main():
    parser = argparse.ArgumentParser(description="Run inference on audio files.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Audio file or folder")
    parser.add_argument("--output", default=None, help="Output CSV for folder input")
    parser.add_argument("--threshold", type=float, default=0.5, help="Target decision threshold")
    parser.add_argument("--device", default=None, help="Device override")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    class_names = get_class_names(config)
    class_to_binary = get_class_to_binary(config)

    device = resolve_device(args.device or config["train"].get("device", "auto"))

    # Load model
    model = AudioCNN(n_mels=model_cfg["n_mels"], num_classes=len(class_names)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    
    print(f"Loaded model from {args.checkpoint}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "best_val_loss" in checkpoint:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    if "best_val_acc_multi" in checkpoint:
        print(f"  Best multi-class acc: {checkpoint['best_val_acc_multi']:.4f}")
    if "best_val_acc_binary" in checkpoint:
        print(f"  Best binary acc: {checkpoint['best_val_acc_binary']:.4f}")

    # Preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=data_cfg["sample_rate"],
        clip_seconds=data_cfg["clip_seconds"],
        n_mels=model_cfg["n_mels"],
        n_fft=model_cfg["n_fft"],
        hop_length=model_cfg["hop_length"],
        training=False,
        fixed_length=True,  # ITFA는 고정 길이
    )

    input_path = Path(args.input)
    audio_ext = data_cfg["audio_ext"].lower()
    
    if input_path.is_dir():
        # Directory: Process all audio files
        output_csv = Path(args.output or "predictions.csv")
        rows = []
        
        print(f"\nProcessing directory: {input_path}")
        audio_files = sorted(input_path.rglob(f"*{audio_ext}"))
        print(f"Found {len(audio_files)} audio files")
        
        for idx, path in enumerate(audio_files, 1):
            if not path.is_file():
                continue
                
            try:
                pred_idx, pred_prob, target_prob, binary_label = predict_file(
                    model, preprocessor, device, path,
                    class_names, class_to_binary, args.threshold,
                )
                
                rows.append({
                    "path": str(path),
                    "class_name": class_names[pred_idx],
                    "class_prob": f"{pred_prob:.6f}",
                    "target_prob": f"{target_prob:.6f}",
                    "binary_label": binary_label,
                })
                
                if idx % 100 == 0:
                    print(f"  Processed {idx}/{len(audio_files)}")
                    
            except Exception as e:
                print(f"  Error processing {path}: {e}")
                continue
        
        # Save results
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["path", "class_name", "class_prob", "target_prob", "binary_label"],
            )
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\nWrote {len(rows)} predictions to {output_csv}")
        
        # Summary statistics
        target_count = sum(1 for r in rows if r["binary_label"] == "target")
        print(f"Summary:")
        print(f"  Target: {target_count}")
        print(f"  Non-target: {len(rows) - target_count}")
        
    else:
        # Single file
        pred_idx, pred_prob, target_prob, binary_label = predict_file(
            model, preprocessor, device, input_path,
            class_names, class_to_binary, args.threshold,
        )
        pred_class = class_names[pred_idx]
        
        print(f"\nPrediction for: {input_path}")
        print(f"  Class: {pred_class}")
        print(f"  Class probability: {pred_prob:.4f}")
        print(f"  Target probability: {target_prob:.4f}")
        print(f"  Binary label: {binary_label}")


if __name__ == "__main__":
    main()