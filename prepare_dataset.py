import argparse
from collections import Counter
from pathlib import Path

from soundclf.config import load_config
from soundclf.utils import get_class_to_index, get_label_map, write_index_csv


def collect_audio_paths(root, label_map, class_to_index, audio_ext):
    rows = []
    for label, binary in label_map.items():
        label_dir = root / label
        if not label_dir.exists():
            print(f"Warning: missing label folder: {label_dir}")
            continue
        for path in label_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() != audio_ext:
                continue
            rel_path = path.relative_to(root)
            rows.append(
                {
                    "path": str(rel_path),
                    "class_name": label,
                    "class_index": str(class_to_index[label]),
                    "binary_label": str(binary),
                }
            )
    rows.sort(key=lambda item: item["path"])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build index CSV from folder labels.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--root", default=None, help="Dataset root (overrides config)")
    parser.add_argument("--output", default=None, help="Output CSV (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.root or config["data"]["root"]).resolve()
    output_csv = Path(args.output or config["data"]["index_csv"])
    audio_ext = config["data"]["audio_ext"].lower()
    label_map = get_label_map(config)
    class_to_index = get_class_to_index(config)

    rows = collect_audio_paths(root, label_map, class_to_index, audio_ext)
    if not rows:
        raise SystemExit("No audio files found. Check your folder names and config.")

    write_index_csv(rows, output_csv)

    class_counts = Counter(row["class_name"] for row in rows)
    target_count = sum(1 for row in rows if row["binary_label"] == "1")
    non_target_count = len(rows) - target_count
    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Target: {target_count} | Non-target: {non_target_count}")
    for name in sorted(class_counts):
        print(f"{name}: {class_counts[name]}")


if __name__ == "__main__":
    main()
