import csv
import random
from pathlib import Path

import torch


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_names(config):
    return list(config["labels"]["target"]) + list(config["labels"]["non_target"])


def get_label_map(config):
    label_map = {}
    for name in config["labels"]["target"]:
        label_map[name] = 1
    for name in config["labels"]["non_target"]:
        label_map[name] = 0
    return label_map


def get_class_to_index(config):
    class_names = get_class_names(config)
    return {name: idx for idx, name in enumerate(class_names)}


def get_class_to_binary(config):
    label_map = get_label_map(config)
    class_names = get_class_names(config)
    return [label_map[name] for name in class_names]


def apply_label_alias(label, config):
    alias_map = config.get("labels", {}).get("aliases", {})
    return alias_map.get(label, label)


def read_index_csv(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def write_index_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "class_name", "class_index", "binary_label"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def stratified_split(rows, label_key, val_split, seed):
    rng = random.Random(seed)
    buckets = {}
    for row in rows:
        label = row[label_key]
        buckets.setdefault(label, []).append(row)

    train_rows = []
    val_rows = []
    for _, items in buckets.items():
        rng.shuffle(items)
        if len(items) <= 1:
            n_val = 0
        else:
            n_val = int(round(len(items) * val_split))
            n_val = max(1, n_val)
            if n_val >= len(items):
                n_val = len(items) - 1
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def resolve_device(device_setting):
    if device_setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting
