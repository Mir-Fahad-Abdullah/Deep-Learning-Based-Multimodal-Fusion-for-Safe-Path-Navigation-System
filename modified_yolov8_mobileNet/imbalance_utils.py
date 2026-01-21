# imbalance_utils.py
from __future__ import annotations
from pathlib import Path
from collections import Counter
import random

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _all_images(img_dir: Path):
    img_dir = Path(img_dir)
    for p in img_dir.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            yield p.resolve()

def _label_path_for(img_path: Path, labels_dir: Path) -> Path:
    return Path(labels_dir) / (img_path.stem + ".txt")

def compute_class_frequencies(train_img_dir: Path, train_lbl_dir: Path):
    """Return (class_counts, per_image_counts)"""
    class_counts = Counter()
    per_image_counts: dict[Path, Counter] = {}
    for img_path in _all_images(Path(train_img_dir)):
        lbl_path = _label_path_for(img_path, train_lbl_dir)
        c = Counter()
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip().split()
                    if len(t) >= 5:
                        try:
                            cid = int(float(t[0]))
                        except Exception:
                            continue
                        class_counts[cid] += 1
                        c[cid] += 1
        per_image_counts[img_path] = c
    return class_counts, per_image_counts

def build_oversampled_manifest(
    train_img_dir: Path,
    train_lbl_dir: Path,
    target_per_class: int = 1000,
    min_multiplier: float = 1.0,
    max_multiplier: float = 6.0,
    rng_seed: int | None = 42,
):
    """Repeat images containing rare classes (size-aware oversampling)."""
    if rng_seed is not None:
        random.seed(rng_seed)

    class_counts, per_image_counts = compute_class_frequencies(train_img_dir, train_lbl_dir)
    if not per_image_counts:
        return []

    weights = {}
    for cid, cnt in class_counts.items():
        cnt = max(cnt, 1)
        weights[cid] = max(1.0, float(target_per_class) / float(cnt))

    manifest: list[str] = []
    for img_path, counts in per_image_counts.items():
        if not counts:
            manifest.append(str(img_path))
            continue
        f = 1.0
        for cid in counts.keys():
            f = max(f, weights.get(cid, 1.0))
        f = max(min_multiplier, min(max_multiplier, f))
        reps = int(round(f))
        if reps < f and random.random() < (f - reps):
            reps += 1
        reps = max(1, min(int(max_multiplier), reps))
        for _ in range(reps):
            manifest.append(str(img_path))
    return manifest

def write_manifest_txt(paths: list[str] | list[Path], out_txt: Path):
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(Path(p).resolve()) + "\n")
