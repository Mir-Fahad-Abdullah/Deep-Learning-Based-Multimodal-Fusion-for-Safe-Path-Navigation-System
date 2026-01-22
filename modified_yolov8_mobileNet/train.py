# train.py
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import yaml
import shutil

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("ULTRALYTICS_WORKERS", "2")

try:
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

import torch
from ultralytics import YOLO
from ultralytics.nn import tasks as yolo_tasks

# (Optional) register custom layers if you have custom_layers.py
try:
    import custom_layers as cl
    for _name in [
        "TimmBackboneC2C5", "TimmGhostBackboneC2C5",
        "BiFPNLite", "C2fDW", "SPPFLite",
        "Index0", "Index1", "Index2", "Index3",
    ]:
        if hasattr(cl, _name):
            setattr(yolo_tasks, _name, getattr(cl, _name))
except Exception:
    pass

# Imbalance helpers
from imbalance_utils import (
    compute_class_frequencies,
    build_oversampled_manifest,
    write_manifest_txt,
)

# ---------------- Paths to edit -----------------
ROOT = Path(__file__).resolve().parent
DATA_YAML = os.environ.get("DATA_YAML", r"/content/drive/MyDrive/modified_yolov8_mobileNetv3/SafeWalkBD-1/data.yaml")
MODEL_YAML = str(ROOT / "yolov8n_mbv3_pan_p2.yaml")   # keep your P2 model
RUNS_DIR   = str(ROOT / "runs")

# Stage names
WARMUP_NAME   = "warmup_448"
STAGE1A_NAME  = "stage1a_512_frozen15"
STAGE1B_NAME  = "stage1b_512_unfrozen"
STAGE2_NAME   = "stage2_ms960"

FAST_WARMUP = True
GLOBAL_SEED = 42

# -------------- utilities --------------
def set_seeds(seed=42):
    try:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def read_dataset_info(data_yaml_path: str):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    names = d.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        d["names"] = names
    nc = int(d.get("nc", len(names)))
    return nc, names, d

def overwrite_nc_in_model_yaml(model_yaml_path: str, nc: int):
    ypath = Path(model_yaml_path)
    if not ypath.exists():
        raise FileNotFoundError(f"Model YAML not found: {ypath}")
    with open(ypath, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    y["nc"] = int(nc)
    if "head" in y and isinstance(y["head"], list):
        for i, layer in enumerate(y["head"]):
            if len(layer) >= 4 and layer[2] == "Detect":
                args = layer[3]
                if isinstance(args, list) and args:
                    args[0] = "nc"
                    y["head"][i][3] = args
    with open(ypath, "w", encoding="utf-8") as f:
        yaml.safe_dump(y, f, sort_keys=False)

def resolve_dataset_paths(data_yaml_path: str):
    ypath = Path(data_yaml_path).resolve()
    if not ypath.is_file():
        raise FileNotFoundError(f"data.yaml not found: {ypath}")
    with open(ypath, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    yaml_dir = ypath.parent

    def to_abs(p):
        if p is None: return None
        p = Path(p)
        return p if p.is_absolute() else (yaml_dir / p).resolve()

    base = d.get("path")
    base_root = to_abs(base) if base else yaml_dir

    def to_abs_with_base(p):
        if p is None: return None
        p = Path(p)
        return p if p.is_absolute() else (base_root / p).resolve()

    train_imgs = to_abs_with_base(d.get("train"))
    val_imgs   = to_abs_with_base(d.get("val") or d.get("valid"))
    if not train_imgs or not val_imgs:
        raise ValueError("data.yaml must define 'train' and 'val' (or 'valid').")

    def derive_labels(images_dir: Path) -> Path:
        return images_dir.parent / "labels"

    train_lbls = derive_labels(train_imgs)
    val_lbls   = derive_labels(val_imgs)
    for p in [train_imgs, train_lbls, val_imgs, val_lbls]:
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset path: {p}")
    return train_imgs, train_lbls, val_imgs, val_lbls, d

def fix_label_file(lbl_path: Path, nc: int) -> tuple[int, int]:
    """Clamp boxes to [0..1], drop malformed rows, fix class ids outside range.
    Returns (#kept, #dropped)."""
    kept = 0; dropped = 0
    if not lbl_path.exists():
        return kept, dropped
    rows = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().split()
            if len(t) < 5: 
                dropped += 1; 
                continue
            try:
                cid = int(float(t[0]))
                x, y, w, h = map(float, t[1:5])
            except Exception:
                dropped += 1; 
                continue
            cid = max(0, min(nc - 1, cid))
            x = min(1.0, max(0.0, x)); y = min(1.0, max(0.0, y))
            w = min(1.0, max(0.0, w)); h = min(1.0, max(0.0, h))
            if w <= 0 or h <= 0:
                dropped += 1; 
                continue
            t = [str(cid), f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"]
            rows.append(" ".join(t))
            kept += 1
    with open(lbl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r + "\n")
    return kept, dropped

def autofix_labels(img_dir: Path, lbl_dir: Path, nc: int):
    total_kept = 0; total_drop = 0
    for img in img_dir.iterdir():
        if img.suffix.lower() not in {".jpg",".jpeg",".png",".bmp",".webp"}:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        k, d = fix_label_file(lbl, nc)
        total_kept += k; total_drop += d
    print(f"[labels] kept={total_kept}  dropped={total_drop}")

def pick_batch(imgsz: int, vram_gb: float) -> int:
    if imgsz <= 448:
        base = 24
    elif imgsz <= 512:
        base = 16
    elif imgsz <= 640:
        base = 12
    elif imgsz <= 800:
        base = 10
    else:
        base = 8
    scale = max(0.75, min(1.5, vram_gb / 6.0))
    b = int(max(4, round(base * scale // 2 * 2)))
    return b

def write_temp_yaml_for_oversample(yaml_dir: Path, names, nc: int, train_manifest_txt: Path, val_img_dir: Path) -> Path:
    tmp_dir = yaml_dir / "_auto_oversample"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_yaml = tmp_dir / "data_oversampled.yaml"
    data_obj = {
        "path": str(yaml_dir),
        "train": str(train_manifest_txt),
        "val": str(val_img_dir),
        "nc": int(nc),
        "names": list(names),
    }
    with open(tmp_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_obj, f, sort_keys=False, allow_unicode=True)
    return tmp_yaml

def run_train(model: YOLO, data: str, overrides: dict, tag: str) -> tuple[Path, Path]:
    model.overrides.update(overrides)
    print(f"\n[{tag}] EFFECTIVE OVERRIDES:")
    for k in [
        "epochs","patience","imgsz","batch","optimizer","lr0","lrf","momentum",
        "cos_lr","resume","fraction","mosaic","mixup","copy_paste",
        "rect","multi_scale","name","close_mosaic"
    ]:
        if k in model.overrides:
            print(f"  - {k}: {model.overrides[k]}")
    print(f"[{tag}] Starting training…")
    result = model.train(data=data)
    try:
        save_dir = Path(model.trainer.save_dir)
    except Exception:
        save_dir = Path(model.overrides.get("project", RUNS_DIR)) / model.overrides.get("name", "train")
    best_pt = save_dir / "weights" / "best.pt"
    print(f"[{tag}] save_dir: {save_dir}")
    print(f"[{tag}] best.pt: {best_pt}  (exists={best_pt.exists()})")
    return save_dir, best_pt

# -------------- main --------------
def main():
    print("PY:", sys.executable)
    print("NumPy:", np.__version__)
    set_seeds(GLOBAL_SEED)

    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"
    amp = True if use_gpu else False

    if use_gpu:
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA device: {name}  (VRAM ≈ {total_mem_gb:.1f} GB)")
        workers = int(os.environ.get("ULTRALYTICS_WORKERS", "2"))
    else:
        workers = 0
        try:
            torch.set_num_threads(2)
        except Exception:
            pass

    print("Using device:", "CUDA" if use_gpu else "CPU")

    # Dataset + nc sync
    nc, names, data_dict = read_dataset_info(DATA_YAML)
    print(f"Dataset: nc={nc}, train={data_dict.get('train')}, val={data_dict.get('val')}")
    overwrite_nc_in_model_yaml(MODEL_YAML, nc)

    train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, _ = resolve_dataset_paths(DATA_YAML)

    # ---- Label health (auto-fix) ----
    print("Autofixing labels (train/val)...")
    autofix_labels(train_img_dir, train_lbl_dir, nc)
    autofix_labels(val_img_dir,   val_lbl_dir,   nc)

    # ---- Class imbalance handling (build oversampled manifest) ----
    print("Computing class frequencies (train set)...")
    class_counts, per_image_counts = compute_class_frequencies(train_img_dir, train_lbl_dir)
    if class_counts:
        max_cls = max(class_counts.values())
        target = max(max_cls, 300)
        print("Class counts (first 15):", dict(list(class_counts.items())[:15]))
        print(f"Oversampling rare classes to ~{target} boxes each (cap repeats per image=6)...")

        manifest = build_oversampled_manifest(
            train_img_dir,
            train_lbl_dir,
            target_per_class=target,
            min_multiplier=1.0,
            max_multiplier=6.0,
            rng_seed=GLOBAL_SEED,
        )
        print(f"Oversampled manifest size: {len(manifest)} images (baseline {len(per_image_counts)})")

        tmp_manifest_dir = Path(RUNS_DIR) / "manifests"
        tmp_manifest_dir.mkdir(parents=True, exist_ok=True)
        tmp_txt = tmp_manifest_dir / "train_oversample.txt"
        write_manifest_txt(manifest, tmp_txt)

        oversampled_yaml = write_temp_yaml_for_oversample(
            Path(DATA_YAML).resolve().parent,
            names,
            nc,
            tmp_txt,
            val_img_dir,
        )
        data_for_training = str(oversampled_yaml)
    else:
        data_for_training = DATA_YAML

    # ---- Stage configs ----
    if use_gpu:
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        BATCH_STAGE1 = pick_batch(512, total_mem_gb)
        BATCH_STAGE2 = pick_batch(960, total_mem_gb)
    else:
        BATCH_STAGE1 = 8
        BATCH_STAGE2 = 8

    CACHE_MODE = "disk"  # safe for Colab/Drive

    # Common augs tuned for small objects
    stage1_common = {
        "project": RUNS_DIR,
        "imgsz": 512,
        "epochs": 90,
        "batch": BATCH_STAGE1,
        "workers": workers,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.1,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "warmup_epochs": 3,
        "cos_lr": True,
        "patience": 60,
        # augs
        "mosaic": 0.6,
        "mixup": 0.05,
        "copy_paste": 0.15,
        "translate": 0.08,
        "scale": 0.5,
        "perspective": 0.01,
        "fliplr": 0.5,
        "flipud": 0.0,
        "rect": False,
        "cache": CACHE_MODE,
        "save": True,
        "save_period": 10,
        "exist_ok": True,
        "resume": False,
        "device": device,
        "amp": amp,
        "plots": True,
        "deterministic": True,
        "val": True,
        # stabilize late epochs
        "close_mosaic": 15,
    }

    # ---------- Stage 0: FAST WARMUP (optional) ----------
    if FAST_WARMUP:
        print(">>> FAST_WARMUP: short, smaller, partial-data pass...")
        warm = stage1_common.copy()
        warm.update({
            "name": WARMUP_NAME,
            "imgsz": 448,
            "epochs": 5,
            "fraction": 0.25,
            "save_period": 1,
            "batch": max(8, min(BATCH_STAGE1, 16)),
        })
        m_warm = YOLO(MODEL_YAML, task="detect")
        _, _ = run_train(m_warm, data_for_training, warm, tag="WARMUP")

    # ---------- Stage 1a: Freeze backbone for 15 epochs ----------
    print(">>> Stage-1a: training with frozen backbone (15 epochs)...")
    stage1a = stage1_common.copy()
    stage1a.update({"name": STAGE1A_NAME, "epochs": 15})
    m1a = YOLO(MODEL_YAML, task="detect")
    try:
        m1a.model.backbone.requires_grad_(False)
    except Exception:
        pass
    save_dir_1a, best_1a = run_train(m1a, data_for_training, stage1a, tag="STAGE-1a")

    # ---------- Stage 1b: Unfreeze and continue ----------
    rem = stage1_common["epochs"] - 15
    print(f">>> Stage-1b: unfreeze and continue ({rem} epochs) from Stage-1a best...")
    stage1b = stage1_common.copy()
    stage1b.update({"name": STAGE1B_NAME, "epochs": rem, "resume": False})
    m1b = YOLO(str(best_1a if best_1a.exists() else MODEL_YAML), task="detect")
    try:
        m1b.model.backbone.requires_grad_(True)
    except Exception:
        pass
    save_dir_1b, best_1b = run_train(m1b, data_for_training, stage1b, tag="STAGE-1b")

    # ---------- Stage 2: Multi-scale fine-tuning at 960 ----------
    print(">>> Stage-2: multi-scale fine-tuning at 960 ...")
    start_ckpt = best_1b if best_1b.exists() else best_1a
    if not start_ckpt.exists():
        raise FileNotFoundError(f"Missing best.pt from Stage 1a/1b. Checked: {best_1a} and {best_1b}")

    m2 = YOLO(str(start_ckpt), task="detect")
    stage2_args = {
        "project": RUNS_DIR,
        "name": STAGE2_NAME,
        "imgsz": 960,
        "epochs": 150,
        "batch": BATCH_STAGE2,
        "multi_scale": True,
        "workers": workers,
        "patience": 80,
        "save_period": 10,
        "cos_lr": True,
        "exist_ok": True,
        "resume": False,
        "device": device,
        "amp": amp,
        "plots": True,
        "deterministic": True,
        "val": True,
        "rect": True,
        "cache": CACHE_MODE,
        # late augs
        "mosaic": 0.3,
        "mixup": 0.05,
        "copy_paste": 0.10,
        "translate": 0.06,
        "scale": 0.4,
        "perspective": 0.005,
        "fliplr": 0.5,
        "close_mosaic": 20,
    }
    save_dir_2, best_2 = run_train(m2, data_for_training, stage2_args, tag="STAGE-2")

    # ---------- Validate + export ----------
    print(">>> Validating with TTA on the original val + export ONNX ...")
    final_best = best_2 if best_2.exists() else (best_1b if best_1b.exists() else best_1a)
    if final_best.exists():
        mv = YOLO(str(final_best))
        # TTA-style validation for a more robust estimate
        mv.val(data=DATA_YAML, imgsz=960, save_json=True, plots=True, augment=True, iou=0.6, conf=0.001)
        try:
            mv.export(format="onnx", dynamic=True)
        except Exception as e:
            print("ONNX export failed:", e)
        print("Artifacts saved under:", save_dir_2)
    else:
        print("Warning: no best.pt found; skipping final val/export.")

if __name__ == "__main__":
    main()
