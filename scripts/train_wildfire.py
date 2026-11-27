#!/usr/bin/env python3
"""
Wildfire detection model training pipeline.

This script provides a scaffold for training custom YOLO models for
wildfire smoke and fire detection. It supports:
- Dataset preparation (COCO/YOLO format)
- Training with Ultralytics YOLO
- Export to ONNX for inference
- Model validation and benchmarking

Usage:
    # Prepare dataset (download or organize)
    python scripts/train_wildfire.py prepare --data-dir data/wildfire

    # Train model
    python scripts/train_wildfire.py train --data data/wildfire.yaml --epochs 100 --model yolov8n

    # Export to ONNX
    python scripts/train_wildfire.py export --weights runs/detect/train/weights/best.pt --output models/wildfire_yolov8n.onnx

    # Validate model
    python scripts/train_wildfire.py validate --weights models/wildfire_yolov8n.onnx --data data/wildfire.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import yaml
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"


def ensure_dirs() -> None:
    """Ensure required directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def create_dataset_yaml(data_dir: Path, output_path: Path) -> None:
    """Create YOLO dataset YAML configuration."""
    config = {
        "path": str(data_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "smoke",
            1: "fire",
        },
        "nc": 2,
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset config created: {output_path}")


def prepare_dataset(args) -> int:
    """Prepare dataset structure for wildfire detection."""
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    for split in ["train", "val", "test"]:
        (data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Create dataset.yaml
    yaml_path = data_dir / "wildfire.yaml"
    create_dataset_yaml(data_dir, yaml_path)

    print(f"""
Dataset structure created at: {data_dir}

Directory structure:
  {data_dir}/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/

Next steps:
1. Add images to images/train, images/val, images/test
2. Add corresponding YOLO format labels to labels/train, labels/val, labels/test
   Label format (one per line): class_id x_center y_center width height (normalized 0-1)
   Classes: 0=smoke, 1=fire

3. Update {yaml_path} if needed

4. Run training:
   python scripts/train_wildfire.py train --data {yaml_path} --epochs 100 --model yolov8n
""")
    return 0


def train_model(args) -> int:
    """Train wildfire detection model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return 1

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"ERROR: Dataset config not found: {data_yaml}")
        return 1

    # Load base model
    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    # Train
    print(f"Starting training for {args.epochs} epochs...")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        save_period=args.save_period,
        project=str(RUNS_DIR / "detect"),
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        pose=args.pose,
        kobj=args.kobj,
        label_smoothing=args.label_smoothing,
        nbs=args.nbs,
        overlap_mask=args.overlap_mask,
        mask_ratio=args.mask_ratio,
        dropout=args.dropout,
        val=args.val,
        plots=args.plots,
        save=args.save,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        half=args.half,
        dnn=args.dnn,
        augment=args.augment,
    )

    print(f"Training complete. Best model: {results.best}")
    return 0


def export_model(args) -> int:
    """Export trained model to ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return 1

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Exporting to ONNX: {output_path}")
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
        opset=args.opset,
        half=args.half,
        dynamic=args.dynamic,
    )

    # Find and move exported model
    exported = weights_path.with_suffix(".onnx")
    if exported.exists():
        shutil.move(str(exported), str(output_path))
    else:
        # Search for exported model
        for p in Path(".").glob("*.onnx"):
            if weights_path.stem in p.name:
                shutil.move(str(p), str(output_path))
                break

    if output_path.exists():
        print(f"Model exported successfully: {output_path}")
        return 0
    else:
        print("ERROR: Failed to find exported ONNX model")
        return 1


def validate_model(args) -> int:
    """Validate model on dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return 1

    weights_path = Path(args.weights)
    data_yaml = Path(args.data)

    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        return 1

    if not data_yaml.exists():
        print(f"ERROR: Dataset config not found: {data_yaml}")
        return 1

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Validating on: {data_yaml}")
    results = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        half=args.half,
        plots=args.plots,
    )

    print(f"Validation results:")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP75: {results.box.map75:.4f}")
    for i, name in enumerate(["smoke", "fire"]):
        print(f"  {name}: mAP50-95={results.box.maps[i]:.4f}")

    return 0


def benchmark_model(args) -> int:
    """Benchmark model inference speed."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return 1

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        return 1

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    import time
    import numpy as np

    # Warmup
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        model(dummy, verbose=False)

    # Benchmark
    times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        model(dummy, verbose=False)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"Benchmark results ({args.iterations} iterations, {args.imgsz}x{args.imgsz}):")
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min: {times.min():.2f} ms")
    print(f"  Max: {times.max():.2f} ms")
    print(f"  Std: {times.std():.2f} ms")
    print(f"  FPS: {1000 / times.mean():.1f}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Wildfire detection model training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset structure
  python scripts/train_wildfire.py prepare --data-dir data/wildfire

  # Train model
  python scripts/train_wildfire.py train --data data/wildfire.yaml --epochs 100 --model yolov8n

  # Export to ONNX
  python scripts/train_wildfire.py export --weights runs/detect/train/weights/best.pt --output models/wildfire_yolov8n.onnx

  # Validate model
  python scripts/train_wildfire.py validate --weights models/wildfire_yolov8n.onnx --data data/wildfire.yaml

  # Benchmark model
  python scripts/train_wildfire.py benchmark --weights models/wildfire_yolov8n.onnx
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Prepare command
    prep = subparsers.add_parser("prepare", help="Prepare dataset structure")
    prep.add_argument("--data-dir", type=str, default="data/wildfire", help="Dataset directory")

    # Train command
    train = subparsers.add_parser("train", help="Train model")
    train.add_argument("--data", type=str, required=True, help="Dataset YAML config")
    train.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt, yolov8s.pt, etc.)")
    train.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train.add_argument("--imgsz", type=int, default=640, help="Image size")
    train.add_argument("--batch", type=int, default=16, help="Batch size")
    train.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    train.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    train.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N epochs")
    train.add_argument("--optimizer", type=str, default="auto", help="Optimizer (SGD, Adam, AdamW, auto)")
    train.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    train.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction")
    train.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    train.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    train.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs")
    train.add_argument("--warmup-momentum", type=float, default=0.8, help="Warmup momentum")
    train.add_argument("--warmup-bias-lr", type=float, default=0.1, help="Warmup bias LR")
    train.add_argument("--box", type=float, default=7.5, help="Box loss gain")
    train.add_argument("--cls", type=float, default=0.5, help="Class loss gain")
    train.add_argument("--dfl", type=float, default=1.5, help="DFL loss gain")
    train.add_argument("--pose", type=float, default=12.0, help="Pose loss gain")
    train.add_argument("--kobj", type=float, default=1.0, help="Obj loss gain")
    train.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing")
    train.add_argument("--nbs", type=int, default=64, help="Nominal batch size")
    train.add_argument("--overlap-mask", action="store_true", default=True, help="Overlap masks")
    train.add_argument("--mask-ratio", type=int, default=4, help="Mask downsample ratio")
    train.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    train.add_argument("--val", action="store_true", default=True, help="Validate during training")
    train.add_argument("--plots", action="store_true", default=True, help="Save plots")
    train.add_argument("--save", action="store_true", default=True, help="Save checkpoints")
    train.add_argument("--save-json", action="store_true", help="Save results to JSON")
    train.add_argument("--save-hybrid", action="store_true", help="Save hybrid labels")
    train.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    train.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    train.add_argument("--max-det", type=int, default=300, help="Max detections")
    train.add_argument("--half", action="store_true", help="Use half precision")
    train.add_argument("--dnn", action="store_true", help="Use OpenCV DNN")
    train.add_argument("--augment", action="store_true", help="Augment inference")

    # Export command
    export = subparsers.add_parser("export", help="Export model to ONNX")
    export.add_argument("--weights", type=str, required=True, help="Path to trained weights (.pt)")
    export.add_argument("--output", type=str, required=True, help="Output ONNX path")
    export.add_argument("--imgsz", type=int, default=640, help="Image size")
    export.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX model")
    export.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    export.add_argument("--half", action="store_true", help="Export in half precision")
    export.add_argument("--dynamic", action="store_true", help="Dynamic batch size")

    # Validate command
    validate = subparsers.add_parser("validate", help="Validate model on dataset")
    validate.add_argument("--weights", type=str, required=True, help="Path to weights (.pt or .onnx)")
    validate.add_argument("--data", type=str, required=True, help="Dataset YAML config")
    validate.add_argument("--imgsz", type=int, default=640, help="Image size")
    validate.add_argument("--batch", type=int, default=16, help="Batch size")
    validate.add_argument("--device", type=str, default="cpu", help="Device")
    validate.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    validate.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    validate.add_argument("--max-det", type=int, default=300, help="Max detections")
    validate.add_argument("--half", action="store_true", help="Use half precision")
    validate.add_argument("--plots", action="store_true", default=True, help="Save plots")

    # Benchmark command
    bench = subparsers.add_parser("benchmark", help="Benchmark model inference speed")
    bench.add_argument("--weights", type=str, required=True, help="Path to weights (.pt or .onnx)")
    bench.add_argument("--imgsz", type=int, default=640, help="Image size")
    bench.add_argument("--iterations", type=int, default=100, help="Number of iterations")

    args = parser.parse_args()

    ensure_dirs()

    if args.command == "prepare":
        return prepare_dataset(args)
    elif args.command == "train":
        return train_model(args)
    elif args.command == "export":
        return export_model(args)
    elif args.command == "validate":
        return validate_model(args)
    elif args.command == "benchmark":
        return benchmark_model(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())