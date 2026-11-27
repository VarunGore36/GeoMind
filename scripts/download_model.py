#!/usr/bin/env python3
"""
Download and prepare wildfire detection models.

This script downloads pretrained models and converts them to ONNX format
for use with the GeoMind Vision Engine.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

# Model registry with known wildfire detection models
MODEL_REGISTRY = {
    "wildfire_yolov8n": {
        "description": "YOLOv8n fine-tuned for wildfire smoke/fire detection",
        "source": "ultralytics/yolov8n",  # Base model, needs fine-tuning
        "classes": ["smoke", "fire"],
        "input_size": 640,
        "format": "onnx",
        "url": None,  # No public pretrained wildfire model available yet
    },
    "wildfire_yolov8s": {
        "description": "YOLOv8s fine-tuned for wildfire smoke/fire detection",
        "source": "ultralytics/yolov8s",
        "classes": ["smoke", "fire"],
        "input_size": 640,
        "format": "onnx",
        "url": None,
    },
}


def ensure_model_dir() -> Path:
    """Ensure model directory exists."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def download_file(url: str, dest: Path, progress: bool = True) -> None:
    """Download a file with optional progress bar."""
    if progress:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                bar_len = 40
                filled = int(bar_len * downloaded / total_size)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stdout.write(f"\r[{bar}] {percent:.1f}% ({downloaded}/{total_size} bytes)")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\rDownloaded: {downloaded} bytes")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest, reporthook)
        sys.stdout.write("\n")
    else:
        urllib.request.urlretrieve(url, dest)


def download_yolo_base(model_name: str, output_path: Path) -> bool:
    """Download a base YOLO model from Ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return False

    print(f"Downloading base YOLO model: {model_name}")
    model = YOLO(model_name)

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    model.export(format="onnx", imgsz=640, simplify=True)

    # Move exported model to target location
    exported = Path(model_name.replace(".pt", "")).with_suffix(".onnx")
    if exported.exists():
        shutil.move(str(exported), str(output_path))
        print(f"Model saved to: {output_path}")
        return True
    else:
        # Try alternative location
        for p in Path(".").glob("*.onnx"):
            if model_name.replace(".pt", "") in p.name:
                shutil.move(str(p), str(output_path))
                print(f"Model saved to: {output_path}")
                return True

    print("ERROR: Failed to find exported ONNX model")
    return False


def create_placeholder_model(model_name: str, output_path: Path) -> bool:
    """Create a placeholder ONNX model for testing."""
    try:
        import numpy as np
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("ERROR: onnx not installed. Run: pip install onnx")
        return False

    # Ensure onnx is available for metadata
    import onnx

    print(f"Creating placeholder ONNX model: {output_path}")

    # Create a minimal ONNX model that outputs dummy detections
    # Input: (1, 3, 640, 640)
    # Output: (1, 8400, 7) - YOLO format: x, y, w, h, obj_conf, class_0, class_1
    input_tensor = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 640, 640]
    )
    output_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 8400, 7]
    )

    # Create a simple graph that just passes through zeros (placeholder)
    node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output0"],
        value=helper.make_tensor(
            name="dummy_output",
            data_type=TensorProto.FLOAT,
            dims=[1, 8400, 7],
            vals=[0.0] * (1 * 8400 * 7),
        ),
    )

    graph = helper.make_graph(
        [node],
        "wildfire_placeholder",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    model.producer_name = "geomind-model-downloader"
    # Set metadata using onnx.helper.set_metadata_props
    onnx.helper.set_metadata_props(model, {
        "model_type": "wildfire",
        "classes": "smoke,fire",
        "input_size": "640",
        "description": "Placeholder model for testing",
    })

    onnx.save(model, output_path)
    print(f"Placeholder model created: {output_path}")
    return True


def list_available_models() -> None:
    """List available models in the registry."""
    print("Available models:")
    for name, info in MODEL_REGISTRY.items():
        url_status = "Available" if info["url"] else "Requires training/conversion"
        print(f"  {name}: {info['description']}")
        print(f"    Classes: {', '.join(info['classes'])}")
        print(f"    Input size: {info['input_size']}")
        print(f"    Format: {info['format']}")
        print(f"    Status: {url_status}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare wildfire detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python scripts/download_model.py --list

  # Download base YOLOv8n and convert to ONNX (for fine-tuning)
  python scripts/download_model.py --model yolov8n --output models/wildfire_yolov8n.onnx

  # Create placeholder model for testing
  python scripts/download_model.py --placeholder --output models/wildfire_placeholder.onnx
        """,
    )
    parser.add_argument(
        "--list", action="store_true", help="List available models in registry"
    )
    parser.add_argument(
        "--model", type=str, help="Base model to download (e.g., yolov8n, yolov8s)"
    )
    parser.add_argument(
        "--output", type=str, help="Output path for the model"
    )
    parser.add_argument(
        "--placeholder", action="store_true", help="Create a placeholder ONNX model for testing"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing model"
    )

    args = parser.parse_args()

    ensure_model_dir()

    if args.list:
        list_available_models()
        return 0

    if args.placeholder:
        output = Path(args.output) if args.output else MODEL_DIR / "wildfire_placeholder.onnx"
        if output.exists() and not args.force:
            print(f"Model already exists: {output}. Use --force to overwrite.")
            return 1
        success = create_placeholder_model("wildfire_placeholder", output)
        return 0 if success else 1

    if args.model:
        output = Path(args.output) if args.output else MODEL_DIR / f"{args.model}.onnx"
        if output.exists() and not args.force:
            print(f"Model already exists: {output}. Use --force to overwrite.")
            return 1
        success = download_yolo_base(args.model, output)
        return 0 if success else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())