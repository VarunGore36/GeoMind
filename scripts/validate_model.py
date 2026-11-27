#!/usr/bin/env python3
"""
Validate ONNX models for GeoMind Vision Engine.

This script validates that an ONNX model is compatible with the
GeoMind Vision Engine's ONNXModel class and produces expected outputs.

Usage:
    # Validate a model
    python scripts/validate_model.py models/wildfire_yolov8n.onnx

    # Validate with custom parameters
    python scripts/validate_model.py models/wildfire_yolov8n.onnx --input-size 640 --classes 0,1 --confidence 0.25

    # Run inference benchmark
    python scripts/validate_model.py models/wildfire_yolov8n.onnx --benchmark --iterations 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def validate_onnx_model(
    model_path: Path,
    input_size: int = 640,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    classes: list[int] | None = None,
    device: str = "cpu",
) -> dict:
    """
    Validate ONNX model compatibility with Vision Engine.

    Returns:
        Dictionary with validation results
    """
    results = {
        "model_path": str(model_path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "metadata": {},
        "inference_test": {},
        "benchmark": {},
    }

    # Check file exists
    if not model_path.exists():
        results["errors"].append(f"Model file not found: {model_path}")
        return results

    # Try to load with onnxruntime
    try:
        import onnxruntime as ort
    except ImportError:
        results["errors"].append("onnxruntime not installed. Run: pip install onnxruntime")
        return results

    # Load model
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )

        results["metadata"]["providers"] = session.get_providers()
        results["metadata"]["input_name"] = session.get_inputs()[0].name
        results["metadata"]["input_shape"] = session.get_inputs()[0].shape
        results["metadata"]["output_names"] = [out.name for out in session.get_outputs()]
        results["metadata"]["output_shapes"] = [out.shape for out in session.get_outputs()]

    except Exception as e:
        results["errors"].append(f"Failed to load model with ONNX Runtime: {e}")
        return results

    # Validate input shape
    input_shape = session.get_inputs()[0].shape
    expected_shape = [1, 3, input_size, input_size] if len(input_shape) == 4 else [None, 3, input_size, input_size]

    if len(input_shape) != 4:
        results["warnings"].append(f"Unexpected input rank: {len(input_shape)} (expected 4)")

    if input_shape[1] != 3:
        results["warnings"].append(f"Unexpected input channels: {input_shape[1]} (expected 3)")

    # Check for dynamic dimensions
    dynamic_dims = [i for i, dim in enumerate(input_shape) if isinstance(dim, str) or dim <= 0]
    if dynamic_dims:
        results["warnings"].append(f"Dynamic input dimensions at indices: {dynamic_dims}")

    # Validate outputs
    outputs = session.get_outputs()
    if len(outputs) == 0:
        results["errors"].append("Model has no outputs")
        return results

    # Check output format (YOLO typically has 1 output with shape [batch, num_boxes, 5+classes])
    output_shape = outputs[0].shape
    results["metadata"]["output_shape"] = output_shape

    if len(output_shape) == 3:
        batch, num_boxes, features = output_shape
        num_classes = features - 5 if features > 5 else 0
        results["metadata"]["num_boxes"] = num_boxes
        results["metadata"]["num_classes"] = num_classes

        if num_classes <= 0:
            results["warnings"].append(f"Output features ({features}) suggests no class predictions (need >5)")

        # Check if classes match
        if classes and num_classes != len(classes):
            results["warnings"].append(f"Model has {num_classes} classes but config specifies {len(classes)}")

    elif len(output_shape) == 2:
        # Some models output [num_boxes, features]
        num_boxes, features = output_shape
        num_classes = features - 5 if features > 5 else 0
        results["metadata"]["num_boxes"] = num_boxes
        results["metadata"]["num_classes"] = num_classes

    else:
        results["warnings"].append(f"Unexpected output shape: {output_shape}")

    # Run inference test
    print("Running inference test...")
    dummy_input = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

    # Preprocess (letterbox)
    import cv2
    h, w = dummy_input.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(dummy_input, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_top = (input_size - new_h) // 2
    pad_left = (input_size - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    input_tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    try:
        start = time.perf_counter()
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        inference_time = (time.perf_counter() - start) * 1000

        results["inference_test"]["success"] = True
        results["inference_test"]["inference_time_ms"] = inference_time
        results["inference_test"]["output_shapes"] = [out.shape for out in outputs]
        results["inference_test"]["num_outputs"] = len(outputs)

        # Try postprocessing
        if len(outputs) > 0:
            output = outputs[0]
            if output.ndim == 3:
                output = output[0]

            if output.shape[1] > 5:
                num_classes = output.shape[1] - 5
                boxes = output[:, :4]
                obj_conf = output[:, 4]
                class_probs = output[:, 5:]

                class_ids = np.argmax(class_probs, axis=1)
                class_conf = class_probs[np.arange(len(class_ids)), class_ids]
                confidences = obj_conf * class_conf

                # Filter by confidence
                mask = confidences >= confidence_threshold
                num_detections = int(np.sum(mask))

                results["inference_test"]["raw_detections"] = int(output.shape[0])
                results["inference_test"]["filtered_detections"] = num_detections
                results["inference_test"]["max_confidence"] = float(np.max(confidences)) if len(confidences) > 0 else 0.0

                if num_detections > 0:
                    results["inference_test"]["sample_classes"] = class_ids[mask][:5].tolist()
                    results["inference_test"]["sample_confidences"] = confidences[mask][:5].tolist()

    except Exception as e:
        results["inference_test"]["success"] = False
        results["inference_test"]["error"] = str(e)
        results["errors"].append(f"Inference test failed: {e}")

    # Check metadata
    try:
        import onnx
        model = onnx.load(str(model_path))
        if model.metadata_props:
            results["metadata"]["onnx_metadata"] = {prop.key: prop.value for prop in model.metadata_props}
    except Exception:
        pass

    # Overall validity
    results["valid"] = len(results["errors"]) == 0

    return results


def benchmark_model(
    model_path: Path,
    input_size: int = 640,
    device: str = "cpu",
    iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """Benchmark model inference speed."""
    import onnxruntime as ort
    import numpy as np
    import cv2

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name

    # Prepare dummy input
    dummy = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
    h, w = dummy.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(dummy, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_top = (input_size - new_h) // 2
    pad_left = (input_size - new_w) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    input_tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: input_tensor})

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_name: input_tensor})
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {
        "iterations": iterations,
        "warmup": warmup,
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "std_ms": float(times.std()),
        "fps": float(1000 / times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
    }


def test_vision_engine_integration(model_path: Path) -> dict:
    """Test model with GeoMind Vision Engine ONNXModel class."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    results = {
        "integration_test": {},
    }

    try:
        from geomind.vision.models_impl import ONNXModel

        model = ONNXModel(
            weights_path=str(model_path),
            device="cpu",
            confidence_threshold=0.25,
            iou_threshold=0.45,
            input_size=640,
        )

        # Test load
        model.load()
        results["integration_test"]["load"] = "success"

        # Test warmup
        model.warmup()
        results["integration_test"]["warmup"] = "success"

        # Test inference
        import numpy as np
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = model.infer(img)

        results["integration_test"]["inference"] = "success"
        results["integration_test"]["detections"] = len(result.detections)
        results["integration_test"]["inference_time_ms"] = result.inference_time_ms
        results["integration_test"]["model_version"] = result.model_version

    except Exception as e:
        results["integration_test"]["error"] = str(e)
        results["integration_test"]["load"] = "failed"

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate ONNX models for GeoMind Vision Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python scripts/validate_model.py models/wildfire_yolov8n.onnx

  # With custom parameters
  python scripts/validate_model.py models/wildfire_yolov8n.onnx --input-size 640 --classes 0,1 --confidence 0.25

  # Benchmark
  python scripts/validate_model.py models/wildfire_yolov8n.onnx --benchmark --iterations 100

  # Full validation including Vision Engine integration
  python scripts/validate_model.py models/wildfire_yolov8n.onnx --integration-test
        """,
    )
    parser.add_argument("model", type=str, help="Path to ONNX model file")
    parser.add_argument("--input-size", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold (default: 0.45)")
    parser.add_argument("--classes", type=str, default="0,1", help="Comma-separated class IDs (default: 0,1)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device (default: cpu)")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations (default: 100)")
    parser.add_argument("--integration-test", action="store_true", help="Test with Vision Engine ONNXModel class")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    model_path = Path(args.model)
    classes = [int(c.strip()) for c in args.classes.split(",")] if args.classes else None

    print(f"Validating model: {model_path}")
    print(f"Input size: {args.input_size}")
    print(f"Classes: {classes}")
    print(f"Confidence: {args.confidence}")
    print(f"Device: {args.device}")
    print()

    # Run validation
    results = validate_onnx_model(
        model_path,
        input_size=args.input_size,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        classes=classes,
        device=args.device,
    )

    # Print results
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Valid: {'✓' if results['valid'] else '✗'}")

    if results["metadata"]:
        print("\nMetadata:")
        for key, value in results["metadata"].items():
            if key != "onnx_metadata":
                print(f"  {key}: {value}")

        if "onnx_metadata" in results["metadata"]:
            print("  ONNX Metadata:")
            for k, v in results["metadata"]["onnx_metadata"].items():
                print(f"    {k}: {v}")

    if results["warnings"]:
        print("\nWarnings:")
        for w in results["warnings"]:
            print(f"  ⚠ {w}")

    if results["errors"]:
        print("\nErrors:")
        for e in results["errors"]:
            print(f"  ✗ {e}")

    if results["inference_test"]:
        print("\nInference Test:")
        it = results["inference_test"]
        if it.get("success"):
            print(f"  ✓ Success ({it.get('inference_time_ms', 0):.2f} ms)")
            print(f"  Raw detections: {it.get('raw_detections', 0)}")
            print(f"  Filtered detections: {it.get('filtered_detections', 0)}")
            print(f"  Max confidence: {it.get('max_confidence', 0):.2f}")
        else:
            print(f"  ✗ Failed: {it.get('error', 'Unknown')}")

    # Benchmark
    if args.benchmark:
        print("\nRunning benchmark...")
        bench = benchmark_model(model_path, args.input_size, args.device, args.iterations)
        results["benchmark"] = bench
        print(f"  Mean: {bench['mean_ms']:.2f} ms")
        print(f"  Median: {bench['median_ms']:.2f} ms")
        print(f"  Min: {bench['min_ms']:.2f} ms")
        print(f"  Max: {bench['max_ms']:.2f} ms")
        print(f"  FPS: {bench['fps']:.1f}")
        print(f"  P95: {bench['p95_ms']:.2f} ms")
        print(f"  P99: {bench['p99_ms']:.2f} ms")

    # Integration test
    if args.integration_test:
        print("\nRunning Vision Engine integration test...")
        integration = test_vision_engine_integration(model_path)
        results.update(integration)
        it = integration["integration_test"]
        if "error" not in it:
            print(f"  ✓ Load: {it.get('load')}")
            print(f"  ✓ Warmup: {it.get('warmup')}")
            print(f"  ✓ Inference: {it.get('inference')} ({it.get('inference_time_ms', 0):.2f} ms)")
            print(f"  Detections: {it.get('detections', 0)}")
        else:
            print(f"  ✗ Failed: {it['error']}")

    # Save output
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print("=" * 60)

    return 0 if results["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())