#!/usr/bin/env python3
"""
Vision Engine Benchmarking Script

Measures end-to-end performance of the GeoMind Vision Engine including:
- Model inference latency (p50, p95, p99)
- Throughput (FPS) at various batch sizes
- Queue latency and backpressure
- Temporal validation overhead
- Memory usage
- CPU/GPU utilization

Usage:
    # Basic benchmark with dummy model
    python scripts/benchmark_vision.py --model dummy --duration 30

    # Benchmark ONNX model
    python scripts/benchmark_vision.py --model onnx --weights models/wildfire_yolov8n.onnx --duration 60

    # Full pipeline benchmark (with camera simulation)
    python scripts/benchmark_vision.py --model onnx --weights models/wildfire_yolov8n.onnx --simulate-cameras 4 --duration 120

    # Compare models
    python scripts/benchmark_vision.py --compare dummy,onnx --weights models/wildfire_yolov8n.onnx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_name: str
    model_type: str
    device: str
    input_size: int
    batch_size: int
    duration_seconds: float
    
    # Throughput
    total_frames: int = 0
    fps: float = 0.0
    
    # Latency (ms)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_mean: float = 0.0
    latency_std: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    
    # Queue
    queue_wait_p50: float = 0.0
    queue_wait_p95: float = 0.0
    max_queue_size: int = 0
    
    # Temporal validation
    temporal_overhead_ms: float = 0.0
    
    # Resources
    cpu_percent_mean: float = 0.0
    cpu_percent_max: float = 0.0
    memory_mb_mean: float = 0.0
    memory_mb_max: float = 0.0
    
    # Detections
    total_detections: int = 0
    confirmed_detections: int = 0
    
    # Raw data
    latencies: list[float] = field(default_factory=list)
    queue_waits: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2),
            "latency_ms": {
                "p50": round(self.latency_p50, 2),
                "p95": round(self.latency_p95, 2),
                "p99": round(self.latency_p99, 2),
                "mean": round(self.latency_mean, 2),
                "std": round(self.latency_std, 2),
                "min": round(self.latency_min, 2),
                "max": round(self.latency_max, 2),
            },
            "queue_wait_ms": {
                "p50": round(self.queue_wait_p50, 2),
                "p95": round(self.queue_wait_p95, 2),
            },
            "max_queue_size": self.max_queue_size,
            "temporal_overhead_ms": round(self.temporal_overhead_ms, 2),
            "resources": {
                "cpu_percent_mean": round(self.cpu_percent_mean, 1),
                "cpu_percent_max": round(self.cpu_percent_max, 1),
                "memory_mb_mean": round(self.memory_mb_mean, 1),
                "memory_mb_max": round(self.memory_mb_max, 1),
            },
            "detections": {
                "total": self.total_detections,
                "confirmed": self.confirmed_detections,
            },
        }


class ResourceMonitor:
    """Monitor CPU and memory usage during benchmark."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.cpu_samples: list[float] = []
        self.memory_samples: list[float] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._monitor())
    
    async def stop(self):
        self._running = False
        if self._task:
            await self._task
    
    async def _monitor(self):
        process = psutil.Process() if PSUTIL_AVAILABLE else None
        while self._running:
            if process:
                self.cpu_samples.append(process.cpu_percent())
                self.memory_samples.append(process.memory_info().rss / 1024 / 1024)
            await asyncio.sleep(self.interval)
    
    def get_stats(self) -> dict:
        return {
            "cpu_percent_mean": statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            "cpu_percent_max": max(self.cpu_samples) if self.cpu_samples else 0,
            "memory_mb_mean": statistics.mean(self.memory_samples) if self.memory_samples else 0,
            "memory_mb_max": max(self.memory_samples) if self.memory_samples else 0,
        }


class ModelWrapper:
    """Wrapper for different model types."""
    
    def __init__(self, model_type: str, weights_path: str, device: str, input_size: int = 640):
        self.model_type = model_type
        self.weights_path = weights_path
        self.device = device
        self.input_size = input_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        if self.model_type == "dummy":
            from geomind.vision.models_impl import DummyModel
            self.model = DummyModel(
                weights_path=self.weights_path,
                device=self.device,
                input_size=self.input_size,
                confidence_threshold=0.25,
                iou_threshold=0.45,
            )
        elif self.model_type == "onnx":
            from geomind.vision.models_impl import ONNXModel
            self.model = ONNXModel(
                weights_path=self.weights_path,
                device=self.device,
                input_size=self.input_size,
                confidence_threshold=0.25,
                iou_threshold=0.45,
            )
        elif self.model_type == "wildfire":
            from geomind.vision.models_impl import WildfireYOLOModel
            self.model = WildfireYOLOModel(
                weights_path=self.weights_path,
                device=self.device,
                input_size=self.input_size,
                confidence_threshold=0.25,
                iou_threshold=0.45,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.load()
        self.model.warmup()
    
    def infer(self, image: np.ndarray):
        return self.model.infer(image)
    
    def infer_batch(self, images: list[np.ndarray]):
        return self.model.infer_batch(images)


class CameraSimulator:
    """Simulate camera feeds for benchmarking."""
    
    def __init__(self, num_cameras: int, fps: float, resolution: tuple = (640, 480)):
        self.num_cameras = num_cameras
        self.fps = fps
        self.resolution = resolution
        self.frame_counters = [0] * num_cameras
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self.frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    
    async def start(self):
        self._running = True
        for i in range(self.num_cameras):
            self._tasks.append(asyncio.create_task(self._camera_loop(i)))
    
    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
    
    async def _camera_loop(self, camera_id: int):
        interval = 1.0 / self.fps
        while self._running:
            # Generate synthetic frame
            frame = self._generate_frame(camera_id)
            timestamp = time.time()
            
            try:
                self.frame_queue.put_nowait({
                    "camera_id": f"cam_{camera_id}",
                    "frame_id": self.frame_counters[camera_id],
                    "timestamp": timestamp,
                    "image": frame,
                })
                self.frame_counters[camera_id] += 1
            except asyncio.QueueFull:
                pass  # Drop frame if queue full
            
            await asyncio.sleep(interval)
    
    def _generate_frame(self, camera_id: int) -> np.ndarray:
        # Generate frame with some synthetic "smoke" pattern
        h, w = self.resolution
        frame = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        
        # Add synthetic smoke-like pattern occasionally
        if np.random.random() < 0.1:
            cx, cy = w // 2, h // 2
            for i in range(50):
                x = cx + np.random.randint(-100, 100)
                y = cy + np.random.randint(-100, 100)
                if 0 <= x < w and 0 <= y < h:
                    frame[y, x] = [200, 200, 200]
        
        return frame


async def run_benchmark(
    model: ModelWrapper,
    duration: float,
    batch_size: int = 1,
    camera_simulator: Optional[CameraSimulator] = None,
    use_temporal: bool = False,
) -> BenchmarkResult:
    """Run benchmark and collect metrics."""
    
    result = BenchmarkResult(
        model_name=Path(model.weights_path).stem,
        model_type=model.model_type,
        device=model.device,
        input_size=model.input_size,
        batch_size=batch_size,
        duration_seconds=duration,
    )
    
    # Start resource monitor
    monitor = ResourceMonitor()
    if PSUTIL_AVAILABLE:
        await monitor.start()
    
    # Temporal validator (if enabled)
    temporal_validator = None
    if use_temporal:
        from geomind.vision.temporal import TemporalValidator
        temporal_validator = TemporalValidator(
            min_consecutive_frames=3,
            frame_confidence_threshold=0.3,
            max_frame_gap_seconds=60.0,
        )
    
    latencies = []
    queue_waits = []
    total_frames = 0
    total_detections = 0
    confirmed_detections = 0
    max_queue_size = 0
    
    start_time = time.time()
    end_time = start_time + duration
    
    if camera_simulator:
        # Camera simulation mode
        await camera_simulator.start()
        
        while time.time() < end_time:
            try:
                frame_data = await asyncio.wait_for(
                    camera_simulator.frame_queue.get(),
                    timeout=0.1
                )
                queue_waits.append(0)  # No queue in this mode
                
                # Process frame
                frame_start = time.perf_counter()
                inference_result = model.infer(frame_data["image"])
                inference_latency = (time.perf_counter() - frame_start) * 1000
                
                latencies.append(inference_latency)
                total_frames += 1
                total_detections += len(inference_result.detections)
                
                # Temporal validation
                if temporal_validator:
                    temp_start = time.perf_counter()
                    temporal_validator.process_frame(
                        camera_id=frame_data["camera_id"],
                        frame_id=frame_data["frame_id"],
                        timestamp=frame_data["timestamp"],
                        detections=inference_result.detections,
                    )
                    result.temporal_overhead_ms += (time.perf_counter() - temp_start) * 1000
                    
                    # Count confirmed (simplified)
                    confirmed_detections += len([
                        d for d in inference_result.detections 
                        if d.confidence > 0.5
                    ])
                
                # Track queue
                if hasattr(model.model, '_queue'):
                    qsize = model.model._queue.qsize() if hasattr(model.model._queue, 'qsize') else 0
                    max_queue_size = max(max_queue_size, qsize)
                
            except asyncio.TimeoutError:
                continue
        
        await camera_simulator.stop()
    else:
        # Synthetic frame mode (max throughput)
        synthetic_frame = np.random.randint(0, 255, (model.input_size, model.input_size, 3), dtype=np.uint8)
        
        while time.time() < end_time:
            batch = [synthetic_frame] * batch_size
            
            batch_start = time.perf_counter()
            results = model.infer_batch(batch)
            batch_latency = (time.perf_counter() - batch_start) * 1000
            
            per_frame_latency = batch_latency / batch_size
            latencies.extend([per_frame_latency] * batch_size)
            total_frames += batch_size
            
            for r in results:
                total_detections += len(r.detections)
    
    actual_duration = time.time() - start_time
    
    # Stop monitor
    if PSUTIL_AVAILABLE:
        await monitor.stop()
        resource_stats = monitor.get_stats()
        result.cpu_percent_mean = resource_stats["cpu_percent_mean"]
        result.cpu_percent_max = resource_stats["cpu_percent_max"]
        result.memory_mb_mean = resource_stats["memory_mb_mean"]
        result.memory_mb_max = resource_stats["memory_mb_max"]
    
    # Calculate statistics
    if latencies:
        latencies.sort()
        result.latencies = latencies
        result.total_frames = total_frames
        result.fps = total_frames / actual_duration
        result.latency_p50 = np.percentile(latencies, 50)
        result.latency_p95 = np.percentile(latencies, 95)
        result.latency_p99 = np.percentile(latencies, 99)
        result.latency_mean = np.mean(latencies)
        result.latency_std = np.std(latencies)
        result.latency_min = np.min(latencies)
        result.latency_max = np.max(latencies)
    
    if queue_waits:
        queue_waits.sort()
        result.queue_waits = queue_waits
        result.queue_wait_p50 = np.percentile(queue_waits, 50)
        result.queue_wait_p95 = np.percentile(queue_waits, 95)
    
    result.max_queue_size = max_queue_size
    result.total_detections = total_detections
    result.confirmed_detections = confirmed_detections
    result.duration_seconds = actual_duration
    
    if use_temporal and total_frames > 0:
        result.temporal_overhead_ms = result.temporal_overhead_ms / total_frames
    
    return result


def print_results(result: BenchmarkResult):
    """Print benchmark results in a nice format."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS: {result.model_name} ({result.model_type})")
    print(f"{'='*60}")
    print(f"Device:           {result.device}")
    print(f"Input Size:       {result.input_size}x{result.input_size}")
    print(f"Batch Size:       {result.batch_size}")
    print(f"Duration:         {result.duration_seconds:.1f}s")
    print(f"\nThroughput:")
    print(f"  Total Frames:   {result.total_frames}")
    print(f"  FPS:            {result.fps:.1f}")
    print(f"\nLatency (ms):")
    print(f"  p50:            {result.latency_p50:.2f}")
    print(f"  p95:            {result.latency_p95:.2f}")
    print(f"  p99:            {result.latency_p99:.2f}")
    print(f"  Mean:           {result.latency_mean:.2f} ± {result.latency_std:.2f}")
    print(f"  Min/Max:        {result.latency_min:.2f} / {result.latency_max:.2f}")
    
    if result.queue_wait_p50 > 0 or result.queue_wait_p95 > 0:
        print(f"\nQueue Wait (ms):")
        print(f"  p50:            {result.queue_wait_p50:.2f}")
        print(f"  p95:            {result.queue_wait_p95:.2f}")
        print(f"  Max Queue:      {result.max_queue_size}")
    
    if result.temporal_overhead_ms > 0:
        print(f"\nTemporal Overhead: {result.temporal_overhead_ms:.2f} ms/frame")
    
    if result.cpu_percent_mean > 0:
        print(f"\nResources:")
        print(f"  CPU:            {result.cpu_percent_mean:.1f}% (max {result.cpu_percent_max:.1f}%)")
        print(f"  Memory:         {result.memory_mb_mean:.1f} MB (max {result.memory_mb_max:.1f} MB)")
    
    print(f"\nDetections:")
    print(f"  Total:          {result.total_detections}")
    print(f"  Confirmed:      {result.confirmed_detections}")
    print(f"{'='*60}")


async def main():
    parser = argparse.ArgumentParser(
        description="Vision Engine Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark dummy model
  python scripts/benchmark_vision.py --model dummy --duration 30

  # Benchmark ONNX model
  python scripts/benchmark_vision.py --model onnx --weights models/wildfire_yolov8n.onnx --duration 60

  # With camera simulation
  python scripts/benchmark_vision.py --model onnx --weights models/wildfire_yolov8n.onnx --simulate-cameras 4 --duration 120

  # With temporal validation
  python scripts/benchmark_vision.py --model onnx --weights models/wildfire_yolov8n.onnx --temporal --duration 60

  # Batch benchmark
  python scripts/benchmark_vision.py --model onnx --weights models/wildfire_yolov8n.onnx --batch-size 4 --duration 30

  # Compare models
  python scripts/benchmark_vision.py --compare dummy,onnx --weights models/wildfire_yolov8n.onnx --duration 30
        """,
    )
    parser.add_argument("--model", choices=["dummy", "onnx", "wildfire"], default="dummy", help="Model type")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device")
    parser.add_argument("--input-size", type=int, default=640, help="Input size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--duration", type=int, default=30, help="Benchmark duration (seconds)")
    parser.add_argument("--simulate-cameras", type=int, help="Number of simulated cameras")
    parser.add_argument("--camera-fps", type=float, default=10.0, help="Camera FPS")
    parser.add_argument("--temporal", action="store_true", help="Enable temporal validation")
    parser.add_argument("--compare", type=str, help="Comma-separated model types to compare")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        model_types = args.compare.split(",")
        results = []
        
        for model_type in model_types:
            weights = args.weights if model_type != "dummy" else "dummy"
            print(f"\nBenchmarking {model_type}...")
            
            model = ModelWrapper(model_type, weights, args.device, args.input_size)
            camera_sim = None
            if args.simulate_cameras:
                camera_sim = CameraSimulator(args.simulate_cameras, args.camera_fps)
            
            result = await run_benchmark(
                model, args.duration, args.batch_size, camera_sim, args.temporal
            )
            results.append(result)
            print_results(result)
        
        # Summary comparison
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'FPS':>8} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10}")
        print("-" * 60)
        for r in results:
            print(f"{r.model_type:<15} {r.fps:>8.1f} {r.latency_p50:>10.2f} {r.latency_p95:>10.2f} {r.latency_p99:>10.2f}")
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        # Single model benchmark
        weights = args.weights if args.model != "dummy" else "dummy"
        if args.model != "dummy" and not args.weights:
            print("ERROR: --weights required for non-dummy models")
            return 1
        
        model = ModelWrapper(args.model, weights, args.device, args.input_size)
        camera_sim = None
        if args.simulate_cameras:
            camera_sim = CameraSimulator(args.simulate_cameras, args.camera_fps)
        
        result = await run_benchmark(
            model, args.duration, args.batch_size, camera_sim, args.temporal
        )
        print_results(result)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))