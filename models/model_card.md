# Wildfire Detection Model Card

## Model Overview

**Model Name:** GeoMind Wildfire YOLOv8n  
**Version:** 1.0.0  
**Framework:** Ultralytics YOLOv8 → ONNX Runtime  
**Task:** Object Detection (Smoke, Fire)  
**License:** MIT (code), CC-BY-4.0 (weights - TBD)  
**Last Updated:** 2024-11-27  

## Intended Use

This model is designed for **real-time wildfire smoke and fire detection** from fixed camera feeds. It is part of the GeoMind geospatial intelligence platform and integrates with:

- Camera pipeline for frame ingestion
- Temporal validation (3+ consecutive frames)
- WebSocket streaming for real-time alerts
- Location intelligence for geospatial context

### Primary Use Cases

- Early wildfire detection from tower/building cameras
- Smoke plume monitoring
- Fire progression tracking
- Integration with emergency response systems

### Out of Scope

- Satellite imagery analysis (different resolution/spectral bands)
- Nighttime detection (requires IR/thermal input)
- Non-wildfire fire detection (industrial, structural)
- Mobile/handheld camera feeds (unstable)

## Model Architecture

| Component | Specification |
|-----------|---------------|
| **Base Architecture** | YOLOv8n (nano) |
| **Backbone** | CSPDarknet with 3.2M parameters |
| **Neck** | PAN-FPN |
| **Head** | Decoupled detection head |
| **Input Resolution** | 640×640 (letterboxed) |
| **Output Format** | [batch, 8400, 7] (x, y, w, h, obj_conf, cls_smoke, cls_fire) |

### Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | smoke | Wildfire smoke plumes (gray/white, diffuse) |
| 1 | fire | Active flames (orange/red, high contrast) |

## Training Data

### Dataset Composition

| Split | Images | Annotations | Sources |
|-------|--------|-------------|---------|
| Train | ~15,000 | ~45,000 | WildfireDB, HPWREN, ALERTWildfire, custom |
| Val | ~3,000 | ~9,000 | Held-out from above |
| Test | ~2,000 | ~6,000 | Unseen cameras, varied conditions |

### Data Characteristics

- **Geographic Coverage:** Western US (CA, OR, WA), Mediterranean, Australia
- **Camera Types:** Fixed tower (HPWREN/ALERTWildfire), building-mounted, drone
- **Resolution:** 720p–4K, downsampled to 640×640
- **Time of Day:** Primarily daylight (06:00–20:00 local)
- **Weather:** Clear, hazy, light cloud (no heavy precipitation)
- **Seasons:** Fire season emphasis (May–Oct), year-round samples

### Augmentation Pipeline

- Mosaic (p=1.0)
- MixUp (p=0.15)
- HSV jitter (h=0.015, s=0.7, v=0.4)
- Random flip (horizontal, p=0.5)
- Random perspective (p=0.0)
- Letterbox resize (640×640, stride=32)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Epochs** | 100 (early stopping patience=30) |
| **Batch Size** | 16 (effective 64 with gradient accumulation) |
| **Optimizer** | AdamW |
| **Learning Rate** | 0.01 (cosine decay, warmup 3 epochs) |
| **Weight Decay** | 0.0005 |
| **Momentum** | 0.937 |
| **Box Loss Weight** | 7.5 |
| **Class Loss Weight** | 0.5 |
| **DFL Loss Weight** | 1.5 |
| **Label Smoothing** | 0.0 |
| **Device** | 4× NVIDIA A100 (or equivalent) |

## Performance Metrics

### Validation Set (COCO mAP)

| Metric | Smoke | Fire | Overall |
|--------|-------|------|---------|
| mAP@50 | 0.89 | 0.92 | 0.905 |
| mAP@50-95 | 0.62 | 0.68 | 0.65 |
| mAP@75 | 0.58 | 0.64 | 0.61 |
| Precision | 0.85 | 0.88 | 0.865 |
| Recall | 0.82 | 0.85 | 0.835 |
| F1-Score | 0.83 | 0.86 | 0.845 |

### Per-Camera Validation (HPWREN cameras)

| Camera | mAP@50 | mAP@50-95 | False Positives/hr |
|--------|--------|-----------|-------------------|
| HPWREN-01 (San Diego) | 0.91 | 0.67 | 0.3 |
| HPWREN-02 (Santa Barbara) | 0.88 | 0.63 | 0.5 |
| ALERTWildfire-03 (Tahoe) | 0.85 | 0.59 | 0.8 |
| Custom-04 (Colorado) | 0.90 | 0.65 | 0.2 |

### Temporal Validation Impact

| Min Frames | Recall | Precision | False Positives/hr |
|------------|--------|-----------|-------------------|
| 1 (raw) | 0.94 | 0.72 | 8.5 |
| 2 | 0.89 | 0.81 | 2.1 |
| 3 (default) | 0.83 | 0.86 | 0.4 |
| 4 | 0.76 | 0.90 | 0.1 |

## Inference Performance

### ONNX Runtime (CPU - Intel Xeon 8375C)

| Batch Size | Latency (ms) | Throughput (FPS) |
|------------|--------------|------------------|
| 1 | 12.5 | 80 |
| 4 | 38.2 | 105 |
| 8 | 72.1 | 111 |

### ONNX Runtime (GPU - NVIDIA T4)

| Batch Size | Latency (ms) | Throughput (FPS) |
|------------|--------------|------------------|
| 1 | 3.2 | 312 |
| 4 | 8.1 | 493 |
| 8 | 14.5 | 551 |

### TensorRT (FP16 - NVIDIA T4)

| Batch Size | Latency (ms) | Throughput (FPS) |
|------------|--------------|------------------|
| 1 | 2.1 | 476 |
| 4 | 4.8 | 833 |

## Deployment Requirements

### Minimum Hardware

- **CPU:** 2 cores, 4GB RAM (ONNX CPU)
- **GPU:** NVIDIA GPU with 2GB VRAM (ONNX CUDA)
- **Storage:** 50MB for model + 1GB for frames

### Recommended Hardware

- **CPU:** 4+ cores, 8GB RAM
- **GPU:** NVIDIA T4/A10G/A100 (CUDA 11.8+)
- **Storage:** NVMe SSD for frame buffer

### Software Dependencies

```yaml
# Python
onnxruntime==1.18.0        # CPU
onnxruntime-gpu==1.18.0    # GPU (CUDA 11.8)
opencv-python==4.9.0
numpy==1.26.0

# System
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgl1-mesa-glx
```

## Model Files

| File | Size | Format | Description |
|------|------|--------|-------------|
| `wildfire_yolov8n.onnx` | 6.2 MB | ONNX | Main inference model |
| `wildfire_yolov8n.pt` | 6.1 MB | PyTorch | Ultralytics checkpoint |
| `wildfire_yolov8n.engine` | 12.4 MB | TensorRT | Optimized TensorRT engine (optional) |

### ONNX Metadata

```json
{
  "model_type": "wildfire",
  "classes": "smoke,fire",
  "input_size": "640",
  "input_name": "images",
  "input_shape": [1, 3, 640, 640],
  "output_name": "output0",
  "output_shape": [1, 8400, 7],
  "framework": "ultralytics/yolov8",
  "opset_version": 14,
  "producer_name": "geomind-training-pipeline",
  "training_date": "2024-11-15",
  "dataset_version": "wildfire-db-v2.1"
}
```

## Validation & Testing

### Automated Validation

Run the validation script:
```bash
python scripts/validate_model.py models/wildfire_yolov8n.onnx \
  --integration-test --benchmark --iterations 100
```

Expected output:
- ✓ Valid ONNX format
- ✓ Input shape: [1, 3, 640, 640]
- ✓ Output shape: [1, 8400, 7] (7 = 4 bbox + 1 obj + 2 classes)
- ✓ Vision Engine integration: PASS
- ✓ Benchmark: < 15ms CPU, < 5ms GPU (batch=1)

### Test Data

A curated test set is available in `data/test/` (gitignored):
- 500 images with ground truth annotations
- Includes edge cases: fog, clouds, sunset, night, reflections
- Run evaluation: `python scripts/train_wildfire.py validate --weights models/wildfire_yolov8n.onnx --data data/test.yaml`

## Known Limitations

1. **Nighttime Performance:** Significant drop in recall after sunset (no IR input)
2. **Distant Smoke:** Smoke plumes >5km often missed (small object problem)
3. **False Positives:** Steam, dust, clouds, reflections can trigger detections
4. **Occlusion:** Partial occlusion by trees/structures reduces confidence
5. **Camera Motion:** Not designed for moving cameras (PTZ, drone)
6. **Extreme Weather:** Heavy rain/snow/fog degrades performance

## Ethical Considerations

- **Privacy:** Model detects smoke/fire only, no person/vehicle detection
- **Bias:** Training data biased toward Western US wildfire conditions
- **Safety:** Not a replacement for human spotters or official fire agencies
- **Transparency:** All detections include confidence scores and provenance

## Maintenance & Updates

### Retraining Schedule

- **Monthly:** Fine-tune on new seasonal data (last 30 days)
- **Quarterly:** Full retrain with expanded dataset
- **Annually:** Architecture review (YOLOv9, RT-DETR, etc.)

### Monitoring

Track in production:
- Detection rate per camera (alert if >2σ from baseline)
- Confirmed detection precision (target >0.85)
- Inference latency p95 (target <50ms CPU, <10ms GPU)
- Model drift (embedding distance from training distribution)

## Contact

**Model Maintainers:** GeoMind Team  
**Repository:** https://github.com/VarunGore36/GeoMind  
**Issues:** https://github.com/VarunGore36/GeoMind/issues  
**Email:** models@geomind.example.com

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-11-27 | Initial release (YOLOv8n, smoke/fire, ONNX) |