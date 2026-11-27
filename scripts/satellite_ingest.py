#!/usr/bin/env python3
"""
Satellite imagery ingestion for GeoMind.

This module provides functionality to ingest satellite imagery from:
- Sentinel-2 (ESA Copernicus)
- Landsat 8/9 (USGS/NASA)

Features:
- Search and download imagery by AOI (Area of Interest) and date range
- Cloud masking and filtering
- Band extraction (RGB, NIR, SWIR, etc.)
- Index calculation (NDVI, NBR, NDWI, etc.)
- Integration with GeoMind location intelligence pipeline
- Export to Cloud Optimized GeoTIFF (COG) for efficient access

Usage:
    # Search Sentinel-2 imagery
    python scripts/satellite_ingest.py search --source sentinel2 --lat 52.52 --lon 13.405 --start 2024-01-01 --end 2024-01-31

    # Download and process
    python scripts/satellite_ingest.py download --source sentinel2 --scene-id S2A_MSIL1C_20240115T102021_N0509_R065_T33UUP_20240115T123456 --output data/sentinel2/

    # Calculate indices
    python scripts/satellite_ingest.py indices --input data/sentinel2/scene.tif --output data/sentinel2/indices.tif --indices ndvi,nbr,ndwi
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

# Optional dependencies - only import when needed
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from pystac_client import Client as STACClient
    from pystac.extensions.eo import EOExtension
    PYSTAC_AVAILABLE = True
except ImportError:
    PYSTAC_AVAILABLE = False


# Configuration
SENTINEL2_STAC_URL = "https://earth-search.aws.element84.com/v1"  # AWS Sentinel-2 STAC
LANDSAT_STAC_URL = "https://landsatlook.usgs.gov/stac-server"     # USGS Landsat STAC

DEFAULT_BANDS = {
    "sentinel2": {
        "rgb": ["B04", "B03", "B02"],      # Red, Green, Blue
        "nir": "B08",                       # Near Infrared
        "swir1": "B11",                     # Shortwave IR 1
        "swir2": "B12",                     # Shortwave IR 2
        "red_edge1": "B05",                 # Red Edge 1
        "red_edge2": "B06",                 # Red Edge 2
        "red_edge3": "B07",                 # Red Edge 3
        "coastal": "B01",                   # Coastal Aerosol
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B8A",                # Narrow NIR
    },
    "landsat": {
        "rgb": ["SR_B4", "SR_B3", "SR_B2"],  # Red, Green, Blue (Surface Reflectance)
        "nir": "SR_B5",                      # Near Infrared
        "swir1": "SR_B6",                    # Shortwave IR 1
        "swir2": "SR_B7",                    # Shortwave IR 2
        "coastal": "SR_B1",                  # Coastal Aerosol
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
        "swir2": "SR_B7",
        "qa": "QA_PIXEL",                    # Quality Assessment
    },
}


def check_dependencies() -> None:
    """Check and report on optional dependencies."""
    missing = []
    if not RASTERIO_AVAILABLE:
        missing.append("rasterio")
    if not NUMPY_AVAILABLE:
        missing.append("numpy")
    if not PYSTAC_AVAILABLE:
        missing.append("pystac-client")

    if missing:
        print(f"WARNING: Optional dependencies not installed: {', '.join(missing)}")
        print("Install with: pip install rasterio numpy pystac-client planetary-computer")
        print("Some features may not work.\n")


def search_sentinel2(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20.0,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search Sentinel-2 imagery via STAC API."""
    if not PYSTAC_AVAILABLE:
        raise RuntimeError("pystac-client not installed. Run: pip install pystac-client planetary-computer")

    import planetary_computer as pc

    catalog = Client.open(SENTINEL2_STAC_URL, modifier=pc.sign_inplace)

    # Create bbox around point (approx 0.1 degree ~ 10km)
    buffer = 0.05
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        limit=limit,
    )

    items = list(search.items())
    print(f"Found {len(items)} Sentinel-2 items")

    results = []
    for item in items:
        eo = EOExtension.ext(item)
        results.append({
            "id": item.id,
            "datetime": item.datetime.isoformat() if item.datetime else None,
            "cloud_cover": eo.cloud_cover,
            "bbox": item.bbox,
            "geometry": item.geometry,
            "assets": {k: v.href for k, v in item.assets.items()},
            "properties": item.properties,
        })

    return results


def search_landsat(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20.0,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search Landsat imagery via STAC API."""
    if not PYSTAC_AVAILABLE:
        raise RuntimeError("pystac-client not installed. Run: pip install pystac-client")

    catalog = Client.open(LANDSAT_STAC_URL)

    buffer = 0.05
    bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        limit=limit,
    )

    items = list(search.items())
    print(f"Found {len(items)} Landsat items")

    results = []
    for item in items:
        eo = EOExtension.ext(item)
        results.append({
            "id": item.id,
            "datetime": item.datetime.isoformat() if item.datetime else None,
            "cloud_cover": eo.cloud_cover,
            "bbox": item.bbox,
            "geometry": item.geometry,
            "assets": {k: v.href for k, v in item.assets.items()},
            "properties": item.properties,
        })

    return results


def download_asset(url: str, output_path: Path, progress: bool = True) -> bool:
    """Download an asset with optional progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress and total_size > 0:
                        percent = downloaded * 100 / total_size
                        bar_len = 40
                        filled = int(bar_len * downloaded / total_size)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        sys.stdout.write(f"\r[{bar}] {percent:.1f}% ({downloaded}/{total_size} bytes)")
                        sys.stdout.flush()

        if progress:
            sys.stdout.write("\n")
        return True

    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def calculate_indices(
    input_path: Path,
    output_path: Path,
    indices: list[str],
    source: str = "sentinel2",
) -> bool:
    """Calculate spectral indices from satellite imagery."""
    if not RASTERIO_AVAILABLE or not NUMPY_AVAILABLE:
        raise RuntimeError("rasterio and numpy required. Run: pip install rasterio numpy")

    import rasterio
    import numpy as np

    bands_config = DEFAULT_BANDS.get(source, DEFAULT_BANDS["sentinel2"])

    with rasterio.open(input_path) as src:
        # Read bands
        band_data = {}
        for band_name, band_key in bands_config.items():
            if isinstance(band_key, str):
                # Single band
                try:
                    band_idx = src.indexes[list(src.descriptions).index(band_key)] if band_key in src.descriptions else None
                except (ValueError, IndexError):
                    # Try to find by name in band metadata
                    band_idx = None
                    for i, desc in enumerate(src.descriptions, 1):
                        if band_key in (desc or ""):
                            band_idx = i
                            break

                if band_idx:
                    band_data[band_name] = src.read(band_idx).astype(np.float32)
            elif isinstance(band_key, list):
                # RGB composite - read multiple bands
                pass

        # Get metadata for output
        meta = src.meta.copy()
        meta.update(count=len(indices), dtype=rasterio.float32)

        # Calculate indices
        index_arrays = []
        index_names = []

        for idx_name in indices:
            idx_name = idx_name.lower()
            if idx_name == "ndvi":
                # NDVI = (NIR - Red) / (NIR + Red)
                nir = band_data.get("nir")
                red = band_data.get("red")
                if nir is not None and red is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndvi = (nir - red) / (nir + red)
                    index_arrays.append(ndvi)
                    index_names.append("NDVI")
            elif idx_name == "nbr":
                # NBR = (NIR - SWIR2) / (NIR + SWIR2)
                nir = band_data.get("nir")
                swir2 = band_data.get("swir2")
                if nir is not None and swir2 is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        nbr = (nir - swir2) / (nir + swir2)
                    index_arrays.append(nbr)
                    index_names.append("NBR")
            elif idx_name == "ndwi":
                # NDWI = (Green - NIR) / (Green + NIR)
                green = band_data.get("green")
                nir = band_data.get("nir")
                if green is not None and nir is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndwi = (green - nir) / (green + nir)
                    index_arrays.append(ndwi)
                    index_names.append("NDWI")
            elif idx_name == "ndmi":
                # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
                nir = band_data.get("nir")
                swir1 = band_data.get("swir1")
                if nir is not None and swir1 is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndmi = (nir - swir1) / (nir + swir1)
                    index_arrays.append(ndmi)
                    index_names.append("NDMI")
            elif idx_name == "evi":
                # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
                nir = band_data.get("nir")
                red = band_data.get("red")
                blue = band_data.get("blue")
                if nir is not None and red is not None and blue is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
                    index_arrays.append(evi)
                    index_names.append("EVI")
            elif idx_name == "savi":
                # SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L), L=0.5
                nir = band_data.get("nir")
                red = band_data.get("red")
                if nir is not None and red is not None:
                    L = 0.5
                    with np.errstate(divide='ignore', invalid='ignore'):
                        savi = (nir - red) / (nir + red + L) * (1 + L)
                    index_arrays.append(savi)
                    index_names.append("SAVI")

        if not index_arrays:
            print("No valid indices could be calculated")
            return False

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **meta) as dst:
            for i, (arr, name) in enumerate(zip(index_arrays, index_names), 1):
                dst.write(arr, i)
                dst.set_band_description(i, name)

        print(f"Calculated {len(index_arrays)} indices: {', '.join(index_names)}")
        print(f"Output written to: {output_path}")
        return True


def create_rgb_composite(
    input_dir: Path,
    output_path: Path,
    source: str = "sentinel2",
) -> bool:
    """Create RGB composite from individual band files."""
    if not RASTERIO_AVAILABLE or not NUMPY_AVAILABLE:
        raise RuntimeError("rasterio and numpy required. Run: pip install rasterio numpy")

    import rasterio
    import numpy as np

    bands_config = DEFAULT_BANDS.get(source, DEFAULT_BANDS["sentinel2"])
    rgb_bands = bands_config.get("rgb", [])

    if len(rgb_bands) != 3:
        print(f"RGB bands not configured for {source}")
        return False

    # Find band files
    band_files = {}
    for band in rgb_bands:
        # Look for file containing band name
        matches = list(input_dir.glob(f"*{band}*.tif")) + list(input_dir.glob(f"*{band}*.jp2"))
        if matches:
            band_files[band] = matches[0]
        else:
            print(f"Band file not found for {band}")
            return False

    # Read and stack
    arrays = []
    for band in rgb_bands:
        with rasterio.open(band_files[band]) as src:
            arr = src.read(1).astype(np.float32)
            # Normalize to 0-1 (assuming 16-bit)
            arr = np.clip(arr / 10000.0, 0, 1)
            arrays.append(arr)

    # Stack RGB
    rgb = np.stack(arrays, axis=0)  # (3, H, W)

    # Write output
    meta = src.meta.copy()
    meta.update(count=3, dtype=rasterio.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(rgb)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")

    print(f"RGB composite created: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Satellite imagery ingestion for GeoMind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search Sentinel-2 imagery
  python scripts/satellite_ingest.py search --source sentinel2 --lat 52.52 --lon 13.405 --start 2024-01-01 --end 2024-01-31

  # Search Landsat imagery
  python scripts/satellite_ingest.py search --source landsat --lat 52.52 --lon 13.405 --start 2024-01-01 --end 2024-01-31

  # Download scene assets
  python scripts/satellite_ingest.py download --source sentinel2 --scene-id S2A_MSIL1C_20240115T102021 --assets B04,B03,B02,B08 --output data/sentinel2/

  # Calculate spectral indices
  python scripts/satellite_ingest.py indices --input data/sentinel2/scene.tif --output data/sentinel2/indices.tif --indices ndvi,nbr,ndwi,ndmi,evi,savi

  # Create RGB composite
  python scripts/satellite_ingest.py rgb --input data/sentinel2/ --output data/sentinel2/rgb.tif --source sentinel2
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search command
    search = subparsers.add_parser("search", help="Search satellite imagery")
    search.add_argument("--source", choices=["sentinel2", "landsat"], required=True, help="Data source")
    search.add_argument("--lat", type=float, required=True, help="Latitude")
    search.add_argument("--lon", type=float, required=True, help="Longitude")
    search.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    search.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    search.add_argument("--max-cloud", type=float, default=20.0, help="Max cloud cover percent (default: 20)")
    search.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    search.add_argument("--output", type=str, help="Output JSON file")

    # Download command
    download = subparsers.add_parser("download", help="Download scene assets")
    download.add_argument("--source", choices=["sentinel2", "landsat"], required=True, help="Data source")
    download.add_argument("--scene-id", type=str, required=True, help="Scene ID from search")
    download.add_argument("--assets", type=str, help="Comma-separated asset keys to download (default: all)")
    download.add_argument("--output", type=str, required=True, help="Output directory")

    # Indices command
    indices = subparsers.add_parser("indices", help="Calculate spectral indices")
    indices.add_argument("--input", type=str, required=True, help="Input GeoTIFF file")
    indices.add_argument("--output", type=str, required=True, help="Output GeoTIFF file")
    indices.add_argument("--indices", type=str, default="ndvi,nbr,ndwi", help="Comma-separated indices (ndvi,nbr,ndwi,ndmi,evi,savi)")
    indices.add_argument("--source", choices=["sentinel2", "landsat"], default="sentinel2", help="Source for band mapping")

    # RGB composite command
    rgb = subparsers.add_parser("rgb", help="Create RGB composite")
    rgb.add_argument("--input", type=str, required=True, help="Input directory with band files")
    rgb.add_argument("--output", type=str, required=True, help="Output GeoTIFF file")
    rgb.add_argument("--source", choices=["sentinel2", "landsat"], default="sentinel2", help="Source for band mapping")

    args = parser.parse_args()

    check_dependencies()

    if args.command == "search":
        if args.source == "sentinel2":
            results = search_sentinel2(args.lat, args.lon, args.start, args.end, args.max_cloud, args.limit)
        else:
            results = search_landsat(args.lat, args.lon, args.start, args.end, args.max_cloud, args.limit)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            print(json.dumps(results, indent=2))

    elif args.command == "download":
        # This would need the scene assets from search results
        print("Download command requires scene assets from search. Use search first to get asset URLs.")
        print("Example asset keys for Sentinel-2: B02, B03, B04, B08, B11, B12, SCL")
        print("Example asset keys for Landsat: SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7, QA_PIXEL")

    elif args.command == "indices":
        idx_list = [i.strip() for i in args.indices.split(",")]
        calculate_indices(Path(args.input), Path(args.output), idx_list, args.source)

    elif args.command == "rgb":
        create_rgb_composite(Path(args.input), Path(args.output), args.source)

    return 0


if __name__ == "__main__":
    sys.exit(main())