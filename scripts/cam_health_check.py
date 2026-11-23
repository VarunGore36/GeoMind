#!/usr/bin/env python3
"""Camera pipeline health check script."""

import argparse
import json
import sys
from datetime import datetime, timedelta

import httpx


def check_health(url: str, timeout: float = 5.0) -> dict:
    try:
        resp = httpx.get(f"{url}/api/v1/cameras/health", timeout=timeout)
        if resp.status_code == 200:
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "data": resp.json()}
        return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--exit-code", action="store_true")
    args = parser.parse_args()

    result = check_health(args.url, args.timeout)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Status: {result['status'].upper()}")
        if "data" in result:
            print(f"Feeds: {result['data'].get('feeds', 0)}")
            for cam_id, cam in result['data'].get('cameras', {}).items():
                print(f"  {cam_id}: frames={cam.get('total_frames', 0)}, errors={cam.get('total_errors', 0)}")

    if args.exit_code and result["status"] != "healthy":
        sys.exit(1)


if __name__ == "__main__":
    main()