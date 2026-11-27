#!/usr/bin/env python3
"""
Generate OpenAPI specification from FastAPI application.

This script generates OpenAPI (Swagger) specification files in JSON and YAML
formats from the GeoMind FastAPI application. It can be used for:
- API documentation generation
- Client SDK generation
- API testing and validation
- Integration with API gateways

Usage:
    # Generate both JSON and YAML
    python scripts/generate_openapi.py --output openapi.json --output-yaml openapi.yaml

    # Generate only JSON
    python scripts/generate_openapi.py --output openapi.json

    # Generate with custom title/version
    python scripts/generate_openapi.py --output openapi.json --title "GeoMind API" --version "1.0.0"

    # Include/exclude specific routes
    python scripts/generate_openapi.py --output openapi.json --include-tags cameras,vision,location
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def generate_openapi(
    output_json: Path | None = None,
    output_yaml: Path | None = None,
    title: str | None = None,
    version: str | None = None,
    description: str | None = None,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    include_path_patterns: list[str] | None = None,
    exclude_path_patterns: list[str] | None = None,
) -> dict:
    """
    Generate OpenAPI specification from the FastAPI app.

    Args:
        output_json: Path to write JSON output
        output_yaml: Path to write YAML output
        title: Override API title
        version: Override API version
        description: Override API description
        include_tags: Only include routes with these tags
        exclude_tags: Exclude routes with these tags
        include_path_patterns: Only include paths matching these patterns
        exclude_path_patterns: Exclude paths matching these patterns

    Returns:
        OpenAPI specification dictionary
    """
    # Import the FastAPI app
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from geomind.main import create_app

    app = create_app()

    # Get OpenAPI schema
    openapi_schema = app.openapi()

    # Override metadata
    if title:
        openapi_schema["info"]["title"] = title
    if version:
        openapi_schema["info"]["version"] = version
    if description:
        openapi_schema["info"]["description"] = description

    # Filter paths based on tags
    if include_tags or exclude_tags:
        filtered_paths = {}
        for path, path_item in openapi_schema.get("paths", {}).items():
            # Check if any operation in this path matches tag filters
            include_path = True

            for method, operation in path_item.items():
                if method in ("get", "post", "put", "patch", "delete", "head", "options", "trace"):
                    tags = operation.get("tags", [])

                    if include_tags and not any(tag in include_tags for tag in tags):
                        include_path = False
                        break

                    if exclude_tags and any(tag in exclude_tags for tag in tags):
                        include_path = False
                        break

            if include_path:
                filtered_paths[path] = path_item

        openapi_schema["paths"] = filtered_paths

    # Filter paths based on path patterns
    if include_path_patterns or exclude_path_patterns:
        import fnmatch
        filtered_paths = {}
        for path, path_item in openapi_schema.get("paths", {}).items():
            include_path = True

            if include_path_patterns:
                if not any(fnmatch.fnmatch(path, pattern) for pattern in include_path_patterns):
                    include_path = False

            if exclude_path_patterns:
                if any(fnmatch.fnmatch(path, pattern) for pattern in exclude_path_patterns):
                    include_path = False

            if include_path:
                filtered_paths[path] = path_item

        openapi_schema["paths"] = filtered_paths

    # Write outputs
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(openapi_schema, f, indent=2, default=str)
        print(f"OpenAPI JSON written to: {output_json}")

    if output_yaml:
        output_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(output_yaml, "w") as f:
            yaml.dump(openapi_schema, f, default_flow_style=False, sort_keys=False)
        print(f"OpenAPI YAML written to: {output_yaml}")

    return openapi_schema


def print_summary(schema: dict) -> None:
    """Print a summary of the OpenAPI schema."""
    info = schema.get("info", {})
    paths = schema.get("paths", {})
    components = schema.get("components", {})

    print("\n" + "=" * 60)
    print("OpenAPI Specification Summary")
    print("=" * 60)
    print(f"Title:       {info.get('title', 'N/A')}")
    print(f"Version:     {info.get('version', 'N/A')}")
    print(f"Description: {info.get('description', 'N/A')[:80]}...")
    print(f"\nPaths:       {len(paths)}")
    print(f"Schemas:     {len(components.get('schemas', {}))}")
    print(f"Security:    {list(components.get('securitySchemes', {}).keys())}")

    # Group paths by tag
    tags = {}
    for path, path_item in paths.items():
        for method, operation in path_item.items():
            if method in ("get", "post", "put", "patch", "delete", "head", "options", "trace"):
                for tag in operation.get("tags", ["untagged"]):
                    if tag not in tags:
                        tags[tag] = []
                    tags[tag].append(f"{method.upper()} {path}")

    print("\nEndpoints by tag:")
    for tag, endpoints in sorted(tags.items()):
        print(f"  {tag}: {len(endpoints)} endpoints")
        for ep in endpoints[:5]:  # Show first 5
            print(f"    {ep}")
        if len(endpoints) > 5:
            print(f"    ... and {len(endpoints) - 5} more")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI specification from GeoMind FastAPI app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both JSON and YAML
  python scripts/generate_openapi.py --output openapi.json --output-yaml openapi.yaml

  # Generate only JSON
  python scripts/generate_openapi.py --output openapi.json

  # Custom metadata
  python scripts/generate_openapi.py --output openapi.json --title "GeoMind API" --version "1.0.0"

  # Filter by tags
  python scripts/generate_openapi.py --output openapi.json --include-tags cameras,vision

  # Exclude internal/debug routes
  python scripts/generate_openapi.py --output openapi.json --exclude-tags internal,debug

  # Filter by path patterns
  python scripts/generate_openapi.py --output openapi.json --include-paths "/api/v1/*"
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file path (default: openapi.json)"
    )
    parser.add_argument(
        "--output-yaml", "-y",
        type=str,
        help="Output YAML file path (optional)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Override API title"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Override API version"
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Override API description"
    )
    parser.add_argument(
        "--include-tags",
        type=str,
        help="Comma-separated list of tags to include"
    )
    parser.add_argument(
        "--exclude-tags",
        type=str,
        help="Comma-separated list of tags to exclude"
    )
    parser.add_argument(
        "--include-paths",
        type=str,
        help="Comma-separated list of path patterns to include (fnmatch)"
    )
    parser.add_argument(
        "--exclude-paths",
        type=str,
        help="Comma-separated list of path patterns to exclude (fnmatch)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of generated schema"
    )

    args = parser.parse_args()

    # Default output
    if not args.output and not args.output_yaml:
        args.output = "openapi.json"

    output_json = Path(args.output) if args.output else None
    output_yaml = Path(args.output_yaml) if args.output_yaml else None

    include_tags = args.include_tags.split(",") if args.include_tags else None
    exclude_tags = args.exclude_tags.split(",") if args.exclude_tags else None
    include_paths = args.include_paths.split(",") if args.include_paths else None
    exclude_paths = args.exclude_paths.split(",") if args.exclude_paths else None

    try:
        schema = generate_openapi(
            output_json=output_json,
            output_yaml=output_yaml,
            title=args.title,
            version=args.version,
            description=args.description,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            include_path_patterns=include_paths,
            exclude_path_patterns=exclude_paths,
        )

        if args.summary:
            print_summary(schema)

        return 0

    except Exception as e:
        print(f"Error generating OpenAPI spec: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())