"""Tests for Overpass query building and response normalisation."""

from __future__ import annotations

from typing import Any

from geomind.providers.overpass import build_overpass_query, parse_overpass_payload


def _way(osm_id: int, tags: dict[str, str], coords: list[tuple[float, float]]) -> dict[str, Any]:
    return {
        "type": "way",
        "id": osm_id,
        "tags": tags,
        "geometry": [{"lat": lat, "lon": lon} for lon, lat in coords],
    }


def test_build_query_contains_expected_clauses() -> None:
    query = build_overpass_query(52.52, 13.405, 500.0, 2500.0, 25)
    assert "[out:json]" in query
    assert 'way["building"](around:500,52.52,13.405);' in query
    assert 'way["highway"](around:500,52.52,13.405);' in query
    assert 'way["natural"="water"](around:2500,52.52,13.405);' in query
    assert "out geom;" in query


def test_parse_building_and_road() -> None:
    payload = {
        "elements": [
            _way(
                1,
                {"building": "yes"},
                [
                    (13.400, 52.520),
                    (13.410, 52.520),
                    (13.410, 52.530),
                    (13.400, 52.530),
                    (13.400, 52.520),
                ],
            ),
            _way(2, {"highway": "residential"}, [(13.400, 52.520), (13.410, 52.520)]),
        ]
    }

    features = parse_overpass_payload(payload)
    by_id = {feature.osm_id: feature for feature in features}

    assert by_id[1].feature_type == "building"
    assert by_id[1].geometry["type"] == "Polygon"
    assert by_id[2].feature_type == "road"
    assert by_id[2].geometry["type"] == "LineString"


def test_parse_vegetation() -> None:
    payload = {
        "elements": [
            _way(
                3,
                {"landuse": "forest"},
                [(13.40, 52.50), (13.41, 52.50), (13.41, 52.51), (13.40, 52.50)],
            )
        ]
    }
    features = parse_overpass_payload(payload)
    assert len(features) == 1
    assert features[0].feature_type == "vegetation"


def test_parse_water_relation_is_assembled_into_polygon() -> None:
    def member(coords: list[tuple[float, float]]) -> dict[str, Any]:
        return {
            "type": "way",
            "role": "outer",
            "geometry": [{"lat": lat, "lon": lon} for lon, lat in coords],
        }

    payload = {
        "elements": [
            {
                "type": "relation",
                "id": 10,
                "tags": {"natural": "water", "name": "Test Lake"},
                "members": [
                    member([(13.400, 52.500), (13.410, 52.500)]),
                    member([(13.410, 52.500), (13.410, 52.510)]),
                    member([(13.410, 52.510), (13.400, 52.510)]),
                    member([(13.400, 52.510), (13.400, 52.500)]),
                ],
            }
        ]
    }

    features = parse_overpass_payload(payload)
    assert len(features) == 1
    feature = features[0]
    assert feature.feature_type == "water"
    assert feature.name == "Test Lake"
    assert feature.geometry["type"] == "Polygon"
    ring = feature.geometry["coordinates"][0]
    assert ring[0] == ring[-1]
    assert len(ring) == 5


def test_parse_skips_irrelevant_elements() -> None:
    payload = {
        "elements": [
            _way(1, {"amenity": "bench"}, [(13.40, 52.50), (13.41, 52.50)]),
            {"type": "node", "id": 2, "tags": {"building": "yes"}},
        ]
    }
    assert parse_overpass_payload(payload) == []


def test_parse_deduplicates_same_feature() -> None:
    way = _way(1, {"building": "yes"}, [(13.40, 52.50), (13.41, 52.50), (13.41, 52.51)])
    payload = {"elements": [way, way]}
    assert len(parse_overpass_payload(payload)) == 1
