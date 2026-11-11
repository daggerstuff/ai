"""Tests for the OVHcloud schema catalog."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from ai.infrastructure.ovh.catalog import (
    API_VERSIONS,
    download_catalog,
    iter_sections,
    load_catalog,
    navigation_url,
    operations_from_spec,
    section_url,
    spec_filename,
)

BASE_URL = "https://api.example.test"
LEAF_SECTIONS = 3
LEAF_OPERATIONS = 3

NAVIGATION = {
    "navigation": [
        {
            "name": "Account",
            "slug": "account",
            "operationCount": 0,
            "children": [
                {"name": "Profile", "slug": "profile", "operationCount": 2, "children": []},
                {
                    "name": "Billing",
                    "slug": "billing",
                    "operationCount": 0,
                    "children": [{"name": "Invoices", "slug": "invoices", "operationCount": 1, "children": []}],
                },
            ],
        },
        {"name": "Public Cloud", "slug": "public-cloud", "operationCount": 0, "children": []},
    ]
}

PROFILE_SPEC = {
    "openapi": "3.0.1",
    "paths": {
        "/me": {"get": {"operationId": "getMe", "summary": "Get this object properties"}},
        "/me/accessRestriction/{id}": {
            "get": {
                "summary": "Get this object properties",
                "parameters": [
                    {"name": "id", "in": "path", "required": True},
                    {"name": "expand", "in": "query"},
                ],
            },
            "put": {
                "summary": "Alter this object properties",
                "requestBody": {
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/nichandle.Restriction"}}}
                },
            },
        },
    },
}

INVOICES_SPEC = {"paths": {"/me/invoices": {"get": {"summary": "List invoices"}}}}

SPECS = {
    "account/profile": PROFILE_SPEC,
    "account/billing/invoices": INVOICES_SPEC,
    "public-cloud": {"paths": {}},
}


def _transport(fail: set[str] | None = None) -> httpx.MockTransport:
    failures = fail or set()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "developer-hub-navigation.json" in path:
            return httpx.Response(200, json=NAVIGATION)
        for section, spec in SPECS.items():
            if path.endswith(f"/{section}.json"):
                if section in failures:
                    return httpx.Response(500)
                return httpx.Response(200, json=spec)
        return httpx.Response(404, json={"class": "Client::NotFound", "message": "nope"})

    return httpx.MockTransport(handler)


# -- url construction ------------------------------------------------------
def test_navigation_url_shape() -> None:
    assert navigation_url(BASE_URL, "v2") == (f"{BASE_URL}/v2/__schemas/developer-hub-navigation.json")


def test_section_url_is_a_slug_chain_under_the_schema_root() -> None:
    assert section_url("account/billing/invoices", BASE_URL, "1.0") == (
        f"{BASE_URL}/1.0/__schemas/developer-hub-navigation/account/billing/invoices.json?format=openapi3"
    )


def test_spec_filename_encodes_the_slug_chain() -> None:
    assert spec_filename("1.0", "account/billing/invoices") == "1.0__account__billing__invoices.json"


# -- navigation ------------------------------------------------------------
def test_iter_sections_returns_only_leaves() -> None:
    sections = list(iter_sections(NAVIGATION, "1.0"))
    assert [s.path for s in sections] == ["account/profile", "account/billing/invoices", "public-cloud"]
    assert sections[1].name == "Invoices"
    assert sections[1].operation_count == 1
    assert sections[0].version == "1.0"


def test_operation_count_matches_the_leaf_totals() -> None:
    sections = list(iter_sections(NAVIGATION, "1.0"))
    assert sum(s.operation_count for s in sections) == LEAF_OPERATIONS


# -- spec parsing ----------------------------------------------------------
def test_operations_from_spec_extracts_methods_summaries_and_parameters() -> None:
    operations = operations_from_spec(PROFILE_SPEC, version="1.0", section="account/profile")
    by_key = {op.key: op for op in operations}
    assert set(by_key) == {"GET /me", "GET /me/accessRestriction/{id}", "PUT /me/accessRestriction/{id}"}
    detail = by_key["GET /me/accessRestriction/{id}"]
    assert detail.operation_id is None
    assert [(p.name, p.location, p.required) for p in detail.parameters] == [
        ("id", "path", True),
        ("expand", "query", False),
    ]
    assert detail.request_schema is None


def test_operations_from_spec_resolves_the_request_schema_name() -> None:
    operations = operations_from_spec(PROFILE_SPEC, version="1.0", section="account/profile")
    put = next(op for op in operations if op.method == "PUT")
    assert put.request_schema == "nichandle.Restriction"


def test_operations_from_spec_ignores_non_http_keys() -> None:
    spec = {"paths": {"/me": {"parameters": [{"name": "x", "in": "query"}], "get": {"summary": "ok"}}}}
    operations = operations_from_spec(spec, version="1.0", section="account/profile")
    assert [op.method for op in operations] == ["GET"]


# -- download and load -----------------------------------------------------
def test_download_catalog_writes_every_spec(tmp_path: Path) -> None:
    summary = download_catalog(tmp_path, base_url=BASE_URL, versions=["1.0"], transport=_transport())
    assert summary["failures"] == []
    assert summary["versions"]["1.0"] == {"sections": 3, "operations": 3, "specs_written": 3}
    assert (tmp_path / "navigation-1.0.json").is_file()
    assert (tmp_path / spec_filename("1.0", "account/profile")).is_file()


def test_download_catalog_records_failed_sections_without_aborting(tmp_path: Path) -> None:
    summary = download_catalog(tmp_path, base_url=BASE_URL, versions=["1.0"], transport=_transport({"account/profile"}))
    assert summary["versions"]["1.0"]["specs_written"] == LEAF_SECTIONS - 1
    assert len(summary["failures"]) == 1
    assert "account/profile" in summary["failures"][0]


def test_load_catalog_indexes_every_operation(tmp_path: Path) -> None:
    download_catalog(tmp_path, base_url=BASE_URL, versions=["1.0"], transport=_transport())
    catalog = load_catalog(tmp_path)
    assert catalog.failures == ()
    assert len(catalog.sections) == LEAF_SECTIONS
    assert {op.key for op in catalog.operations} == {
        "GET /me",
        "GET /me/accessRestriction/{id}",
        "PUT /me/accessRestriction/{id}",
        "GET /me/invoices",
    }


def test_load_catalog_reports_a_missing_spec(tmp_path: Path) -> None:
    download_catalog(tmp_path, base_url=BASE_URL, versions=["1.0"], transport=_transport())
    (tmp_path / spec_filename("1.0", "account/profile")).unlink()
    catalog = load_catalog(tmp_path)
    assert any("missing spec" in failure for failure in catalog.failures)


def test_catalog_find_matches_path_method_and_summary(tmp_path: Path) -> None:
    download_catalog(tmp_path, base_url=BASE_URL, versions=["1.0"], transport=_transport())
    catalog = load_catalog(tmp_path)
    assert [op.key for op in catalog.find("invoices")] == ["GET /me/invoices"]
    assert [op.key for op in catalog.find("PUT")] == ["PUT /me/accessRestriction/{id}"]
    assert catalog.find("nothing-here") == []


def test_catalog_for_path_ignores_the_trailing_slash(tmp_path: Path) -> None:
    download_catalog(tmp_path, base_url=BASE_URL, versions=["1.0"], transport=_transport())
    catalog = load_catalog(tmp_path)
    assert [op.method for op in catalog.for_path("/me/")] == ["GET"]


def test_load_catalog_survives_an_empty_cache(tmp_path: Path) -> None:
    catalog = load_catalog(tmp_path)
    assert catalog.operations == ()
    assert catalog.sections == ()


def test_default_versions_cover_the_documented_surface() -> None:
    assert API_VERSIONS == ("1.0", "v2")


def test_catalog_rows_are_json_serialisable(tmp_path: Path) -> None:
    download_catalog(tmp_path, base_url=BASE_URL, versions=["1.0"], transport=_transport())
    catalog = load_catalog(tmp_path)
    json.dumps([op.as_dict() for op in catalog.operations])
