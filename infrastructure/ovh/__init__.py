"""OVHcloud API client and offline OpenAPI catalog."""

from ai.infrastructure.ovh.catalog import (
    API_VERSIONS,
    Catalog,
    Operation,
    Section,
    download_catalog,
    iter_sections,
    load_catalog,
    navigation_url,
    operations_from_spec,
    section_url,
    spec_filename,
)
from ai.infrastructure.ovh.client import (
    DEFAULT_API_VERSION,
    DEFAULT_BASE_URL,
    MissingTokenError,
    OvhApiError,
    OvhClient,
    find_dotenv,
    read_env_file,
    resolve_token,
    resolve_token_with_source,
)

__all__ = [
    "API_VERSIONS",
    "DEFAULT_API_VERSION",
    "DEFAULT_BASE_URL",
    "Catalog",
    "MissingTokenError",
    "Operation",
    "OvhApiError",
    "OvhClient",
    "Section",
    "download_catalog",
    "find_dotenv",
    "iter_sections",
    "load_catalog",
    "navigation_url",
    "operations_from_spec",
    "read_env_file",
    "resolve_token",
    "resolve_token_with_source",
    "section_url",
    "spec_filename",
]
