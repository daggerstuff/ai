from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class AddMemoryRequest(BaseModel):
    content: str
    user_id: str
    org_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    visibility: str | None = "private"
    include_shared: bool | None = True
    category: str | None = None
    metadata: dict[str, Any] | None = None


class SearchMemoryRequest(BaseModel):
    query: str
    user_id: str
    org_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    include_shared: bool | None = True
    limit: int | None = 10


class UpdateMemoryRequest(BaseModel):
    content: str | None = None
    user_id: str
    org_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    include_shared: bool | None = True
    metadata: dict[str, Any] | None = None


class ForesightRetainItem(BaseModel):
    content: str
    document_id: str | None = None
    context: str | None = None
    tags: list[str] = Field(default_factory=list)


class ForesightRetainRequest(BaseModel):
    items: list[ForesightRetainItem]


class ForesightRecallRequest(BaseModel):
    query: str
    limit: int | None = 10
    tags: list[str] | None = None
    tags_match: str | None = "any"


class ScopeRequest(Protocol):
    user_id: str
    org_id: str | None
    project_id: str | None
    session_id: str | None
    agent_id: str | None
    run_id: str | None
    include_shared: bool | None
