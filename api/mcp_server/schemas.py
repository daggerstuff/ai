from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field


class AddMemoryRequest(BaseModel):
    content: str
    user_id: str
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    visibility: Optional[str] = "private"
    include_shared: Optional[bool] = True
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchMemoryRequest(BaseModel):
    query: str
    user_id: str
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    include_shared: Optional[bool] = True
    limit: Optional[int] = 10


class UpdateMemoryRequest(BaseModel):
    content: Optional[str] = None
    user_id: str
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    include_shared: Optional[bool] = True
    metadata: Optional[Dict[str, Any]] = None


class HindsightRetainItem(BaseModel):
    content: str
    document_id: Optional[str] = None
    context: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class HindsightRetainRequest(BaseModel):
    items: List[HindsightRetainItem]


class HindsightRecallRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    tags: Optional[List[str]] = None
    tags_match: Optional[str] = "any"


class ScopeRequest(Protocol):
    user_id: str
    org_id: Optional[str]
    project_id: Optional[str]
    session_id: Optional[str]
    agent_id: Optional[str]
    run_id: Optional[str]
    include_shared: Optional[bool]
