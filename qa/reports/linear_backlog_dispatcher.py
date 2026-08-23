#!/usr/bin/env python3
"""Dispatch Linear backlog actions from gap-conversion payloads."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _format_payload(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


class LinearBacklogDispatcher:
    """Write actionable Linear artifacts and optionally submit them via GraphQL."""

    def __init__(
        self,
        *,
        queue_path: str = "monitoring/linear_backlog_artifacts/pending_actions.jsonl",
        linear_api_url: str = "https://api.linear.app/graphql",
        linear_api_key_env: str = "LINEAR_API_KEY",
        linear_team_key: str | None = None,
        linear_team_id_env: str = "LINEAR_TEAM_ID",
        linear_project_id_env: str = "LINEAR_PROJECT_ID",
        linear_parent_issue_id_env: str = "LINEAR_PARENT_ISSUE_ID",
        request_timeout: int = 12,
        max_retries: int = 3,
        issue_state_path: str | None = None,
    ):
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.linear_api_url = linear_api_url
        self.linear_api_key = os.getenv(linear_api_key_env, "").strip()
        self.linear_team_key = linear_team_key or os.getenv("LINEAR_TEAM_KEY", "").strip()
        self.linear_team_id = os.getenv(linear_team_id_env, "").strip()
        self.linear_project_id = os.getenv(linear_project_id_env, "").strip()
        self.linear_parent_issue_id = os.getenv(linear_parent_issue_id_env, "").strip()
        self.request_timeout = request_timeout
        self.max_retries = max(1, max_retries)
        self.state_path = (
            Path(issue_state_path) if issue_state_path else self.queue_path.parent / "linear_backlog_issue_state.json"
        )
        self._cached_team_id = None
        # Circuit breaker settings
        self.failure_threshold = 5
        self.recovery_timeout = 30  # seconds
        self.failure_count = 0
        self.state = "closed"  # closed, open, half-open
        self.last_failure_time = None

    def _load_issue_state(self) -> dict[str, str]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_issue_state(self, state: dict[str, str]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @property
    def has_linear_credentials(self) -> bool:
        return bool(self.linear_api_key)

    def _post_graphql(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_data = _format_payload(payload).encode("utf-8")
        req = urllib.request.Request(
            self.linear_api_url,
            data=request_data,
            headers={
                "Authorization": f"Bearer {self.linear_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        last_error: urllib.error.URLError | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                    raw = response.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    raise
                time.sleep(min(2**attempt, 4))
        if last_error is not None:
            raise last_error
        raise RuntimeError("GraphQL request failed without exception")

    def _append_queue_record(self, record: dict[str, Any]) -> str:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with self.queue_path.open("a", encoding="utf-8") as queue_file:
            queue_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return str(self.queue_path)

    def _fetch_team_id(self) -> str | None:
        if self._cached_team_id is not None:
            return self._cached_team_id
        if not self.linear_team_key:
            return None
        query = {
            "query": """
                query GetTeamByKey {
                    teams {
                        nodes {
                            id
                            key
                        }
                    }
                }
            """
        }
        payload = {"query": query["query"]}
        response = self._post_graphql(payload)
        teams = response.get("data", {}).get("teams", {}).get("nodes", [])
        for team in teams:
            if team.get("key") == self.linear_team_key:
                self._cached_team_id = team.get("id")
                return self._cached_team_id
        return None

    def _build_linear_issue_input(self, action: dict[str, Any]) -> dict[str, Any]:
        issue_input: dict[str, Any] = {
            "title": action.get("title", "Untitled backlog action"),
            "description": action.get("description", ""),
        }

        team_id = self.linear_team_id or self._fetch_team_id()
        if team_id:
            issue_input["teamId"] = team_id

        if self.linear_project_id:
            issue_input["projectId"] = self.linear_project_id

        if self.linear_parent_issue_id:
            issue_input["parentId"] = self.linear_parent_issue_id

        if action.get("priority"):
            issue_input["priority"] = action["priority"]

        return issue_input

    def _build_linear_issue_update_input(self, action: dict[str, Any], issue_id: str) -> dict[str, Any]:
        update_input: dict[str, Any] = {
            "id": issue_id,
            "title": action.get("title", "Untitled backlog action"),
            "description": action.get("description", ""),
        }
        if action.get("priority"):
            update_input["priority"] = action["priority"]
        return update_input

    def _mutation_create(self, action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return (
            """
            mutation($input: IssueCreateInput!) {
              issueCreate(input: $input) {
                success
                issue {
                  id
                  title
                }
              }
            }
            """,
            {"input": self._build_linear_issue_input(action)},
        )

    def _mutation_update(self, action: dict[str, Any], issue_id: str) -> tuple[str, dict[str, Any]]:
        return (
            """
            mutation($input: IssueUpdateInput!) {
              issueUpdate(input: $input) {
                success
                issue {
                  id
                  title
                }
              }
            }
            """,
            {"input": self._build_linear_issue_update_input(action, issue_id)},
        )

    def _extract_issue(self, response: dict[str, Any], action: str) -> dict[str, Any] | None:
        payload = response.get("data", {})
        container = payload.get(f"issue{action}", {})
        if container.get("success") and container.get("issue"):
            return container["issue"]
        return None

    def _handle_api_failure(self, action: dict[str, Any], record: dict[str, Any]) -> None:
        error = action.get("error") or action.get("errors") or "Create/update call unsuccessful"
        record["status"] = "failed"
        record["error"] = error

    @staticmethod
    def _is_stale_issue_error(error: str | None) -> bool:
        if not error:
            return False
        normalized = error.lower()
        return "not found" in normalized or "does not exist" in normalized or "cannot find" in normalized

    def _dispatch_update(self, action: dict[str, Any], issue_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        mutation, variables = self._mutation_update(action, issue_id)
        mutation_payload = {
            "query": mutation,
            "variables": variables,
        }
        response = self._post_graphql(mutation_payload)
        issue = self._extract_issue(response, "Update")
        error = response.get("errors") if isinstance(response, dict) else None
        return (
            "updated" if issue else "failed",
            issue,
            str(error) if error else None,
        )

    def _dispatch_create(self, action: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        mutation, variables = self._mutation_create(action)
        mutation_payload = {
            "query": mutation,
            "variables": variables,
        }
        response = self._post_graphql(mutation_payload)
        issue = self._extract_issue(response, "Create")
        error = response.get("errors") if isinstance(response, dict) else None
        return (
            "created" if issue else "failed",
            issue,
            str(error) if error else None,
        )

    def _resolve_dispatch_status(
        self,
        action: dict[str, Any],
        issue_state: dict[str, str],
    ) -> tuple[str, dict[str, Any] | None, str | None]:
        existing_issue_id = issue_state.get(action.get("change_id") or "")
        if existing_issue_id:
            status, issue, error = self._dispatch_update(action, existing_issue_id)
            if status == "failed" and self._is_stale_issue_error(error):
                issue_state.pop(action.get("change_id") or "", None)
                return self._dispatch_create(action)
            return status, issue, error
        return self._dispatch_create(action)

    def dispatch_backlog_actions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist Linear-ready actions to queue and create issues when credentials exist."""
        actions = payload.get("actions", []) if isinstance(payload, dict) else []
        if not isinstance(actions, list):
            actions = []

        result = {
            "dispatched_at": datetime.now(UTC).isoformat(),
            "attempted": len(actions),
            "created": 0,
            "queued": 0,
            "failed": 0,
            "updated": 0,
            "mode": "create" if self.has_linear_credentials else "queue_only",
            "queue_file": str(self.queue_path),
            "state_file": str(self.state_path),
            "items": [],
        }

        issue_state = self._load_issue_state()

        for index, action in enumerate(actions):
            # Check circuit breaker state
            if self.state == "open":
                # Check if recovery timeout has passed
                if (
                    self.last_failure_time
                    and (datetime.now(UTC) - self.last_failure_time).total_seconds() > self.recovery_timeout
                ):
                    self.state = "half-open"
                else:
                    # Circuit is open, fail fast
                    record = {
                        "action_index": index,
                        "change_id": action.get("change_id"),
                        "title": action.get("title"),
                        "status": "failed",
                        "queued_at": datetime.now(UTC).isoformat(),
                        "error": "Circuit breaker is open",
                    }
                    self._append_queue_record({**record, "payload": action})
                    result["failed"] += 1
                    result["items"].append(record)
                    continue

            record = {
                "action_index": index,
                "change_id": action.get("change_id"),
                "title": action.get("title"),
                "status": "queued",
                "queued_at": datetime.now(UTC).isoformat(),
                "error": None,
            }

            if not self.has_linear_credentials:
                record["status"] = "queued"
                self._append_queue_record({**record, "payload": action})
                result["queued"] += 1
                result["items"].append(record)
                continue

            try:
                status, issue, error = self._resolve_dispatch_status(action, issue_state)

                if status == "updated":
                    record["status"] = "updated"
                    record["linear_issue"] = issue or {}
                    result["updated"] += 1
                    if issue is None:
                        self._handle_api_failure(
                            {"error": "Issue is None despite updated status"},
                            record,
                        )
                        self._append_queue_record({**record, "payload": action})
                        result["failed"] += 1
                        result["items"].append(record)
                        continue
                    if action.get("change_id"):
                        existing_issue_id = issue_state.get(action["change_id"])
                        issue_state[action["change_id"]] = issue.get("id", existing_issue_id)
                    self.failure_count = 0
                    self.state = "closed"
                elif status == "created":
                    record["status"] = "created"
                    record["linear_issue"] = issue or {}
                    result["created"] += 1
                    if issue is None:
                        self._handle_api_failure(
                            {"error": "Issue is None despite created status"},
                            record,
                        )
                        self._append_queue_record({**record, "payload": action})
                        result["failed"] += 1
                        result["items"].append(record)
                        continue
                    if action.get("change_id"):
                        issue_state[action["change_id"]] = issue.get("id", "")
                    self.failure_count = 0
                    self.state = "closed"
                else:
                    self._handle_api_failure(
                        {"error": error, "errors": error},
                        record,
                    )
                    self._append_queue_record({**record, "payload": action})
                    result["failed"] += 1
            except (urllib.error.URLError, OSError, ValueError, TypeError, KeyError) as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                self._append_queue_record({**record, "payload": action})
                result["failed"] += 1

                self.failure_count += 1
                self.last_failure_time = datetime.now(UTC)
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"

            result["items"].append(record)

        self._save_issue_state(issue_state)

        return result


__all__ = ["LinearBacklogDispatcher"]
