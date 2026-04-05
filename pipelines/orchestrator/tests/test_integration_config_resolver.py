from __future__ import annotations

from pathlib import Path

from ai.pipelines.orchestrator.orchestration.integration_config_resolver import (
    IntegrationConfigResolver,
)


def test_resolver_prefers_master_training_gap_closure_from_task_sync_projects(
    tmp_path: Path,
):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "integration": {
            "asana": {
              "project_id": "111",
              "task_sync_projects": {
                "active_sprint": "222",
                "master_training_gap_closure": "333"
              },
              "all_projects": {
                "active_sprint": "444",
                "master_training_epic": "555"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    resolver = IntegrationConfigResolver(config_path=config_path)

    assert resolver.resolve_training_asana_project_gid() == "333"


def test_resolver_falls_back_to_legacy_project_id(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "integration": {
            "asana": {
              "project_id": "999"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    resolver = IntegrationConfigResolver(config_path=config_path)

    assert resolver.resolve_training_asana_project_gid() == "999"


def test_resolver_retries_after_invalid_json(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{invalid-json", encoding="utf-8")

    resolver = IntegrationConfigResolver(config_path=config_path)

    assert resolver.resolve_training_asana_project_gid() is None

    config_path.write_text(
        """
        {
          "integration": {
            "asana": {
              "project_id": "777"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    assert resolver.resolve_training_asana_project_gid() == "777"
