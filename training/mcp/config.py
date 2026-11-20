"""Configuration for the training-infra MCP service.

Env resolution order: process environment, then the first existing dotenv
file among ``TRAINING_MCP_ENV_FILE``, ``<pkg>/.env``, and the pixelated
workspace ``.env``. Secrets never live in this package's tracked files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
AI_ROOT = PKG_DIR.parents[1]
WORKSPACE_ROOT = AI_ROOT.parent

DEFAULT_ENV_FILES = (
    PKG_DIR / ".env",
    WORKSPACE_ROOT / ".env",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw = stripped.partition("=")
        values[key.strip()] = raw.strip().strip("'\"")
    return values


def merged_env() -> dict[str, str]:
    """Process env overlaid on dotenv files, later files losing to earlier."""
    env = dict(os.environ)
    files = []
    override = os.environ.get("TRAINING_MCP_ENV_FILE")
    if override:
        files.append(Path(override))
    files.extend(DEFAULT_ENV_FILES)
    # Lowest precedence first: workspace .env, then package .env.
    for path in reversed(files):
        if path.exists():
            for key, value in _parse_env_file(path).items():
                env.setdefault(key, value)
    return env


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    db_path: Path
    env: dict[str, str] = field(default_factory=dict)

    @property
    def wandb_api_key(self) -> str:
        return self.env.get("WANDB_API_KEY", "")

    @property
    def wandb_project(self) -> str:
        return self.env.get("WANDB_PROJECT", "pixelated-empathy-kan28")

    @property
    def wandb_entity(self) -> str:
        return self.env.get("WANDB_ENTITY", "wutang")

    @property
    def mcp_token(self) -> str:
        return self.env.get("TRAINING_INFRA_MCP_TOKEN", self.env.get("TRAINING_MCP_TOKEN", ""))

    @property
    def max_concurrent_jobs(self) -> int:
        return int(self.env.get("TRAINING_MCP_MAX_CONCURRENT_JOBS", "1"))

    @property
    def daily_job_cap(self) -> int:
        return int(self.env.get("TRAINING_MCP_DAILY_JOB_CAP", "5"))

    @property
    def monthly_cost_cap_usd(self) -> float:
        return float(self.env.get("TRAINING_MCP_MONTHLY_COST_CAP_USD", "100.0"))

    @property
    def allow_remote(self) -> bool:
        """Explicit opt-in required before binding to a non-loopback address."""
        return self.env.get("TRAINING_MCP_ALLOW_REMOTE", "").lower() in {"1", "true", "yes"}

    @property
    def foresight_url(self) -> str:
        return self.env.get("FORESIGHT_URL", "http://127.0.0.1:8764/mcp")

    @property
    def foresight_enabled(self) -> bool:
        """Write terminal-state provenance to Foresight (epic acceptance criterion)."""
        return self.env.get("TRAINING_MCP_FORESIGHT", "1").lower() not in {"0", "false", "no", "off"}

    @property
    def binding_is_loopback(self) -> bool:
        return self.host in {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}

    @property
    def r2_configured(self) -> bool:
        return all(self.env.get(name) for name in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT"))

    @property
    def ovhai_available(self) -> bool:
        return os.path.exists(os.path.expanduser("~/.config/ovhai/config.json"))


def load_settings() -> Settings:
    env = merged_env()
    return Settings(
        host=env.get("TRAINING_MCP_HOST", "127.0.0.1"),
        port=int(env.get("TRAINING_MCP_PORT", "8766")),
        db_path=Path(env.get("TRAINING_MCP_DB", str(PKG_DIR / "state" / "jobs.db"))),
        env=env,
    )
