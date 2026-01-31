#!/usr/bin/env python3
"""
Configuration Locking System
Freezes configuration, seeds, and git commit info for reproducibility
"""

import json
import random
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib


@dataclass
class GitInfo:
    """Git repository information"""
    commit_sha: str
    commit_message: str
    branch: str
    is_dirty: bool
    remote_url: Optional[str] = None

    @classmethod
    def capture(cls, repo_path: Optional[Path] = None) -> "GitInfo":
        """Capture current git state"""
        repo_path = repo_path or Path.cwd()

        try:
            # Get commit SHA
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            commit_sha = result.stdout.strip()

            # Get commit message
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            commit_message = result.stdout.strip()

            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            branch = result.stdout.strip()

            # Check if working directory is dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            is_dirty = len(result.stdout.strip()) > 0

            # Get remote URL
            remote_url = None
            try:
                result = subprocess.run(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                remote_url = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass

            return cls(
                commit_sha=commit_sha,
                commit_message=commit_message,
                branch=branch,
                is_dirty=is_dirty,
                remote_url=remote_url
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Git not available or not a git repo
            return cls(
                commit_sha="unknown",
                commit_message="unknown",
                branch="unknown",
                is_dirty=False,
                remote_url=None
            )


@dataclass
class LockedConfig:
    """Locked configuration with reproducibility info"""
    # Timestamp
    created_at: str

    # Git information
    git_info: GitInfo

    # Random seed
    random_seed: int

    # Configuration snapshot
    config_snapshot: Dict[str, Any]

    # Environment info
    python_version: str
    platform: str

    # Config hash for verification
    config_hash: str = field(default="")

    def __post_init__(self):
        """Calculate config hash after initialization"""
        if not self.config_hash:
            # Create hash from config snapshot
            config_str = json.dumps(self.config_snapshot, sort_keys=True)
            self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'created_at': self.created_at,
            'git_info': asdict(self.git_info),
            'random_seed': self.random_seed,
            'config_snapshot': self.config_snapshot,
            'python_version': self.python_version,
            'platform': self.platform,
            'config_hash': self.config_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LockedConfig":
        """Create from dictionary"""
        git_info = GitInfo(**data['git_info'])
        return cls(
            created_at=data['created_at'],
            git_info=git_info,
            random_seed=data['random_seed'],
            config_snapshot=data['config_snapshot'],
            python_version=data['python_version'],
            platform=data['platform'],
            config_hash=data.get('config_hash', '')
        )

    def save(self, path: Path) -> None:
        """Save locked config to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LockedConfig":
        """Load locked config from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def lock_config(config: Dict[str, Any], seed: Optional[int] = None,
                repo_path: Optional[Path] = None) -> LockedConfig:
    """Lock a configuration with reproducibility info"""
    import sys
    import platform

    # Generate seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    # Set random seed
    random.seed(seed)

    # Capture git info
    git_info = GitInfo.capture(repo_path)

    # Create locked config
    locked = LockedConfig(
        created_at=datetime.utcnow().isoformat() + "Z",
        git_info=git_info,
        random_seed=seed,
        config_snapshot=config,
        python_version=sys.version,
        platform=platform.platform()
    )

    return locked


def apply_locked_config(locked_config: LockedConfig) -> None:
    """Apply a locked configuration (set random seed)"""
    random.seed(locked_config.random_seed)
    # Note: Config snapshot should be applied by the caller

