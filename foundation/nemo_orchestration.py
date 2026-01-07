"""
NeMo Microservices Orchestration and Configuration.

Handles:
- Docker Compose service coordination
- Data Designer for synthetic therapeutic datasets
- Guardrails for conversation safety
- Evaluator for quality metrics
- Customizer for therapeutic approach fine-tuning
- Safe Synthesizer for data generation
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class NeMoMicroservicesManager:
    """Orchestrates NeMo microservices for therapeutic enhancement."""

    def __init__(self):
        self.quickstart_path = Path(
            "/home/vivi/pixelated/ngc_public_therapeutic_resources/microservices/nemo-microservices-quickstart_v25.10"
        )
        self.docker_compose_file = self.quickstart_path / "docker-compose.yaml"
        self.services_dir = self.quickstart_path / "services"

    def validate_installation(self) -> Dict[str, bool]:
        """Validate NeMo microservices are correctly installed."""
        return {
            "quickstart_path_exists": self.quickstart_path.exists(),
            "docker_compose_exists": self.docker_compose_file.exists(),
            "services_directory_exists": self.services_dir.exists(),
        }

    def list_available_services(self) -> List[str]:
        """List available NeMo services."""
        if not self.services_dir.exists():
            return []

        services = [item.name for item in self.services_dir.iterdir() if item.is_dir()]
        return sorted(services)

    def get_service_config(self, service_name: str) -> Optional[Dict]:
        """Load configuration for a specific service."""
        service_config_path = self.services_dir / service_name / "config.json"
        if not service_config_path.exists():
            return None

        with open(service_config_path) as f:
            return json.load(f)

    def status(self) -> str:
        """Generate status report of NeMo infrastructure."""
        lines = [
            "=== NeMo Microservices Status ===",
            "\nValidation:",
        ]

        checks = self.validate_installation()
        lines.extend(
            f"  {'✓' if result else '✗'} {check_name}"
            for check_name, result in checks.items()
        )

        services = self.list_available_services()
        lines.append(f"\nAvailable Services ({len(services)}):")
        lines.extend(f"  - {service}" for service in services)

        return "\n".join(lines)


if __name__ == "__main__":
    manager = NeMoMicroservicesManager()
    print(manager.status())
