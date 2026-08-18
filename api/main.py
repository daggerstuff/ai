"""Container entrypoint for the AKS Helm deployment.

This module exists because the production Docker image launches
`python -m ai.api.main`. Re-export the lightweight ASGI app used by the
project and provide a local uvicorn runner for container startup.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the ASGI app on the port expected by the Helm chart."""
    port = int(os.getenv("PIXEL_API_PORT", "8000"))
    uvicorn.run("ai.api.index:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
