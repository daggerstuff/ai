import json
import logging
from pathlib import Path
from rich.console import Console

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_data():
    console.print("[bold red]Data generation is currently paused pending strategy review.[/bold red]")

if __name__ == "__main__":
    generate_data()
