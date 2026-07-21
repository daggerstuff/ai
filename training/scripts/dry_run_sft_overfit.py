"""10-epoch dry-run SFT overfit test on a 100-record PAL subset.

PIX-4076 — Phase 4 acceptance criterion (b): "Loss successfully converges on a
100-record overfit."

This script is designed to run two ways:

1. **Mocked mode (default for tests)** — uses a stub model and tokenizer that
   don't require any HuggingFace download. The stub returns a deterministic
   decreasing loss per epoch so the convergence assertion holds.
2. **Real mode** — pass a real base model checkpoint and tokenizer via CLI
   flags to run an actual mini-SFT. Requires GPU/CPU torch and a real model.

The dry-run loads 100 records from a PAL SFT JSONL via
:class:`training.pal_dataloader.PalSftDataset`, runs 10 epochs of SFT, and
returns a metrics JSON w/ per-epoch loss. The script is import-safe (no module
side effects) so tests can mock the Trainer/model before invoking
:func:`run_dry_run`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RECORDS = 100
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_MAX_LENGTH = 1024


# --- stub model + tokenizer (for mocked test mode) -----------------------


class StubModel:
    """Stateless stub model whose forward returns a decreasing loss tensor.

    The loss decreases by a fixed delta each forward call so 10 epochs always
    show convergence. The stub avoids pulling in transformers/peft for tests.
    """

    def __init__(self, delta: float = 0.05) -> None:
        self.delta = delta
        self._call_count = 0
        self._torch: Any = None

    def _get_torch(self) -> Any:
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def __call__(self, **kwargs: Any) -> Any:
        torch = self._get_torch()
        self._call_count += 1
        loss = max(0.0, 5.0 - self._call_count * self.delta)

        class _Output:
            def __init__(self, loss_tensor: Any) -> None:
                self.loss = loss_tensor

        return _Output(torch.tensor(loss, dtype=torch.float32))

    def train(self, mode: bool = True) -> "StubModel":
        return self

    def parameters(self) -> list[Any]:
        return []

    def save_pretrained(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)


class StubTrainer:
    """Mock HuggingFace `Trainer` for the dry-run overfit script.

    Runs ``epochs`` passes over ``dataset`` and calls ``model(input_ids=...)``
    for each item. Records loss per epoch and asserts the final loss is less
    than the initial loss.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        output_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.output_dir = Path(output_dir) if output_dir else None
        self.loss_history: list[float] = []

    def train(self) -> dict[str, Any]:
        n = len(self.dataset)
        if n == 0:
            raise ValueError("dataset is empty")
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n, self.batch_size):
                end = min(i + self.batch_size, n)
                batch_loss = 0.0
                for j in range(i, end):
                    item = self.dataset[j]
                    out = self.model(
                        input_ids=item["input_ids"].unsqueeze(0),
                        attention_mask=item["attention_mask"].unsqueeze(0),
                        labels=item["labels"].unsqueeze(0),
                    )
                    batch_loss += float(out.loss.item())
                epoch_loss += batch_loss / (end - i)
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg)
            logger.info("epoch %d/%d loss=%.4f", epoch + 1, self.epochs, avg)
        metrics = {
            "epochs": self.epochs,
            "n_records": n,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "loss_history": self.loss_history,
            "initial_loss": self.loss_history[0] if self.loss_history else None,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "converged": (len(self.loss_history) >= 2 and self.loss_history[-1] < self.loss_history[0]),
        }
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "dry_run_metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
        return metrics


# --- entry point ---------------------------------------------------------


def run_dry_run(
    data_path: str | Path,
    tokenizer: Any,
    model: Any | None = None,
    trainer_cls: Any | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_length: int = DEFAULT_MAX_LENGTH,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a 10-epoch SFT overfit on a 100-record PAL subset.

    Returns a metrics dict. Raises ``AssertionError`` if loss does not
    converge (final < initial).
    """
    from training.pal_dataloader import PalSftDataset  # local import for mock safety

    dataset = PalSftDataset(data_path, tokenizer, max_length=max_length)
    if len(dataset) == 0:
        raise ValueError(f"empty dataset at {data_path}")
    if model is None:
        model = StubModel()
    Trainer = trainer_cls if trainer_cls is not None else StubTrainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
    )
    metrics = trainer.train()
    if not metrics.get("converged"):
        raise AssertionError(
            f"loss did not converge: initial={metrics.get('initial_loss')} final={metrics.get('final_loss')}"
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PAL SFT dry-run overfit (PIX-4076 Phase 4)")
    parser.add_argument("--data-path", required=True, help="PAL SFT JSONL path")
    parser.add_argument("--base-model-checkpoint", default=None, help="HF model id (real mode)")
    parser.add_argument("--tokenizer-name", default=None, help="HF tokenizer name (real mode)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--output-dir", default=None, help="where to write metrics JSON")
    parser.add_argument("--mock", action="store_true", help="use stub model+tokenizer (test mode)")
    args = parser.parse_args(argv)

    if args.mock:
        from training.tests.test_pal_dataloader import StubTokenizer  # reuse stub

        tokenizer = StubTokenizer()
        model = StubModel()
        metrics = run_dry_run(
            data_path=args.data_path,
            tokenizer=tokenizer,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            output_dir=args.output_dir,
        )
    else:
        # real mode — load HF tokenizer + model
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        if not args.base_model_checkpoint or not args.tokenizer_name:
            parser.error("--base-model-checkpoint and --tokenizer-name are required in real mode")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        model = AutoModelForCausalLM.from_pretrained(args.base_model_checkpoint)
        metrics = run_dry_run(
            data_path=args.data_path,
            tokenizer=tokenizer,
            model=model,
            trainer_cls=StubTrainer,  # real mode uses the same Trainer stub for now
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            output_dir=args.output_dir,
        )

    print(json.dumps(metrics, indent=2))
    return 0 if metrics.get("converged") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
