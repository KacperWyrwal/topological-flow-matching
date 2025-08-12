from __future__ import annotations

from typing import Dict, List, Optional

import wandb


class WandBLogger:
    """Thin wrapper around a W&B run with helpers for common plots.

    Responsibilities:
    - Hold a reference to an active `wandb.Run`
    - Log per-epoch training scalars (loss, validation metrics)
    - Create two line plots from history at the end of training:
      1) loss vs epoch
      2) W1/W2 vs epoch (if available)
    - Create a bar plot for final test metrics
    - Store arbitrary summary fields that should not be plotted
    """

    def __init__(self, run: Optional[wandb.Run]):
        self._run = run
        self._make_plots = False

    @property
    def run(self) -> Optional[wandb.Run]:
        return self._run

    def is_enabled(self) -> bool:
        return self._run is not None

    def log_training_scalars(self, *, epoch: int, loss: float, eval_metrics: Dict[str, float] | None = None) -> None:
        if not self.is_enabled():
            return
        payload: Dict[str, float | int] = {
            "epoch": epoch,
            "train/loss": float(loss),
        }
        if eval_metrics:
            payload.update({f"val/{k}": float(v) for k, v in eval_metrics.items()})
        self._run.log(payload)

    def log_training_curves(self, history: Dict[str, List[float]]) -> None:
        """Create two line plots from the accumulated history.

        - loss vs epoch
        - W1/W2 vs epoch (if present)
        """
        if not self.is_enabled():
            return

        # loss vs epoch
        if "loss" in history and len(history["loss"]) > 0:
            xs = list(range(1, len(history["loss"]) + 1))
            ys = [history["loss"]]
            keys = ["loss"]
            loss_plot = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=keys,
                title="Training Loss vs Epoch",
                xname="epoch",
            )
            self._run.log({"plots/loss_vs_epoch": loss_plot})

        # validation curves (W1/W2) if available
        val_keys = [k for k in ("W1", "W2") if k in history and len(history[k]) > 0]
        if len(val_keys) > 0:
            xs = list(range(1, len(history[val_keys[0]]) + 1))
            ys = [history[k] for k in val_keys]
            val_plot = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=val_keys,
                title="Validation Metrics vs Epoch",
                xname="epoch",
            )
            self._run.log({"plots/val_vs_epoch": val_plot})

    def log_training(
        self,
        *,
        epoch: int,
        loss: float,
        eval_metrics: Dict[str, float] | None,
        history: Dict[str, List[float]],
    ) -> None:
        """Log both scalars for the current epoch and refresh the line plots."""
        if not self.is_enabled():
            return
        self.log_training_scalars(epoch=epoch, loss=loss, eval_metrics=eval_metrics)
        if self._make_plots:
            self.log_training_curves(history)

    def log_test_scalars(self, metrics: Dict[str, float]) -> None:
        """Log final test metrics as scalars."""
        if not self.is_enabled():
            return
        scalar_payload = {f"test/{k}": float(v) for k, v in metrics.items()}
        self._run.log(scalar_payload)

    def log_test_barplot(self, metrics: Dict[str, float]) -> None:
        """Log final test metrics as a bar plot."""
        if not self.is_enabled():
            return
        table = wandb.Table(data=[[k, float(v)] for k, v in metrics.items()], columns=["metric", "value"])
        bar = wandb.plot.bar(table, "metric", "value", title="Test Metrics")
        self._run.log({"plots/test_metrics": bar})

    def log_test(self, metrics: Dict[str, float]) -> None:
        """Log both test scalars and bar plot."""
        if not self.is_enabled():
            return
        self.log_test_scalars(metrics)
        if self._make_plots:
            self.log_test_barplot(metrics)

    def log_image(self, name: str, path: str) -> None:
        if not self.is_enabled():
            return
        self._run.log({name: wandb.Image(path)})

    def set_summary(self, key: str, value) -> None:
        if not self.is_enabled():
            return
        self._run.summary[key] = value

    def finish(self) -> None:
        if not self.is_enabled():
            return
        self._run.finish()


