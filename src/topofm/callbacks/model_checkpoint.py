import torch
import logging
from topofm.models.model import Model
from pathlib import Path


log = logging.getLogger(__name__)

class ModelCheckpoint:
    def __init__(self, enabled: bool, dirpath: str, filename: str, monitor: str, mode: str = "min"):
        self.enabled = enabled
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
        
        if self.enabled:
            self.dirpath.mkdir(parents=True, exist_ok=True)

    def save_if_best(self, model: Model, metrics: dict, step: int):
        if not self.enabled: 
            return
        
        score = metrics.get(self.monitor)
        if score is None: 
            log.warning(f"[ModelCheckpoint] Metric {self.monitor} not found in metrics. Skipping checkpoint.")
            return

        improved = (score < self.best_score) if self.mode == "min" else (score > self.best_score)
        if improved:
            self.best_score = score
            save_path = self.dirpath / self.filename
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": step,
                "metrics": metrics,
            }, save_path)
            
            log.info(f"💾 [ModelCheckpoint] Saved new best model ({self.monitor}={score:.4f}) to {save_path}")