import logging


log = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, enabled: bool, monitor: str, mode: str = "min", patience: int = 5) -> None:
        self.enabled = enabled
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.should_stop = False

    def check(self, metrics: dict[str, float]) -> None:
        if not self.enabled: 
            return
        
        score = metrics.get(self.monitor)
        if score is None: 
            return

        improved = (score < self.best_score) if self.mode == "min" else (score > self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                log.info(f"🛑 [EarlyStopping] Triggered. Best {self.monitor}: {self.best_score:.4f}")
