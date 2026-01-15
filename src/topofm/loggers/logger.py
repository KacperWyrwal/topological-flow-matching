from abc import ABC, abstractmethod

class Logger(ABC):
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
    
    @abstractmethod
    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass