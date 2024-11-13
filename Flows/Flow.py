from abc import ABC, abstractmethod

class Flow(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def evaluate(self, design_rtl):
        pass