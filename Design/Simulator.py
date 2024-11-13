from abc import ABC, abstractmethod

class Simulator(ABC):
    def __init__(self,
                 simulator_path : str = None,
                 simulator_output : str = None) -> None:
        
        self.simulator_path = simulator_path
        self.simulator_output = simulator_output
        self.simulate_time = -1
        self.report = None
        super().__init__()

    @abstractmethod
    def simulate(self, X):
        pass

    def update_design_space(self, design_space : dict):
        
        self.design_space = design_space

        self.ub = [len(options) for _, options in self.design_space.items()]
        self.lb = [0] * len(self.design_space)
        return self.design_space