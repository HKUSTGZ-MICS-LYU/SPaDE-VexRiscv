from abc import ABC, abstractmethod

import numpy as np
class Generator(ABC):

    lb = list
    ub = list
    design_space = dict

    def __init__(self,
                 generator_path : str = None,
                 generator_output : str = None) -> None:
        
        self.generator_path = generator_path
        self.generator_output = generator_output
        self.generator_time = -1
        super().__init__()
    
    # Generate a design from an embedding vector
    @abstractmethod
    def generate(self, X):
        pass

    # Validate (and correct) the embedding vector
    @abstractmethod
    def validate(self, X, return_corrected: bool = False):
        pass

    def update_design_space(self, design_space : dict):
        
        self.design_space = design_space

        self.ub = [len(options) for _, options in self.design_space.items()]
        self.lb = [0] * len(self.design_space)
        return self.design_space