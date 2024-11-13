from abc import ABC, abstractmethod
from Problems import Problem

class Optimizer(ABC):
    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.sampled_X = []
        self.sampled_Y = []

        self.ref_X = []
        self.ref_Y = []

        self.dataset_data()
        
    @abstractmethod
    def suggest(self):
        pass
    
    @abstractmethod
    def observe(self, X, Y):
        pass

    @abstractmethod
    def get_pareto(self):
        pass

    @abstractmethod
    def get_hv(self):
        pass

    def dataset_data(self):
        if self.problem.dataset == {}:
            return
        else:
            self.ref_X = self.problem.input_space
            self.ref_Y = self.problem.evaluate(self.ref_X)
            ref_max = [max(self.ref_Y[i] for i in range(self.problem.nobjs))]
            print(f"Maximum Reference Point: {ref_max}")
        return