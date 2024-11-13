import json
import argparse

from Flows import VivadoFlow
from Design.riscv import VexRiscvGenerator, VexRiscvSimulator
from Problems import VexRiscvProblem
from Optimizers import SpadeOptimizer

'''
Basic Example: Optimizing VexRiscv for TinyML Benchmarks (with presampled dataset) 
You can change the benchmark and the norm_ref values to:
| Benchmark     |   norm_ref   |
|---------------|--------------|
| tinyml_AD     | [0.50, 5000] |
| tinyml_KWS    | [10.0, 5000] |
| tinyml_VWW    | [50.0, 5000] |
| tinyml_RESNET | [50.0, 5000] |
'''

if __name__ == "__main__":

    vexgen = VexRiscvGenerator()
    vexsim = VexRiscvSimulator(benchmark="tinyml_AD")
    vvdflow = VivadoFlow("xcau25p-sfvb784-1-i")

    dataset_path = "PresampledDataset/VexRiscv/tinyml_AD_1963.json"

    with open(dataset_path, "r") as dataset:
        datadict = json.load(dataset)

    problem = VexRiscvProblem(vexgen, vexsim, vvdflow, datadict, norm_ref=[0.5, 5000])

    opt = SpadeOptimizer(problem=problem, 
                         ninits = 10, 
                         leaf_size = 10,
                         method = 'mcts-ehvi',
                         weight_method = 'global-hv',
                         cluster_method = 'hybrid',
                         minimize=True)
    
    test_iter = 21

    for _ in range(test_iter):
        X = opt.suggest()
        Y = problem.evaluate(X)
        opt.observe(X, Y)