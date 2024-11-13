import numpy as np

from Flows import VivadoFlow
from Design.riscv import VexRiscvGenerator, VexRiscvSimulator
from Problems import VexRiscvProblem
from Optimizers import SpadeOptimizer

'''
Basic Example: Optimizing VexRiscv for TinyML Benchmarks (without presampled dataset) 

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

    problem = VexRiscvProblem(vexgen, vexsim, vvdflow, 
                              norm_ref=[0.5, 5000], allow_invalid=True)

    opt = SpadeOptimizer(problem=problem, 
                         ninits = 1, 
                         leaf_size = 10,
                         method = 'ehvi',
                         weight_method = 'global-hv',
                         cluster_method = 'hybrid',
                         minimize=True)
    
    test_iter = 2

    for _ in range(test_iter):
        X = opt.suggest()

        # online optimization is applied on continuous space
        # float vector will be suggested by the optimizer
        # use floor() to discretize it
        X = np.floor(X).astype(int).tolist()

        Y = problem.evaluate(X)
        opt.observe(X, Y)
        problem.save("data.json")