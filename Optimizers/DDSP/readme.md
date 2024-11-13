# Diversity-Driven Space Partition Learning (DDSP)

# Requirements

Main Requirements:
+ Python 3.10

Python Dependencies:
+ numpy
+ matplotlib
+ scikit-learn
+ botorch
+ pygmo
+ iccad-contest (optional, for BOOM DSE problem)

And all dependencies of above packages.

You can use conda to create a new environment. **(Highly Recommended**, for pygmo installation)

# Usage

Run multi-objective optimization problem experiments:
```
$ python ddsp_expr.py
```

Run BOOM microprocessor design space exploration experiments:
```
$ python ddsp_expr_boom.py
```

Modify `ddsp_expr.py` or `ddsp_expr_boom.py` to tune the hyperparameters and methods, or choice other MOO problems. 

# Reference

C. Bai, Q. Sun, J. Zhai, Y. Ma, B. Yu, and M. D. F. Wong, “BOOM-Explorer: RISC-V BOOM Microarchitecture Design Space Exploration”.

https://github.com/baichen318/boom-explorer-public

L. Wang, R. Fonseca, and Y. Tian, “Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search”.

https://github.com/facebookresearch/LaMCTS

Y. Zhao, L. Wang, K. Yang, T. Zhang, T. Guo, and Y. Tian, “Multi-objective Optimization by Learning Space Partitions”.

https://github.com/aoiang/LaMOO