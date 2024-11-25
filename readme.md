# SPaDE: Space Partition Aided Design (Space) Exploration (For VexRiscv)

## Setup

For offline DSE:
Offline datasets are stored in `PresampledDataset`. 
You can check `examples/VexRiscv_offline.py` for a simple example.

For online DSE, you need to install the required tools for the hardware generation and evaluation, such as *Vivado* or *Yosys*.

You will need the VexRiscv generator for online DSE as well:

```shell
git submodule init
git submodule update Design/riscv/VexRiscv
```

VexRiscv requires some tools like *Verilator*, please check the original repo for detailed guide:
https://github.com/SpinalHDL/VexRiscv

## Run Examples

Offline DSE:
```
python examples/VexRiscv_offline.py
```

Online DSE:
```
python examples/VexRiscv_online.py
```

## Algorithm Settings

SpadeOptimizer is highly configurable, the default setting of our proposed method is as set in the examples:
```Python
method = 'ehvi'
weight_method = 'global-hv'
cluster_method = 'hybrid'
```
If you want to change them, please check the details in `Optimizers/DDSP/mcts/mcts.py`.

## Cite Us

```LaTeX
@inproceedings{space_partition_dse_iccd2024,
  title     = {Efficient Microprocessor Design Space Exploration via Space Partitioning},
  author    = {Zijun Jiang and Yangdi Lyu},
  booktitle = {The 42nd IEEE International Conference on Computer Design (ICCD)},
  year      = {2024}
}
```

## Acknowledgement

C. Bai, Q. Sun, J. Zhai, Y. Ma, B. Yu, and M. D. F. Wong, “BOOM-Explorer: RISC-V BOOM Microarchitecture Design Space Exploration”.

https://github.com/baichen318/boom-explorer-public

L. Wang, R. Fonseca, and Y. Tian, “Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search”.

https://github.com/facebookresearch/LaMCTS

Y. Zhao, L. Wang, K. Yang, T. Zhang, T. Guo, and Y. Tian, “Multi-objective Optimization by Learning Space Partitions”.

https://github.com/aoiang/LaMOO