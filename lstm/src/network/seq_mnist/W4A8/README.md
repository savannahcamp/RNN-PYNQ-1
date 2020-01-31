# W4A8 Weights and Folding description

Weights and Folding description of the W4A8 neural network.

## Utilization reports

Utilization reports using Vivado Design Suite 2019.2:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints  THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints  
    -------      -------  ---------------------  -------------------      -------      -------  ---------------------  -------------------     --------     --------  ----------------------  --------------------  
      0.545        0.000                      0                89388        0.011        0.000                      0                89388        3.750        0.000                       0                 38331  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 34791 |     0 |     53200 | 65.40 |
|   LUT as Logic             | 34487 |     0 |     53200 | 64.83 |
|   LUT as Memory            |   304 |     0 |     17400 |  1.75 |
|     LUT as Distributed RAM |    10 |     0 |           |       |
|     LUT as Shift Register  |   294 |     0 |           |       |
| Slice Registers            | 37655 |     0 |    106400 | 35.39 |
|   Register as Flip Flop    | 37655 |     0 |    106400 | 35.39 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |     3 |     0 |     26600 |  0.01 |
| F8 Muxes                   |     0 |     0 |     13300 |  0.00 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    | 69.5 |     0 |       140 | 49.64 |
|   RAMB36/FIFO*    |   48 |     0 |       140 | 34.29 |
|     RAMB36E1 only |   48 |       |           |       |
|   RAMB18          |   43 |     0 |       280 | 15.36 |
|     RAMB18E1 only |   43 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |    9 |     0 |       220 |  4.09 |
|   DSP48E1 only |    9 |       |           |       |
+----------------+------+-------+-----------+-------+
```
