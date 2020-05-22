# W2A2 Weights and Folding description

Weights and Folding description of the W2A2 neural network.

## Utilization reports

Utilization reports using Vivado Design Suite 2019.2:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints  THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints  
    -------      -------  ---------------------  -------------------      -------      -------  ---------------------  -------------------     --------     --------  ----------------------  --------------------  
      1.481        0.000                      0                31351        0.021        0.000                      0                31351        3.750        0.000                       0                 11349  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 |  7326 |     0 |     53200 | 13.77 |
|   LUT as Logic             |  6581 |     0 |     53200 | 12.37 |
|   LUT as Memory            |   745 |     0 |     17400 |  4.28 |
|     LUT as Distributed RAM |    10 |     0 |           |       |
|     LUT as Shift Register  |   735 |     0 |           |       |
| Slice Registers            | 10116 |     0 |    106400 |  9.51 |
|   Register as Flip Flop    | 10116 |     0 |    106400 |  9.51 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |   238 |     0 |     26600 |  0.89 |
| F8 Muxes                   |    68 |     0 |     13300 |  0.51 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    |   23 |     0 |       140 | 16.43 |
|   RAMB36/FIFO*    |    4 |     0 |       140 |  2.86 |
|     RAMB36E1 only |    4 |       |           |       |
|   RAMB18          |   38 |     0 |       280 | 13.57 |
|     RAMB18E1 only |   38 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |   69 |     0 |       220 | 31.36 |
|   DSP48E1 only |   69 |       |           |       |
+----------------+------+-------+-----------+-------+
```
