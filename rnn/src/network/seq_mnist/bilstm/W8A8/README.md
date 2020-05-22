# W8A8 Weights and Folding description

Weights and Folding description of the W8A8 neural network.

## Utilization reports

Utilization reports using Vivado Design Suite 2019.2:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints  THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints  
    -------      -------  ---------------------  -------------------      -------      -------  ---------------------  -------------------     --------     --------  ----------------------  --------------------  
      0.296        0.000                      0                95387        0.021        0.000                      0                95387        3.750        0.000                       0                 41110  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 46139 |     0 |     53200 | 86.73 |
|   LUT as Logic             | 45837 |     0 |     53200 | 86.16 |
|   LUT as Memory            |   302 |     0 |     17400 |  1.74 |
|     LUT as Distributed RAM |    10 |     0 |           |       |
|     LUT as Shift Register  |   292 |     0 |           |       |
| Slice Registers            | 40398 |     0 |    106400 | 37.97 |
|   Register as Flip Flop    | 40398 |     0 |    106400 | 37.97 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |     3 |     0 |     26600 |  0.01 |
| F8 Muxes                   |     0 |     0 |     13300 |  0.00 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    | 87.5 |     0 |       140 | 62.50 |
|   RAMB36/FIFO*    |   66 |     0 |       140 | 47.14 |
|     RAMB36E1 only |   66 |       |           |       |
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
