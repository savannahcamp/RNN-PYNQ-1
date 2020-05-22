# W4A4 Weights and Folding description

Weights and Folding description of the W4A4 neural network.

## Utilization reports

Utilization reports using Vivado Design Suite 2019.2:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints  THS Total Endpoints     WPWS(ns)     TPWS(ns)  TPWS Failing Endpoints  TPWS Total Endpoints  
    -------      -------  ---------------------  -------------------      -------      -------  ---------------------  -------------------     --------     --------  ----------------------  --------------------  
      0.169        0.000                      0                62957        0.022        0.000                      0                62957        3.750        0.000                       0                 27711  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 35534 |     0 |     53200 | 66.79 |
|   LUT as Logic             | 34846 |     0 |     53200 | 65.50 |
|   LUT as Memory            |   688 |     0 |     17400 |  3.95 |
|     LUT as Distributed RAM |    10 |     0 |           |       |
|     LUT as Shift Register  |   678 |     0 |           |       |
| Slice Registers            | 26396 |     0 |    106400 | 24.81 |
|   Register as Flip Flop    | 26396 |     0 |    106400 | 24.81 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |     3 |     0 |     26600 |  0.01 |
| F8 Muxes                   |     0 |     0 |     13300 |  0.00 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    | 59.5 |     0 |       140 | 42.50 |
|   RAMB36/FIFO*    |   11 |     0 |       140 |  7.86 |
|     RAMB36E1 only |   11 |       |           |       |
|   RAMB18          |   97 |     0 |       280 | 34.64 |
|     RAMB18E1 only |   97 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |    6 |     0 |       220 |  2.73 |
|   DSP48E1 only |    6 |       |           |       |
+----------------+------+-------+-----------+-------+
```
