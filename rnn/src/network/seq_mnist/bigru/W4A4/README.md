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
      0.280        0.000                      0                38138        0.014        0.000                      0                38138        3.750        0.000                       0                 16779  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 16764 |     0 |     53200 | 31.51 |
|   LUT as Logic             | 15841 |     0 |     53200 | 29.78 |
|   LUT as Memory            |   923 |     0 |     17400 |  5.30 |
|     LUT as Distributed RAM |    10 |     0 |           |       |
|     LUT as Shift Register  |   913 |     0 |           |       |
| Slice Registers            | 15201 |     0 |    106400 | 14.29 |
|   Register as Flip Flop    | 15201 |     0 |    106400 | 14.29 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |   224 |     0 |     26600 |  0.84 |
| F8 Muxes                   |     0 |     0 |     13300 |  0.00 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    |   37 |     0 |       140 | 26.43 |
|   RAMB36/FIFO*    |    4 |     0 |       140 |  2.86 |
|     RAMB36E1 only |    4 |       |           |       |
|   RAMB18          |   66 |     0 |       280 | 23.57 |
|     RAMB18E1 only |   66 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |    5 |     0 |       220 |  2.27 |
|   DSP48E1 only |    5 |       |           |       |
+----------------+------+-------+-----------+-------+
```