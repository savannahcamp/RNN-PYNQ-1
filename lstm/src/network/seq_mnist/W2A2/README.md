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
      1.698        0.000                      0                45198        0.020        0.000                      0                45198        3.750        0.000                       0                 15526  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 10971 |     0 |     53200 | 20.62 |
|   LUT as Logic             |  9864 |     0 |     53200 | 18.54 |
|   LUT as Memory            |  1107 |     0 |     17400 |  6.36 |
|     LUT as Distributed RAM |    10 |     0 |           |       |
|     LUT as Shift Register  |  1097 |     0 |           |       |
| Slice Registers            | 13578 |     0 |    106400 | 12.76 |
|   Register as Flip Flop    | 13578 |     0 |    106400 | 12.76 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |   323 |     0 |     26600 |  1.21 |
| F8 Muxes                   |   128 |     0 |     13300 |  0.96 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    | 45.5 |     0 |       140 | 32.50 |
|   RAMB36/FIFO*    |    4 |     0 |       140 |  2.86 |
|     RAMB36E1 only |    4 |       |           |       |
|   RAMB18          |   83 |     0 |       280 | 29.64 |
|     RAMB18E1 only |   83 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |  137 |     0 |       220 | 62.27 |
|   DSP48E1 only |  137 |       |           |       |
+----------------+------+-------+-----------+-------+
```
