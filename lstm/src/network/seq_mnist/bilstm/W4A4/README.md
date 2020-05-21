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
      0.420        0.000                      0                52043        0.016        0.000                      0                52043        3.750        0.000                       0                 22917  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 16545 |     0 |     53200 | 31.10 |
|   LUT as Logic             | 16041 |     0 |     53200 | 30.15 |
|   LUT as Memory            |   504 |     0 |     17400 |  2.90 |
|     LUT as Distributed RAM |    10 |     0 |           |       |
|     LUT as Shift Register  |   494 |     0 |           |       |
| Slice Registers            | 21949 |     0 |    106400 | 20.63 |
|   Register as Flip Flop    | 21949 |     0 |    106400 | 20.63 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |     3 |     0 |     26600 |  0.01 |
| F8 Muxes                   |     0 |     0 |     13300 |  0.00 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    | 42.5 |     0 |       140 | 30.36 |
|   RAMB36/FIFO*    |   19 |     0 |       140 | 13.57 |
|     RAMB36E1 only |   19 |       |           |       |
|   RAMB18          |   47 |     0 |       280 | 16.79 |
|     RAMB18E1 only |   47 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |    9 |     0 |       220 |  4.09 |
|   DSP48E1 only |    9 |       |           |       |
+----------------+------+-------+-----------+-------+
```
