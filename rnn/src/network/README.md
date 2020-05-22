# lstm-pynq

Uni and Bi GRU/LSTM topologies, working on different datasets and multiple precision, namely:
 
 - "plain": plain-text dataset from [Insiders Technologies GmbH](https://www.insiders-technologies.de/home.html).
 - "seq_mnist": sequence of mnist digits from [my ctc-ocr-pytorch repo](https://github.com/ussamazahid96/ctc-ocr-pytorch).

For each dataset, there is a folder structure like this:

 - `<dataset>/<network>/WxAy/` contains the exported weights for a specific precision for both weigths (x bits) and activations (y bits)
 	Note: To use the unidirectional version, when running inference on hardware, uncomment the line for `bidirectional_enabled=False` in the [test_ocr.py](https://github.com/ussamazahid96/RNN-PYNQ/blob/master/tests/test_ocr.py)
 - `<dataset>/<network>/top.cpp` is the top level file, synthesized by Vivado HLS, of the RNN accelerator
 - `<dataset>/<network>/hw_config.hpp` contains defines for datatypes used within the RNN cell

The root folder contains the common host main.cpp file and the make-hw.sh script that allows to reproduce the complete HW flow (calling hls-syn.tcl for HLS synthesis)
