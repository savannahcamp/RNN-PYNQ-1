# lstm-pynq

Uni and Bi LSTM topologies, working on different datasets and multiple precision, namely:
 
 - "plain": plain-text dataset from [Insiders Technologies GmbH](https://www.insiders-technologies.de/home.html).
 - "seq_mnist": sequence of mnist digits from [my ctc-ocr-pytorch repo](https://github.com/ussamazahid96/ctc-ocr-pytorch).

For each dataset, there is a folder structure like this:

 - `<dataset>/WxAy/` contains the exported weights for a specific precision for both weigths (x bits) and activations (y bits)
 	Note: To use the unidirectional version, repalace the file as `mv r_model_fw_bw_uni.hpp r_model_fw_bw.hpp` and launch the synthesis. Similarly when running inference on hardware, uncomment the line for `bidirectional_enabled=False` in the [test_ocr.py](https://github.com/ussamazahid96/LSTM-PYNQ/blob/ec0ac07a4414f055099805e496931193d92956aa/tests/test_ocr.py#L56)
 - `<dataset>/top.cpp` is the top level file, synthesized by Vivado HLS, of the LSTM accelerator
 - `<dataset>/hw_config.hpp` contains defines for datatypes used within the LSTM cell

The root folder contains the common host main.cpp file and the make-hw.sh script that allows to reproduce the complete HW flow (calling hls-syn.tcl for HLS synthesis)
