# Upgrades in this fork:

1. Added Support for Ultra96 and ZC706
2. Replaced all the .so host code with python (no .so files needed anymore)
3. Includes the seq_mnist example (trained from my <a href="https://github.com/ussamazahid96/ctc-ocr-pytorch" target="_blank"> ctc-ocr-pytorch repo</a>)
4. Added Support and example for unidirectional LSTM
5. Added Uni and Bidirectional GRU

# RNN-PYNQ Pip Installable Package

This repo contains the pip install package for Quantized RNNs on PYNQ. 
Currently one overlay is included, that performs Optical Character Recognition (OCR) of a plain-text dataset provided by [Insiders Technologies GmbH](https://www.insiders-technologies.de/home.html) and sequential mnist.

If you find it useful, we would appreciate a citation to:

**FINN-L: Library Extensions and Design Trade-off Analysis for Variable Precision LSTM Networks on FPGAs**,
V. Rybalkin, A. Pappalardo, M. M. Ghaffar, G. Gambardella, N. Wehn, M. Blott.
*Accepted for publication, 28th International Conference on Field Programmable Logic and Applications (FPL), August, 2018, Dublin, Ireland.*

BibTeX:

``` bibtex
@ARTICLE{2018arXiv180704093R,
   author = {{Rybalkin}, V. and {Pappalardo}, A. and {Mohsin Ghaffar}, M. and 
	{Gambardella}, G. and {Wehn}, N. and {Blott}, M.},
    title = "{FINN-L: Library Extensions and Design Trade-off Analysis for Variable Precision LSTM Networks on FPGAs}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1807.04093},
 primaryClass = "cs.CV",
 keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Hardware Architecture, Computer Science - Machine Learning},
     year = 2018,
    month = jul
}

```

This repo is a joint release of University of Kaiserslautern, [Microelectronic System Design Research Group](https://ems.eit.uni-kl.de/en/start/): Vladimir Rybalkin, Mohsin Ghaffar, Norbert Wehn in cooperation with [Xilinx, Inc.:](https://www.xilinx.com/) Alessandro Pappalardo, Giulio Gambardella, Michael Gross, Michaela Blott.

## Quick Start

In order to install it to your PYNQ (on PYNQ v2.5), connect to the board, open a terminal and type:

```
sudo pip3.6 install git+https://github.com/ussamazahid96/RNN-PYNQ.git
```

This will install the RNN-PYNQ package to your board, and create a **rnn** directory in the Jupyter home area. You will find the Jupyter notebooks to test the RNN in this directory. 
 
## Repo organization 

The repo is organized as follows:
-   *rnn*: contains the pip installed package
    -	*bitstreams*: bitstreams for the plain and sequential mnist OCR overlay.
    -	*datasets*: contains support files for working with a given dataset.
    -	*src*: contains the sources and scripts to regenerate the available overlays
        - *library*: FINN library for HLS RNN descriptions, host code, script to rebuilt and drivers for the PYNQ (please refer to README for more details)
        - *network*: RNN topologies HLS top functions on multiple datasets, host code and make script for HW and SW built (please refer to README for more details)
-	*notebooks*: lists a set of python notebooks examples, that during installation will be moved in `/home/xilinx/jupyter_notebooks/rnn/` folder.
-	*tests*: contains test scripts and test images

## Hardware design rebuilt

In order to rebuild the hardware designs, the repo should be cloned in a machine with installation of the Vivado Design Suite (tested with 2019.2). 
Following the step-by-step instructions:

1.	Clone the repository on your linux machine: git clone https://github.com/ussamazahid96/RNN-PYNQ.git;
2.	Move to `<clone_path>/RNN-PYNQ/rnn/src/network/`
3.	Set the RNN_ROOT environment variable to `<clone_path>/RNN-PYNQ/rnn/src/`
4.	Launch the shell script make-hw.sh with parameters the target dataset, target network, target platform and mode, with the command `./make-hw.sh {dataset} {network} {precision} {platform} {mode}` where:
	- dataset can be plain or seq_mnist;
  - network can be bilstm, bigru, unilstm, unigru
	- precision depends on the precision you want for weights and activations (e.g., WxAy features x bits for Weights and y bits for activations) - check the available configuration in the dataset folder at `<clone_path>/RNN-PYNQ/rnn/src/network/<dataset>`;
	- platform is pynq;
	- mode can be `h` to launch Vivado HLS synthesis, `b` to launch the Vivado project (needs HLS synthesis results), `a` to launch both.
5.	The results will be visible in `<clone_path>/RNN-PYNQ/rnn/src/network/output/` that is organized as follows:
	- bitstream: contains the generated bitstream(s);
	- hls-syn: contains the Vivado HLS generated RTL and IP (in the subfolder named as the target network);
	- report: contains the Vivado and Vivado HLS reports;
	- vivado: contains the Vivado project.
6.	Copy the generated bitstream and hwh script on the PYNQ board `<pip_installation_path>/rnn/bitstreams/`
