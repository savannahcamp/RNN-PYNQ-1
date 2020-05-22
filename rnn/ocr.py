#   Copyright (c) 2018, TU Kaiserslautern
#   Copyright (c) 2018, Xilinx, Inc.
#   All rights reserved.
# 
#   Redistribution and use in source and binary forms, with or without 
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright 
#       notice, this list of conditions and the following disclaimer in the 
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import time
import numpy as np
from .input_handling import InputImage, Alphabets
from abc import ABCMeta, abstractmethod, abstractproperty
from rnn import PynqRNN, RUNTIME_HW, RNN_DATA_DIR, PlainImagePreprocessor

MAX_OCR_LENGTH = 1024

class PynqOCR(PynqRNN):
    __metaclass__ = ABCMeta

    def __init__(self, runtime, dataset, network, precision, load_overlay, preprocessor, bitstream_path=None):
        super(PynqOCR, self).__init__(runtime, dataset, network, precision, load_overlay, bitstream_path)
        self.alphabet_path = os.path.join(LSTM_DATA_DIR, dataset, "alphabet.txt")
        self.preprocessor = preprocessor
        self.input_bitwidth = int(precision[-1])

    @property
    def ops_per_seq_element(self):
        return self.lstm_ops_per_seq_element + self.fc_ops_per_seq_element

    @property
    def fc_ops_per_seq_element(self):
        return 2 * self.hidden_size * 2 * self.alphabet_size if self.peepholes_enabled else 2 * self.hidden_size * self.alphabet_size

    def inference(self, input_data):

        input_data = self.preprocessor.preprocess(input_data)
        input_data_post_process_width = int(len(input_data) / self.input_size)       

        input_data_processed = InputImage(input_data, self.input_size, self.bidirectional_enabled)
        alphabets = Alphabets(self.alphabet_path, self.alphabet_size)
        start = time.time()
        predictions = self.hw_inference(input_data_processed)        
        end = time.time() - start
        string = alphabets.ReturnString(predictions)
        self.cleanup()
        mops_per_s = 0.001 * self.ops_per_seq_element * input_data_post_process_width / end
        return mops_per_s, end, string

    @abstractproperty
    def alphabet_size(self):
        pass

class PynqPlainOCR(PynqOCR):

    def __init__(self, runtime=RUNTIME_HW, network="bilstm", precision="W2A2", load_overlay=True, bitstream_path=None):
        super(PynqPlainOCR, self).__init__(runtime, 
                                           "plain",
                                           network, 
                                           precision,
                                           load_overlay, 
                                           PlainImagePreprocessor(self.input_size, int(precision[-1])),
                                           bitstream_path=bitstream_path)

    @property
    def alphabet_size(self):
        return 82

    @property
    def input_size(self):
        return 32

    @property
    def hidden_size(self):
        return 128

    @property
    def peepholes_enabled(self):
        return False

    @property
    def bias_enabled(self):
        return True

    @property
    def bidirectional_enabled(self):
        return True

class PynqSeqMnistOCR(PynqOCR):

    def __init__(self, runtime=RUNTIME_HW, network="bigru", precision="W2A2", load_overlay=True, bitstream_path=None):
        super(PynqSeqMnistOCR, self).__init__(runtime, 
                                           "seq_mnist",
                                           network, 
                                           precision,
                                           load_overlay, 
                                           PlainImagePreprocessor(self.input_size, int(precision[-1])),
                                           bitstream_path=bitstream_path)
        self.bidirectional = True

    @property
    def alphabet_size(self):
        return 11

    @property
    def input_size(self):
        return 32

    @property
    def hidden_size(self):
        return 128

    @property
    def peepholes_enabled(self):
        return False

    @property
    def bias_enabled(self):
        return True

    @property
    def bidirectional_enabled(self):
        return self.bidirectional

    @bidirectional_enabled.setter
    def bidirectional_enabled(self, directions):
        self.bidirectional = directions

    @bidirectional_enabled.deleter
    def bidirectional_enabled(self):
        del self.bidirectional

