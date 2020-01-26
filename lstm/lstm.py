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
import cffi
import numpy as np
from pynq import Overlay, PL, Xlnk, allocate
from abc import ABCMeta, abstractmethod, abstractproperty

if os.environ['BOARD'] == 'Pynq-Z1' or os.environ['BOARD'] == 'Pynq-Z2':
    PLATFORM="pynqZ1-Z2"
elif os.environ['BOARD'] == 'ZC706':
    PLATFORM="zc706"
elif os.environ['BOARD'] == 'Ultra96':
    PLATFORM="ultra96"
else:
    raise RuntimeError("Board not supported")


RUNTIME_HW = "libhw"
RUNTIME_SW = "libsw"

LSTM_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
LSTM_LIB_DIR = os.path.join(LSTM_ROOT_DIR, 'libraries')
LSTM_BIT_DIR = os.path.join(LSTM_ROOT_DIR, 'bitstreams')
LSTM_DATA_DIR = os.path.join(LSTM_ROOT_DIR, 'datasets')


class PynqLSTM(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, runtime, dataset, network, load_overlay, bitstream_path=None):
        self.bitstream_name="{}-{}-{}.bit".format(dataset, network, PLATFORM)
        if runtime == RUNTIME_HW:
            if bitstream_path is None:
                self.bitstream_path=os.path.join(LSTM_BIT_DIR, dataset, network, self.bitstream_name)
            else:
                self.bitstream_path=bitstream_path
            self.overlay = Overlay(self.bitstream_path, download=load_overlay)
            if PL.bitfile_name != self.bitstream_path:
                raise RuntimeError("Incorrect Overlay loaded")
        self.BLSTM_CTC = self.overlay.topLevel_BLSTM_CTC_0.register_map

        self.input_bitwidth = int(network[-1])
        self.accel_input_buffer = None
        self.accel_output_buffer = None

        # self._ffi = cffi.FFI()
        # self._libraries = {}
        # dllname = "{}-{}-{}-ocr-{}.so".format(runtime, dataset, network, PLATFORM)
        # if dllname not in self._libraries:
        #     self._libraries[dllname] = self._ffi.dlopen(
        # os.path.join(LSTM_LIB_DIR, dataset, network, dllname))
        # self.interface = self._libraries[dllname]
        # self._ffi.cdef(self.ffi_interface)


    def pack(self, img):
        img = img*2**(self.input_bitwidth-1)
        img = np.where(img < 0, img+(1 << self.input_bitwidth), img).astype(np.uint64)
        img = img.T
        pix_per_64_entry = 64//self.input_bitwidth
        img = img.reshape(img.shape[0], -1, pix_per_64_entry)
        factor = 1 << self.input_bitwidth*np.arange(img.shape[-1], dtype=np.uint64)
        img = img.dot(factor)
        return img.reshape(-1)


    def hw_inference(self, input_data_processed):
        packed_input = self.pack(input_data_processed.image_array)
        
        self.accel_input_buffer = allocate(shape=packed_input.shape, dtype=np.uint64)
        self.accel_output_buffer = allocate(shape=(128,), dtype=np.uint64)
        np.copyto(self.accel_input_buffer, packed_input)
        
        bytes_read = np.ceil((self.input_bitwidth*self.input_size*8)/64)
        bytes_read = int(bytes_read*2*input_data_processed.width)

        self.BLSTM_CTC.numberColumns_V = input_data_processed.width
        self.BLSTM_CTC.numberColumnsTwice_V = 2*input_data_processed.width
        self.BLSTM_CTC.numberBytesRead_V = bytes_read
        self.BLSTM_CTC.input_buffer_V_1 = self.accel_input_buffer.physical_address & 0xffffffff
        self.BLSTM_CTC.input_buffer_V_2 = (self.accel_input_buffer.physical_address >> 32) & 0xffffffff
        self.BLSTM_CTC.output_buffer_V_1 = self.accel_output_buffer.physical_address & 0xffffffff
        self.BLSTM_CTC.output_buffer_V_2 = (self.accel_output_buffer.physical_address >> 32) & 0xffffffff
        self.ExecAccel()
        
        predictions =  np.copy(np.frombuffer(self.accel_output_buffer, dtype=np.uint64))
        return predictions


    def ExecAccel(self):
        self.BLSTM_CTC.CTRL.AP_START = 1
        while not self.BLSTM_CTC.CTRL.AP_DONE:
            pass

    @property
    def ops_per_seq_element(self):
        return self.lstm_ops_per_seq_element
    
    @property
    def lstm_ops_per_seq_element(self):
        gate_input_size = self.input_size + self.hidden_size + 1 if self.bias_enabled else self.input_size + self.hidden_size
        #2 accounts for mul and add separately, 4 is the number of gates
        ops = 2 * gate_input_size * 4 * self.hidden_size
        #element wise muls and peepholes
        ops = ops + 3 * self.hidden_size * 2 if self.peepholes_enabled else ops + 3 * self.hidden_size
        #directions
        return ops * 2 if self.bidirectional_enabled else ops

    def cleanup(self):
        xlnk = Xlnk()
        xlnk.xlnk_reset()

    @abstractproperty
    def input_size(self):
        pass

    @abstractproperty
    def hidden_size(self):
        pass

    @abstractproperty
    def peepholes_enabled(self):
        pass

    @abstractproperty
    def bias_enabled(self):
        pass

    @abstractproperty
    def bidirectional_enabled(self):
        pass

    @abstractproperty
    def ffi_interface(self):
        pass
        
    @abstractmethod
    def inference(self, input_data):
        pass


        

    
