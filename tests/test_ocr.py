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
import rnn
import pytest
import numpy as np
from PIL import Image

def test_plain_ocr():
    precisions = ["W2A2", "W4A8"]
    test_dir = os.path.dirname(os.path.realpath(__file__))
    for precision in precisions:
        hw_ocr = rnn.PynqPlainOCR(runtime=rnn.RUNTIME_HW, precision=precision)
        im = Image.open(os.path.join(test_dir, 'Test_images', 'plain', precision, '010077.bin.png'))
        with open(os.path.join(test_dir, 'Test_images', 'plain', precision, 'test_image_gt.txt'), 'r') as f:
            print("Prec  = {}".format(precision))            
            gt = f.read().replace('\n', '')
            hw_result = hw_ocr.inference(im)
            _, _, hw_recognized_text = hw_result
            hw_ocr.cleanup()
            print("Label = {}".format(gt))
            print("Pred  = {}\n".format(hw_recognized_text))
            assert gt == hw_recognized_text

def test_seq_mnist_ocr():
    network = "bigru"
    precisions = ["W2A2", "W4A4", "W8A8"]#,"W4A4","W4A8", "W8A8"]
    test_dir = os.path.dirname(os.path.realpath(__file__))
    for precision in precisions:
        hw_ocr = rnn.PynqSeqMnistOCR(runtime=rnn.RUNTIME_HW, network=network, precision=precision)
        # uncomment this if using uni birectional rnn
        # hw_ocr.bidirectional_enabled = False
        im = Image.open(os.path.join(test_dir, 'Test_images', 'seq_mnist', precision, 'test_image.png'))
        with open(os.path.join(test_dir, 'Test_images', 'seq_mnist', precision, 'test_image_gt.txt'), 'r') as f:
            print("Prec  = {}".format(precision))
            gt = f.read().replace('\n', '')
            hw_result = hw_ocr.inference(im)
            _, _, hw_recognized_text = hw_result
            hw_ocr.cleanup()
            print("Label = {}".format(gt))
            print("Pred  = {}\n".format(hw_recognized_text))
            assert gt == hw_recognized_text

if __name__ == '__main__':
    test_seq_mnist_ocr()
