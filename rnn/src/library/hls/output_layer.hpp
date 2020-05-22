/*
 *  Copyright (c) 2018, TU Kaiserslautern
 *	Copyright (c) 2018, Xilinx
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef OUTPUT_LAYER_HPP
#define OUTPUT_LAYER_HPP

#include <ap_int.h>
#include <hls_stream.h>

//===================================================================================================================================================================================
// OUTPUT LAYER
//===================================================================================================================================================================================
template
<
unsigned int DIRECTIONS,
typename Bias_fc_t,
unsigned int BiasWidth_fc,
typename Weight_fc_t,
unsigned int WeightWidth_fc,
typename OutputActivationHiddenLayer_t,
unsigned int OutputActivationHiddenLayerWidth,
typename OutputActivationOutputLayer_t,
unsigned int OutputActivationOutputLayerWidth,
typename NumberHiddenUnits_t,
unsigned int NumberHiddenUnits,
typename NumberOutputUnits_t,
unsigned int NumberOutputUnits,
typename NumberColumns_t
>
void OutputLayer(const ap_uint<BiasWidth_fc> biases[1][DIRECTIONS * NumberOutputUnits],
				 const ap_uint<WeightWidth_fc> weights[NumberHiddenUnits][DIRECTIONS * NumberOutputUnits],
				 NumberColumns_t numberOfColumns,
		 	 	 hls::stream<ap_uint<OutputActivationHiddenLayerWidth * NumberHiddenUnits>> & input_stream,
				 hls::stream<OutputActivationOutputLayer_t> & output_stream)
{
	ap_uint<OutputActivationHiddenLayerWidth * NumberHiddenUnits> input_stream_temp;
	OutputActivationOutputLayer_t mul[NumberHiddenUnits];
	OutputActivationOutputLayer_t sum;

	#pragma HLS ARRAY_PARTITION variable=mul complete dim=1

	for(NumberColumns_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
		for (ap_uint<DIRECTIONS> count = 0; count < DIRECTIONS; count++)
		{
			input_stream.read(input_stream_temp);

			for(NumberOutputUnits_t currentClass = 0; currentClass < NumberOutputUnits; currentClass++)
			{
			#pragma HLS PIPELINE II=1 

				ap_int<BiasWidth_fc> bias_temp = biases[0][count * NumberOutputUnits + currentClass];
				Bias_fc_t bias = *reinterpret_cast<Bias_fc_t *>(&bias_temp);

				sum = (OutputActivationOutputLayer_t)bias;

				for(NumberHiddenUnits_t currentHiddenUnit = 0; currentHiddenUnit < NumberHiddenUnits; currentHiddenUnit++)
				{
				#pragma HLS UNROLL

					ap_int<OutputActivationHiddenLayerWidth> input_temp = input_stream_temp((currentHiddenUnit + 1) * OutputActivationHiddenLayerWidth - 1, currentHiddenUnit * OutputActivationHiddenLayerWidth);
					OutputActivationHiddenLayer_t input = *reinterpret_cast<OutputActivationHiddenLayer_t *>(&input_temp);
					ap_int<WeightWidth_fc> weigth_temp = weights[currentHiddenUnit][count * NumberOutputUnits + currentClass];
					Weight_fc_t weigth = *reinterpret_cast<Weight_fc_t *>(&weigth_temp);
					mul[currentHiddenUnit] = input * weigth;
				}

				for(NumberHiddenUnits_t currentHiddenUnit = 0; currentHiddenUnit < NumberHiddenUnits; currentHiddenUnit++)
				{
				#pragma HLS UNROLL

					sum += mul[currentHiddenUnit];
				}

				output_stream.write(sum);
				
			}
		}
	}
}

#endif