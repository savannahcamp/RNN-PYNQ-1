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

#ifndef DM2V_HPP
#define DM2V_HPP

#include <ap_int.h>

//===================================================================================================================================================================================
// DOT PRODUCT FUNCTIONS
//===================================================================================================================================================================================
template
< 
unsigned int DIRECTIONS,
unsigned int PE,
unsigned int SIMD_INPUT, 			// Number of parallel MAC performed in the gates on input pixels
unsigned int SIMD_RECURRENT, 		// Number of parallel MAC performed in the gates on recurrent path
typename Pixel_t, 
unsigned int PixelWidth, 
typename OutputActivation_t,
unsigned int OutputActivationWidth,
typename Bias_t,
unsigned int BiasWidth, 
typename Weight_t,
unsigned int WeightWidth, 
typename DotProductResult_t, 
typename ColumnHeight_t,
unsigned int ColumnHeight,
typename NumberHiddenUnits_t,
unsigned int NumberHiddenUnits
>
DotProductResult_t DotVectorToMatrix(const ap_uint<BiasWidth> biases_i[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
				     const ap_uint<BiasWidth> biases_h[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
				     const ap_uint<WeightWidth> weights_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],
				     const ap_uint<WeightWidth> weights_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],
				     ap_uint<ColumnHeight * PixelWidth> image_column,
				     ap_uint<OutputActivationWidth * NumberHiddenUnits> inputs,
				     NumberHiddenUnits_t currentHiddenUnit,
				     NumberHiddenUnits_t PE_count)
{	
	constexpr unsigned int FoldingInput = ColumnHeight / SIMD_INPUT;
	constexpr unsigned int FoldingRecurrent = NumberHiddenUnits / SIMD_RECURRENT;
	DotProductResult_t mul_pix[SIMD_INPUT][FoldingInput];
	DotProductResult_t mul_neuron[SIMD_RECURRENT][FoldingRecurrent];
	DotProductResult_t sum_pix = 0.0;
	DotProductResult_t sum_neuron = 0.0;
	DotProductResult_t sum = 0.0;
	ap_int<BiasWidth> bias_i_temp = biases_i[PE_count][currentHiddenUnit];
	ap_int<BiasWidth> bias_h_temp = biases_h[PE_count][currentHiddenUnit];
	Bias_t bias_i = *reinterpret_cast<Bias_t *>(&bias_i_temp);
	Bias_t bias_h = *reinterpret_cast<Bias_t *>(&bias_h_temp);
	
#pragma HLS ARRAY_PARTITION variable=mul_pix complete dim=1
#pragma HLS ARRAY_PARTITION variable=mul_neuron complete dim=1
#pragma HLS ARRAY_RESHAPE variable=weights_i complete dim=1
#pragma HLS ARRAY_RESHAPE variable=weights_h complete dim=1
#pragma HLS ARRAY_RESHAPE variable=weights_i complete dim=3
#pragma HLS ARRAY_RESHAPE variable=weights_h complete dim=3
	for(ColumnHeight_t j = 0; j < FoldingInput; j++) {
#pragma HLS PIPELINE II=1 rewind
		for(ColumnHeight_t i = 0; i < SIMD_INPUT; i++)		
		{			
	#pragma HLS UNROLL	
			unsigned int PixelInColumn = j*SIMD_INPUT+i;
			ap_int<PixelWidth> pixel_temp = image_column((PixelInColumn+1)*PixelWidth-1, PixelInColumn*PixelWidth);
			Pixel_t pixel = *reinterpret_cast<Pixel_t *>(&pixel_temp);

			ap_int<WeightWidth> weight_temp = weights_i[i][j][PE_count][currentHiddenUnit];
			Weight_t weigth = *reinterpret_cast<Weight_t *>(&weight_temp);
			mul_pix[i][j] = pixel * weigth;
		}
	}
	for(NumberHiddenUnits_t j = 0; j < FoldingRecurrent; j++) {
#pragma HLS PIPELINE II=1 rewind
		for(NumberHiddenUnits_t  i = 0; i < SIMD_RECURRENT; i++)
		{
	#pragma HLS UNROLL
			unsigned int ActivationInRecurrent = j*SIMD_RECURRENT+i;
			ap_int<OutputActivationWidth> input_temp = inputs((ActivationInRecurrent + 1) * OutputActivationWidth - 1, ActivationInRecurrent * OutputActivationWidth);
			OutputActivation_t input = *reinterpret_cast<OutputActivation_t *>(&input_temp);

			ap_int<WeightWidth> weight_temp = weights_h[i][j][PE_count][currentHiddenUnit];
			Weight_t weigth = *reinterpret_cast<Weight_t *>(&weight_temp);
			mul_neuron[i][j] = input * weigth;
		}
	}
	for(ColumnHeight_t j = 0; j < FoldingInput; j++) {
#pragma HLS PIPELINE II=1 rewind
		for(ColumnHeight_t  i = 0; i < SIMD_INPUT; i++)
		{
	#pragma HLS UNROLL
			sum_pix += mul_pix[i][j];
		}
	}
	for(NumberHiddenUnits_t j = 0; j < FoldingRecurrent; j++) {
#pragma HLS PIPELINE II=1 rewind
		for(NumberHiddenUnits_t  i = 0; i < SIMD_RECURRENT; i++)
		{
	#pragma HLS UNROLL
			sum_neuron += mul_neuron[i][j];
		}
	}
	sum = (DotProductResult_t)bias_i + (DotProductResult_t)bias_h  + sum_pix + sum_neuron;

	return sum;
}

#endif