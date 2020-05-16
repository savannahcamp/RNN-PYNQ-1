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

#ifndef GRU_HPP
#define GRU_HPP

#include <ap_int.h>
#include <iostream>
#include "dv2m.hpp"
#include <hls_stream.h>
#include "activations.hpp"


//===================================================================================================================================================================================
//NEURON
//===================================================================================================================================================================================
template
<
unsigned int DIRECTIONS, 
unsigned int PE,					// Number of neurons to be executed in parallel
unsigned int SIMD_INPUT, 			// Number of parallel MAC performed in the gates on input pixels
unsigned int SIMD_RECURRENT, 		// Number of parallel MAC performed in the gates on recurrent path
typename Pixel_t,     				// Type of the input pixel
unsigned int PixelWidth, 			// number of bits of the input pixel
typename Weight_t,				// Type of the weights for gate i
unsigned int WeightWidth,		// number of bits of each weight (gate i)
typename Bias_t,					// Type of the bias for gate i
unsigned int BiasWidth,			// number of bits of each bias (gate i)
typename DotProductResult_t, 	// type of the result for MAC with weight of gate i
typename gix_accumulator_t,
typename gi_ci_accumulator_t, 
typename OutputActivation_t,
unsigned int OutputActivationWidth,
typename ColumnHeight_t,
unsigned int ColumnHeight,
typename NumberHiddenUnits_t,
unsigned int NumberHiddenUnits, 
typename State_t, 
typename Sigmoid_out_t,
unsigned int Lut_Entries_Sigmoid, 
typename Sigmoid_limit_t,
typename Sigmoid_step_t,
typename Tanh_out_t, 
unsigned int Lut_Entries_Tanh,
typename Tanh_limit_t,
typename Tanh_step_t
>
void GRUCell(uint16_t currentColumn,
			NumberHiddenUnits_t currentHiddenUnit,
			NumberHiddenUnits_t PE_count,
			ap_uint<ColumnHeight  * PixelWidth> image,
			ap_uint<OutputActivationWidth * NumberHiddenUnits> h_prev,
			OutputActivation_t & h_next, 
			
			const ap_uint<WeightWidth> weights_r_ih[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			const ap_uint<WeightWidth> weights_r_hh[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			const ap_uint<BiasWidth> biases_r_ih[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			const ap_uint<BiasWidth> biases_r_hh[PE][(DIRECTIONS * NumberHiddenUnits)/PE],			  

			const ap_uint<WeightWidth> weights_c_ih[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			const ap_uint<WeightWidth> weights_c_hh[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			const ap_uint<BiasWidth> biases_c_ih[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			const ap_uint<BiasWidth> biases_c_hh[PE][(DIRECTIONS * NumberHiddenUnits)/PE],

			const ap_uint<WeightWidth> weights_n_ih[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][(DIRECTIONS * NumberHiddenUnits)/PE], 
			const ap_uint<WeightWidth> weights_n_hh[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],  
			const ap_uint<BiasWidth> biases_n_ih[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			const ap_uint<BiasWidth> biases_n_hh[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			
			Sigmoid_out_t lut_sigmoid_1[Lut_Entries_Sigmoid], 
			Tanh_out_t lut_tanh_1[Lut_Entries_Tanh]
			)

{

	// gi_ci_accumulator_t ci_gi_mul; 
	gix_accumulator_t grx_sum, gcx_sum, gnx_sum; 
	
	// Sigmoid_out_t gr, gz;
	// Tanh_out_t ci;
	// Tanh_out_t tanhf_o;

	// State_t tmp_c_next;
	// State_t gf_state_mul;

	DotProductResult_t grx, gcx, gnx, gnx1, gnx2;
	
	// grx = DotVectorToMatrix<DIRECTIONS, PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, 
	// 						OutputActivation_t, OutputActivationWidth, Bias_t, BiasWidth, 
	// 						Weight_t, WeightWidth, DotProductResult_t, ColumnHeight_t, 
	// 						ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>
	// 						(biases_r_ih, biases_r_hh, weights_r_ih, weights_r_hh, image, h_prev, 
	// 						 currentHiddenUnit, PE_count);

	gcx = DotVectorToMatrix<DIRECTIONS, PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, 
							OutputActivation_t, OutputActivationWidth, Bias_t, BiasWidth, 
							Weight_t, WeightWidth, DotProductResult_t, ColumnHeight_t, 
							ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>
							(biases_c_ih, biases_c_hh, weights_c_ih, weights_c_hh, image, h_prev, 
							 currentHiddenUnit, PE_count);

	// gnx1 = DotVectorToOneMatrix<DIRECTIONS, PE, SIMD_INPUT, Pixel_t, PixelWidth, 
	// 							Bias_t, BiasWidth,Weight_t, WeightWidth, 
	// 							DotProductResult_t, ColumnHeight_t, ColumnHeight, 
	// 							NumberHiddenUnits_t, NumberHiddenUnits>
	// 							(biases_n_ih, weights_n_ih, image, 
	// 							 currentHiddenUnit, PE_count);

	// gnx2 = DotVectorToOneMatrix<DIRECTIONS, PE, SIMD_RECURRENT, OutputActivation_t, OutputActivationWidth, 
	// 							Bias_t, BiasWidth, Weight_t, WeightWidth, 
	// 							DotProductResult_t, NumberHiddenUnits_t, NumberHiddenUnits, 
	// 							NumberHiddenUnits_t, NumberHiddenUnits>
	// 							(biases_n_hh, weights_n_hh, h_prev, 
	// 							 currentHiddenUnit, PE_count);

	grx_sum = grx;
	gcx_sum = gcx;
	// gnx_sum = gnx;	
	
	std::cout << "GRX = " << grx_sum << '\n';
	std::cout << "GCX = " << gcx_sum << '\n';
	// std::cout << "GNX = " << gnx_sum << '\n';
	
	exit(1); 

	// gi = sigmoid_lut<Lut_Entries_Sigmoid,gix_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gix_sum, lut_sigmoid_1);
	// gf = sigmoid_lut<Lut_Entries_Sigmoid,gfx_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gfx_sum, lut_sigmoid_1);
	// go = sigmoid_lut<Lut_Entries_Sigmoid,gox_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gox_sum, lut_sigmoid_1);
	// ci = tanh_lut<Lut_Entries_Tanh,DotProductResult_t_ci,Tanh_limit_t,Tanh_step_t,Tanh_out_t>(cix,lut_tanh_1);

	// ci_gi_mul = ci * gi;

	// if(currentColumn > 0)
	// {
	// 	gf_state_mul = gf * c_prev;
	// 	tmp_c_next = ci_gi_mul + gf_state_mul;
	// }
	// else
	// {
	// 	tmp_c_next = ci_gi_mul;
	// }


	// tanhf_o = tanh_lut<Lut_Entries_Tanh,State_t,Tanh_limit_t,Tanh_step_t,Tanh_out_t>(tmp_c_next,lut_tanh_1);

	// h_next = tanhf_o * go;
	// c_next = tmp_c_next;

}




//===================================================================================================================================================================================
// HIDDEN LAYER
//===================================================================================================================================================================================
template<
unsigned int DIRECTIONS, 
unsigned int PE,					// Number of neurons to be executed in parallel
unsigned int SIMD_INPUT, 			// Number of parallel MAC performed in the gates on input pixels
unsigned int SIMD_RECURRENT, 		// Number of parallel MAC performed in the gates on recurrent path
typename Pixel_t,     				// Type of the input pixel
unsigned int PixelWidth, 			// number of bits of the input pixel
typename Weight_t,				// Type of the weights for gate i
unsigned int WeightWidth,		// number of bits of each weight (gate i)
typename Bias_t,					// Type of the bias for gate i
unsigned int BiasWidth,			// number of bits of each bias (gate i)
typename DotProductResult_t, 	// type of the result for MAC with weight of gate i
typename gix_accumulator_t,
typename gi_ci_accumulator_t,
typename OutputActivation_t,
unsigned int OutputActivationWidth,
typename ColumnHeight_t,
unsigned int ColumnHeight,
typename NumberHiddenUnits_t,
unsigned int NumberHiddenUnits,
unsigned int MaxNumberColumns,
typename State_t, 
typename Sigmoid_out_t,
unsigned int Lut_Entries_Sigmoid, 
typename Sigmoid_limit_t,
typename Sigmoid_step_t,
typename Tanh_out_t, 
unsigned int Lut_Entries_Tanh,
typename Tanh_limit_t,
typename Tanh_step_t
>	
void GRULayer(uint32_t numberOfColumns,
			  hls::stream<ap_uint<ColumnHeight * PixelWidth> > &image_stream,					  
			  hls::stream<ap_uint<OutputActivationWidth*PE> > &result_stream,
			  
			  const ap_uint<WeightWidth> weights_r_ih[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			  const ap_uint<WeightWidth> weights_r_hh[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			  const ap_uint<BiasWidth> biases_r_ih[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			  const ap_uint<BiasWidth> biases_r_hh[PE][(DIRECTIONS * NumberHiddenUnits)/PE],			  
			  
			  const ap_uint<WeightWidth> weights_c_ih[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			  const ap_uint<WeightWidth> weights_c_hh[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			  const ap_uint<BiasWidth> biases_c_ih[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			  const ap_uint<BiasWidth> biases_c_hh[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			  
			  const ap_uint<WeightWidth> weights_n_ih[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][(DIRECTIONS * NumberHiddenUnits)/PE], 
			  const ap_uint<WeightWidth> weights_n_hh[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][(DIRECTIONS * NumberHiddenUnits)/PE],  
			  const ap_uint<BiasWidth> biases_n_ih[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			  const ap_uint<BiasWidth> biases_n_hh[PE][(DIRECTIONS * NumberHiddenUnits)/PE],
			  
			  Sigmoid_out_t lut_sigmoid_1[Lut_Entries_Sigmoid], 
			  Tanh_out_t lut_tanh_1[Lut_Entries_Tanh]
			)
{
	if(NumberHiddenUnits % PE != 0) {
		std::cout << "Error: NumberHiddenUnits has to be a multiple of PE" << std::endl;
	}
	if(ColumnHeight % SIMD_INPUT != 0) {
		std::cout << "Error: ColumnHeight size has to be multiple of SIMD_INPUT" << std::endl;
	}
	if(NumberHiddenUnits % SIMD_RECURRENT != 0) {
		std::cout << "Error: NumberHiddenUnits size has to be multiple of SIMD_RECURRENT" << std::endl;
	}

	OutputActivation_t output;
	ap_uint<ColumnHeight * PixelWidth> local_image;
	ap_uint<OutputActivationWidth * NumberHiddenUnits> local_input;
	hls::stream<ap_uint<OutputActivationWidth * NumberHiddenUnits> > recurrent_stream("recurrent_stream");
#pragma HLS STREAM variable=recurrent_stream depth=4

	ap_uint<OutputActivationWidth * NumberHiddenUnits> output_reg = 0;

	if (DIRECTIONS==2)
	{
		for(NumberHiddenUnits_t path = 0; path < DIRECTIONS; path++)
		{
	#pragma HLS PIPELINE II=1 rewind
			recurrent_stream.write(output_reg);	
		}		
	}
	else
	{
		recurrent_stream.write(output_reg);
	}

	for(uint16_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
		for (ap_uint<DIRECTIONS> count = 0; count <DIRECTIONS; count ++) 
		{			
			image_stream.read(local_image);
			recurrent_stream.read(local_input);
			for(NumberHiddenUnits_t currentHiddenUnit = 0; currentHiddenUnit < NumberHiddenUnits/PE; currentHiddenUnit++)
			{
			constexpr unsigned int FoldingInput = ColumnHeight / SIMD_INPUT;
			#pragma HLS PIPELINE II=FoldingInput rewind
				ap_uint<OutputActivationWidth*PE> temp_output_packed;
				
				for (NumberHiddenUnits_t PE_count = 0; PE_count < PE; PE_count++)
				{
				#pragma HLS UNROLL
					NumberHiddenUnits_t actual_hidden_unit_address = count*NumberHiddenUnits/PE + currentHiddenUnit;
					
					GRUCell
					<
					DIRECTIONS, PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, 
					Weight_t, WeightWidth, 
					Bias_t, BiasWidth, 
					DotProductResult_t, gix_accumulator_t, 
					gi_ci_accumulator_t,
					OutputActivation_t, OutputActivationWidth,
					ColumnHeight_t, ColumnHeight, 
					NumberHiddenUnits_t, NumberHiddenUnits,
					State_t, 
					Sigmoid_out_t, Lut_Entries_Sigmoid, Sigmoid_limit_t, Sigmoid_step_t,
					Tanh_out_t, Lut_Entries_Tanh, Tanh_limit_t, Tanh_step_t
					>
					(currentColumn,
					actual_hidden_unit_address, PE_count,
					local_image,
					local_input,
					output,
					weights_r_ih, weights_r_hh, biases_r_ih, biases_r_hh, 
					weights_c_ih, weights_c_hh, biases_c_ih, biases_c_hh, 
					weights_n_ih, weights_n_hh, biases_n_ih, biases_n_hh, 
					lut_sigmoid_1,lut_tanh_1);

					temp_output_packed = temp_output_packed >> OutputActivationWidth;
					ap_uint<OutputActivationWidth> temp_output = *reinterpret_cast<ap_uint<OutputActivationWidth> *>(&output); 
					temp_output_packed(OutputActivationWidth*PE-1,OutputActivationWidth*(PE-1)) = temp_output;
					output_reg(((currentHiddenUnit*PE + PE_count) + 1) * OutputActivationWidth - 1, (currentHiddenUnit*PE + PE_count) * OutputActivationWidth) = temp_output;
				} // pe
				result_stream.write(temp_output_packed);
			}//neurons	
			if(currentColumn < numberOfColumns - 1)
				recurrent_stream.write(output_reg);
		}// backward/forward	
	}//column
}

#endif
