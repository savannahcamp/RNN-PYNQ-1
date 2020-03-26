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

#ifndef LSTM_PH_HPP
#define LSTM_PH_HPP

#include <ap_int.h>
#include <iostream>
#include "dv2m.hpp"
#include <hls_stream.h>
#include "activations.hpp"



//===================================================================================================================================================================================
// NEURON
//===================================================================================================================================================================================
template
< 
unsigned int PE,					// Number of neurons to be executed in parallel
unsigned int SIMD_INPUT, 			// Number of parallel MAC performed in the gates on input pixels
unsigned int SIMD_RECURRENT, 		// Number of parallel MAC performed in the gates on recurrent path
typename Pixel_t,     				// Type of the input pixel
unsigned int PixelWidth, 			// number of bits of the input pixel
typename Bias_t_gi,					// Type of the bias for gate i
unsigned int BiasWidth_gi,			// number of bits of each bias (gate i)
typename Weight_t_gi,				// Type of the weights for gate i
unsigned int WeightWidth_gi,		// number of bits of each weight (gate i)
typename DotProductResult_t_gi, 	// type of the result for MAC with weight of gate i
typename gix_accumulator_t,
typename Bias_t_gf,					// Type of the bias for gate f
unsigned int BiasWidth_gf,			// number of bits of each bias (gate f)
typename Weight_t_gf,				// Type of the weights for gate f
unsigned int WeightWidth_gf,		// number of bits of each weight (gate f)
typename DotProductResult_t_gf, 	// type of the result for MAC with weight of gate f
typename gfx_accumulator_t,
typename Bias_t_go,					// Type of the bias for gate o
unsigned int BiasWidth_go,			// number of bits of each bias (gate o)
typename Weight_t_go,				// Type of the weights for gate o
unsigned int WeightWidth_go,		// number of bits of each weight (gate o)
typename DotProductResult_t_go, 	// type of the result for MAC with weight of gate o
typename gox_accumulator_t,
typename Bias_t_ci,					// Type of the bias for gate ci
unsigned int BiasWidth_ci,			// number of bits of each bias (gate ci)
typename Weight_t_ci,				// Type of the weights for gate ci
unsigned int WeightWidth_ci,		// number of bits of each weight (gate ci)
typename DotProductResult_t_ci, 	// type of the result for MAC with weight of gate ci
typename gi_ci_accumulator_t, 
typename Weight_wip_t,
unsigned int WeightWidth_wip,
typename Wip_mul_t,
typename Weight_wfp_t,
unsigned int WeightWidth_wfp,
typename Wfp_mul_t,
typename Weight_wop_t,
unsigned int WeightWidth_wop,
typename Wop_mul_t,
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
void LSTMCell(uint16_t currentColumn,
			NumberHiddenUnits_t currentHiddenUnit,
			NumberHiddenUnits_t PE_count,
			ap_uint<ColumnHeight  * PixelWidth> image,
			ap_uint<OutputActivationWidth * NumberHiddenUnits> h_prev,
			State_t c_prev,
			State_t & c_next,
			OutputActivation_t & h_next, 
			const ap_uint<BiasWidth_gi> biases_gii[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_gi> biases_gih[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_gi> weights_gi_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_gi> weights_gi_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_gf> biases_gfi[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_gf> biases_gfh[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_gf> weights_gf_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],			
			const ap_uint<WeightWidth_gf> weights_gf_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],			
			const ap_uint<BiasWidth_go> biases_goi[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_go> biases_goh[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_go> weights_go_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],			
			const ap_uint<WeightWidth_go> weights_go_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],			
			const ap_uint<BiasWidth_ci> biases_cii[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_ci> biases_cih[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_ci> weights_ci_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE], 
			const ap_uint<WeightWidth_ci> weights_ci_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE], 
			Sigmoid_out_t lut_sigmoid_1[Lut_Entries_Sigmoid], 
			Tanh_out_t lut_tanh_1[Lut_Entries_Tanh],
			const ap_uint<WeightWidth_wip> peepholes_ip[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_wfp> peepholes_fp[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_wop> peepholes_op[PE][2 * NumberHiddenUnits/PE]
			)

{
	Wip_mul_t wip_state_mul; 
	Wfp_mul_t wfp_state_mul;
	Wop_mul_t wop_state_mul; 
	gi_ci_accumulator_t ci_gi_mul; 
	gix_accumulator_t gix_sum; 
	gfx_accumulator_t gfx_sum; 
	gox_accumulator_t gox_sum; 
	
	Sigmoid_out_t gi;
	Sigmoid_out_t gf;
	Sigmoid_out_t go;
	Tanh_out_t ci;
	Tanh_out_t tanhf_o;

	State_t tmp_c_prev, tmp_c_next;
	State_t gf_state_mul;

	DotProductResult_t_gi gix;
	DotProductResult_t_gf gfx;
	DotProductResult_t_go gox;
	DotProductResult_t_ci cix;			
	
	gix = DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_gi, BiasWidth_gi, Weight_t_gi, WeightWidth_gi, DotProductResult_t_gi, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(biases_gii, biases_gih, weights_gi_i, weights_gi_h, image, h_prev, currentHiddenUnit, PE_count);
	gfx = DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_gf, BiasWidth_gf, Weight_t_gf, WeightWidth_gf, DotProductResult_t_gf, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(biases_gfi, biases_gfh, weights_gf_i, weights_gf_h, image, h_prev, currentHiddenUnit, PE_count);
	gox = DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_go, BiasWidth_go, Weight_t_go, WeightWidth_go, DotProductResult_t_go, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(biases_goi, biases_goh, weights_go_i, weights_go_h, image, h_prev, currentHiddenUnit, PE_count);
	cix = DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_ci, BiasWidth_ci, Weight_t_ci, WeightWidth_ci, DotProductResult_t_ci, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(biases_cii, biases_cih, weights_ci_i, weights_ci_h, image, h_prev, currentHiddenUnit, PE_count);

	tmp_c_prev = c_prev;

	if(currentColumn > 0)
	{
		ap_uint<WeightWidth_wip> wip_temp = peepholes_ip[PE_count][currentHiddenUnit];
		Weight_wip_t wip_internal = *reinterpret_cast<Weight_wip_t *>(&wip_temp);
		wip_state_mul = wip_internal * tmp_c_prev;
		gix_sum = gix + wip_state_mul;
	}
	else{
		gix_sum = gix;
	}
	if(currentColumn > 0) {
		
		ap_uint<WeightWidth_wfp> wfp_temp = peepholes_fp[PE_count][currentHiddenUnit];
		Weight_wfp_t wfp_internal = *reinterpret_cast<Weight_wfp_t *>(&wfp_temp);
		wfp_state_mul = wfp_internal * tmp_c_prev;
		gfx_sum = gfx + wfp_state_mul;
	}
	else{	
		gfx_sum = gfx;
	}

	gi = sigmoid_lut<Lut_Entries_Sigmoid,gix_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gix_sum, lut_sigmoid_1);
	gf = sigmoid_lut<Lut_Entries_Sigmoid,gfx_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gfx_sum, lut_sigmoid_1);

	ci = tanh_lut<Lut_Entries_Tanh,DotProductResult_t_ci,Tanh_limit_t,Tanh_step_t,Tanh_out_t>(cix,lut_tanh_1);
	ci_gi_mul = ci * gi;

	if(currentColumn > 0)
	{

		gf_state_mul = gf * tmp_c_prev;
		tmp_c_next = ci_gi_mul + gf_state_mul;

		ap_uint<WeightWidth_wop> wop_temp = peepholes_op[PE_count][currentHiddenUnit];
		Weight_wop_t wop_internal = *reinterpret_cast<Weight_wop_t *>(&wop_temp);

		wop_state_mul = wop_internal * tmp_c_next;
		gox_sum = gox + wop_state_mul;

	}
	else
	{
		tmp_c_next = ci_gi_mul;
		gox_sum = gox;
	}

	go = sigmoid_lut<Lut_Entries_Sigmoid,gox_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gox_sum, lut_sigmoid_1);

	tanhf_o = tanh_lut<Lut_Entries_Tanh,State_t,Tanh_limit_t,Tanh_step_t,Tanh_out_t>(tmp_c_next,lut_tanh_1);
	h_next = tanhf_o * go;

	c_next = tmp_c_next;
}




//===================================================================================================================================================================================
// HIDDEN LAYER
//===================================================================================================================================================================================
template< 
unsigned int PE,				// Number of neurons to be executed in parallel
unsigned int SIMD_INPUT, 			// Number of parallel MAC performed in the gates on input pixels
unsigned int SIMD_RECURRENT, 			// Number of parallel MAC performed in the gates on recurrent path
typename Pixel_t,     				// Type of the input pixel
unsigned int PixelWidth, 			// number of bits of the input pixel
typename Bias_t_gi,				// Type of the bias for gate i
unsigned int BiasWidth_gi,			// number of bits of each bias (gate i)
typename Weight_t_gi,				// Type of the weights for gate i
unsigned int WeightWidth_gi,			// number of bits of each weight (gate i)
typename DotProductResult_t_gi, 		// type of the result for MAC with weight of gate i
typename gix_accumulator_t,
typename Bias_t_gf,				// Type of the bias for gate f
unsigned int BiasWidth_gf,			// number of bits of each bias (gate f)
typename Weight_t_gf,				// Type of the weights for gate f
unsigned int WeightWidth_gf,			// number of bits of each weight (gate f)
typename DotProductResult_t_gf, 		// type of the result for MAC with weight of gate f
typename gfx_accumulator_t,
typename Bias_t_go,				// Type of the bias for gate o
unsigned int BiasWidth_go,			// number of bits of each bias (gate o)
typename Weight_t_go,				// Type of the weights for gate o
unsigned int WeightWidth_go,			// number of bits of each weight (gate o)
typename DotProductResult_t_go, 		// type of the result for MAC with weight of gate o
typename gox_accumulator_t,
typename Bias_t_ci,				// Type of the bias for gate ci
unsigned int BiasWidth_ci,			// number of bits of each bias (gate ci)
typename Weight_t_ci,				// Type of the weights for gate ci
unsigned int WeightWidth_ci,			// number of bits of each weight (gate ci)
typename DotProductResult_t_ci, 		// type of the result for MAC with weight of gate ci
typename gi_ci_accumulator_t,
typename Weight_wip_t,
unsigned int WeightWidth_wip,
typename Wip_mul_t,
typename Weight_wfp_t,
unsigned int WeightWidth_wfp,
typename Wfp_mul_t,
typename Weight_wop_t,
unsigned int WeightWidth_wop,
typename Wop_mul_t,
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
void HiddenLayer(uint32_t numberOfColumns,
			  hls::stream<ap_uint<ColumnHeight * PixelWidth> > &image_stream,					  
			  hls::stream<ap_uint<OutputActivationWidth> > &result_stream,
			  const ap_uint<BiasWidth_gi> biases_gii[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_gi> biases_gih[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_gi> weights_gi_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_gi> weights_gi_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_gf> biases_gfi[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_gf> biases_gfh[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_gf> weights_gf_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],			
			  const ap_uint<WeightWidth_gf> weights_gf_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],			
			  const ap_uint<BiasWidth_go> biases_goi[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_go> biases_goh[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_go> weights_go_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],			
			  const ap_uint<WeightWidth_go> weights_go_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],			
			  const ap_uint<BiasWidth_ci> biases_cii[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_ci> biases_cih[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_ci> weights_ci_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE], 
			  const ap_uint<WeightWidth_ci> weights_ci_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE], 
			  Sigmoid_out_t lut_sigmoid_1[Lut_Entries_Sigmoid], 
			  Tanh_out_t lut_tanh_1[Lut_Entries_Tanh],
			  const ap_uint<WeightWidth_wip> peepholes_ip[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_wfp> peepholes_fp[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_wop> peepholes_op[PE][2 * NumberHiddenUnits/PE]
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

	State_t c_prev, c_next, c_reg[2 * NumberHiddenUnits/PE][PE];
#pragma HLS ARRAY_PARTITION variable=c_reg complete dim=2
	OutputActivation_t output;

	ap_uint<ColumnHeight * PixelWidth> local_image;
	ap_uint<OutputActivationWidth * NumberHiddenUnits> local_input;
	
	hls::stream<ap_uint<OutputActivationWidth * NumberHiddenUnits> > recurrent_stream;
#pragma HLS STREAM variable=recurrent_stream depth=4

	ap_uint<OutputActivationWidth * NumberHiddenUnits> output_reg = 0;
	for(NumberHiddenUnits_t path = 0; path < 2; path++)
	{
#pragma HLS PIPELINE II=1 rewind
		recurrent_stream.write(output_reg);	
	}		

	for(uint16_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
		for (ap_uint<2> count = 0; count <2; count ++) 
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
					c_prev = c_reg[actual_hidden_unit_address][PE_count];
					
					LSTMCell
					<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, 
					Bias_t_gi, BiasWidth_gi, Weight_t_gi, WeightWidth_gi, DotProductResult_t_gi, gix_accumulator_t,
					Bias_t_gf, BiasWidth_gf, Weight_t_gf, WeightWidth_gf, DotProductResult_t_gf, gfx_accumulator_t,
					Bias_t_go, BiasWidth_go, Weight_t_go, WeightWidth_go, DotProductResult_t_go, gox_accumulator_t,
					Bias_t_ci, BiasWidth_ci, Weight_t_ci, WeightWidth_ci, DotProductResult_t_ci, gi_ci_accumulator_t,
					Weight_wip_t, WeightWidth_wip, Wip_mul_t,
					Weight_wfp_t, WeightWidth_wfp, Wfp_mul_t,
					Weight_wop_t, WeightWidth_wop, Wop_mul_t,
					OutputActivation_t, OutputActivationWidth,
					ColumnHeight_t, ColumnHeight, 
					NumberHiddenUnits_t, NumberHiddenUnits,
					State_t, 
					Sigmoid_out_t, Lut_Entries_Sigmoid, Sigmoid_limit_t, Sigmoid_step_t,
					Tanh_out_t, Lut_Entries_Tanh, Tanh_limit_t, Tanh_step_t>
					(currentColumn,
					actual_hidden_unit_address, PE_count,
					local_image,
					local_input,
					c_prev,
					c_next,
					output, 
					biases_gii, biases_gih, weights_gi_i, weights_gi_h,
					biases_gfi, biases_gfh, weights_gf_i, weights_gf_h,
					biases_goi, biases_goh, weights_go_i, weights_go_h,
					biases_cii, biases_cih, weights_ci_i, weights_ci_h,
					lut_sigmoid_1,lut_tanh_1,
					peepholes_ip,peepholes_fp,peepholes_op);

					c_reg[actual_hidden_unit_address][PE_count] = c_next;
					
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
