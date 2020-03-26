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

#ifndef UTILS_HPP
#define UTILS_HPP

#include <ap_int.h>
#include <hls_stream.h>

#define CASSERT_DATAFLOW(x);	

using namespace hls;

//===========================================================================================================================================================================================
// TAKEN FROM FINN
//===========================================================================================================================================================================================
template<unsigned int DataWidth>
void Mem2Stream(ap_uint<DataWidth> * in, stream<ap_uint<DataWidth> > & out, const unsigned int numBytes)
{
	CASSERT_DATAFLOW(DataWidth % 8 == 0);
	const unsigned int numWords = numBytes / (DataWidth / 8);
	CASSERT_DATAFLOW(numWords != 0);
	for (unsigned int i = 0; i < numWords; i++) 
	{
#pragma HLS PIPELINE II=1
		ap_uint<DataWidth> e = in[i];
		out.write(e);
	}
}

template<typename InT, typename OutT>
void StreamingCast(stream<InT> & in, stream<OutT> & out, unsigned int numReps) 
{
  for(unsigned int i = 0; i < numReps; i++) 
  {
#pragma HLS PIPELINE II=1	
	out.write((OutT) in.read());
  }
}

template<unsigned int InWidth, unsigned int OutWidth, unsigned int NumInWords>
void StreamingDataWidthConverter_Batch(stream<ap_uint<InWidth> > & in, stream<ap_uint<OutWidth> > & out, const unsigned int numReps) 
{
	if (InWidth > OutWidth) 
	{
		// emit multiple output words per input word read
		CASSERT_DATAFLOW(InWidth % OutWidth == 0);
		const unsigned int outPerIn = InWidth / OutWidth;
		const unsigned int totalIters = NumInWords * outPerIn * numReps;
		unsigned int o = 0;
		ap_uint<InWidth> ei = 0;
		for (unsigned int t = 0; t < totalIters; t++) 
		{
	#pragma HLS PIPELINE II=1
			// read new input word if current out count is zero
			if (o == 0)
				ei = in.read();
			// pick output word from the rightmost position
			ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
			out.write(eo);
			// shift input to get new output word for next iteration
			ei = ei >> OutWidth;
			// increment written output count
			o++;
			// wraparound indices to recreate the nested loop structure
			if (o == outPerIn) 
			{
				o = 0;
			}
		}
	} 
	else if (InWidth == OutWidth) 
	{
		// straight-through copy
		for (unsigned int i = 0; i < NumInWords * numReps; i++) 
		{
	#pragma HLS PIPELINE II=1
			ap_uint<InWidth> e = in.read();
			out.write(e);
		}

	}
	else 
	{ // InWidth < OutWidth
		// read multiple input words per output word emitted
		CASSERT_DATAFLOW(OutWidth % InWidth == 0);
		//const unsigned int inPerOut = OutWidth / InWidth;
		const ap_uint<8> inPerOut = OutWidth / InWidth;
		const unsigned int totalIters = NumInWords * numReps;
		//unsigned int i = 0;
		ap_uint<8> i = 0;
		ap_uint<OutWidth> eo = 0;
		for (unsigned int t = 0; t < totalIters; t++) 
		{
	#pragma HLS PIPELINE II=1
			// read input and shift into output buffer
			ap_uint<InWidth> ei = in.read();
			eo = eo >> InWidth;
			eo(OutWidth - 1, OutWidth - InWidth) = ei;
			// increment read input count
			i++;
			// wraparound logic to recreate nested loop functionality
			if (i == inPerOut) 
			{
				i = 0;
				out.write(eo);
			}
		}
	}
}	

//===================================================================================================================================================================================
// MEMORY ROUTER
//===================================================================================================================================================================================
template
<		
typename OutputActivationOutputLayer_t,
unsigned int OutputActivationOutputLayerWidth,
typename NumberOutputUnits_t,
unsigned int NumberOutputUnits,
typename NumberColumns_t,
unsigned int MaxNumberColumns
>
void Concatenator(NumberColumns_t numberOfColumns,
			       hls::stream<OutputActivationOutputLayer_t> & input_stream,
			       hls::stream<OutputActivationOutputLayer_t> & output_stream)
{
	static OutputActivationOutputLayer_t fw_bw_mem[MaxNumberColumns * NumberOutputUnits];
	OutputActivationOutputLayer_t input, mem, sum;
	//ap_uint<OutputActivationOutputLayerWidth> output;

	for(ap_int<16> currentColumn = numberOfColumns - 1; currentColumn >= 0; currentColumn--)
	{
		for(NumberOutputUnits_t currentClass = 0; currentClass < NumberOutputUnits; currentClass++)
		{
		#pragma HLS PIPELINE II=1 rewind

		#pragma HLS loop_flatten off

			input_stream.read(input);

			ap_uint<32> offset = currentColumn * NumberOutputUnits + currentClass;
			fw_bw_mem[offset] = input;
		}
	}

	for(NumberColumns_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
		for(NumberOutputUnits_t currentClass = 0; currentClass < NumberOutputUnits; currentClass++)
		{
		#pragma HLS PIPELINE II=1 rewind

		#pragma HLS loop_flatten off

			input_stream.read(input);

			ap_uint<32> offset = currentColumn * NumberOutputUnits + currentClass;
			mem = fw_bw_mem[offset];

			sum = mem + input;

			//output = *reinterpret_cast<ap_uint<OutputActivationOutputLayerWidth> *>(&sum);

			output_stream.write(sum);

		}
	}
}
//===================================================================================================================================================================================
// MAX PER COLUMN
//===================================================================================================================================================================================
template
<
typename Input_t,
unsigned int InputWidth,
typename Max_t,
typename NumberOutputUnits_t,
unsigned int NumberOutputUnits,
typename NumberColumns_t
>
void MaxPerColumn(NumberColumns_t numberOfColumns,
				  hls::stream<Input_t> & input_stream,//hls::stream<ap_uint<InputWidth>> & input_stream,
				  hls::stream<Max_t> & output_stream)
{
	Input_t input;
	//ap_uint<InputWidth> input_temp;
	Max_t max;
	max.value = 0.0;
	max.label = 0;

	for(NumberColumns_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
		for(NumberOutputUnits_t currentClass = 0; currentClass < NumberOutputUnits; currentClass++)
		{
		#pragma HLS PIPELINE II=1 rewind

		#pragma HLS loop_flatten off

			input_stream.read(input);

			//input = *reinterpret_cast<Input_t *>(&input_temp);

			if(input > max.value)
			{
				max.value = input;
				max.label = currentClass;
			}
		}

		output_stream.write(max);
		max.value = 0.0;

	}
}
//===================================================================================================================================================================================
// FINAL LABELING
//===================================================================================================================================================================================
template
<
typename Max_t, 
typename NumberOutputUnits_t,
unsigned int NumberOutputUnits,
typename NumberColumns_t,
unsigned int MaxNumberColumns
>
void FinalLabeling(NumberColumns_t numberOfColumns,
			   	   hls::stream<Max_t> & input_stream,
				   hls::stream<NumberOutputUnits_t > & output_stream)
{

	static Max_t max_global[MaxNumberColumns];

	Max_t max_per_previous_column;
	max_per_previous_column.value = 0.0;
	max_per_previous_column.label = 0;

	NumberColumns_t center = (numberOfColumns >> 1);
	ap_int<16> pointer = 0;

	for(NumberColumns_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
	#pragma HLS PIPELINE II=1 rewind

		Max_t max_per_current_column;
		input_stream.read(max_per_current_column);

		if(max_per_current_column.label == 0)
			max_per_current_column.value = 0.0;

		max_global[center + pointer] = max_per_current_column;

		//if(currentColumn % 2 == 0)
		if(currentColumn(0,0) == 0)
		{
			pointer++;
			pointer = -pointer;
		}
		else
		{
			pointer = -pointer;
		}
	}

	for(NumberColumns_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
	#pragma HLS PIPELINE II=1 rewind

		Max_t max_per_current_column = max_global[currentColumn];

		if(max_per_current_column.value > max_per_previous_column.value)
		{
			max_per_previous_column = max_per_current_column;
		}
		else if(max_per_previous_column.label != 0 && max_per_current_column.label == 0)
		{
			output_stream.write(max_per_previous_column.label);
			max_per_previous_column = max_per_current_column;
		}
	}

	output_stream.write((NumberOutputUnits_t)NumberOutputUnits);
}
//===================================================================================================================================================================================
// DMA OUTPUT
//===================================================================================================================================================================================
template
<
unsigned int NumberOutputUnits,
unsigned int DataWidth,
unsigned int MaxSizeOutputString,
typename NumberOutputUnits_t
>
void Stream2Mem(hls::stream<NumberOutputUnits_t > & input_stream,
				ap_uint<DataWidth> * output_mem)
{

	NumberOutputUnits_t input = 0;
	NumberOutputUnits_t counter = 0;

	while(input != NumberOutputUnits)
	{
	#pragma HLS PIPELINE II=1

		input_stream.read(input);
		counter++;
		if(counter > MaxSizeOutputString - 1)
		{
			break;
		}
		else
		{
			output_mem[counter] = (ap_uint<DataWidth>)input;
		}
	}
	output_mem[0] = (ap_uint<DataWidth>)(counter - 1);
}

#endif
