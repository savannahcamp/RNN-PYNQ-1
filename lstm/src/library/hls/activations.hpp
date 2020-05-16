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

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <stdint.h>

template
<
unsigned int NUMBER_OF_LUT_ENTRIES,
typename Input_t,
typename Limit_t,
typename RecipStep_t,
typename Output_t
>
Output_t sigmoid_lut(Input_t & input, Output_t lut_sigmoid[NUMBER_OF_LUT_ENTRIES])
{
	Limit_t lower_limit = -5.0;
	Limit_t upper_limit = 5.0;
	RecipStep_t recip_step = 25.5;

	Input_t input_temp = input;
	Output_t output;

	// If we are outside of LUT range
	if (input_temp <= lower_limit)
	{
		output = lut_sigmoid[0];
	}
	else if (input_temp >= upper_limit)
	{
		output = lut_sigmoid[NUMBER_OF_LUT_ENTRIES-1];
	}
	else
	{
		// Scale from [lower, upper] to [0, N]
		Input_t t = input_temp - lower_limit;
		uint16_t index = t * recip_step;

		output = lut_sigmoid[index];
	}

	return output;
}

template
<
unsigned int NUMBER_OF_LUT_ENTRIES,
typename Input_t,
typename Limit_t,
typename RecipStep_t,
typename Output_t
>
Output_t tanh_lut(Input_t & input, Output_t lut_tanh[NUMBER_OF_LUT_ENTRIES])
{
	Limit_t lower_limit = -3.0;
	Limit_t upper_limit = 3.0;
	RecipStep_t recip_step = 42.5;

	Input_t input_temp = input;
	Output_t output;

	// If we are outside of LUT range
	if (input_temp <= lower_limit)
	{
		output = lut_tanh[0];
	}
	else if (input_temp >= upper_limit)
	{
		output = lut_tanh[NUMBER_OF_LUT_ENTRIES-1];
	}
	else
	{
		// Scale from [lower, upper] to [0, N]
		Input_t t = input_temp - lower_limit;
		uint16_t index = t * recip_step;

		output = lut_tanh[index];
	}

	return output;
}



//=============================================================================================
//=============================================================================================
//============================= TAKEN FROM FINN-HLSLIB ========================================
//=============================================================================================
//=============================================================================================
#include "interpret.hpp"

/**
 * General contract for activation functions.
 *
 * This class itself has no formal significance for the implementation
 * of the MVAU. Implementations of activation functions are encouraged
 * to implement it nonetheless to guarantee appropriate function
 * signatures.
 */
template<typename TA, typename TO>
class Activation {
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

  /**
   * Compute the activation of the passed accumulator value accu in row idx.
   */
  TO activate(unsigned const  nf, unsigned const  pe, TA const &accu) const;
};

/**
 * A no-op activation that simply outputs the computed accumulator
 * output as the final result.
 */
template<typename T>
class PassThroughActivation : public Activation<T, T> {
public:
  T activate(unsigned const  nf, unsigned const  pe, T const &accu) const {
#pragma HLS inline
    return  accu;
  }
};

/**
 * Use a simple global threshold comparison as activation function.
 *
 * The constant threshold is initialized at construction.
 * The default comparison returns true if the threshold value is
 * smaller than the passed accumulator value.
 */
template<typename TA, typename Compare = std::less<TA>>
class ThresholdActivation : public Activation<TA, bool> {
  TA const  m_threshold;
public:
  ThresholdActivation(TA const &threshold) : m_threshold(threshold) {
#pragma HLS inline
  }

public:
  bool activate(unsigned const  nf, unsigned const  pe, TA const &accu) const {
#pragma HLS inline
    return  Compare()(m_threshold, accu);
  }
};

/**
 * Use a simple per-row threshold comparison as activation function.
 *
 * The thresholds are taken from an array indexed by output row.
 * It is currently public to allow direct initialization and
 * to make its name accessible for top-level HLS pragmas.
 *
 * The default comparison returns true if the threshold value defined for
 * the indexed row is smaller than the passed accumulator value.
 */
template<unsigned NF, unsigned numPEs, unsigned NumTH, 
	 typename TA, typename TR, int ActVal = 0, typename Compare = std::less<TA>>
class ThresholdsActivation {
public:
  TA m_thresholds[numPEs][NF][NumTH];
  
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

public:
  TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu) const {
#pragma HLS inline
    TR result=ActVal;
	for(unsigned int i=0; i< NumTH; i++){
#pragma HLS unroll
      result+=Compare()(m_thresholds[pe][nf][i], accu);
    }
    return result;
  }
};

/**
 * \brief Thresholding function for multiple images
 *
 * The function performs thresholds comparison with input activation vector, 
 * and generating output based on the comparison results
 *
 * \tparam ImgDim         Width and Heigth of the Input Feature Map (assumed square)
 * \tparam NumChannels    Heigth of the input matrix
 * \tparam PE             Number of output rows computed in parallel
 * \tparam TSrcI          DataType of the input activation (as used in the MAC)
 * \tparam TDstI          DataType of the output activation (as generated by the activation)
 * \tparam TI             DataType of the input stream - safely deducible from the paramaters
 * \tparam TO             DataType of the output stream - safely deducible from the paramaters
 * \tparam TA             DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param activation      Activation class
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template <
    unsigned ImgDim, unsigned NumChannels, unsigned numPEs,
    typename TSrcI = Identity, typename TDstI = Identity,
    typename TI, typename TO, typename TA>
void Thresholding_Batch(hls::stream<TI> &in,
                        hls::stream<TO> &out,
                        TA const &activation,
                        int const reps)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const NF = NumChannels / numPEs;

  unsigned nf = 0;
  unsigned tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  for (unsigned i = 0; i < reps * ImgDim * ImgDim * NF; i++)
  {
    TI inElem;
    inElem = in.read();
    auto outElem = TDstI().template operator()<TO>();
    for (unsigned pe = 0; pe < numPEs; pe++)
    {
#pragma HLS UNROLL
      auto const act = TSrcI()(inElem);
      outElem(pe,0,1) = activation.activate(nf, pe, act(pe,0));
    }
    out.write(outElem);
    if (++nf == NF)
    {
      nf = 0;
    }
  }
}
#endif
