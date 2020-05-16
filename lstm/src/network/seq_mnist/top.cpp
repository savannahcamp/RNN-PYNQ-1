#define AP_INT_MAX_W 2048
#include "hardware_lstm.hpp"
#include "hw_config.hpp"
#include "r_model_fw_bw.hpp"
#include <fstream>

void DoCompute(ap_uint<16> numberColumns,
	    	   ap_uint<16> numberColumnsTwice,
		       ap_uint<32> numberBytesRead,
			   ap_uint<DATAWIDTH> *input_buffer,
			   ap_uint<DATAWIDTH> *output_buffer)
{
#pragma HLS ALLOCATION instances=DSP48 limit=160 core
	constexpr unsigned int StreamPerColumn = (HEIGHT_IN_PIX*PIXELWIDTH) / DATAWIDTH + (((HEIGHT_IN_PIX*PIXELWIDTH) % DATAWIDTH)>0); // CEILING
#pragma HLS DATAFLOW
	
	hls::stream<ap_uint<DATAWIDTH> >output_stream_dma_input("output_stream_dma_input");
#pragma HLS STREAM variable=output_stream_dma_input depth=2

	hls::stream<ap_uint<HEIGHT_IN_PIX * PIXELWIDTH> > output_stream_columns("output_stream_columns");
#pragma HLS STREAM variable=output_stream_columns depth=2

	hls::stream<ap_uint<DATAWIDTH * StreamPerColumn> > stream_column_padded("stream_column_padded");
#pragma HLS STREAM variable=stream_column_padded depth=2

	hls::stream<ap_uint<OUTPUTACTIVATIONHIDDENLAYERWIDTH*PE> > output_stream_hidden_layer("output_stream_hidden_layer");
#pragma HLS STREAM variable=output_stream_hidden_layer depth=2

	hls::stream<ap_uint<OUTPUTACTIVATIONHIDDENLAYERWIDTH * NUMBER_OF_NEURONS> > output_stream_input_streamer("output_stream_input_streamer");
#pragma HLS STREAM variable=output_stream_input_streamer depth=2

	hls::stream<t_fixed_sum_fc> output_stream_mac("output_stream_mac");
#pragma HLS STREAM variable=output_stream_mac depth=2

	hls::stream<t_fixed_sum_fc> output_stream_concatenator("output_stream_concatenator");
#pragma HLS STREAM variable=output_stream_concatenator depth=2

	hls::stream<maxx> output_stream_div_max_per_column("output_stream_div_max_per_column");
#pragma HLS STREAM variable=output_stream_div_max_per_column depth=2

	hls::stream<ap_uint<8> > output_stream_final_labeling("output_stream_final_labeling");
#pragma HLS STREAM variable=output_stream_final_labeling depth=2
	
	Mem2Stream<DATAWIDTH>(input_buffer, output_stream_dma_input, numberBytesRead);
	
	// Converts data widths of streams into multiple or submultiples	
	StreamingDataWidthConverter_Batch<DATAWIDTH, StreamPerColumn * DATAWIDTH, StreamPerColumn>(output_stream_dma_input, stream_column_padded, numberColumnsTwice);
	
	// This cast will remove the padding from the MSBs..casts intput to output	
	StreamingCast< ap_uint<StreamPerColumn * DATAWIDTH>, ap_uint<HEIGHT_IN_PIX * PIXELWIDTH> >(stream_column_padded, output_stream_columns, numberColumnsTwice);
	
	// std::ofstream ofs("/home/uzahid/personal/ctc-ocr-brevitas/hls.txt");
	
	// int s = output_stream_columns.size();
	// std::cout << "Size of the Stream = " << s << '\n';
	// for(int i=0; i<s; i++)
	// {
	// 	ap_uint<HEIGHT_IN_PIX*PIXELWIDTH> temp = output_stream_columns.read();	
	// 	for(int j=0; j<HEIGHT_IN_PIX; j++)
	// 	{
	// 		ap_int<8> pix = temp((j+1)*8-1, j*8);
	// 		t_fixed_image fpix = *reinterpret_cast<t_fixed_image*>(&pix); 
	// 		std::cout << fpix << '\n';
	// 		ofs << fpix << '\n'; 
	// 	}
	// 	ofs << '\n'; 
	// }
	// for(auto nf=0; nf<NUMBER_OF_NEURONS/PE; nf++)
	// {
	// 	for(auto pe=0; pe<PE; pe++)
	// 	{
	// 		for(auto sf=0; sf<NUMBER_OF_NEURONS/SIMD_RECURRENT; sf++)
	// 		{
	// 			for (auto simd=0; simd<SIMD_RECURRENT; simd++)
	// 			{
	// 				ap_int<WEIGHTWIDTH> wei = wn_hh[simd][sf][pe][nf];
	// 				t_fixed_wr fwei = *reinterpret_cast<t_fixed_wr*>(&wei);
	// 				ofs << fwei << '\n';
	// 			}
	// 		}
	// 	}
	// }
	// ofs.close();
	// exit(1);

	GRULayer
	<DIRECTIONS, PE, SIMD_INPUT, SIMD_RECURRENT, t_fixed_image, PIXELWIDTH,
	t_fixed_wr, WEIGHTWIDTH, t_fixed_br, BIASWIDTH, t_fixed_sum_wr, t_fixed_gix_sum,
	t_fixed_ci_gi_mul, t_fixed_recurrent, OUTPUTACTIVATIONHIDDENLAYERWIDTH,
	ap_uint<HEIGHT_IN_PIX_TYPEWIDTH>, HEIGHT_IN_PIX,
	ap_uint<NUMBER_OF_NEURONS_TYPEWIDTH>, NUMBER_OF_NEURONS,
	MAX_NUMBER_COLUMNS_TEST_SET,
	t_fixed_state, 
	t_fixed_sigma_o, NUMBER_OF_LUT_ETRIES_SIGMOID_1, t_fixed_lut_sigmoid_limit, t_fixed_lut_sigmoid_recip_step,
	t_fixed_tanh_o, NUMBER_OF_LUT_ETRIES_TANH_1, t_fixed_lut_tanh_limit, t_fixed_lut_tanh_recip_step
	>
	(numberColumns, output_stream_columns, output_stream_hidden_layer,
	 wr_ih, wr_hh, br_ih, br_hh,  
	 wc_ih, wc_hh, bc_ih, bc_hh,  
	 wn_ih, wn_hh, bn_ih, bn_hh,
	 lut_sigmoid_1, lut_tanh_1);
				
	exit(1);
	StreamingDataWidthConverter_Batch<OUTPUTACTIVATIONHIDDENLAYERWIDTH*PE, OUTPUTACTIVATIONHIDDENLAYERWIDTH * NUMBER_OF_NEURONS, NUMBER_OF_NEURONS/PE>(output_stream_hidden_layer, output_stream_input_streamer, numberColumnsTwice);

	OutputLayer
	<DIRECTIONS,
	t_fixed_bfc, FCBIASWIDTH,
	t_fixed_wfc, FCWEIGHTWIDTH,
	t_fixed_recurrent, OUTPUTACTIVATIONHIDDENLAYERWIDTH,
	t_fixed_sum_fc, OUTPUTACTIVATIONOUTPUTLAYERWIDTH,
	ap_uint<NUMBER_OF_NEURONS_TYPEWIDTH>, NUMBER_OF_NEURONS,
	ap_uint<NUMBER_OF_CLASSES_TYPEWIDTH>, NUMBER_OF_CLASSES,
	ap_uint<MAX_NUMBER_COLUMNS_TEST_SET_TYPEWIDTH>
	>			
	(bfc, wfc, numberColumns, output_stream_input_streamer, output_stream_mac);

	Concatenator
	<
	DIRECTIONS,
	t_fixed_sum_fc, OUTPUTACTIVATIONOUTPUTLAYERWIDTH,
	ap_uint<NUMBER_OF_CLASSES_TYPEWIDTH>, NUMBER_OF_CLASSES,
	ap_uint<MAX_NUMBER_COLUMNS_TEST_SET_TYPEWIDTH>, MAX_NUMBER_COLUMNS_TEST_SET
	>
	(numberColumns, output_stream_mac, output_stream_concatenator);
	
	MaxPerColumn
	<
	t_fixed_sum_fc, OUTPUTACTIVATIONOUTPUTLAYERWIDTH,
	maxx, 
	ap_uint<NUMBER_OF_CLASSES_TYPEWIDTH>, NUMBER_OF_CLASSES,
	ap_uint<16>
	>
	(numberColumns, output_stream_concatenator, output_stream_div_max_per_column);

	FinalLabeling
	<
	DIRECTIONS, maxx, 
	ap_uint<NUMBER_OF_CLASSES_TYPEWIDTH>, NUMBER_OF_CLASSES,
	ap_uint<MAX_NUMBER_COLUMNS_TEST_SET_TYPEWIDTH>, MAX_NUMBER_COLUMNS_TEST_SET
	>
	(numberColumns, output_stream_div_max_per_column, output_stream_final_labeling);
	
	Stream2Mem
	<
	NUMBER_OF_CLASSES,
	DATAWIDTH,
	SIZE_OF_OUTPUT_BUFFER,
	ap_uint<NUMBER_OF_CLASSES_TYPEWIDTH>
	>
	(output_stream_final_labeling, output_buffer);
	
}

//===================================================================================================================================================================================
// TOP LEVEL
//===================================================================================================================================================================================
void topLevel_BLSTM_CTC(ap_uint<32> numberColumns,
					    ap_uint<32> numberColumnsTwice,
						ap_uint<32> numberBytesRead,
						ap_uint<DATAWIDTH> * input_buffer,
						ap_uint<DATAWIDTH> * output_buffer)
{
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=numberColumns bundle=control
#pragma HLS INTERFACE s_axilite port=numberColumnsTwice bundle=control
#pragma HLS INTERFACE s_axilite port=numberBytesRead bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=input_buffer bundle=hostmem depth=4096
#pragma HLS INTERFACE s_axilite port=input_buffer bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=output_buffer bundle=hostmem depth=2048
#pragma HLS INTERFACE s_axilite port=output_buffer bundle=control

/*#pragma HLS ARRAY_PARTITION variable=wgi_ih complete dim=3
#pragma HLS ARRAY_PARTITION variable=wgi_hh complete dim=3
#pragma HLS ARRAY_RESHAPE variable=wgi_ih complete dim=1
#pragma HLS ARRAY_RESHAPE variable=wgi_hh complete dim=1

#pragma HLS ARRAY_PARTITION variable=wgf_ih complete dim=3
#pragma HLS ARRAY_PARTITION variable=wgf_hh complete dim=3
#pragma HLS ARRAY_RESHAPE variable=wgf_ih complete dim=1
#pragma HLS ARRAY_RESHAPE variable=wgf_hh complete dim=1


#pragma HLS ARRAY_PARTITION variable=wgo_ih complete dim=3
#pragma HLS ARRAY_PARTITION variable=wgo_hh complete dim=3
#pragma HLS ARRAY_RESHAPE variable=wgo_ih complete dim=1
#pragma HLS ARRAY_RESHAPE variable=wgo_hh complete dim=1


#pragma HLS ARRAY_PARTITION variable=wci_ih complete dim=3
#pragma HLS ARRAY_PARTITION variable=wci_hh complete dim=3
#pragma HLS ARRAY_RESHAPE variable=wci_ih complete dim=1
#pragma HLS ARRAY_RESHAPE variable=wci_hh complete dim=1*/

#pragma HLS ARRAY_RESHAPE variable=bgi_ih complete dim=1
#pragma HLS ARRAY_RESHAPE variable=bgi_hh complete dim=1
#pragma HLS ARRAY_RESHAPE variable=bgf_ih complete dim=1
#pragma HLS ARRAY_RESHAPE variable=bgf_hh complete dim=1
#pragma HLS ARRAY_RESHAPE variable=bgo_ih complete dim=1
#pragma HLS ARRAY_RESHAPE variable=bgo_hh complete dim=1
#pragma HLS ARRAY_RESHAPE variable=bci_ih complete dim=1
#pragma HLS ARRAY_RESHAPE variable=bci_hh complete dim=1

#pragma HLS ARRAY_RESHAPE variable=wfc complete dim=1

	DoCompute(numberColumns, numberColumnsTwice, numberBytesRead, input_buffer, output_buffer);

}
