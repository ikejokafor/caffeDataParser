#pragma once


#include <fcntl.h>
#include <stdint.h>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <string>
#include <iostream>
#include <unistd.h>

// Google protobufs
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>


// caffe
#include "caffe.pb.h"


namespace caffeDataParser
{
    //	LayerTypes:
    //      Input
    //      Convolution
    //      ReLU
    //      Pooling_MAX
    //      Pooling_AVE
    //      Permute
    //      Flatten
    //      Eltwise
    //      BatchNorm
    //      Scale
    //      DetectionOutput
    //      PriorBox
    //      Reshape
    //      InnerProduct
    //      Softmax
    //      Concat
    //      PSROIPoolingLayer

    typedef struct
    {
        std::string layerName;
        std::vector<std::string> topLayerNames;
        std::vector<std::string> bottomLayerNames;
        std::string layerType;
        int numInputRows;
        int numInputCols;
        int inputDepth;
        int outputDepth;
        int numKernelRows;
        int numKernelCols;
        int stride;
        int padding;
        bool globalPooling;
        float *filterData;
        float *biasData;
        int group;
        int localSize;
        float alpha;
        float beta;
        int numFilterValues;
        int numBiasValues;
    } layerInfo_t;
}


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void GetLayerFilterAndBias(caffeDataParser::layerInfo_t *layerInfo, caffe::NetParameter wparam);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<caffeDataParser::layerInfo_t> parseCaffeData(std::string protoFileName, std::string modelFileName);


// --------------------------------------------------------------------------------------------------------------------------------------------------
/**
 *		@brief			function description
 *		@param	param0	param0 description
 *		@param	param1	param1 description
 *		@return			0 success, 1 failure
 */
// --------------------------------------------------------------------------------------------------------------------------------------------------
void printModelProtocalBuffer(std::string protoFileName, std::string modelFileName);
