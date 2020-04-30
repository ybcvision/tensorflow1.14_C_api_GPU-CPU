#pragma once

#include <iostream>
#include "tensorflow/c/c_api.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <utility>
#include <regex>
#include <numeric>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cstddef>
#include <cstdint>


#ifndef _DNN_TENSORFLOW_H
#define _DNN_TENSORFLOW_H



class dnn_tensorflow
{
public:
	dnn_tensorflow();
	~dnn_tensorflow();

	void InitTFEnvironment(const std::vector<uint8_t> config);
	void CleanTFEnv();

	void LoadGraph(const char* pb_filename, bool log_info);

	void BuildSession();

	TF_Tensor* CreateTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, const void* data, std::size_t len);
	template <typename T>
	TF_Tensor* CreateTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, const std::vector<T>& data) {
		return CreateTensor(data_type,
			dims.data(), dims.size(),
			data.data(), data.size() * sizeof(T));
	};

	TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims);
	TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims);

	void CreateIO_Ops(std::vector<const char*> node_names_vec, TF_Output ops[]);

	void DeleteTensor(TF_Tensor* tensor);
	void DeleteTensor(std::vector<TF_Tensor*> tensors);

	void RunSession(TF_Output* inputs, TF_Tensor* const *input_values, int ninputs,
		TF_Output* outputs, TF_Tensor** output_values, int noutputs);



	template <typename T>
	std::vector<T> GetTensorData(const TF_Tensor* tensor) {
		auto data = static_cast<T*>(TF_TensorData(tensor));
		if (data == nullptr) {
			return{};
		}

		return{ data, data + TF_TensorElementCount(tensor) };
		//return{ data};
	}

	template <typename T>
	std::vector<std::vector<T>> GetTensorsData(const std::vector<TF_Tensor*>& tensors) {
		std::vector<std::vector<T>> data;
		data.reserve(tensors.size());
		for (auto t : tensors) {
			data.push_back(GetTensorData<T>(t));
		}

		return data;
	}

private:
	TF_Graph * _graph;
	TF_Status* _status;
	TF_SessionOptions* _sess_opts;
	TF_Session* _session;

};


#endif










