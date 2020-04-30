#include "VideoSurveillanceSys/dnn_tensorflow.hpp"
#include <assert.h>
#include <algorithm>



dnn_tensorflow::dnn_tensorflow()
{

}

dnn_tensorflow::~dnn_tensorflow()
{
}

void dnn_tensorflow::InitTFEnvironment(const std::vector<uint8_t> config)
{
	_status = TF_NewStatus();
	_sess_opts = TF_NewSessionOptions();
	TF_SetConfig(_sess_opts, (void*)config.data(), config.size(), _status);

	printf("[INFO]TF version: %s\n", TF_Version());
}



void dnn_tensorflow::CleanTFEnv()
{
	TF_CloseSession(_session, _status);
	TF_DeleteSession(_session, _status);
	TF_DeleteSessionOptions(_sess_opts);
	TF_DeleteGraph(_graph);
	TF_DeleteStatus(_status);

}

void dnn_tensorflow::LoadGraph(const char* pb_filename, bool log_info)
{

	_graph = TF_NewGraph();

	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

	FILE* fp = fopen(pb_filename, "rb");

	if (fp == NULL)
	{
		printf("[ERROR] Cannot open PB file! => %s\n", pb_filename);
		return;
	}

	fseek(fp, 0L, SEEK_END);
	int file_size = ftell(fp);
	rewind(fp);

	char* graph_content = (char*)malloc(file_size);

	fread(graph_content, file_size, 1, fp);
	fclose(fp);

	TF_Buffer tfBuffer;
	tfBuffer.data = graph_content;
	tfBuffer.length = file_size;

	TF_GraphImportGraphDef(_graph, &tfBuffer, opts, _status);

	free(graph_content);

	assert(TF_GetCode(_status) == TF_OK);
	if (log_info)
	{
		printf("[INFO]Loaded graph successfully.\n");
	}

}


void dnn_tensorflow::BuildSession()
{
	_session = TF_NewSession(_graph, _sess_opts, _status);
	assert(TF_GetCode(_status) == TF_OK);
}

TF_Tensor* dnn_tensorflow::CreateTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, const void* data, std::size_t len)
{
	if (dims == nullptr) {
		return nullptr;
	}

	auto tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
	if (tensor == nullptr) {
		return nullptr;
	}

	auto tensor_data = TF_TensorData(tensor);
	if (tensor_data == nullptr) {
		TF_DeleteTensor(tensor);
		return nullptr;
	}

	len = std::min(len, TF_TensorByteSize(tensor));
	if (data != nullptr && len != 0) {
		std::memcpy(tensor_data, data, len);
	}

	return tensor;
}

TF_Tensor* dnn_tensorflow::CreateEmptyTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims)
{
	return CreateTensor(data_type, dims, num_dims, nullptr, 0);
}

TF_Tensor* dnn_tensorflow::CreateEmptyTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims)
{
	return CreateEmptyTensor(data_type, dims.data(), dims.size());
}

void dnn_tensorflow::CreateIO_Ops(std::vector<const char*> node_names_vec, TF_Output ops[])
{
	assert(node_names_vec.size() > 0);

	for (int i = 0; i < node_names_vec.size(); i++)
	{
		TF_Operation* op = TF_GraphOperationByName(_graph, node_names_vec[i]);
		ops[i] = { op, 0 };
	}
}

void dnn_tensorflow::DeleteTensor(TF_Tensor* tensor)
{
	if (tensor != nullptr)
		TF_DeleteTensor(tensor);
}
void dnn_tensorflow::DeleteTensor(std::vector<TF_Tensor*> tensors)
{
	for (int i = 0; i < tensors.size(); i++)
	{
		if (tensors[i] != nullptr)
			TF_DeleteTensor(tensors[i]);
	}
}

void dnn_tensorflow::RunSession(TF_Output* inputs, TF_Tensor* const *input_values, int ninputs,
	TF_Output* outputs, TF_Tensor** output_values, int noutputs)
{
	TF_SessionRun(
		_session,
		nullptr,
		inputs, input_values, ninputs,
		outputs, output_values, noutputs,
		nullptr,
		0,
		nullptr,
		_status);
	if (TF_GetCode(_status) != TF_OK) {

		printf("[ERROR]Running failed!%s\n", TF_Message(_status));
	}
	else
	{
		std:: cout << "[INFO]run session successfully.\n" << std::endl;
	}
}




