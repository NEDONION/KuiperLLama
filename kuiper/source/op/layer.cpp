// Layer基础类实现文件
// 提供神经网络层的基础功能，包括输入输出管理、参数管理和前向传播接口

#include "op/layer.h"
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>

namespace op {
// BaseLayer构造函数
// 初始化层的基本属性：设备类型、层类型、数据类型和层名称
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type,
                     base::DataType data_type, std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {
}

// 获取层的数据类型
base::DataType BaseLayer::data_type() const {
  return data_type_;
}

// 获取层的类型
LayerType BaseLayer::layer_type() const {
  return layer_type_;
}

// 获取层的名称
const std::string& BaseLayer::get_layer_name() const {
  return layer_name_;
}

// 设置层的名称
void BaseLayer::set_layer_name(const std::string& layer_name) {
  layer_name_ = layer_name;
}

// 获取层的设备类型（CPU/CUDA等）
base::DeviceType BaseLayer::device_type() const {
  return device_type_;
}

// 设置层的设备类型
void BaseLayer::set_device_type(base::DeviceType device_type) {
  device_type_ = device_type;
}

// Layer构造函数
// 默认使用FP32数据类型，继承BaseLayer并初始化基本属性
Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32,
                std::move(layer_name)) {
}

// 初始化函数
// 由子类重写实现具体的初始化逻辑
base::Status Layer::init() {
  return base::error::Success();
}

// 基础前向传播函数
// 需要子类重写实现具体的计算逻辑
base::Status Layer::base_forward() {
  return base::error::FunctionNotImplement("");
}

// 检查张量的基本有效性
// 验证张量是否为空、设备类型和数据类型是否匹配
base::Status Layer::check_tensor(const tensor::Tensor& tensor,
                                 base::DeviceType device_type,
                                 base::DataType data_type) const {
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }
  return base::error::Success();
}

// 检查张量的有效性并验证维度
// 使用可变参数列表验证张量的每个维度是否匹配预期值
base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                          base::DeviceType device_type,
                                          base::DataType data_type, ...) const {
  std::va_list args;
  // 检查张量是否为空
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor parameter is empty.");
  }
  // 检查设备类型是否匹配
  if (tensor.device_type() != device_type) {
    return base::error::InvalidArgument("The tensor has a wrong device type.");
  }
  // 检查数据类型是否匹配
  if (tensor.data_type() != data_type) {
    return base::error::InvalidArgument("The tensor has a wrong data type.");
  }

  // 使用可变参数逐个验证每个维度
  va_start(args, data_type);
  int32_t dims = tensor.dims_size();
  for (int32_t i = 0; i < dims; ++i) {
    int32_t dim = va_arg(args, int32_t);
    if (dim != tensor.get_dim(i)) {
      return base::error::InvalidArgument("The tensor has a wrong dim in dim" +
                                        std::to_string(i));
    }
  }
  va_end(args);
  return base::error::Success();
}

// 设置指定索引的输入张量
void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

// 设置指定索引的输出张量
void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

// 获取指定索引的输入张量（const版本）
const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

// 获取指定索引的输入张量（非const版本）
tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

// 获取指定索引的输出张量（非const版本）
tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

// 检查层的配置是否正确
// 由子类重写实现具体的检查逻辑
base::Status Layer::check() const {
  return base::error::FunctionNotImplement("The check function is not implement yet");
}

// 获取指定索引的输出张量（const版本）
const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

// 重置输入张量数组的大小
void Layer::reset_input_size(size_t size) {
  inputs_.resize(size);
}

// 重置输出张量数组的大小
void Layer::reset_output_size(size_t size) {
  outputs_.resize(size);
}

// 获取输入张量的数量
size_t Layer::input_size() const {
  return inputs_.size();
}

// 获取输出张量的数量
size_t Layer::output_size() const {
  return outputs_.size();
}

// LayerFp32Param构造函数
// 带FP32参数的层，用于存储权重等参数
LayerFp32Param::LayerFp32Param(base::DeviceType device_type, LayerType layer_type,
                               std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)) {
}

// 设置指定索引的权重张量
void LayerFp32Param::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  weights_.at(idx) = weight;
}

// 获取指定索引的权重张量（const版本）
const tensor::Tensor& LayerFp32Param::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

// 从原始数据指针设置权重张量
// 根据维度信息和数据指针创建Buffer并关联到权重张量
void LayerFp32Param::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                const float* weight_ptr, base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());

  // 计算权重数据的总字节数
  size_t size =
      std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
  // 创建Buffer对象，使用外部内存指针
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, (void*)(weight_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  // 创建张量并关联Buffer
  tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
  CHECK(weight.assign(buffer));
  weights_.at(idx) = weight;
}

// 重置权重张量数组的大小
void LayerFp32Param::reset_weight_size(size_t size) {
  weights_.resize(size);
}

// 获取权重张量的数量
size_t LayerFp32Param::weight_size() const {
  return weights_.size();
}

// 1输入1输出的前向传播接口
// 设置输入输出张量后调用base_forward执行计算
base::Status Layer::forward_i1o1(const tensor::Tensor& input1,
                                 const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->base_forward();
}

// 2输入1输出的前向传播接口
base::Status Layer::forward_i2o1(const tensor::Tensor& input1,
                                 const tensor::Tensor& input2,
                                 const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->base_forward();
}

// 3输入1输出的前向传播接口
base::Status Layer::forward_i3o1(const tensor::Tensor& input1,
                                 const tensor::Tensor& input2,
                                 const tensor::Tensor& input3,
                                 const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);

  this->set_output(0, output1);
  return this->base_forward();
}

// 4输入1输出的前向传播接口
base::Status Layer::forward_i4o1(const tensor::Tensor& input1,
                                 const tensor::Tensor& input2,
                                 const tensor::Tensor& input3,
                                 const tensor::Tensor& input4,
                                 const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);

  this->set_output(0, output1);
  return this->base_forward();
}

// 5输入1输出的前向传播接口
base::Status Layer::forward_i5o1(const tensor::Tensor& input1,
                                 const tensor::Tensor& input2,
                                 const tensor::Tensor& input3,
                                 const tensor::Tensor& input4,
                                 const tensor::Tensor& input5,
                                 const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->base_forward();
}

// 获取指定索引的权重张量（非const版本）
tensor::Tensor& LayerFp32Param::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

}  // namespace op