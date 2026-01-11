// 旋转位置编码（RoPE）层实现文件
// 应用旋转位置编码到Query和Key向量，为模型提供位置信息

#include "op/rope.h"
#include <cmath>
#include "kernels/rope_kernel.h"
namespace op {
// RoPELayer构造函数
// 参数:
//   device_type: 设备类型（CPU/CUDA）
//   dim: Query向量的维度
//   kv_dim: Key向量的维度
//   head_size: 每个注意力头的维度
RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim,
                     int32_t head_size)
    : Layer(device_type, LayerType::kLayerRoPe, "RoPe"),
      dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size) {
  reset_input_size(3);   // 3个输入：Query向量、Key向量、位置索引
  reset_output_size(1);  // 输出直接修改输入张量，无需额外输出
}

// 前向传播函数
// 对Query和Key向量应用旋转位置编码
// RoPE原理：将向量每两个维度看作一个复数，根据位置旋转相应角度
// 公式：(x, y) -> (x*cos(θ) - y*sin(θ), x*sin(θ) + y*cos(θ))
// 其中θ = pos / (10000^(2i/d))
base::Status RoPELayer::base_forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }

  tensor::Tensor input_q = this->get_input(0);    // Query向量
  tensor::Tensor input_k = this->get_input(1);    // Key向量
  tensor::Tensor input_pos = this->get_input(2);  // 当前位置索引

  // 调用对应设备的RoPE kernel
  kernel::get_rope_kernel(device_type_)(dim_, kv_dim_, head_size_, input_q, input_k,
                                        input_pos);
  return base::error::Success();
}

// 检查输入张量的形状是否符合要求
base::Status RoPELayer::check() const {
  // 检查位置索引为int32类型，维度为[1]
  auto status = check_tensor_with_dim(get_input(2), device_type_,
                                      base::DataType::kDataTypeInt32, 1);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the add layer.";
    return status;
  }

  // 检查Key向量的维度为[kv_dim]
  status = check_tensor_with_dim(get_input(1), device_type_, data_type_, kv_dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return status;
  }

  // 检查Query向量的维度为[dim]
  status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor 0 error in the add layer.";
    return status;
  }
  return base::error::Success();
}

}  // namespace op