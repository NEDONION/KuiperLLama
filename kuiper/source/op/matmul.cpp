// 矩阵乘法层实现文件
// 执行矩阵向量乘法运算：output = weight × input

#include "op/matmul.h"
#include "kernels/matmul_kernel.h"
namespace op {
// MatmulLayer构造函数
// 参数:
//   device_type: 设备类型（CPU/CUDA）
//   dim0: 权重矩阵的行数，也是输出向量的维度
//   dim1: 权重矩阵的列数，也是输入向量的维度
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1)
    : LayerFp32Param(device_type, LayerType::kLayerMatmul, "Matmul"),
      dim0_(dim0),
      dim1_(dim1) {
  reset_input_size(1);   // 1个输入：输入向量
  reset_output_size(1);  // 1个输出：输出向量
  reset_weight_size(1);  // 1个权重：权重矩阵
}

// 检查张量的形状是否符合要求
// 输入张量shape: [dim1]
// 权重张量shape: [dim0, dim1]
// 输出张量shape: [dim0]
base::Status MatmulLayer::check() const {
  // 检查输入向量的维度是否为dim1
  auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim1_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the matmul layer.";
    return status;
  }

  // 检查权重矩阵的维度是否为[dim0, dim1]
  status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the matmul layer.";
    return status;
  }

  // 检查输出向量的维度是否为dim0
  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim0_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the matmul layer.";
    return status;
  }
  return base::error::Success();
}

// 前向传播函数
// 执行矩阵向量乘法：output[i] = sum(weight[i][j] * input[j])
base::Status MatmulLayer::base_forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  // 调用对应设备的矩阵乘法kernel
  kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0));
  return base::error::Success();
}
}  // namespace op