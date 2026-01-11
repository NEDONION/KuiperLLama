// 多头注意力机制层实现文件
// 实现Transformer中的多头自注意力机制，支持KV缓存优化

#include "op/mha.h"
#include "kernels/mha_kernel.h"
namespace op {
// MultiHeadAttention构造函数
// 参数:
//   device_type: 设备类型（CPU/CUDA）
//   layer_index: 当前层的索引，用于访问对应的KV缓存
//   kv_mul: KV头数与Q头数的比例（用于分组查询注意力GQA）
//   kv_dim: KV向量的总维度
//   seq_len: 序列的最大长度
//   head_num: 注意力头的数量
//   head_size: 每个注意力头的维度
MultiHeadAttention::MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                                       int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                       int32_t head_num, int32_t head_size)
    : Layer(device_type, LayerType::kLayerMHA, "MultiHead"),
      layer_index_(layer_index),
      kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      seq_len_(seq_len),
      head_num_(head_num),
      head_size_(head_size) {
  reset_input_size(5);   // 5个输入：query, score, key_cache, value_cache, key
  reset_output_size(1);  // 1个输出：attention输出
}

// 前向传播函数
// 执行多头注意力计算：
// 1. 更新KV缓存
// 2. 计算注意力分数
// 3. 应用softmax
// 4. 加权求和得到输出
base::Status MultiHeadAttention::base_forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  const tensor::Tensor& mha_out = this->get_output(0);
  const tensor::Tensor& query_tensor = this->get_input(0);       // Query向量
  const tensor::Tensor& score_tensor = this->get_input(1);       // 注意力分数缓存
  const tensor::Tensor& key_cache_tensor = this->get_input(2);   // Key缓存
  const tensor::Tensor& value_cache_tensor = this->get_input(3); // Value缓存
  const tensor::Tensor& key_tensor = this->get_input(4);         // 当前位置的Key

  // 调用对应设备的多头注意力kernel
  kernel::get_mha_kernel(device_type_)(
      pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_, head_size_, mha_out,
      query_tensor, score_tensor, key_cache_tensor, value_cache_tensor, key_tensor);
  return base::error::Success();
}

// 设置当前token的位置索引
// 用于在KV缓存中定位当前token的存储位置
void MultiHeadAttention::set_pos(int32_t pos) {
  this->pos_ = pos;
}

// 检查所有输入输出张量的有效性
base::Status MultiHeadAttention::check() const {
  base::Status status;
  const int32_t input_tensor_num = 5;
  // 检查所有输入张量
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    status = check_tensor(get_input(i), device_type_, data_type_);
    if (!status) {
      LOG(ERROR) << "The input tensor " << std::to_string(i)
                 << " error in the matmul layer.";
      return status;
    }
  }
  // 检查输出张量
  return check_tensor(get_output(0), device_type_, data_type_);
}

}  // namespace op