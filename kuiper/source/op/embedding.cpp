// 词嵌入层实现文件
// 将token ID序列转换为对应的词向量表示

#include "op/embedding.h"
#include "kernels/emb_kernel.h"
#include "op/layer.h"
namespace op {
// EmbeddingLayer构造函数
// 参数:
//   device_type: 设备类型（CPU/CUDA）
//   dim: 词向量的维度
//   seq_len: 序列的最大长度
//   vocab_size: 词汇表的大小
EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                               int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerFp32Param(device_type, LayerType::kLayerEmbedding, "Embedding") {
  reset_weight_size(1);  // 1个权重：嵌入矩阵
  reset_input_size(2);   // 2个输入：token ID序列和实际token数量
  reset_output_size(1);  // 1个输出：词向量序列
}

// 检查张量的形状是否符合要求
// 输入0: token ID序列，shape: [token_size]，类型: int32
// 输入1: 实际token数量，用于指示有效token的个数
// 权重: 嵌入矩阵，shape: [vocab_size, dim]
// 输出: 词向量序列，shape: [seq_len, dim]
base::Status EmbeddingLayer::check() const {
  const auto& input_tensor = get_input(0);
  const auto& token_size = get_input(1).size();
  // 检查实际token数量不能超过输入张量的大小
  if (token_size > input_tensor.size()) {
    return base::error::InvalidArgument(
        "The number of input tensor is greater than seq len.");
  }

  // 检查输入张量为int32类型，维度为[token_size]
  base::Status status = check_tensor_with_dim(input_tensor, device_type_,
                                              base::DataType::kDataTypeInt32, token_size);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the embedding layer.";
    return status;
  }

  // 检查嵌入矩阵的维度为[vocab_size, dim]
  status =
      check_tensor_with_dim(get_weight(0), device_type_, data_type_, vocab_size_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding layer.";
    return status;
  }

  // 检查输出张量的维度为[seq_len, dim]
  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, seq_len_, dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding layer.";
    return status;
  }
  return base::error::Success();
}

// 前向传播函数
// 根据token ID从嵌入矩阵中查找对应的词向量
// 例如：token_id=5 -> output = embedding_matrix[5]
base::Status EmbeddingLayer::base_forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }
  // 调用对应设备的嵌入kernel
  kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0),
                                       vocab_size_);
  return base::StatusCode::kSuccess;
}
}  // namespace op