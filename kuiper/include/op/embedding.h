// 嵌入层：将token ID转换为稠密的embedding向量
// 通过查表操作从embedding矩阵中提取对应token的向量表示
#ifndef KUIPER_INCLUDE_OP_EMBEDDING_H_
#define KUIPER_INCLUDE_OP_EMBEDDING_H_
#include "layer.h"
namespace op {
// EmbeddingLayer：嵌入层
// 输入：token序列（整数）
// 输出：embedding向量序列（浮点数）
class EmbeddingLayer : public LayerFp32Param {
 public:
  // 构造函数
  // @param device_type 运行设备类型
  // @param dim embedding维度
  // @param seq_len 最大序列长度
  // @param vocab_size 词汇表大小
  explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                          int32_t vocab_size);

  // 检查层的有效性
  base::Status check() const override;

  // 执行前向传播（查表操作）
  base::Status base_forward() override;

 private:
  int32_t dim_ = 0;         // embedding维度
  int32_t seq_len_ = 0;     // 最大序列长度
  int32_t vocab_size_ = 0;  // 词汇表大小
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_EMBEDDING_H_
