// 多头注意力层：实现Transformer的多头自注意力机制
// 支持分组查询注意力（Grouped Query Attention, GQA）和KV缓存
#ifndef KUIPER_INLCUDE_MHA_H
#define KUIPER_INLCUDE_MHA_H
#include "layer.h"
namespace op {
// MultiHeadAttention：多头注意力层
// 实现自注意力机制：Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
// 支持KV缓存以加速自回归生成
class MultiHeadAttention : public op::Layer {
 public:
  // 构造函数
  // @param device_type 运行设备类型
  // @param layer_index 层索引（用于定位KV缓存）
  // @param kv_mul KV重复倍数（用于分组查询注意力）
  // @param kv_dim KV缓存维度
  // @param seq_len 最大序列长度
  // @param head_num 注意力头数量
  // @param head_size 每个注意力头的维度
  explicit MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                              int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                              int32_t head_num, int32_t head_size);

  // 检查层的有效性
  base::Status check() const override;

  // 设置当前处理位置（用于KV缓存索引）
  void set_pos(int32_t pos);

  // 执行前向传播（计算多头注意力）
  base::Status base_forward() override;

 private:
  int32_t layer_index_ = 0;  // 层索引
  int32_t pos_ = 0;          // 当前位置
  int32_t kv_mul_ = 0;       // KV重复倍数（head_num / kv_head_num）
  int32_t kv_dim_ = 0;       // KV缓存维度
  int32_t seq_len_ = 0;      // 最大序列长度
  int32_t head_num_ = 0;     // 注意力头数量
  int32_t head_size_ = 0;    // 每个头的维度
};
}  // namespace op
#endif  // KUIPER_INLCUDE_MHA_H
