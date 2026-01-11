// RoPE层：旋转位置编码（Rotary Position Embedding）
// 通过旋转变换为Query和Key向量注入位置信息，无需额外的位置编码
#ifndef KUIPER_INCLUDE_OP_ROPE_H_
#define KUIPER_INCLUDE_OP_ROPE_H_
#include "layer.h"
namespace op {
// RoPELayer：旋转位置编码层
// 对Query和Key向量进行旋转变换，使模型能够感知token的相对位置
// 公式：RoPE(x, pos) = R(pos) @ x，其中R为旋转矩阵
class RoPELayer : public Layer {
 public:
  // 构造函数
  // @param device_type 运行设备类型
  // @param dim Query维度
  // @param kv_dim Key/Value维度
  // @param head_size 每个注意力头的维度
  explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size);

  // 检查层的有效性
  base::Status check() const override;

  // 执行前向传播（应用旋转位置编码）
  base::Status base_forward() override;

 private:
  int32_t dim_ = 0;       // Query维度
  int32_t kv_dim_ = 0;    // Key/Value维度
  int32_t head_size_ = 0; // 每个头的维度
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_ROPE_H_
