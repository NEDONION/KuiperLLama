// RMSNorm层：Root Mean Square归一化
// 用于替代LayerNorm的归一化方法，计算简单且效果相当
#ifndef KUIPER_INCLUDE_OP_RMSNORM_H_
#define KUIPER_INCLUDE_OP_RMSNORM_H_
#include "layer.h"
namespace op {
// RmsNormLayer：RMS归一化层
// 公式：RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
// 对输入进行归一化，稳定训练和推理过程
class RmsNormLayer : public LayerFp32Param {
 public:
  // 构造函数
  // @param device_type 运行设备类型
  // @param dim 输入维度
  explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

  // 检查层的有效性
  base::Status check() const override;

  // 执行前向传播（RMS归一化）
  base::Status base_forward() override;

 private:
  int32_t dim_ = 0;  // 输入维度
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_RMSNORM_H_
