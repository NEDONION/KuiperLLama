// 向量加法层：实现元素级向量加法运算 output = input1 + input2
// 用于LLama2模型中的残差连接（residual connection）
#ifndef KUIPER_INCLUDE_OP_ADD_H
#define KUIPER_INCLUDE_OP_ADD_H
#include "base/base.h"
#include "layer.h"
namespace op {
// VecAddLayer：向量加法层
// 执行逐元素加法：output[i] = input1[i] + input2[i]
class VecAddLayer : public op::Layer {
 public:
  // 构造函数
  // @param device_type 运行设备类型
  explicit VecAddLayer(base::DeviceType device_type);

  // 检查层的有效性
  base::Status check() const override;

  // 执行前向传播（向量加法）
  base::Status base_forward() override;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_ADD_H