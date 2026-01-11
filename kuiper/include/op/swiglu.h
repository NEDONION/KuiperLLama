// SwiGLU层：Swish-Gated Linear Unit激活函数
// 用于LLama2前馈网络的激活函数，结合了Swish和GLU的优点
#ifndef LLAMA_INFER_INCLUDE_OP_SWIGLU_H_
#define LLAMA_INFER_INCLUDE_OP_SWIGLU_H_
#include "layer.h"
namespace op {
// SwiGLULayer：SwiGLU激活函数层
// 公式：SwiGLU(x1, x2) = Swish(x1) * x2
// 其中 Swish(x) = x * sigmoid(x)
// 用于FFN：output = SwiGLU(w1(x), w3(x))
class SwiGLULayer : public op::Layer {
 public:
  // 构造函数
  // @param device_type 运行设备类型
  // @param hidden_dim FFN隐藏层维度
  explicit SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim);

  // 检查层的有效性
  base::Status check() const override;

  // 执行前向传播（SwiGLU激活）
  base::Status base_forward() override;

 private:
  int32_t hidden_dim_ = 0;  // FFN隐藏层维度
};
}  // namespace op
#endif  // LLAMA_INFER_INCLUDE_OP_SWIGLU_H_
