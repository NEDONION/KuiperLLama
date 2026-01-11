// 矩阵乘法层：实现矩阵乘法运算 output = weight @ input
// 用于LLama2模型中的所有线性变换（wq, wk, wv, wo, w1, w2, w3, cls）

#ifndef KUIPER_INCLUDE_OP_MATMUL_H_
#define KUIPER_INCLUDE_OP_MATMUL_H_
#include "layer.h"
namespace op {
// MatmulLayer：矩阵乘法层
// 执行 output(dim0) = weight(dim0, dim1) @ input(dim1) 运算
class MatmulLayer : public LayerFp32Param {
 public:
  // 构造函数
  // @param device_type 运行设备类型
  // @param dim0 输出维度
  // @param dim1 输入维度
  explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1);

  // 检查层的有效性
  base::Status check() const override;

  // 执行前向传播
  base::Status base_forward() override;

 private:
  int32_t dim0_ = 0;  // 输出维度（权重矩阵的行数）
  int32_t dim1_ = 0;  // 输入维度（权重矩阵的列数）
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_MATMUL_H_
