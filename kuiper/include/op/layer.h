// 算子层基类：定义神经网络层的抽象接口
// 该文件包含层的基类和通用功能，所有具体的算子层都继承自这些基类
#ifndef KUIPER_INCLUDE_OP_LAYER_H_
#define KUIPER_INCLUDE_OP_LAYER_H_
#include <string>
#include <utility>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace op {
// LayerType：层类型枚举
// 用于标识不同类型的神经网络层
enum class LayerType : uint8_t {
  kLayerUnknown = 0,    // 未知类型
  kLayerLinear = 1,     // 线性层
  kLayerEncode = 2,     // 编码层（tokenizer）
  kLayerEmbedding = 3,  // 嵌入层
  kLayerRMSNorm = 4,    // RMS归一化层
  kLayerMatmul = 5,     // 矩阵乘法层
  kLayerRoPe = 6,       // 旋转位置编码层
  kLayerMHA = 7,        // 多头注意力层
  kLayerSoftmax = 8,    // Softmax激活层
  kLayerAdd = 9,        // 向量加法层
  kLayerSwiGLU = 10,    // SwiGLU激活层
};

// BaseLayer：神经网络层的抽象基类
// 定义了所有层的通用接口，包括前向传播、输入输出管理等
class BaseLayer {
 public:
  // 构造函数
  // @param device_type 运行设备类型
  // @param layer_type 层类型
  // @param data_type 数据类型
  // @param layer_name 层名称（可选）
  explicit BaseLayer(base::DeviceType device_type, LayerType layer_type,
                     base::DataType data_type, std::string layer_name = "");

  // 获取数据类型
  base::DataType data_type() const;

  // 获取层类型
  LayerType layer_type() const;

  // 初始化层（纯虚函数，由子类实现）
  virtual base::Status init() = 0;

  // 执行基础前向传播（纯虚函数，由子类实现）
  virtual base::Status base_forward() = 0;

  // 1输入1输出的前向传播
  virtual base::Status forward_i1o1(const tensor::Tensor& input1,
                                    const tensor::Tensor& output1) = 0;

  // 2输入1输出的前向传播
  virtual base::Status forward_i2o1(const tensor::Tensor& input1,
                                    const tensor::Tensor& input2,
                                    const tensor::Tensor& output1) = 0;

  // 3输入1输出的前向传播
  virtual base::Status forward_i3o1(const tensor::Tensor& input1,
                                    const tensor::Tensor& input2,
                                    const tensor::Tensor& input3,
                                    const tensor::Tensor& output1) = 0;

  // 4输入1输出的前向传播
  virtual base::Status forward_i4o1(const tensor::Tensor& input1,
                                    const tensor::Tensor& input2,
                                    const tensor::Tensor& input3,
                                    const tensor::Tensor& input4,
                                    const tensor::Tensor& output1) = 0;

  // 5输入1输出的前向传播
  virtual base::Status forward_i5o1(const tensor::Tensor& input1,
                                    const tensor::Tensor& input2,
                                    const tensor::Tensor& input3,
                                    const tensor::Tensor& input4,
                                    const tensor::Tensor& input5,
                                    const tensor::Tensor& output1) = 0;

  // 设置输入张量
  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  // 设置输出张量
  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  // 获取输入数量
  virtual size_t input_size() const = 0;

  // 获取输出数量
  virtual size_t output_size() const = 0;

  // 检查层的有效性
  virtual base::Status check() const = 0;

  // 获取输入张量（可修改）
  virtual tensor::Tensor& get_input(int32_t idx) = 0;

  // 获取输出张量（可修改）
  virtual tensor::Tensor& get_output(int32_t idx) = 0;

  // 获取输入张量（只读）
  virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

  // 获取输出张量（只读）
  virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

  // 获取层名称
  const std::string& get_layer_name() const;

  // 设置层名称
  void set_layer_name(const std::string& layer_name);

  // 获取设备类型
  base::DeviceType device_type() const;

  // 设置设备类型
  void set_device_type(base::DeviceType device_type);

 protected:
  std::string layer_name_;  // 层名称
  LayerType layer_type_ = LayerType::kLayerUnknown;  // 层类型
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;  // 数据类型
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;  // 设备类型
};

// Layer：通用层类
// 继承自BaseLayer，提供输入输出张量管理和前向传播的默认实现
class Layer : public BaseLayer {
 public:
  // 构造函数（默认使用fp32数据类型）
  explicit Layer(base::DeviceType device_type, LayerType layer_type,
                 std::string layer_name = "");

  // 初始化层
  base::Status init() override;

  // 检查张量的有效性（设备类型和数据类型）
  base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                            base::DataType data_type) const;

  // 检查张量的有效性（包括维度检查）
  base::Status check_tensor_with_dim(const tensor::Tensor& tensor,
                                     base::DeviceType device_type,
                                     base::DataType data_type, ...) const;

  base::Status check() const override;

  base::Status base_forward() override;

  base::Status forward_i1o1(const tensor::Tensor& input1,
                            const tensor::Tensor& output1) override;

  base::Status forward_i2o1(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) override;

  base::Status forward_i3o1(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3,
                            const tensor::Tensor& output1) override;

  base::Status forward_i4o1(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output1) override;

  base::Status forward_i5o1(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5,
                            const tensor::Tensor& output1) override;

  void set_input(int32_t idx, const tensor::Tensor& input) override;

  void set_output(int32_t idx, const tensor::Tensor& output) override;

  const tensor::Tensor& get_input(int32_t idx) const override;

  const tensor::Tensor& get_output(int32_t idx) const override;

  tensor::Tensor& get_input(int32_t idx) override;

  tensor::Tensor& get_output(int32_t idx) override;

  size_t input_size() const override;

  size_t output_size() const override;

  void reset_input_size(size_t size);

  void reset_output_size(size_t size);

 private:
  std::vector<tensor::Tensor> inputs_;   // 输入张量列表
  std::vector<tensor::Tensor> outputs_;  // 输出张量列表
};

// LayerFp32Param：带fp32参数的层
// 继承自Layer，增加了权重管理功能，用于需要加载模型权重的层
class LayerFp32Param : public Layer {
 public:
  // 构造函数
  explicit LayerFp32Param(base::DeviceType device_type, LayerType layer_type,
                          std::string layer_name = "");

  // 获取权重数量
  size_t weight_size() const;

  // 重置权重数量
  void reset_weight_size(size_t size);

  // 获取指定索引的权重（可修改）
  tensor::Tensor& get_weight(int32_t idx);

  // 获取指定索引的权重（只读）
  const tensor::Tensor& get_weight(int32_t idx) const;

  // 设置权重张量
  void set_weight(int32_t idx, const tensor::Tensor& weight);

  // 从指针设置权重（从模型文件加载）
  // @param idx 权重索引
  // @param dims 权重维度
  // @param weight_ptr 权重数据指针
  // @param device_type 设备类型
  void set_weight(int32_t idx, const std::vector<int32_t>& dims, const float* weight_ptr,
                  base::DeviceType device_type = base::DeviceType::kDeviceUnknown);

 private:
  std::vector<tensor::Tensor> weights_;  // 权重张量列表
  std::vector<tensor::Tensor> inputs_;   // 输入张量列表
  std::vector<tensor::Tensor> outputs_;  // 输出张量列表
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_LAYER_H_
