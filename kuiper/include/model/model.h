// 模型基类：定义通用的语言模型接口和基础功能
// 该文件包含模型的抽象基类、缓冲区管理和原始模型数据的处理
#ifndef KUIPER_INCLUDE_MODEL_MODEL_H_
#define KUIPER_INCLUDE_MODEL_MODEL_H_
#include <map>
#include <string>
#include "config.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/encode.h"
#include "op/layer.h"
#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"
#include "op/rope.h"
#include "sampler/argmax_sampler.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"
#include "config.h"

namespace model {
// ModelBufferType：模型内部缓冲区类型枚举
// 用于标识和管理模型推理过程中各个阶段的中间结果张量
enum class ModelBufferType {
  kInputTokens = 0,       // 输入token序列
  kInputEmbeddings = 1,   // 输入token的embedding向量
  kOutputRMSNorm = 2,     // RMSNorm层的输出
  kKeyCache = 3,          // 注意力机制的Key缓存
  kValueCache = 4,        // 注意力机制的Value缓存
  kQuery = 5,             // 查询向量（Q）
  kInputPos = 6,          // 当前位置索引
  kScoreStorage = 7,      // 注意力分数存储
  kOutputMHA = 8,         // 多头注意力输出
  kAttnOutput = 9,        // 注意力输出经过wo投影后的结果
  kW1Output = 10,         // FFN中w1层的输出
  kW2Output = 11,         // FFN中w2层的输出
  kW3Output = 12,         // FFN中w3层的输出
  kFFNRMSNorm = 13,       // FFN前的RMSNorm输出
  kKeyStorage = 14,       // Key临时存储空间

  kForwardOutput = 15,    // 最终的前向传播输出（logits）
};

// RawModelData：原始模型数据管理结构
// 负责模型文件的内存映射和权重数据的访问
struct RawModelData {
  int32_t fd = -1;            // 模型文件描述符
  size_t file_size = 0;       // 模型文件大小（字节）
  float* data = nullptr;      // 内存映射后的数据起始地址
  float* weight_data = nullptr;  // 权重数据起始地址（跳过文件头）

  // 析构函数：释放内存映射和关闭文件描述符
  ~RawModelData();

  // 获取指定偏移处的权重指针
  // @param offset 权重偏移量（以float为单位）
  // @return 权重数据指针
  const float* weight(size_t offset) const;

  // 检查指定偏移的权重是否有效（是否越界）
  // @param peek 要检查的偏移量（以float为单位）
  // @return 是否有效
  bool is_weight_valid(size_t peek) const;
};

// Model：语言模型抽象基类
// 定义了所有语言模型的通用接口和基础功能，包括模型初始化、前向推理、缓冲区管理等
class Model {
 public:
  // 构造函数
  // @param model_type 模型类型（如LLama2）
  // @param token_path tokenizer文件路径
  // @param model_path 模型权重文件路径
  explicit Model(base::ModelType model_type, std::string token_path,
                 std::string model_path);

  // 初始化模型（纯虚函数，由子类实现）
  // @param device_type 运行设备类型（CPU/GPU）
  // @return 操作状态
  virtual base::Status init(base::DeviceType device_type) = 0;

  // 执行前向推理（纯虚函数，由子类实现）
  // @param tokens 输入token序列
  // @param start_pos 起始位置
  // @return 操作状态
  virtual base::Status forward(const std::vector<int>& tokens, int start_pos) = 0;

  // 获取模型类型
  // @return 模型类型
  base::ModelType model_type() const;

  // 获取tokenizer路径
  // @return tokenizer文件路径
  const std::string& token_path() const;

  // 获取模型路径
  // @return 模型文件路径
  const std::string& model_path() const;

 protected:
  // 获取指定类型的缓冲区（可修改版本）
  // @param buffer_idx 缓冲区类型索引
  // @return 缓冲区张量引用
  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  // 获取指定类型的缓冲区（只读版本）
  // @param buffer_idx 缓冲区类型索引
  // @return 缓冲区张量引用
  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  // 插入新的缓冲区
  // @param buffer_idx 缓冲区类型索引
  // @param tensor 要插入的张量
  // @return 操作状态
  virtual base::Status insert_buffer(ModelBufferType buffer_idx,
                                     const tensor::Tensor& tensor);

 protected:
  // 读取模型文件并进行内存映射
  // @return 操作状态
  virtual base::Status read_model_file();

  // 创建tokenizer编码层
  // @return 操作状态
  virtual base::Status create_encode_layer();

  // 从文件生成模型（包括读取配置、权重和创建层）
  // @return 操作状态
  virtual base::Status gen_model_from_file();

  // 生成模型配置信息（从原始配置派生运行时配置）
  // @param config 原始模型配置
  // @return 操作状态
  virtual base::Status generate_model_infos(const ModelConfig& config) const;

  // 后处理：对模型输出进行采样和解码（纯虚函数，由子类实现）
  // @param pos 当前位置
  // @param next 输出的下一个token（通过引用返回）
  // @param tokens 输入token序列
  // @return 解码后的字符串
  virtual std::string post_processing(int32_t pos, int32_t& next,
                                      const std::vector<int32_t>& tokens) const = 0;

 private:
  // 初始化模型内存（缓冲区）（纯虚函数，由子类实现）
  virtual void init_mem() = 0;

  // 创建模型层（纯虚函数，由子类实现）
  // @return 操作状态
  virtual base::Status create_layers() = 0;

  // 将文本编码为token序列（纯虚函数，由子类实现）
  // @param sentence 输入文本
  // @return token序列
  virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

  // 切片获取指定层和位置的KV缓存（纯虚函数，由子类实现）
  // @param layer_idx 层索引
  // @param token_pos token位置
  // @return key和value缓存的张量对
  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(
      int32_t layer_idx, int32_t token_pos) const = 0;

  // 创建带参数的层（如矩阵乘法层）（纯虚函数，由子类实现）
  virtual void create_param_layers() = 0;

  // 创建不带参数的层（如激活函数层）（纯虚函数，由子类实现）
  virtual void create_nonparam_layers() = 0;

 protected:
  std::unique_ptr<TransformerConfig> config_;  // Transformer配置

  std::string token_path_;   // tokenizer文件路径
  std::string model_path_;   // 模型权重文件路径
  std::unique_ptr<op::EncodeLayer> encode_layer_;  // 编码解码层
  std::map<ModelBufferType, tensor::Tensor> buffers_;  // 缓冲区映射表
  std::unique_ptr<sampler::Sampler> sampler_;  // 采样器
  std::shared_ptr<RawModelData> raw_model_data_;  // 原始模型数据
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;  // 设备类型
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;  // 模型类型
};
}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_MODEL_H_
