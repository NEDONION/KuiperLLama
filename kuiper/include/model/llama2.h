// LLama2模型实现：LLama2语言模型的具体实现
// 该文件包含LLama2模型的网络结构定义、层管理和前向推理逻辑
#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <map>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {

// EmbeddingOutput：Embedding层的输出结构
// 包含输入token、对应的embedding向量和token数量
struct EmbeddingOutput {
  tensor::Tensor input_tokens;       // 输入的token序列
  tensor::Tensor input_embeddings;   // token对应的embedding向量
  tensor::Tensor input_token_num;    // token的数量
};

// LLama2Layers：LLama2模型的所有层组件
// 该结构体管理LLama2模型中的所有神经网络层
struct LLama2Layers {
  // 非参数层（无需加载权重的层）
  std::shared_ptr<op::VecAddLayer> add_layer_;       // 向量加法层（用于残差连接）
  std::shared_ptr<op::RoPELayer> rope_layer_;        // 旋转位置编码层
  std::shared_ptr<op::SwiGLULayer> swiglu_layer_;    // SwiGLU激活函数层

  // 多头注意力层（每个Transformer块一个）
  std::vector<std::shared_ptr<op::MultiHeadAttention>> mha_layers_;

  // 注意力机制的权重矩阵（每个Transformer块一组）
  std::vector<std::shared_ptr<op::MatmulLayer>> wq_layers_;  // Query投影矩阵
  std::vector<std::shared_ptr<op::MatmulLayer>> wk_layers_;  // Key投影矩阵
  std::vector<std::shared_ptr<op::MatmulLayer>> wv_layers_;  // Value投影矩阵
  std::vector<std::shared_ptr<op::MatmulLayer>> wo_layers_;  // 输出投影矩阵

  // 前馈网络的权重矩阵（每个Transformer块一组）
  std::vector<std::shared_ptr<op::MatmulLayer>> w1_layers_;  // FFN第一层（用于SwiGLU的gate）
  std::vector<std::shared_ptr<op::MatmulLayer>> w2_layers_;  // FFN第二层（降维）
  std::vector<std::shared_ptr<op::MatmulLayer>> w3_layers_;  // FFN第三层（用于SwiGLU的up）

  // RMSNorm归一化层（每个Transformer块两个：注意力前和FFN前，再加最终输出的一个）
  std::vector<std::shared_ptr<op::RmsNormLayer>> rmsnorm_layers_;

  // 分类层（输出logits）
  std::shared_ptr<op::MatmulLayer> cls_layer_;

  // Embedding层（将token转换为向量）
  std::shared_ptr<op::EmbeddingLayer> embedding_layer_;
};

// LLama2Model：LLama2模型的主类
// 继承自Model基类，实现LLama2的具体推理逻辑
class LLama2Model : public Model {
 public:
  // 构造函数
  // @param token_path tokenizer文件路径
  // @param model_path 模型权重文件路径
  explicit LLama2Model(std::string token_path, std::string model_path);

  // 初始化模型（加载权重、创建层、分配缓冲区）
  // @param device_type 运行设备类型
  // @return 操作状态
  base::Status init(base::DeviceType device_type) override;

  // 执行前向推理（生成文本）
  // @param tokens 输入token序列
  // @param total_steps 最大生成步数
  // @return 操作状态
  base::Status forward(const std::vector<int>& tokens, int32_t total_steps) override;

  // 将文本编码为token序列
  // @param sentence 输入文本
  // @return token序列
  std::vector<int32_t> encode(const std::string& sentence) const override;

  // 切片获取指定层和位置的KV缓存
  // @param layer_idx 层索引
  // @param token_pos token位置
  // @return key和value缓存的张量对
  std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(
      int32_t layer_idx, int32_t token_pos) const override;

 private:
  // 初始化模型内存（分配所有缓冲区）
  void init_mem() override;

  // 创建所有模型层
  // @return 操作状态
  base::Status create_layers() override;

  // 创建带参数的层（加载权重）
  void create_param_layers() override;

  // 创建不带参数的层
  void create_nonparam_layers() override;

  // 执行多头注意力计算
  // @param layer_idx 层索引
  // @param pos_tensor 当前位置张量
  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  // 执行embedding操作（将token转换为向量）
  // @param tokens 输入token序列
  // @return embedding输出
  EmbeddingOutput embedding(const std::vector<int>& tokens) const;

  // 执行注意力前的RMSNorm归一化
  // @param layer_idx 层索引
  // @param input 输入张量
  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  // 执行前馈网络计算
  // @param layer_idx 层索引
  // @param input 输入张量（同时作为输出，通过残差连接更新）
  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  // 填充输入张量（处理prefill和generation两个阶段）
  // @param next 下一个token（用于generation阶段）
  // @param pos_tensor 当前位置张量
  // @param tokens 输入token序列
  // @param input 输出的输入张量
  // @param embedding_output embedding层的输出
  void fill_input(int32_t next, const tensor::Tensor& pos_tensor,
                  const std::vector<int32_t>& tokens, tensor::Tensor& input,
                  const EmbeddingOutput& embedding_output) const;

  // 执行注意力机制的QKV计算
  // @param layer_idx 层索引
  // @param pos_tensor 当前位置张量
  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  // 计算最终的分类logits
  // @param input 输入张量
  void cls_logits(const tensor::Tensor& input) const;

  // 后处理：采样下一个token并解码为文本
  // @param pos 当前位置
  // @param next 输出的下一个token（通过引用返回）
  // @param tokens 输入token序列
  // @return 解码后的字符串
  std::string post_processing(int32_t pos, int32_t& next,
                              const std::vector<int32_t>& tokens) const override;

 private:
  std::unique_ptr<LLama2Layers> llama_layers_;  // LLama2模型的所有层
};
}  // namespace model

#endif