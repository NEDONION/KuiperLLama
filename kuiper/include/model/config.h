// 配置文件：定义模型配置结构体，用于存储LLama模型的超参数
#ifndef KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
namespace model {
// ModelConfig：从模型文件中直接读取的原始配置信息
// 该结构体与模型文件中保存的配置格式完全对应
struct ModelConfig {
  int32_t dim = 0;          // 模型的嵌入维度（embedding dimension）
  int32_t hidden_dim = 0;   // 前馈网络的隐藏层维度
  int32_t layer_num = 0;    // Transformer层的数量
  int32_t head_num = 0;     // 多头注意力机制中头的数量
  int32_t kv_head_num = 0;  // KV缓存的头数量（用于分组查询注意力）
  int32_t vocab_size = 0;   // 词汇表大小（正数表示共享权重，负数表示不共享）
  int32_t seq_len = 0;      // 最大序列长度
};

// TransformerConfig：运行时使用的扩展配置信息
// 在ModelConfig基础上增加了一些派生参数，用于模型计算
struct TransformerConfig {
  int32_t kv_dim_ = 0;      // KV缓存的维度（dim * kv_head_num / head_num）
  int32_t kv_mul_ = 0;      // KV重复倍数（head_num / kv_head_num）
  int32_t head_size_ = 0;   // 每个注意力头的维度（dim / head_num）
  int32_t vocab_size_ = 0;  // 词汇表大小（绝对值）

  int32_t dim_ = 0;         // 模型的嵌入维度
  int32_t hidden_dim_ = 0;  // 前馈网络的隐藏层维度
  int32_t layer_num_ = 0;   // Transformer层的数量
  int32_t head_num_ = 0;    // 多头注意力机制中头的数量
  int32_t kv_head_num_ = 0; // KV缓存的头数量
  int32_t seq_len_ = 0;     // 最大序列长度
  bool is_shared_weight_ = false;  // 是否共享embedding和输出层的权重
};
}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
