// 编码层：使用SentencePiece进行文本和token之间的转换
// 负责将输入文本编码为token序列，以及将生成的token解码为文本
#ifndef KUIPER_INCLUDE_OP_ENCODE_H_
#define KUIPER_INCLUDE_OP_ENCODE_H_
#include <sentencepiece_processor.h>
#include "layer.h"
namespace op {
// EncodeLayer：文本编码解码层
// 封装SentencePiece tokenizer，提供文本和token之间的双向转换
class EncodeLayer : public Layer {
 public:
  // 默认构造函数
  explicit EncodeLayer(base::DeviceType device_type);

  // 构造函数（带SentencePiece处理器）
  // @param device_type 运行设备类型
  // @param has_bos 是否添加开始符（BOS）
  // @param has_eos 是否添加结束符（EOS）
  // @param sentence_piece_processor SentencePiece处理器
  explicit EncodeLayer(
      base::DeviceType device_type, bool has_bos, bool has_eos,
      std::unique_ptr<sentencepiece::SentencePieceProcessor> sentence_piece_processor);

  // 将文本编码为token序列
  // @param sentence 输入文本
  // @return token ID序列
  std::vector<int32_t> encode(const std::string& sentence) const;

  // 将token ID解码为文本
  // @param token_id token ID
  // @return 解码后的文本片段
  std::string decode(int32_t token_id) const;

  // 获取结束符（EOS）的token ID
  // @return EOS的token ID
  int32_t eos() const;

 private:
  bool has_bos_ = true;   // 是否添加开始符
  bool has_eos_ = false;  // 是否添加结束符
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spe;  // SentencePiece处理器
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_ENCODE_H_
