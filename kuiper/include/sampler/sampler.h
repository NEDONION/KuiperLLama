// 采样器基类：定义从模型输出logits中采样下一个token的接口
// 不同的采样策略（如argmax、多项式采样等）继承此基类实现
#ifndef LLAMA_INFER_SAMPLER_H
#define LLAMA_INFER_SAMPLER_H
#include <cstddef>
#include <cstdint>
namespace sampler {
// Sampler：采样器抽象基类
// 用于从模型输出的logits（未归一化的概率分布）中选择下一个token
class Sampler {
 public:
  // 从logits中采样一个token
  // @param logits 模型输出的logits数组
  // @param size logits数组的大小（词汇表大小）
  // @return 采样得到的token ID
  virtual int32_t sample(const float* logits, int32_t size) = 0;
};
}  // namespace sampler
#endif  // LLAMA_INFER_SAMPLER_H
