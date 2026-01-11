// Argmax采样器：贪心采样策略
// 总是选择概率最大的token，结果确定性，适合生成稳定的输出

#ifndef LLAMA_INFER_NON_SAMPLER_H
#define LLAMA_INFER_NON_SAMPLER_H
#include "sampler.h"
namespace sampler {
// ArgmaxSampler：Argmax采样器
// 实现贪心采样：选择logits中值最大的token
// 优点：结果确定、稳定；缺点：缺乏多样性
class ArgmaxSampler : public Sampler {
 public:
  // 采样函数：返回logits中最大值对应的索引
  // @param logits 模型输出的logits数组
  // @param size logits数组的大小
  // @return 最大logit对应的token ID
  int32_t sample(const float* logits, int32_t size) override;
};
}  // namespace sampler
#endif  // LLAMA_INFER_NON_SAMPLER_H
