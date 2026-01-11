// 多项式采样器：随机采样策略
// 根据概率分布随机选择token，增加输出的多样性和创造性
#ifndef LLAMA_INFER_MULT_SAMPLER_H
#define LLAMA_INFER_MULT_SAMPLER_H
#include "sampler.h"
#include <random>
namespace sampler {
// MultSampler：多项式采样器
// 实现随机采样：根据softmax后的概率分布进行采样
// 优点：输出多样化、有创造性；缺点：结果不确定
class MultSampler : public Sampler {
 public:
  // 构造函数：初始化随机数生成器
  explicit MultSampler();

  // 采样函数：根据概率分布随机采样
  // @param logits 模型输出的logits数组
  // @param size logits数组的大小
  // @return 采样得到的token ID
  int32_t sample(const float* logits, int32_t size) override;

 private:
  std::mt19937 mt_;  // Mersenne Twister随机数生成器
  std::uniform_real_distribution<float> dist_;  // 均匀分布（0,1）
};
}  // namespace sampler
#endif  // LLAMA_INFER_MULT_SAMPLER_H
