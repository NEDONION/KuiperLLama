// KuiperLLama主程序：LLama2模型推理演示程序
// 功能：加载LLama2模型，对输入文本进行编码，执行自回归生成，输出生成的文本
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include "base/alloc.h"
#include "base/buffer.h"
#include "model/llama2.h"
#include "tensor/tensor.h"
#include "base/tick.h"

// 主函数：演示LLama2模型的推理流程
// @param argc 命令行参数数量
// @param argv 命令行参数数组
//   argv[1]: 模型权重文件路径（如 out/model.bin）
//   argv[2]: tokenizer文件路径（如 tokenizer.model）
// @return 执行状态码
int main(int argc, char* argv[]) {
  // 检查命令行参数
  if (argc != 3) {
    LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path ";
    return -1;
  }

  // 创建CPU内存分配器（未使用，可能用于未来扩展）
  std::shared_ptr<base::CPUDeviceAllocator> alloc =
      std::make_shared<base::CPUDeviceAllocator>();

  // 获取模型和tokenizer路径
  const char* checkpoint_path = argv[1];  // 模型权重文件路径
  const char* tokenizer_path = argv[2];   // tokenizer文件路径

  // 创建并初始化LLama2模型
  model::LLama2Model model(tokenizer_path, checkpoint_path);
  model.init(base::DeviceType::kDeviceCPU);

  // 准备输入文本并编码为token序列
  std::string sentence = "Hi everyone";
  const auto& tokens = model.encode(sentence);

  // 执行前向推理（生成最多32个token）
  TICK(A)  // 开始计时
  const auto s = model.forward(tokens, 32);
  TOCK(A)  // 结束计时并输出耗时

  // 输出生成结果
  LOG(INFO) << s;
  return 0;
}