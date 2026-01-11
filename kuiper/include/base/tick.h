// 性能计时工具：提供TICK/TOCK宏用于测量代码执行时间
#ifndef KUIPER_INFER_INCLUDE_TICK_HPP_
#define KUIPER_INFER_INCLUDE_TICK_HPP_
#include <chrono>
#include <iostream>

#ifndef __ycm__
// TICK宏：记录起始时间点
// 用法: TICK(my_operation)
// 会创建一个名为bench_my_operation的时间点变量
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();

// TOCK宏：计算并输出从TICK到现在的耗时
// 用法: TOCK(my_operation)
// 会输出 "my_operation: X.XXXXXs"
#define TOCK(x)                                                     \
  printf("%s: %lfs\n", #x,                                          \
         std::chrono::duration_cast<std::chrono::duration<double>>( \
             std::chrono::steady_clock::now() - bench_##x)          \
             .count());
#else
// YCM（YouCompleteMe）模式下禁用计时宏
#define TICK(x)
#define TOCK(x)
#endif
#endif  // KUIPER_INFER_INCLUDE_TICK_HPP_
