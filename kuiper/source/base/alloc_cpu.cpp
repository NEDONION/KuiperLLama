// CPU设备分配器实现：提供CPU内存的分配、释放和拷贝功能
#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

// 检查是否支持POSIX对齐内存分配
#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif

namespace base {
// CPUDeviceAllocator构造函数：设置设备类型为CPU
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

// 分配CPU内存
// 优先使用对齐内存分配（POSIX），否则使用标准malloc
// 大于1024字节时使用32字节对齐，否则使用16字节对齐
void* CPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
#ifdef KUIPER_HAVE_POSIX_MEMALIGN
  void* data = nullptr;
  // 根据分配大小选择对齐方式：>=1024字节用32字节对齐，否则用16字节对齐
  const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
  int status = posix_memalign((void**)&data,
                              ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
                              byte_size);
  if (status != 0) {
    return nullptr;
  }
  return data;
#else
  // 如果不支持POSIX对齐分配，使用标准malloc
  void* data = malloc(byte_size);
  return data;
#endif
}

// 释放CPU内存
void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

// CPU内存拷贝
// 使用标准库的memcpy函数
void CPUDeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t size) const {
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);
  if (!size) {
    return;
  }
  std::memcpy(dest_ptr, src_ptr, size);
}

// CPUDeviceAllocatorFactory单例实例
std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base