// 设备内存分配器头文件：定义设备内存分配器的抽象接口和CPU实现
#ifndef KUIPER_INCLUDE_BASE_ALLOC_H_
#define KUIPER_INCLUDE_BASE_ALLOC_H_
#include <memory>
#include "base.h"
namespace base {
// DeviceAllocator：设备内存分配器的抽象基类
// 定义了内存分配、释放、拷贝的统一接口
class DeviceAllocator {
 public:
  // 构造函数
  // @param device_type 设备类型
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {
  }

  // 获取设备类型
  virtual DeviceType device_type() const {
    return device_type_;
  }

  // 释放内存（纯虚函数，由子类实现）
  // @param ptr 要释放的内存指针
  virtual void release(void* ptr) const = 0;

  // 分配内存（纯虚函数，由子类实现）
  // @param size 要分配的字节数
  // @return 分配的内存指针
  virtual void* allocate(size_t size) const = 0;

  // 内存拷贝（纯虚函数，由子类实现）
  // @param src_ptr 源内存指针
  // @param dest_ptr 目标内存指针
  // @param size 要拷贝的字节数
  virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t size) const = 0;

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;  // 设备类型
};

// CPUDeviceAllocator：CPU设备的内存分配器实现
class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  // 在CPU上分配内存
  // @param size 要分配的字节数
  // @return 分配的内存指针
  void* allocate(size_t size) const override;

  // 释放CPU内存
  // @param ptr 要释放的内存指针
  void release(void* ptr) const override;

  // CPU内存拷贝
  // @param src_ptr 源内存指针
  // @param dest_ptr 目标内存指针
  // @param size 要拷贝的字节数
  void memcpy(const void* src_ptr, void* dest_ptr, size_t size) const override;
};

// CPUDeviceAllocatorFactory：CPU分配器的单例工厂
class CPUDeviceAllocatorFactory {
 public:
  // 获取CPU分配器的单例实例
  // @return CPU分配器的shared_ptr
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;  // 单例实例
};

}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_ALLOC_H_