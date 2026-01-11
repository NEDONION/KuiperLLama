// 内存缓冲区头文件：定义Buffer类用于管理设备内存
#ifndef KUIPER_INCLUDE_BASE_BUFFER_H_
#define KUIPER_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"
namespace base {
// Buffer类：管理设备内存的缓冲区
// 支持内部分配和外部内存，不可拷贝但可以通过shared_ptr共享
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 private:
  size_t byte_size_ = 0;                                    // 缓冲区字节大小
  void* ptr_ = nullptr;                                      // 内存指针
  bool use_external_ = false;                                // 是否使用外部内存
  DeviceType device_type_ = DeviceType::kDeviceUnknown;      // 设备类型
  std::shared_ptr<DeviceAllocator> allocator_;               // 内存分配器

 public:
  explicit Buffer() = default;

  // 构造函数
  // @param byte_size 缓冲区字节大小
  // @param allocator 内存分配器，默认为nullptr
  // @param ptr 外部内存指针，默认为nullptr（表示内部分配）
  // @param use_external 是否使用外部内存，默认为false
  explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  virtual ~Buffer();

  // 分配内存（仅当未使用外部内存时）
  // @return 分配成功返回true，否则返回false
  bool allocate();

  // 从另一个Buffer拷贝数据
  // @param buffer 源Buffer
  void copy_from(const Buffer& buffer) const;

  // 从另一个Buffer指针拷贝数据
  // @param buffer 源Buffer指针
  void copy_from(const Buffer* buffer) const;

  // 获取内存指针（非const）
  void* ptr();

  // 获取内存指针（const）
  const void* ptr() const;

  // 获取缓冲区字节大小
  size_t byte_size() const;

  // 获取内存分配器
  std::shared_ptr<DeviceAllocator> allocator() const;

  // 获取设备类型
  DeviceType device_type() const;

  // 设置设备类型
  void set_device_type(DeviceType device_type);

  // 获取shared_ptr指向自身
  std::shared_ptr<Buffer> get_shared_from_this();

  // 判断是否使用外部内存
  bool is_external() const;
};
}  // namespace base

#endif