// Buffer类实现：设备内存缓冲区的管理实现
#include "base/buffer.h"
#include <glog/logging.h>

namespace base {
// Buffer构造函数：初始化缓冲区
// 如果未提供外部内存指针，且提供了分配器，则自动分配内存
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr,
               bool use_external)
    : byte_size_(byte_size),
      allocator_(allocator),
      ptr_(ptr),
      use_external_(use_external) {
  if (!ptr_ && allocator_) {
    device_type_ = allocator_->device_type();
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size);  // 自动分配内存
  }
}

// Buffer析构函数：释放非外部内存
Buffer::~Buffer() {
  if (!use_external_) {  // 只释放内部分配的内存
    if (ptr_ && allocator_) {
      LOG(INFO) << "Release...";
      allocator_->release(ptr_);
      ptr_ = nullptr;
    }
  }
}

// 获取内存指针（非const版本）
void* Buffer::ptr() {
  return ptr_;
}

// 获取内存指针（const版本）
const void* Buffer::ptr() const {
  return ptr_;
}

// 获取缓冲区字节大小
size_t Buffer::byte_size() const {
  return byte_size_;
}

// 手动分配内存
// 要求分配器和字节大小都不为0
bool Buffer::allocate() {
  if (allocator_ && byte_size_ != 0) {
    use_external_ = false;
    ptr_ = allocator_->allocate(byte_size_);
    if (!ptr_) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

// 获取分配器
std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
  return allocator_;
}

// 从另一个Buffer拷贝数据（引用版本）
// 拷贝大小为两个Buffer中较小的那个
void Buffer::copy_from(const Buffer& buffer) const {
  CHECK(allocator_ != nullptr && buffer.allocator_ != nullptr);
  CHECK(this->device_type() == buffer.device_type());  // 设备类型必须相同

  size_t copy_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
  return allocator_->memcpy(this->ptr_, buffer.ptr_, copy_size);
}

// 从另一个Buffer拷贝数据（指针版本）
// 拷贝大小为两个Buffer中较小的那个
void Buffer::copy_from(const Buffer* buffer) const {
  if (!buffer) {
    return;
  }
  CHECK(allocator_ != nullptr && buffer->allocator_ != nullptr);
  CHECK(this->device_type() == buffer->device_type());  // 设备类型必须相同

  size_t src_size = byte_size_;
  size_t dest_size = buffer->byte_size_;
  size_t copy_size = src_size < dest_size ? src_size : dest_size;
  return allocator_->memcpy(buffer->ptr_, this->ptr_, copy_size);
}

// 获取设备类型
DeviceType Buffer::device_type() const {
  return device_type_;
}

// 设置设备类型
void Buffer::set_device_type(DeviceType device_type) {
  device_type_ = device_type;
}

// 获取shared_ptr指向自身
std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
  return shared_from_this();
}

// 判断是否使用外部内存
bool Buffer::is_external() const {
  return this->use_external_;
}

}  // namespace base