// Tensor类实现：多维张量的具体实现
#include "tensor/tensor.h"
#include <glog/logging.h>
#include <numeric>

namespace tensor {
// 辅助函数：计算维度向量的乘积
// 用于计算张量的总元素数
// @param begin 维度向量的起始迭代器
// @param end 维度向量的结束迭代器
// @param init 初始值（通常为1）
// @return 所有维度的乘积
template <typename T, typename Tp>
static inline size_t ReduceDimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}

// 1D张量构造函数
Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
  if (need_alloc && alloc) {
    allocate(alloc);  // 如果需要，立即分配内存
  }
}

// 2D张量构造函数
Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
  if (need_alloc && alloc) {
    allocate(alloc);  // 如果需要，立即分配内存
  }
}

// 3D张量构造函数
Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
  if (need_alloc && alloc) {
    allocate(alloc);  // 如果需要，立即分配内存
  }
}

// 4D张量构造函数
Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
               int32_t dim3, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
  if (need_alloc && alloc) {
    allocate(alloc);  // 如果需要，立即分配内存
  }
}

// 任意维度张量构造函数
// 通过维度向量构造张量
Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = ReduceDimension(dims_.begin(), dims_.end(), 1);
}

// 获取张量元素总数
size_t Tensor::size() const {
  return this->size_;
}

// 获取指定维度的大小
// 检查索引合法性后返回对应维度
int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

// 获取设备类型
// 如果buffer为空，返回未知类型
base::DeviceType Tensor::device_type() const {
  if (!buffer_) {
    return base::DeviceType::kDeviceUnknown;
  }
  return buffer_->device_type();
}

// 分配外部Buffer给张量
// 检查Buffer大小是否足够容纳张量数据
bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (byte_size > buffer->byte_size()) {
    LOG(ERROR) << "The size of buffer is too small for the tensor!";
    return false;
  }
  buffer_ = buffer;
  return true;
}

// 为张量分配内存
// @param allocator 内存分配器
// @param need_realloc 是否强制重新分配（默认false）
// 如果已有足够大小的buffer且不强制重分配，则复用现有buffer
bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                      bool need_realloc) {
  if (!allocator) {
    LOG(ERROR) << "The allocator parameter in the allocate function is null "
                  "pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (!byte_size) {
    LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
    return false;
  }

  // 如果已有足够大小的buffer，且不需要重新分配，则直接返回
  if (buffer_ && byte_size <= buffer_->byte_size()) {
    if (!need_realloc) {
      return true;
    }
  }

  // 创建新的Buffer
  buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
  if (!buffer_->ptr()) {
    LOG(ERROR) << "The memory allocated is a null pointer!";
    return false;
  }
  return true;
}

// 获取维度向量
const std::vector<int32_t>& Tensor::dims() const {
  return this->dims_;
}

// 设置设备类型
// 通过Buffer设置设备类型
void Tensor::set_device_type(base::DeviceType device_type) {
  if (buffer_) {
    buffer_->set_device_type(device_type);
  }
}

// 重置张量
// 清除现有buffer，设置新的数据类型和维度
void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
  this->data_type_ = data_type;
  this->dims_ = dims;
  this->size_ = ReduceDimension(dims.begin(), dims.end(), 1);
  this->buffer_ = nullptr;  // 清除buffer
}

// 获取维度数量
int32_t Tensor::dims_size() const {
  return static_cast<int32_t>(dims_.size());
}

// 获取数据类型
base::DataType Tensor::data_type() const {
  return data_type_;
}

// 重塑张量形状
// 如果新的元素总数大于当前大小，会创建新buffer并拷贝数据
void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t size = ReduceDimension(dims.begin(), dims.end(), 1);
  if (!buffer_) {
    this->dims_ = dims;
    this->size_ = size;
    return;
  }

  // 如果新大小超过当前大小，需要重新分配
  if (size > size_) {
    auto new_buffer = std::make_shared<base::Buffer>(
        size * base::DataTypeSize(this->data_type_), buffer_->allocator());
    CHECK(new_buffer->allocate());
    new_buffer->copy_from(buffer_.get());  // 拷贝旧数据
    this->buffer_ = new_buffer;
  }
  this->dims_ = dims;
  this->size_ = size;
}

// 获取张量占用的字节数
size_t Tensor::byte_size() const {
  return this->size() * DataTypeSize(data_type_);
}

// 计算各维度的步长
// 步长表示在该维度上移动一个单位需要跨越的元素数
std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    // 每个维度的步长 = 后续所有维度大小的乘积
    for (int32_t i = 0; i < dims_.size() - 1; ++i) {
      size_t stride = ReduceDimension(dims_.begin() + i + 1, dims_.end(), 1);
      strides.push_back(stride);
    }
    strides.push_back(1);  // 最后一维的步长为1
  }
  return strides;
}

// 判断张量是否为空
// 元素数为0，或buffer为空，或buffer指针为空时返回true
bool Tensor::is_empty() const {
  return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}
}  // namespace tensor