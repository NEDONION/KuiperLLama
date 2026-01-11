// 张量头文件：定义多维张量类，用于存储和操作多维数组数据
#ifndef KUIPER_INCLUDE_TENSOR_TENSOR_H_
#define KUIPER_INCLUDE_TENSOR_TENSOR_H_
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"
namespace tensor {

// Tensor类：多维张量，支持1D到4D的张量操作
class Tensor {
 public:
  explicit Tensor() = default;

  // 构造1D张量
  // @param data_type 数据类型
  // @param dim0 第一维大小
  // @param need_alloc 是否需要立即分配内存
  // @param alloc 内存分配器
  explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr);

  // 构造2D张量
  // @param data_type 数据类型
  // @param dim0 第一维大小
  // @param dim1 第二维大小
  // @param need_alloc 是否需要立即分配内存
  // @param alloc 内存分配器
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr);

  // 构造3D张量
  // @param data_type 数据类型
  // @param dim0 第一维大小（通道数）
  // @param dim1 第二维大小（行数）
  // @param dim2 第三维大小（列数）
  // @param need_alloc 是否需要立即分配内存
  // @param alloc 内存分配器
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                  bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr);

  // 构造4D张量
  // @param data_type 数据类型
  // @param dim0 第一维大小
  // @param dim1 第二维大小
  // @param dim2 第三维大小
  // @param dim3 第四维大小
  // @param need_alloc 是否需要立即分配内存
  // @param alloc 内存分配器
  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                  int32_t dim3, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr);

  // 构造任意维度张量
  // @param data_type 数据类型
  // @param dims 各维度大小向量
  explicit Tensor(base::DataType data_type, std::vector<int32_t> dims);

  // 判断张量是否为空
  // @return 元素个数为0或缓冲区为空时返回true
  bool is_empty() const;

  // 获取数据指针（模板版本，非const）
  template <typename T>
  T* ptr();

  // 获取数据指针（模板版本，const）
  template <typename T>
  const T* ptr() const;

  // 重塑张量形状
  // 如果新形状大于当前大小，会重新分配内存
  // @param dims 新的维度向量
  void reshape(const std::vector<int32_t>& dims);

  // 获取张量元素总数
  size_t size() const;

  // 获取张量占用的字节数
  size_t byte_size() const;

  // 获取张量维度数量
  int32_t dims_size() const;

  // 获取数据类型
  base::DataType data_type() const;

  // 获取指定维度的大小
  // @param idx 维度索引
  // @return 该维度的大小
  int32_t get_dim(int32_t idx) const;

  // 获取所有维度信息
  const std::vector<int32_t>& dims() const;

  // 获取各维度的步长
  // @return 步长向量
  std::vector<size_t> strides() const;

  // 分配外部Buffer给张量
  // @param buffer 外部Buffer指针
  // @return 分配成功返回true
  bool assign(std::shared_ptr<base::Buffer> buffer);

  // 重置张量的数据类型和维度
  // 会清除现有的Buffer
  // @param data_type 新的数据类型
  // @param dims 新的维度向量
  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  // 设置设备类型
  void set_device_type(base::DeviceType device_type);

  // 获取设备类型
  base::DeviceType device_type() const;

  // 分配内存
  // @param allocator 内存分配器
  // @param need_realloc 是否强制重新分配
  // @return 分配成功返回true
  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                bool need_realloc = false);

  // 获取偏移后的数据指针（非const）
  // @param index 偏移量
  template <typename T>
  T* ptr(int64_t index);

  // 获取偏移后的数据指针（const）
  // @param index 偏移量
  template <typename T>
  const T* ptr(int64_t index) const;

  // 通过偏移量访问元素（非const）
  // @param offset 偏移量
  template <typename T>
  T& index(int64_t offset);

  // 通过偏移量访问元素（const）
  // @param offset 偏移量
  template <typename T>
  const T& index(int64_t offset) const;

  // 转置第1维和第2维（用于3D张量）
  // 将(C, H, W)转换为(C, W, H)
  // @param dst 目标张量
  template <typename T>
  void transpose_dim12(Tensor dst);

 private:
  size_t size_ = 0;                                        // 元素总数
  std::vector<int32_t> dims_;                              // 各维度大小
  std::shared_ptr<base::Buffer> buffer_;                   // 数据缓冲区
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;  // 数据类型
};

// 通过偏移量访问元素（非const版本）
// 直接返回指定偏移量处元素的引用
template <typename T>
T& Tensor::index(int64_t offset) {
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

// 通过偏移量访问元素（const版本）
// 直接返回指定偏移量处元素的const引用
template <typename T>
const T& Tensor::index(int64_t offset) const {
  const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

// 获取数据指针（const版本）
// 返回类型为T的const指针
template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

// 获取数据指针（非const版本）
// 返回类型为T的指针
template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T*>(buffer_->ptr());
}

// 获取偏移后的数据指针（非const版本）
// 返回指向偏移index个元素后位置的指针
template <typename T>
T* Tensor::ptr(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

// 获取偏移后的数据指针（const版本）
// 返回指向偏移index个元素后位置的const指针
template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

// 转置第1维和第2维（用于3D张量）
// 将形状(C, H, W)的张量转换为(C, W, H)
// 使用Armadillo库的矩阵转置功能，逐通道进行转置
template <typename T>
void Tensor::transpose_dim12(Tensor dst) {
  // 检查输入输出张量的有效性
  CHECK_EQ(dims_size(), 3);          // 源张量必须是3D
  CHECK_EQ(is_empty(), false);       // 源张量不能为空
  CHECK_EQ(dst.dims_size(), 3);      // 目标张量必须是3D
  CHECK_EQ(dst.is_empty(), false);   // 目标张量不能为空
  CHECK_EQ(get_dim(0), dst.get_dim(0));    // 通道数必须相同
  CHECK_EQ(get_dim(1), dst.get_dim(2));    // 源的行数 = 目标的列数
  CHECK_EQ(get_dim(2), dst.get_dim(1));    // 源的列数 = 目标的行数
  CHECK(device_type() == dst.device_type());         // 设备类型必须相同
  CHECK(device_type() == base::DeviceType::kDeviceCPU);  // 只支持CPU

  int32_t src_ch = this->get_dim(0);   // 通道数
  int32_t src_row = this->get_dim(1);  // 源行数
  int32_t src_col = this->get_dim(2);  // 源列数
  int32_t dst_row = dst.get_dim(1);    // 目标行数
  int32_t dst_col = dst.get_dim(2);    // 目标列数
  int32_t plane_size = src_col * src_row;  // 每个通道的元素数

  T* src_ptr = this->ptr<T>();
  T* dst_ptr = dst.ptr<T>();
  // 逐通道进行转置
  for (int32_t ch = 0; ch < src_ch; ++ch) {
    T* src_ch_ptr = src_ptr + ch * plane_size;
    T* dst_ch_ptr = dst_ptr + ch * plane_size;
    // 使用Armadillo矩阵封装（不复制数据）
    arma::Mat<T> src_mat = arma::Mat<T>(src_ch_ptr, src_col, src_row, false, true);
    arma::Mat<T> dst_mat = arma::Mat<T>(dst_ch_ptr, dst_col, dst_row, false, true);
    dst_mat = src_mat.t();  // 执行转置
  }
}
}  // namespace tensor
#endif  // KUIPER_INCLUDE_TENSOR_TENSOR_H_
