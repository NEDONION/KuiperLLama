// 基础定义头文件：包含设备类型、数据类型、模型类型、状态码等核心定义
#ifndef KUIPER_INCLUDE_BASE_BASE_H_
#define KUIPER_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>
namespace base {
// 设备类型枚举：定义运算设备类型
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,  // 未知设备
  kDeviceCPU = 1,      // CPU设备
};

// 数据类型枚举：定义张量支持的数据类型
enum class DataType : uint8_t {
  kDataTypeUnknown = 0,  // 未知类型
  kDataTypeFp32 = 1,     // 32位浮点数
  kDataTypeInt8 = 2,     // 8位整数
  kDataTypeInt32 = 3,    // 32位整数
};

// 模型类型枚举：定义支持的模型类型
enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,  // 未知模型
  kModelTypeLLama2 = 1,   // LLaMA2模型
};

// 获取数据类型的字节大小
// @param data_type 数据类型
// @return 该类型的字节大小，未知类型返回0
inline size_t DataTypeSize(DataType data_type) {
  if (data_type == DataType::kDataTypeFp32) {
    return sizeof(float);
  } else if (data_type == DataType::kDataTypeInt8) {
    return sizeof(int8_t);
  } else if (data_type == DataType::kDataTypeInt32) {
    return sizeof(int32_t);
  } else {
    return 0;
  }
}

// 不可拷贝基类：禁止拷贝构造和拷贝赋值
class NoCopyable {
 protected:
  NoCopyable() = default;

  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;              // 禁止拷贝构造

  NoCopyable& operator=(const NoCopyable&) = delete;   // 禁止拷贝赋值
};

// 状态码枚举：定义各种错误状态码
enum StatusCode : uint8_t {
  kSuccess = 0,              // 成功
  kFunctionUnImplement = 1,  // 功能未实现
  kPathNotValid = 2,         // 路径无效
  kModelParseError = 3,      // 模型解析错误
  kInternalError = 5,        // 内部错误
  kKeyValueHasExist = 6,     // 键值已存在
  kInvalidArgument = 7,      // 无效参数
};

// 状态类：用于封装函数返回的状态码和错误信息
class Status {
 public:
  // 构造函数
  // @param code 状态码，默认为成功
  // @param err_message 错误信息
  Status(int code = StatusCode::kSuccess, std::string err_message = "");

  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;

  // 赋值状态码
  Status& operator=(int code);

  // 比较状态码
  bool operator==(int code) const;

  bool operator!=(int code) const;

  // 转换为整数状态码
  operator int() const;

  // 转换为布尔值（成功返回true，失败返回false）
  operator bool() const;

  // 获取错误信息
  const std::string& get_err_msg() const;

  // 设置错误信息
  void set_err_msg(const std::string& err_msg);

 private:
  int code_ = StatusCode::kSuccess;  // 状态码
  std::string message_;              // 错误信息
};

namespace error {
// 状态检查宏：检查函数调用的返回状态，失败时打印错误并终止程序
#define STATUS_CHECK(call)                                                                  \
  do {                                                                                     \
    const base::Status& status = call;                                                     \
    if (!status) {                                                                         \
      const size_t buf_size = 512;                                                         \
      char buf[buf_size];                                                                  \
      snprintf(buf, buf_size - 1,                                                          \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
               __LINE__, int(status), status.get_err_msg().c_str());                       \
      LOG(FATAL) << buf;                                                                   \
    }                                                                                      \
  } while (0)

// 以下函数用于创建各种状态对象

// 创建成功状态
Status Success(const std::string& err_msg = "");

// 创建功能未实现状态
Status FunctionNotImplement(const std::string& err_msg = "");

// 创建路径无效状态
Status PathNotValid(const std::string& err_msg = "");

// 创建模型解析错误状态
Status ModelParseError(const std::string& err_msg = "");

// 创建内部错误状态
Status InternalError(const std::string& err_msg = "");

// 创建键已存在状态
Status KeyHasExits(const std::string& err_msg = "");

// 创建无效参数状态
Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

// 输出流操作符重载：用于打印Status对象
std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_BASE_H_
