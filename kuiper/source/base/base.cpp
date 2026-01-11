// 基础类实现：Status类的实现及错误工厂函数
#include "base/base.h"

#include <string>
namespace base {
// Status构造函数：初始化状态码和错误信息
Status::Status(int code, std::string err_message)
    : code_(code), message_(std::move(err_message)) {}

// 赋值操作符：仅赋值状态码
Status& Status::operator=(int code) {
  code_ = code;
  return *this;
};

// 相等比较操作符：比较状态码是否相等
bool Status::operator==(int code) const {
  if (code_ == code) {
    return true;
  } else {
    return false;
  }
};

// 不等比较操作符：比较状态码是否不等
bool Status::operator!=(int code) const {
  if (code_ != code) {
    return true;
  } else {
    return false;
  }
};

// int转换操作符：将Status转换为整数状态码
Status::operator int() const { return code_; }

// bool转换操作符：成功返回true，失败返回false
Status::operator bool() const { return code_ == kSuccess; }

// 获取错误信息
const std::string& Status::get_err_msg() const { return message_; }

// 设置错误信息
void Status::set_err_msg(const std::string& err_msg) { message_ = err_msg; }

namespace error {
// 创建成功状态
Status Success(const std::string& err_msg) { return Status{kSuccess, err_msg}; }

// 创建功能未实现状态
Status FunctionNotImplement(const std::string& err_msg) {
  return Status{kFunctionUnImplement, err_msg};
}

// 创建路径无效状态
Status PathNotValid(const std::string& err_msg) {
  return Status{kPathNotValid, err_msg};
}

// 创建模型解析错误状态
Status ModelParseError(const std::string& err_msg) {
  return Status{kModelParseError, err_msg};
}

// 创建内部错误状态
Status InternalError(const std::string& err_msg) {
  return Status{kInternalError, err_msg};
}

// 创建无效参数状态
Status InvalidArgument(const std::string& err_msg) {
  return Status{kInvalidArgument, err_msg};
}

// 创建键已存在状态
Status KeyHasExits(const std::string& err_msg) {
  return Status{kKeyValueHasExist, err_msg};
}
}  // namespace error

// 输出流操作符：输出错误信息到流
std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.get_err_msg();
  return os;
}

}  // namespace base