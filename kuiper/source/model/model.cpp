// 模型基类实现：提供模型文件加载、缓冲区管理等基础功能
#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
namespace model {
// 析构函数：释放内存映射的模型数据并关闭文件
RawModelData::~RawModelData() {
  // 解除内存映射
  if (data != nullptr && data != MAP_FAILED) {
    munmap(data, file_size);
    data = nullptr;
  }
  // 关闭文件描述符
  if (fd != -1) {
    close(fd);
    fd = -1;
  }
}

// 获取指定偏移处的权重数据指针
// @param offset 权重偏移量（以float为单位）
// @return 权重数据指针
const float* RawModelData::weight(size_t offset) const {
  return weight_data + offset;
}

// 检查指定偏移的权重是否在文件范围内
// @param peek 要检查的偏移量（以float为单位）
// @return 是否有效（未越界）
bool RawModelData::is_weight_valid(size_t peek) const {
  if (peek * sizeof(float) < file_size) {
    return true;
  } else {
    return false;
  }
}

// Model构造函数
// @param model_type 模型类型
// @param token_path tokenizer文件路径
// @param model_path 模型权重文件路径
Model::Model(base::ModelType model_type, std::string token_path, std::string model_path)
    : model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)) {
}

// 获取模型类型
// @return 模型类型枚举
base::ModelType Model::model_type() const {
  return model_type_;
}

// 获取tokenizer文件路径
// @return tokenizer路径
const std::string& Model::token_path() const {
  return token_path_;
}

// 获取模型文件路径
// @return 模型文件路径
const std::string& Model::model_path() const {
  return model_path_;
}

// 插入新的缓冲区张量
// @param buffer_idx 缓冲区类型索引
// @param tensor 要插入的张量
// @return 操作状态（如果键已存在或张量为空则返回错误）
base::Status Model::insert_buffer(ModelBufferType buffer_idx,
                                  const tensor::Tensor& tensor) {
  // 检查缓冲区是否已存在
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) +
                                    " has exits in the buffers");
  }
  // 检查张量是否为空
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

// 获取指定类型的缓冲区（可修改版本）
// @param buffer_idx 缓冲区类型索引
// @return 缓冲区张量引用
tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

// 获取指定类型的缓冲区（只读版本）
// @param buffer_idx 缓冲区类型索引
// @return 缓冲区张量引用
const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

// 读取模型文件并进行内存映射
// 该函数完成以下步骤：
// 1. 打开模型文件
// 2. 读取文件头中的配置信息
// 3. 使用mmap将整个文件映射到内存
// 4. 设置权重数据的起始指针
// @return 操作状态
base::Status Model::read_model_file() {
  using namespace base;
  // 检查模型路径是否为空
  if (model_path_.empty()) {
    return error::PathNotValid(
        "Failed to open the weight file, the model path is empty!");
  }
  // 以只读方式打开模型文件
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  // 使用FILE*读取配置信息
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  // 读取模型配置（文件头部）
  auto config = ModelConfig{};
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model "
        "file.");
  }

  // 生成运行时配置信息
  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    return gen_status;
  }

  // 获取文件大小
  raw_model_data_ = std::make_shared<RawModelData>();
  fseek(file, 0, SEEK_END);
  raw_model_data_->file_size = ftell(file);
  fclose(file);

  // 使用mmap将整个文件映射到内存（只读、私有映射）
  raw_model_data_->fd = fd;
  raw_model_data_->data =
      static_cast<float*>(mmap(nullptr, raw_model_data_->file_size, PROT_READ,
                               MAP_PRIVATE, raw_model_data_->fd, 0));

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory.");
  }

  // 设置权重数据起始指针（跳过配置头部）
  raw_model_data_->weight_data =
      raw_model_data_->data + sizeof(ModelConfig) / sizeof(float);
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError(
        "Failed to map the weight file " + model_path_ +
        " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

// 从文件配置生成运行时模型配置信息
// 该函数将ModelConfig转换为TransformerConfig，并计算派生参数
// @param config 从文件读取的原始配置
// @return 操作状态
base::Status Model::generate_model_infos(const ModelConfig& config) const {
  // 复制基础配置
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.head_num;
  config_->seq_len_ = config.seq_len;

  // 计算派生参数
  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;  // KV缓存维度
  config_->kv_mul_ = config.head_num / config.kv_head_num;  // KV重复倍数（用于分组查询注意力）
  config_->head_size_ = config.dim / config.head_num;  // 每个注意力头的维度

  // 判断是否共享权重（vocab_size为正表示共享embedding和输出层权重）
  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // 验证词汇表大小是否匹配
  if (std::abs(config.vocab_size) != config_->vocab_size_) {
    return base::error::ModelParseError(
        "Vocabulary size mismatch between the model file and the token list.");
  }
  return base::error::Success();
}

// 创建tokenizer编码解码层
// 使用SentencePiece进行文本和token之间的转换
// @return 操作状态
base::Status Model::create_encode_layer() {
  using namespace base;
  // 创建SentencePiece处理器
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spe =
      std::make_unique<sentencepiece::SentencePieceProcessor>();
  const auto& status = spe->Load(token_path_);
  if (!status.ok()) {
    return error::PathNotValid(token_path_);
  }

  // 获取词汇表大小
  config_->vocab_size_ = spe->GetPieceSize();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }

  // 创建编码解码层
  encode_layer_ =
      std::make_unique<op::EncodeLayer>(device_type_, true, false, std::move(spe));
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }
  return error::Success();
}

// 从文件生成完整的模型
// 该函数协调整个模型加载流程：
// 1. 初始化配置
// 2. 创建tokenizer层
// 3. 读取模型文件并内存映射
// 4. 创建所有模型层
// @return 操作状态
base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // 初始化SentencePiece处理器
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }

  // 读取模型文件并内存映射
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }

  // 创建所有模型层
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

}  // namespace model