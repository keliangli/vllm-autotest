# vLLM 吞吐性能参数文档 - 项目总结

## 项目概述

根据需求，本项目开发了一套完整的工具和文档，用于搜索和记录所有能够影响 vLLM 推理模型吞吐性能的参数。

## 交付物

### 1. 参数提取脚本 (`generate_throughput_params_doc.py`)
- **大小**: 18KB
- **行数**: ~450 行
- **功能**: 
  - 自动提取和分类 vLLM 的性能参数
  - 无需安装 vLLM 依赖即可运行
  - 自动生成 Markdown 格式文档

### 2. 参数文档 (`vllm_throughput_parameters.md`)
- **大小**: 15KB
- **行数**: 281 行
- **内容**:
  - 53 个影响吞吐性能的参数
  - 6 个主要配置类别
  - 详细的参数描述和使用场景
  - 多个实际配置示例
  - 性能调优建议

### 3. 使用说明 (`README_THROUGHPUT_PARAMS.md`)
- **大小**: 4.5KB
- **行数**: 138 行
- **内容**:
  - 工具使用指南
  - 常见优化模式
  - 维护和更新说明

## 参数分类统计

### 配置参数 (41个)
1. **调度器配置** (SchedulerConfig) - 9个参数
   - `max_num_batched_tokens`: 单次迭代处理的最大 token 数
   - `max_num_seqs`: 单次迭代处理的最大序列数
   - `enable_chunked_prefill`: 是否启用分块预填充
   - 等...

2. **缓存配置** (CacheConfig) - 10个参数
   - `gpu_memory_utilization`: GPU 内存利用率 (0-1)
   - `block_size`: KV 缓存块大小
   - `cache_dtype`: 缓存数据类型 (fp8/bfloat16)
   - `enable_prefix_caching`: 是否启用前缀缓存
   - 等...

3. **并行配置** (ParallelConfig) - 9个参数
   - `tensor_parallel_size`: 张量并行大小
   - `pipeline_parallel_size`: 流水线并行大小
   - `data_parallel_size`: 数据并行大小
   - `enable_expert_parallel`: 是否启用专家并行 (MoE)
   - 等...

4. **模型配置** (ModelConfig) - 7个参数
   - `max_model_len`: 最大序列长度
   - `dtype`: 模型数据类型
   - `quantization`: 量化方法 (AWQ/GPTQ/FP8)
   - `enforce_eager`: 是否强制使用 eager 执行
   - 等...

5. **编译配置** (CompilationConfig) - 3个参数
   - `cudagraph_capture_sizes`: CUDA 图捕获的批次大小列表
   - `max_cudagraph_capture_size`: 最大 CUDA 图捕获大小
   - `optimization_level`: 优化级别 (O0-O3)

6. **其他性能参数** - 3个参数
   - `seed`: 随机种子
   - `disable_log_stats`: 是否禁用统计日志
   - `load_format`: 模型加载格式

### 环境变量 (12个)
- `VLLM_ATTENTION_BACKEND`: 注意力后端选择
- `VLLM_FUSED_MOE_CHUNK_SIZE`: MoE 融合操作块大小
- `VLLM_USE_FLASHINFER_SAMPLER`: 使用 FlashInfer 采样器
- `VLLM_FLASHINFER_MOE_BACKEND`: FlashInfer MoE 后端模式
- `VLLM_LOG_BATCHSIZE_INTERVAL`: 批次大小日志间隔
- 等...

## 使用示例

### 运行脚本生成文档
```bash
python3 generate_throughput_params_doc.py
```

### 最大吞吐配置示例
```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,           # 使用 2 GPU 张量并行
    gpu_memory_utilization=0.95,      # 使用 95% GPU 内存
    max_num_batched_tokens=8192,      # 大批次提高吞吐
    max_num_seqs=256,                 # 并行处理更多序列
    enable_prefix_caching=True,       # 启用前缀缓存
    enable_chunked_prefill=True,      # 启用分块预填充
)
```

### 命令行使用
```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --enable-prefix-caching
```

### 环境变量配置
```bash
# 使用 FlashInfer 后端
export VLLM_ATTENTION_BACKEND=FLASHINFER

# 优化 MoE 模型吞吐
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_FUSED_MOE_CHUNK_SIZE=32768

# 禁用日志提高性能
export VLLM_LOG_BATCHSIZE_INTERVAL=-1

vllm serve model --tensor-parallel-size 2
```

## 性能调优关键点

### 1. 内存利用率最大化
- 生产环境设置 `gpu_memory_utilization=0.9-0.95`
- 开发环境使用较低值 (0.7-0.8) 避免 OOM

### 2. 批次大小优化
- `max_num_batched_tokens`: 从 2048 开始，提高到 8192+ 获得更高吞吐
- `max_num_seqs`: 从 128 开始，根据请求模式调整

### 3. CUDA 图加速
- 默认自动启用
- 显著提升解码阶段吞吐
- 可通过 `cudagraph_capture_sizes` 优化特定批次大小

### 4. 前缀缓存
- 对于共享前缀的请求提供 2-10x 加速
- 特别适合聊天应用

### 5. 并行策略
- **张量并行**: 用于超大模型
- **数据并行**: 提高整体吞吐，运行多个副本
- **流水线并行**: 用于超大层数模型

### 6. 量化技术
- AWQ/GPTQ: 4-bit 量化，约 2x 内存减少
- FP8: 8-bit 量化，性能和精度平衡
- 支持更大批次大小，提高吞吐

### 7. 工作负载调优
- **批处理**: 最大化 `max_num_batched_tokens` 和 `max_num_seqs`
- **交互式/聊天**: 使用分块预填充、前缀缓存
- **长上下文**: 考虑滑动窗口注意力、CPU offloading

## 技术亮点

1. **完整性**: 覆盖所有主要影响吞吐的参数类别
2. **实用性**: 提供多种场景的实际配置示例
3. **可维护性**: 脚本可重复运行，保持文档更新
4. **易用性**: 无需 vLLM 依赖即可运行脚本
5. **中英文**: 支持中文使用说明和英文技术文档

## 文件位置

```
vllm-autotest/
├── generate_throughput_params_doc.py   # 参数提取脚本
├── vllm_throughput_parameters.md      # 参数文档 (英文)
├── README_THROUGHPUT_PARAMS.md        # 使用说明 (英文)
└── SUMMARY_CN.md                      # 本文件 (中文总结)
```

## 后续使用建议

1. **查看文档**: 阅读 `vllm_throughput_parameters.md` 了解所有参数
2. **运行脚本**: 在 vLLM 版本更新后重新运行脚本更新文档
3. **实际测试**: 根据自己的硬件和模型选择合适的参数配置
4. **性能监控**: 实时监控 GPU 利用率、内存使用和吞吐量
5. **迭代优化**: 基于实际负载调整参数设置

## 参考资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [引擎参数参考](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- [性能基准测试](https://github.com/vllm-project/vllm/tree/main/benchmarks)

## 总结

本项目成功完成了需求目标，提供了：
- ✅ 自动化的参数搜索和提取工具
- ✅ 53 个吞吐性能相关参数的完整文档
- ✅ 6 大类参数的详细分类和说明
- ✅ 丰富的使用示例和调优建议
- ✅ 易于维护和更新的文档生成系统

所有文件已提交到 Git 仓库，可以立即使用。
