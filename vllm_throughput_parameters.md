# vLLM Throughput Performance Parameters

This document lists all vLLM parameters that can affect the inference throughput performance of LLM models.

## Overview

vLLM provides numerous configuration parameters that can significantly impact the throughput (tokens/second) of your LLM inference workloads. These parameters are organized into several categories based on their configuration class.

## Key Concepts

- **Throughput**: The number of tokens generated per second across all requests
- **Latency**: The time taken to complete a single request
- **Batch Size**: Number of sequences processed in parallel
- **Memory Utilization**: How efficiently GPU memory is used

## Configuration Categories

### Scheduler Configuration

| Parameter | Description | Config Class |
|-----------|-------------|-------------|
| `max_num_batched_tokens` | Maximum number of tokens to be processed in a single iteration. Higher values increase throughput but require more memory. | `SchedulerConfig` |
| `max_num_seqs` | Maximum number of sequences to be processed in a single iteration. Controls batch size. | `SchedulerConfig` |
| `enable_chunked_prefill` | If True, prefill requests can be chunked based on the remaining max_num_batched_tokens. Can improve throughput for mixed workloads. | `SchedulerConfig` |
| `max_num_partial_prefills` | For chunked prefill, the maximum number of sequences that can be partially prefilled concurrently. | `SchedulerConfig` |
| `max_long_partial_prefills` | For chunked prefill, the maximum number of prompts longer than long_prefill_token_threshold that will be prefilled concurrently. | `SchedulerConfig` |
| `long_prefill_token_threshold` | For chunked prefill, a request is considered long if the prompt is longer than this number of tokens. | `SchedulerConfig` |
| `policy` | Scheduling policy (fcfs, priority). Affects how requests are prioritized. | `SchedulerConfig` |
| `async_scheduling` | Whether to enable asynchronous scheduling for better throughput. | `SchedulerConfig` |
| `stream_interval` | Interval for streaming output tokens, affects latency vs throughput tradeoff. | `SchedulerConfig` |

### Cache Configuration

| Parameter | Description | Config Class |
|-----------|-------------|-------------|
| `block_size` | Size of a contiguous cache block in number of tokens. Valid range: 1-256 (inclusive). Default varies by platform. Affects memory efficiency and throughput. | `CacheConfig` |
| `gpu_memory_utilization` | Fraction of GPU memory to be used (0-1). Higher values allow more sequences in parallel, increasing throughput. | `CacheConfig` |
| `swap_space` | Size of the CPU swap space per GPU (in GiB). Allows handling more sequences but may reduce throughput. | `CacheConfig` |
| `cache_dtype` | Data type for kv cache storage (auto, fp8, bfloat16). fp8 reduces memory and increases throughput. | `CacheConfig` |
| `enable_prefix_caching` | Whether to enable prefix caching. Can significantly improve throughput for requests with common prefixes. | `CacheConfig` |
| `cpu_offload_gb` | Space in GiB to offload to CPU per GPU. Increases effective memory but may reduce throughput. | `CacheConfig` |
| `kv_cache_memory_bytes` | Size of KV Cache per GPU in bytes. Allows fine-grain control of memory allocation. | `CacheConfig` |
| `sliding_window` | Sliding window size for the KV cache. Affects memory usage and throughput for long sequences. | `CacheConfig` |
| `kv_offloading_size` | Size of the KV cache offloading buffer in GiB. Enables offloading but may impact throughput. | `CacheConfig` |
| `kv_offloading_backend` | Backend for KV cache offloading (native, lmcache). | `CacheConfig` |

### Parallel Configuration

| Parameter | Description | Config Class |
|-----------|-------------|-------------|
| `tensor_parallel_size` | Number of tensor parallel groups. Distributes model across GPUs for larger models, enabling higher throughput. | `ParallelConfig` |
| `pipeline_parallel_size` | Number of pipeline parallel groups. Splits model into stages for better GPU utilization. | `ParallelConfig` |
| `data_parallel_size` | Number of data parallel groups. Increases overall throughput by running multiple replicas. | `ParallelConfig` |
| `prefill_context_parallel_size` | Number of prefill context parallel groups. Accelerates prefill phase. | `ParallelConfig` |
| `decode_context_parallel_size` | Number of decode context parallel groups. Accelerates decode phase. | `ParallelConfig` |
| `enable_expert_parallel` | Whether to enable expert parallelism for MoE models. Can improve throughput for large MoE models. | `ParallelConfig` |
| `enable_dbo` | Whether to enable Disaggregated Batch Offloading for better resource utilization. | `ParallelConfig` |
| `disable_custom_all_reduce` | Disable custom all reduce implementation. May affect throughput in multi-GPU setups. | `ParallelConfig` |
| `worker_cls` | Worker class to use. Different implementations may have different performance characteristics. | `ParallelConfig` |

### Model Configuration

| Parameter | Description | Config Class |
|-----------|-------------|-------------|
| `max_model_len` | Maximum length of a sequence (including prompt and generated text). Affects memory and throughput. | `ModelConfig` |
| `dtype` | Data type for model weights (auto, float16, bfloat16, float32). Lower precision can increase throughput. | `ModelConfig` |
| `quantization` | Quantization method to use. Reduces memory and increases throughput (AWQ, GPTQ, SqueezeLLM, FP8, etc.). | `ModelConfig` |
| `enforce_eager` | Whether to enforce eager execution. Disabling CUDA graphs may reduce throughput but save memory. | `ModelConfig` |
| `trust_remote_code` | Whether to trust remote code. Some models require this for optimal performance. | `ModelConfig` |
| `max_logprobs` | Maximum number of log probabilities to return. Lower values reduce computation. | `ModelConfig` |
| `disable_sliding_window` | Whether to disable sliding window attention. May affect throughput for long sequences. | `ModelConfig` |

### Compilation Configuration

| Parameter | Description | Config Class |
|-----------|-------------|-------------|
| `cudagraph_capture_sizes` | List of batch sizes to capture CUDA graphs for. CUDA graphs significantly improve throughput. | `CompilationConfig` |
| `max_cudagraph_capture_size` | Maximum batch size for CUDA graph capture. | `CompilationConfig` |
| `optimization_level` | Optimization level (O0-O3). Higher levels enable more optimizations for better throughput. | `CompilationConfig` |

### Other Performance Parameters

| Parameter | Description | Config Class |
|-----------|-------------|-------------|
| `seed` | Random seed for generation. Affects reproducibility but not throughput directly. | `Various` |
| `disable_log_stats` | Whether to disable logging statistics. May marginally improve throughput. | `Various` |
| `load_format` | Format to load model weights (auto, pt, safetensors, etc.). Affects loading time but not runtime throughput. | `Various` |

### Environment Variables

| Parameter | Description | Config Class |
|-----------|-------------|-------------|
| `VLLM_ATTENTION_BACKEND` | Specify attention backend (e.g., 'FLASHINFER' or 'FLASH_ATTN'). Note: values are case-sensitive. Can significantly affect performance. | `Environment` |
| `VLLM_FUSED_MOE_CHUNK_SIZE` | Chunk size for fused MoE operations (default: 16384). Affects MoE model throughput. | `Environment` |
| `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` | Enable overlapping communication in Ray (default: False). Can improve multi-node throughput. | `Environment` |
| `VLLM_WORKER_MULTIPROC_METHOD` | Multiprocessing method (fork/spawn). Fork is generally faster but spawn is safer. | `Environment` |
| `VLLM_USE_FLASHINFER_SAMPLER` | Use FlashInfer sampler for better sampling performance. | `Environment` |
| `VLLM_LOG_BATCHSIZE_INTERVAL` | Interval for logging batch size statistics. Set to -1 to disable for better performance. | `Environment` |
| `VLLM_DISABLE_FLASHINFER_PREFILL` | Disable FlashInfer for prefill (default: False). May affect prefill throughput. | `Environment` |
| `VLLM_FLASHINFER_MOE_BACKEND` | FlashInfer MoE backend mode (throughput/latency/masked_gemm). 'throughput' optimizes for throughput. | `Environment` |
| `VLLM_ENABLE_CUDAGRAPH_GC` | Enable CUDA graph garbage collection. May help with memory management. | `Environment` |
| `VLLM_DISABLE_COMPILE_CACHE` | Disable compilation cache (default: False). Disabling may slow down startup. | `Environment` |
| `VLLM_USE_TRITON_AWQ` | Use Triton-based AWQ kernels. May improve quantized model performance. | `Environment` |
| `MAX_JOBS` | Maximum parallel compilation jobs. Can speed up compilation at startup. | `Environment` |

## Usage Examples

### High Throughput Configuration

For maximum throughput in production serving:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,           # Use 2 GPUs with tensor parallelism
    gpu_memory_utilization=0.95,      # Use 95% of GPU memory
    max_num_batched_tokens=8192,      # Large batch size for throughput
    max_num_seqs=256,                 # Process many sequences in parallel
    enable_prefix_caching=True,       # Enable prefix caching
    enable_chunked_prefill=True,      # Enable chunked prefill
    block_size=16,                    # Optimal block size
)
```

### Balanced Configuration

For a balance between throughput and latency:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,           # Single GPU
    gpu_memory_utilization=0.9,       # 90% GPU memory
    max_num_batched_tokens=4096,      # Moderate batch size
    max_num_seqs=128,                 # Moderate parallelism
    enable_prefix_caching=True,       # Enable prefix caching
    enable_chunked_prefill=True,      # Enable chunked prefill
)
```

### Memory-Constrained Configuration

For systems with limited GPU memory:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,       # Conservative memory usage
    max_num_batched_tokens=2048,      # Smaller batch size
    max_num_seqs=64,                  # Fewer sequences in parallel
    enable_prefix_caching=False,      # Disable prefix caching to save memory
    cpu_offload_gb=4,                 # Offload some KV cache to CPU
)
```

### Quantized Model Configuration

For models using quantization to increase throughput:

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="awq",               # Use AWQ quantization
    dtype="float16",                  # Use float16
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,      # Can use more memory due to quantization
    max_num_batched_tokens=8192,      # Larger batch size possible
    max_num_seqs=256,
)
```

## Command Line Usage

When using `vllm serve` for OpenAI-compatible API serving:

```bash
# High throughput configuration
vllm serve meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --enable-chunked-prefill

# With quantization
vllm serve meta-llama/Llama-2-7b-hf \
    --quantization awq \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 8192
```

## Environment Variables

Environment variables can be set to further tune performance:

```bash
# Use FlashInfer backend for better attention performance
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Optimize for throughput in MoE models
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_FUSED_MOE_CHUNK_SIZE=32768

# Disable logging for production
export VLLM_LOG_BATCHSIZE_INTERVAL=-1

# Use FlashInfer sampler
export VLLM_USE_FLASHINFER_SAMPLER=1

# Enable overlapping communication for multi-node setups
export VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=1

# Then run vllm
vllm serve meta-llama/Llama-2-7b-hf --tensor-parallel-size 2
```

## Performance Tuning Tips

### 1. Maximize GPU Memory Utilization
- Set `gpu_memory_utilization` to 0.9-0.95 for production
- Lower values (0.7-0.8) for development to avoid OOM errors

### 2. Optimize Batch Size
- `max_num_batched_tokens`: Start with 2048, increase to 8192+ for throughput
- `max_num_seqs`: Start with 128, adjust based on request pattern

### 3. Enable CUDA Graphs
- CUDA graphs are automatically enabled by default
- Significantly improve throughput for decode phase
- Use `cudagraph_capture_sizes` to optimize for specific batch sizes

### 4. Use Prefix Caching
- Enable `enable_prefix_caching` for requests with common prefixes
- Can provide 2-10x speedup for chat applications

### 5. Leverage Parallelism
- **Tensor Parallelism**: Use for large models that don't fit on single GPU
- **Data Parallelism**: Use for higher overall throughput with multiple replicas
- **Pipeline Parallelism**: Use for very large models with many layers

### 6. Consider Quantization
- AWQ, GPTQ: 4-bit quantization, ~2x memory reduction, minimal accuracy loss
- FP8: 8-bit quantization, good balance of speed and accuracy
- Enables larger batch sizes and higher throughput

### 7. Tune for Workload Pattern
- **Batch Workloads**: Maximize `max_num_batched_tokens` and `max_num_seqs`
- **Interactive/Chat**: Use chunked prefill, prefix caching
- **Long Context**: Consider sliding window attention, CPU offloading

### 8. Monitor and Profile
- Use `--disable-log-stats` in production to reduce overhead
- Monitor GPU utilization, memory usage, and throughput
- Adjust parameters based on observed bottlenecks

## Important Notes

1. **Memory vs Throughput Tradeoff**: Higher throughput often requires more memory. Find the right balance for your hardware.

2. **Latency vs Throughput**: Optimizing for maximum throughput may increase latency per request. Consider your use case requirements.

3. **Model-Specific Considerations**: Some parameters work better with certain model architectures. Test with your specific model.

4. **Hardware Dependencies**: Optimal settings vary by GPU type (A100, H100, etc.) and available VRAM.

## References

- [vLLM Official Documentation](https://docs.vllm.ai/)
- [Engine Arguments Reference](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- [Performance Benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)

## Generated Information

This document was automatically generated by analyzing vLLM's configuration classes.
For the most up-to-date information, please refer to the official vLLM documentation.
