# vLLM Throughput Parameters Documentation Generator

This directory contains a script and documentation for understanding and optimizing vLLM inference throughput.

## Files

- `generate_throughput_params_doc.py` - Python script that automatically extracts and documents all vLLM parameters affecting inference throughput
- `vllm_throughput_parameters.md` - Generated comprehensive markdown documentation of throughput-related parameters

## Quick Start

### Generating the Documentation

To regenerate the documentation with the latest parameters:

```bash
python3 generate_throughput_params_doc.py
```

This will create/update `vllm_throughput_parameters.md` with:
- Categorized list of all throughput-affecting parameters
- Detailed descriptions and default values
- Usage examples for different scenarios
- Performance tuning tips
- Environment variable configurations

### Viewing the Documentation

Simply open `vllm_throughput_parameters.md` in any markdown viewer or text editor.

## What's Documented

The script extracts and documents parameters from the following vLLM configuration classes:

1. **Scheduler Configuration** - Controls batching and scheduling policies
2. **Cache Configuration** - Manages KV cache and memory allocation
3. **Parallel Configuration** - Settings for distributed inference
4. **Model Configuration** - Model-specific performance settings
5. **Compilation Configuration** - CUDA graph and compilation options
6. **Environment Variables** - System-level performance tuning

## Use Cases

### For Developers
- Understanding how different parameters affect throughput
- Finding the right parameters to optimize for specific workloads
- Learning about trade-offs between throughput, latency, and memory

### For Operations
- Configuring production deployments for maximum throughput
- Troubleshooting performance issues
- Capacity planning and resource allocation

### For Researchers
- Benchmarking different configurations
- Understanding vLLM's performance characteristics
- Comparing different optimization strategies

## Parameter Categories

### High Impact on Throughput
- `max_num_batched_tokens` - Directly controls batch size
- `max_num_seqs` - Limits concurrent sequences
- `tensor_parallel_size` - Enables multi-GPU parallelism
- `gpu_memory_utilization` - Affects available memory for batching
- `enable_prefix_caching` - Can provide 2-10x speedup for certain workloads

### Medium Impact
- `enable_chunked_prefill` - Improves mixed workload throughput
- `cache_dtype` - fp8 saves memory allowing larger batches
- `quantization` - Reduces memory and increases throughput
- `cudagraph_capture_sizes` - Optimizes decode phase

### Configuration-Specific
- Environment variables for backend selection
- MoE-specific parameters for Mixture-of-Experts models
- Multi-node parameters for distributed setups

## Common Optimization Patterns

### Maximum Throughput (Batch Workloads)
```bash
vllm serve model \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 512 \
    --enable-prefix-caching
```

### Balanced (Production Serving)
```bash
vllm serve model \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --enable-chunked-prefill
```

### Memory-Constrained
```bash
vllm serve model \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.80 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 64 \
    --cpu-offload-gb 4
```

## Updates and Maintenance

The script is designed to be run periodically to keep the documentation in sync with vLLM's evolving API. When vLLM is updated with new parameters or changes:

1. Run the script to regenerate documentation
2. Review the changes to ensure accuracy
3. Update examples if needed

## Contributing

To add more parameters or improve descriptions:

1. Edit the `get_throughput_parameters()` function in `generate_throughput_params_doc.py`
2. Add parameters to the appropriate category
3. Provide clear, concise descriptions
4. Run the script to regenerate documentation
5. Review the output for accuracy

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Performance Tuning Guide](https://docs.vllm.ai/en/latest/serving/performance.html)

## License

This documentation generator follows the same license as the vLLM project (Apache 2.0).
