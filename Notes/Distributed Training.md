# Key Reasons for Multi-GPU Training
- Typically, multi-GPU training is required when the model is too big to fit on a single GPU.
- However, even for models that fit on a single GPU, multi-GPU will speed up training as it trains data in parallel.
- **Memory limitations**: A single GPU may not fit large models or datasets.
- **Speed benefits**: Multi-GPU strategies can accelerate training, even for smaller models.

# Approaches to Multi-GPU Training

1. Model Replication: Distributed Data-Parallel (DDP) - PyTorch
- **How it works**:
  - Copies the model to each GPU.
  - Splits and processes data batches in parallel across GPUs.
  - Synchronizes results to update identical models on all GPUs.
- **Advantages**:
  - Enables faster training by leveraging parallel computations.
- **Limitations**:
  - All model weights, gradients, and optimizer states must fit on a single GPU.

![image](https://github.com/user-attachments/assets/97381788-d17f-473e-b757-a069d595c113)

2. Model Sharding: Fully Sharded Data Parallel (FSDP) - PyTorch
- **Motivation**: Inspired by Microsoft's 2019 ZeRO (Zero Redundancy Optimizer) paper.
- **Goal**: Eliminate memory redundancy by distributing (sharding) the model parameters, gradients, and optimizer states across GPUs instead of replicating them.

## ZeRO Optimization Stages
- **Stage 1**: Shards only optimizer states across GPUs.
  - Reduces memory footprint by up to 4x.
- **Stage 2**: Shards gradients in addition to optimizer states.
  - Combined with Stage 1, reduces memory by up to 8x.
- **Stage 3**: Shards all components, including model parameters.
  - Memory reduction scales linearly with the number of GPUs (e.g., 64 GPUs = 64x savings).

## FSDP Mechanics
- In contrast to DDP, where each GPU has all of the model states required for processing each batch of data locally, FSDP requires data collection from all GPUs before the forward and backward pass.
- **Sharding process**:
  - Data, model parameters, gradients, and optimizer states are distributed across GPUs.
  - On-demand data retrieval for forward/backward passes.
  - After computation, data is either released or retained for future operations.
- In the final step after the backward pass, FSDP synchronizes the gradients across the GPUs in the same way as DDP.
- **Performance vs. memory trade-off**:
  - Helps reduce overall GPU memory utilization.
  - Supports offloading to CPU if required.
  - Higher sharding factors save more memory but increase GPU communication overhead.
  - Configure the level of sharing via the *sharding factor*.
- **Configurable sharding**:
  - **Sharding factor = 1**: No sharding, similar to DDP.
  - **Maximum sharding factor**: Full sharding for maximum memory savings.
  - **Intermediate values**: Hybrid sharding for balanced performance and memory use.

![image](https://github.com/user-attachments/assets/a5cd2caf-23d4-4c0e-946d-0b8cc59f65b0)

---

# Performance Comparison: DDP vs. FSDP
- **FSDP advantages**:
  - Handles models larger than 2.28 billion parameters, unlike DDP.
  - Achieves higher performance (teraflops) by supporting lower precision (e.g., FP16).
- **Scaling impacts**:
  - Performance decreases slightly as the number of GPUs increases (due to communication overhead).

---

# Key Insights for LLM Training
- **FSDP vs. DDP**:
  - Comparable performance for smaller models (e.g., 611M and 2.28B parameters).
  - Superior for larger models (e.g., 11.3B+ parameters).
- **Precision optimizations**:
  - Lowering model precision (e.g., FP16) boosts FSDP performance.
- **Communication trade-offs**:
  - Larger GPU clusters increase communication volume, slowing performance for massive models.

---

# Summary of Multi-GPU Scaling
- **DDP**: Ideal for smaller models where memory fits on a single GPU.
- **FSDP**: Necessary for larger models, enabling seamless scaling across GPUs.
- **Use Cases**:
  - Train large models beyond a single GPU’s memory capacity.
  - Efficiently scale small models for faster training.
