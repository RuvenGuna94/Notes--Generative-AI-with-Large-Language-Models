# Growth of Model Sizes
- **Larger models** are typically more capable across tasks without needing additional fine-tuning or in-context learning.
- **Drivers of model growth**:
  - Scalable transformer architecture.
  - Access to vast amounts of training data.
  - More powerful compute resources.
- Hypothesized **"Moore’s Law" for LLMs**:
  - Increasing parameters enhances performance.
  - Raises questions about scalability and feasibility due to the high costs of training enormous models.

![image](https://github.com/user-attachments/assets/b0ff414a-dd13-4fdc-9c47-cef77142aa92)

# Challenges with Large Models
- Training is difficult and expensive, which may limit further growth.
- Ongoing research focuses on addressing these challenges to balance performance with feasibility.

---

## Memory Challenges in Training LLMs
- **Out-of-Memory Issues:**
    - Common issue when training or loading large language models on Nvidia GPUs.
    - CUDA (Compute Unified Device Architecture):
        - A collection of libraries and tools for Nvidia GPUs.
        - Used by frameworks like PyTorch and TensorFlow to enhance performance on matrix operations and deep learning tasks.

### Memory Usage in Training:
- A single parameter is typically represented as a **32-bit float (FP32)**, which uses **4 bytes of memory**.
- For 1 billion parameters:
  - Model weights require **4 GB** of GPU RAM (FP32).
  - Training overhead (optimizers, gradients, activations, etc.) increases the memory requirement by **6x**, leading to **24 GB GPU RAM** needed.
- Consumer GPUs and even some data center GPUs struggle to handle models with this memory demand on a single processor.

![image](https://github.com/user-attachments/assets/d39934a6-d3cd-48f9-8d9b-d21bd2850ea4)
![image](https://github.com/user-attachments/assets/c2eee4fc-b09b-4485-adca-9356934c8b1a)

---

# Quantization: Reducing Memory Requirements
- **Definition:**
    - Reduces memory usage by lowering the precision of model weights from **FP32** to **FP16** or **INT8**.
    - Projects 32-bit floating-point numbers into lower precision spaces using scaling factors.

- Data Types for Quantization:
    - **FP32 (32-bit full precision):**
        - 1 bit for sign, 8 bits for exponent, 23 bits for fraction.
        - High precision but large memory footprint (**4 bytes per parameter**).
    - **FP16 (16-bit half precision):**
        - 1 bit for sign, 5 bits for exponent, 10 bits for fraction.
        - Memory usage reduced by **50%** (**2 bytes per parameter**).
        - Smaller range of representable values: -65,504 to +65,504.
        - Slight loss of precision, typically acceptable in most cases.
    - **BFLOAT16 (16-bit precision, short for Brain floating point format developed at Google Brain):**
        - Hybrid of FP32 and FP16.
        - Many LLMs including **FLAN-T5** have been pretrained with BF16.
        - Retains full dynamic range of FP32 with 8 bits for the exponent and 7 bits for the fraction.
        - Improves memory efficiency and training stability.
        - Supported by newer GPUs (e.g., NVIDIA A100).
    - **INT8 (8-bit integers):**
        - 1 bit for sign, remaining 7 bits for representation.
        - Dramatically reduces memory usage (**1 byte per parameter**).
        - Significant precision loss (e.g., Pi approximated as 3).

- Memory Savings Example:
    - **1 billion parameters**:
        - FP32: **4 GB**
        - FP16: **2 GB** (50% memory saving).
        - INT8: **1 GB** (additional 50% saving over FP16).

- Modern deep learning frameworks and libraries support quantization-aware training (QAT):
    - Learns the quantization scaling factors during scaling.
![image](https://github.com/user-attachments/assets/e40f2132-d686-410a-a216-c21047c93969)

---

## Scaling Challenges with Larger Models
- **Models with billions of parameters** (e.g., **50B** or **100B**):
  - Memory demands scale drastically:
    - Training a **100B model** with FP32 requires up to **500 times** more GPU memory (thousands of GBs).
- **Solution: Distributed Training**:
  - Distributes training across multiple GPUs (potentially hundreds).
  - Very expensive and resource-intensive.

---

## Fine-Tuning and Practical Use
- Fine-tuning requires storing all parameters in memory.
- Likely to fine-tune a pre-trained model instead of training one from scratch.
