# Pre-training
- During pre-training, the model weights get updated to minimize the loss of the training objective.
- Encoder generates an embedding or vector representation of each token.
- Requires large amounts of compute and GPUs.

# Encoder-Only Models
- Encoder-only models are also known as **Autoencoding models**.
- They are pre-trained using **masked language modeling (MLM)**:
  - Random tokens in the input sequence are masked.
  - **Training objective**: Predict masked tokens to reconstruct the original sentence (denoising objective).
- Builds bi-directional representations of the input sequence, understanding the full context of a token (both preceding and succeeding words).
- **Applications**:
  - Sentence classification (e.g., sentiment analysis).
  - Token-level tasks (e.g., named entity recognition, word classification).
- **Examples**:
  - BERT
  - RoBERTa

---

# Decoder-Only Models (Autoregressive Models)
- **Key Features**:
  - Pre-trained using **causal language modeling (CLM)**:
    - **Training objective**: Predict the next token based on the previous sequence of tokens (full language modeling).
    - Unidirectional context (only considers preceding tokens).
  - Builds a statistical representation of language by iterating token by token over input sequences.
  - Utilizes the decoder component of the transformer architecture (without an encoder).
- **Applications**:
  - Text generation.
  - Larger models show strong zero-shot inference abilities and can handle various tasks.
- **Examples**:
  - GPT
  - BLOOM

---

# Sequence-to-Sequence Models (Encoder-Decoder Models)
- **Key Features**:
  - Combines both encoder and decoder components of the transformer architecture.
  - Pre-training objectives vary across models:
    - **Example**: T5 uses **span corruption**:
      - Random input token sequences are masked and replaced with a unique Sentinel token.
      - Decoder reconstructs masked sequences autoregressively.
- **Applications**:
  - Translation.
  - Summarization.
  - Question-answering.
  - Generally used when both input and output are text bodies.
- **Examples**:
  - T5
  - BART

---

# Summary of Architectures and Training Objectives

1. **Autoencoding Models (Encoder-only)**:
   - **Pre-training**: Masked language modeling.
   - **Focus**: Bi-directional context.
   - **Tasks**: Sentence classification, token classification.
   - **Examples**: BERT, RoBERTa.

2. **Autoregressive Models (Decoder-only)**:
   - **Pre-training**: Causal language modeling.
   - **Focus**: Unidirectional context.
   - **Tasks**: Text generation, zero-shot inference.
   - **Examples**: GPT, BLOOM.

3. **Sequence-to-Sequence Models (Encoder-Decoder)**:
   - **Pre-training**: Varies (e.g., span corruption in T5).
   - **Focus**: Input-to-output text tasks.
   - **Tasks**: Translation, summarization, question-answering.
   - **Examples**: T5, BART.
![image](https://github.com/user-attachments/assets/63eda803-a8a8-403a-9f8e-bfbc410ca349)
