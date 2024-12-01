# LLM Overview

![LLM examples](https://github.com/user-attachments/assets/56ed78bc-1ae5-4069-b4be-c057797d40d1)


- LLMs where size reflects number of parameters utilized by the model
- More parameters, more memory, more sophisticated tasks

# Text Generation before Transformers
- Before LLMs, text generation was done with RNNs
- However, RNNs need to scale in order to take in more words or context
- Furthermore, they were not very accurate
- 2017, Google and the university of Toronto published the paper "Attention is all you need" which introduced the transformer architecture
    - It can scale efficiently using multi-core GPUs
    - It can process input data in parallel
    - It is able to learn to "pay attention" to the meaning of the words it is processing
    - The paper proposes a neural network architecture that replaces traditional recurrent neural networks (RNNs) and convolutional neural networks (CNNs) with an entirely attention-based mechanism. 

# Transformer Architecture 

Significance: The power of the transformer architecture lies in its ability to learn the relevance and context of all of the words in a sentence. To apply attention weights to those relationships so that the model learns the relevance of each word to each other words no matter where they are in the input. 

![Transformer Detail](https://github.com/user-attachments/assets/56463dfe-0fba-417a-a8ff-7be63bd70d41)
![Transformer Overview](https://github.com/user-attachments/assets/f5dda45e-c4de-4830-ba59-97652cb34a24)


- Words must be tokenized before going into the transformer
- Token IDs can match complete words or parts of a word
- Tokenizer used must be the same for model training and text generation
- These tokens are then parsed through the embedding layer to represent the tokens as vectors
- Each word maps to a token ID and each token ID maps to a vector
- Words typically need to be tokenized before they can be processed by an embedding model. However, some embedding models, like Word2Vec and GloVe, work directly with words as tokens.
- Once the tokens are embedded into vectors, the 'self-attention' mechanism is applied. This step allows the model to weigh the importance of each token relative to others in the sequence, capturing dependencies and relationships between tokens.
- Original transformer paper had a vector size of 512
- In the transformer architecture, self-attention happens multiple times, referred to as multi-headed self-attention
- Multi-Headed Self-Attention:
    ○ Multiple sets of self-attention weights (heads) are learned in parallel.
    ○ Each head operates independently.
    ○ Commonly, models have between 12 to 100 attention heads.
- Purpose of Multiple Heads:
    ○ Each head captures different aspects of the language.
    ○ Example:
    ○ One head might focus on relationships between entities (e.g., people).
    ○ Another head might focus on the activities described in the sentence.
- After this layer, the outputs are passed to the feed forward network (FFN) which outputs a vector of logics proportional to the probability score for each token
- This vector is then passed to the softmax layer where they are normalized where each word in the model's vocabulary has a probability. One word will have the highest probability and this is likely the next predicted token. 

## Translation task data flow
![Transformer sq to sq example](https://github.com/user-attachments/assets/abc75d7b-76af-4615-bca6-03619e6c731d)

1. **Tokenization**: The input phrase is tokenized using the same tokenizer that trained the network.
2. **Encoder**:
    - Tokens are passed through the embedding layer.
    - Processed through multi-headed attention layers.
    - Outputs are fed through a feed-forward network, to the output of the encoder.
    - The encoder produces a deep representation of the input sequence's structure and meaning.
    - This is then passed to the middle of the decoder
3. **Decoder**:
    - The encoder's output influences the decoder's self-attention mechanisms.
    - A start-of-sequence token triggers the decoder to predict the next token, based on the contextual understanding provided by the encoder.
    - Similarly, the output of the decoder's self-attention layer gets passed through the decoder's FFN and finally, softmax output layer. 
    - The process continues in a loop until an end-of-sequence token is predicted.
    - The final sequence of tokens is detokenized into words to produce the output.

**Summary**: The encoder encodes input sequences into a deep representation of the structure and meaning of the input. The decoder, working from input token triggers, uses the encoder's contextual understanding to generate new tokens

**Model Variations**:
- Encoder-Only Models: Used for tasks like classification like sentiment analysis (e.g., BERT).
- Encoder-Decoder Models: Used for sequence-to-sequence tasks. Commonly used for general text generation. (e.g., BART, T5).
- Decoder-Only Models: Generalized for most tasks (e.g., GPT family, BLOOM, Jurassic, LLaMA)
