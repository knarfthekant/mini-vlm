##### Project description
- This project implements a mini VLM model from scratch, using PyTorch and Hugging Face Transformers. 
- The vision and language backbone are implemented using Hugging Face Transformers and Encoders. The projection layer is a MLP implemented using pure PyTorch.

##### Important considerations
- The project will be developed in multiple phases. For the first phase, we will focus on the data pipeline, the projection layer, the training loop, and the evaluation.
- For the second phase, we will implement the full VLM model using pytorch, by swapping the backbone with a pure pytorch implementation.
- For the third phase, we plan to implement specific kernels for the attention and MLP layers to accelerate the training and inference.

##### Architecture
- Vision backbone: siglip2-base-patch16-512
- Language backbone: Qwen3-0.6B

##### Enhancements from original implementation
1. The original implementation uses a cursor arithmetic to pre-compute the mask index. However, this can lead to drifting during byte pair encoding (BPE) tokenization if multiple messages are merged.
    - Implemented token search approach to ensure no drifting happens.

