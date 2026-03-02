# Models
### SigLIP-2 Base
##### Shape
- Input: (B, C, H, W) where B is batch size, C is the number of channels, H is the height, and W is the width.
- (B, 3, 512, 512)
- Output: (B, L, D) where B is batch size, L is the number of patches, and D is the hidden size.
- (B, 1024, 768)

### Qwen3-0.6B
##### Shape
- Input: (B, L) where B is batch size, L is the sequence length.
- (B, 5)
- Output: (B, L, V) where B is batch size, L is the sequence length, and V is the vocabulary size.
- (B, 5, 151936)

### VLM
##### Extra VLM tokens
```json
{"image_token": "<|image|>", "global_image_token": "<|global_image|>",
"r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>", "r1c5": "<row_1_col_5>", "r1c6": "<row_1_col_6>", "r1c7": "<row_1_col_7>", "r1c8": "<row_1_col_8>",
"r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>", "r2c5": "<row_2_col_5>", "r2c6": "<row_2_col_6>", "r2c7": "<row_2_col_7>", "r2c8": "<row_2_col_8>",
"r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>", "r3c5": "<row_3_col_5>", "r3c6": "<row_3_col_6>", "r3c7": "<row_3_col_7>", "r3c8": "<row_3_col_8>",
"r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>", "r4c5": "<row_4_col_5>", "r4c6": "<row_4_col_6>", "r4c7": "<row_4_col_7>", "r4c8": "<row_4_col_8>",
"r5c1": "<row_5_col_1>", "r5c2": "<row_5_col_2>", "r5c3": "<row_5_col_3>", "r5c4": "<row_5_col_4>", "r5c5": "<row_5_col_5>", "r5c6": "<row_5_col_6>", "r5c7": "<row_5_col_7>", "r5c8": "<row_5_col_8>",
"r6c1": "<row_6_col_1>", "r6c2": "<row_6_col_2>", "r6c3": "<row_6_col_3>", "r6c4": "<row_6_col_4>", "r6c5": "<row_6_col_5>", "r6c6": "<row_6_col_6>", "r6c7": "<row_6_col_7>", "r6c8": "<row_6_col_8>",
"r7c1": "<row_7_col_1>", "r7c2": "<row_7_col_2>", "r7c3": "<row_7_col_3>", "r7c4": "<row_7_col_4>", "r7c5": "<row_7_col_5>", "r7c6": "<row_7_col_6>", "r7c7": "<row_7_col_7>", "r7c8": "<row_7_col_8>",
"r8c1": "<row_8_col_1>", "r8c2": "<row_8_col_2>", "r8c3": "<row_8_col_3>", "r8c4": "<row_8_col_4>", "r8c5": "<row_8_col_5>", "r8c6": "<row_8_col_6>", "r8c7": "<row_8_col_7>", "r8c8": "<row_8_col_8>"}
```
##### Modality Projector
- Using **pixel_shuffle** algorithm to losslessly compress image tokens into fewer tokens with more features
##### Interleaved Multimodal Processing
- Use placeholder for image tokens
- The modal swaps out the placeholders for image tokens

# Data pipeline
### Dataset
##### Data Source
- DatasetDict is a dictionary returned by Hugging face load_dataset that contains multiple datasets, indexed by 'training', 'validation', 'test', ...
##### Logic Layer
- TrainingDataset(Dataset) is a pytorch arrow table dataset that handles random access via `dataset[index]`
    - Handles image preprocessing, text tokenization, and data augmentation.
```
Dataset
  ├── data (Arrow Table)
  │     ├── column "texts"  -> array/list of length N
  │     ├── column "images" -> array/list of length N
  │     ├── column "labels" -> array/list of length N
  │     └── ...
  ├── features (schema / types for each column)
  ├── format / transforms (how to return items: python/numpy/torch, etc.)
  └── indices / fingerprint (optional indexing + caching metadata)
```
##### Optimization Layer
- ConstantLengthDatasets(IterableDataset) packs samples of similar lengths together to minimize padding.
- IterableDataset is a pytorch dataset that handles streaming data. It is sequential and it implements iter.
##### Delivery Layer
- DataLoader makes data available
### Collator
- Collator handles padding and batching, converts the list of python dictionaries into a single pytorch tensor
### Data Workers

# Shapes
### Image
##### 0. Configuration Hyperparameters
Based on `models/config.py`:
- `vit_img_size`: 512 (The size of a single **Tile** passed to the Vision Transformer)
- `vit_patch_size`: 16 (The size of each small **Token Patch** inside the ViT)
- `mp_pixel_shuffle_factor`: 4 (Downsampling/Concentration factor for the modality projector)
- `mp_image_token_length`: 64 (Final number of tokens per **Tile**)
- `max_img_size`: 1536 (Maximum side length allowed for the dynamic resize)
- `vit_hidden_dim`: 768 (Embedding dimension of the ViT)
- `lm_hidden_dim`: 960 (Embedding dimension of the LLM)
##### 1. Raw Data Loading
- **Input**: PIL Image with arbitrary dimensions `(H_raw, W_raw, 3)`.
##### 2. Image Processing (`data/processors.py` & `data/custom_transforms.py`)
###### A. Dynamic Resize (`DynamicResize`)
The image is resized to fit within `max_img_size` while maintaining aspect ratio and ensuring dimensions are divisible by the Tile size (512).
- **Output Shape**: `(new_H, new_W, 3)` where both `new_H` and `new_W` are multiples of 512.
###### B. Tensor Conversion (`ToTensor`)
- **Output Shape**: `(3, new_H, new_W)` (Values normalized to [0, 1]).
###### C. Tiling (`GlobalAndSplitImages`)
To handle high-resolution images, the project "tiles" the resized image into square **Tiles** of size `(512, 512)`.
- **Logic**:
  - `n_h = new_H // 512`
  - `n_w = new_W // 512`
  - Total local Tiles: `N = n_h * n_w`
  - If `N > 1`: A "global Tile" is created by resizing the entire `(new_H, new_W)` image down to `(512, 512)`.
- **Output Shape**: 
  - If `N = 1`: `(1, 3, 512, 512)` (Only one Tile)
  - If `N > 1`: `(1 + N, 3, 512, 512)` (The first element is the global overview Tile).
##### 3. Model Forward Pass (`models/vision_language_model.py`)
###### A. Batch Processing (`_process_images`)
All Tiles from all images in the batch are flattened into a single sequence.
- **Output Shape**: `(B_total_tiles, 3, 512, 512)`
###### B. Vision Encoder (`ViT`)
The Vision Transformer processes each `(512, 512)` Tile and splits it into small **Token Patches**.
- **Token Patching**: 
  - Each Tile is divided into `16x16` patches.
  - `(512 / 16) * (512 / 16) = 32 * 32 = 1024` tokens per Tile.
- **Output Shape**: `(B_total_tiles, 1024, 768)`.
###### C. Modality Projector (`ModalityProjector`)
The projector applies "Pixel Shuffle" to concentrate the `32x32` grid of tokens into a smaller `8x8` grid, increasing the features per position.
- **Pixel Shuffle (factor 4)**:
  - Reshape: `(B_total_tiles, 32, 32, 768)`
  - Rearrange: `(B_total_tiles, 32/4, 32/4, 768 * 4 * 4) = (B_total_tiles, 8, 8, 12288)`
  - Flatten spatial: `(B_total_tiles, 64, 12288)`
- **Linear Projection**:
  - `nn.Linear(12288, 960)`
- **Output Shape**: `(B_total_tiles, 64, 960)`.
  - Note: `64` is the final `mp_image_token_length` for each Tile.
###### D. Token Replacement (`_replace_img_tokens_with_embd`)
The LLM text includes `64` copies of the `<|image|>` token for every Tile. These placeholders are replaced by the embeddings generated above.
- **Final Input to LLM**: `(Batch, Seq_Len, 960)`.

# Training

# Evaluation
