First research JiRack on java with similar BERT Google architecture with Amazon DJL Framework version 0.35.0
Note : Use JiRack PyTorch for GPT-5 class models 
Url : https://huggingface.co/CMSManhattan/JiRack_GPT5_1b


# JiRack GPT-2 Initial Weights

This file is strictly intended for saving the **initial weights (checkpoint)** of the JiRack GPT model.  
The model is **"clean"**: it contains no data and has never undergone any pre-training.
- Powered by CMS Manhattan’s cutting-edge Vision-BERT architecture.

It is engineered to be a maximally safe and robust base for **training from scratch** for specialized, smaller models, such as:

- **SPAM Detection Systems**
- **FRAUD Detection Models**
- **Background Check (BG Check) Models**

_A product of CMS Manhattan._

---

## Tokenizer Choices

- For English: **GPT-2 Hugging Face tokenizer**
- For multilingual use: **BERT tokenizer** from the Hugging Face library

---

## Model Architecture Details

### GPT-2 Architecture (Classic, Transformer-like)

```
CustomEmbedding
FrozenSignatureLayer
LearnedPositionalEmbedding
[TransformerBlock]
    ├── MultiHeadAttention
    ├── LayerNorm
    ├── LayerNorm
    ├── FFN
          ├── Linear
          ├── Activation: GELU
          └── Linear
LayerNorm
Linear
```

---

## Model Checkpoint File Explanations

### **12-head Attention Model**

**Parameters:**
- `VOCAB_SIZE = 50257`
- `MODEL_DIM = 768`
- `NUM_HEADS = 12`
- `NUM_LAYERS = 6`
- `MAX_SEQ_LEN = 8192`
- `FFN_HIDDEN_DIM = 4 * MODEL_DIM`
- `HEAD_DIM = MODEL_DIM // NUM_HEADS`

**File:**  
`JiRack_H12_L6_V50257_D768_MSL8192_FF768x4.pt`

---

### **6-head Attention Model**

**Parameters:**
- `VOCAB_SIZE = 50257`
- `MODEL_DIM = 768`
- `NUM_HEADS = 6`
- `NUM_LAYERS = 6`
- `MAX_SEQ_LEN = 8192`
- `FFN_HIDDEN_DIM = 4 * MODEL_DIM`
- `HEAD_DIM = MODEL_DIM // NUM_HEADS`

**File:**  
`JiRack_H6_L6_V50257_D768_MSL8192_FF768x4.pt`



- So About PyTorch script . You can use Pytorch script for AI classification task . 
- Do not Jit for Chatbot task . Use just state dict PyTorch for  GPT  (Chatbot) tasks


---

See other models with same patterns for read parameters 

# install tokenizer before run 
---
- mkdir -p tokenizer
- wget -O tokenizer/tokenizer.json https://huggingface.co/gpt2/resolve/main/tokenizer.json
- wget -O tokenizer/vocab.json https://huggingface.co/gpt2/resolve/main/vocab.json
- wget -O tokenizer/merges.txt https://huggingface.co/gpt2/resolve/main/merges.txt
- wget -O tokenizer/tokenizer_config.json https://huggingface.co/gpt2/resolve/main/tokenizer_config.json

---
Welcome to ask to design your corp model over 33B or 70B or more parameters
## 

CMS Manhattan  
Copyright © 2002–2026
