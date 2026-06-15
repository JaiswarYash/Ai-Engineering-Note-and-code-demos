# Fine-tuning Sessions — Ollama, PEFT, Hugging Face & Unsloth

**Course:** Krish Naik Academy — Generative AI Engineering
**Sessions:** Fine-tuning Parts 1, 2, and 3

---

## What These Sessions Covered

- **Part 1** — Ollama (local model setup) + Introduction to fine-tuning: what it is, the 3-stage LLM training pipeline, which model to pick and why
- **Part 2** — PyTorch foundations, Hugging Face deep dive, Unsloth introduction, Google Colab GPU setup
- **Part 3** — Full fine-tuning pipeline: dataset, tokenization, LoRa config, SFT training, DPO preference tuning using Unsloth on Colab

---

## Part 1 — Ollama

### What is Ollama?

An AI platform (AI operating system) that lets you run LLMs locally on your machine — no API key needed. Also provides cloud-hosted model access for free-tier models.

### Local vs Cloud

| Mode | How it works | Pros | Cons |
|---|---|---|---|
| Local (`ollama pull`) | Downloads model to your machine | No API cost, full data privacy, works offline | Needs 16GB RAM minimum |
| Ollama Cloud | Model stays on Ollama's server | No download needed | Internet required, limited free models |

### Key Terminal Commands

```bash
ollama -v                  # check version
ollama list                # see downloaded models
ollama pull model-name     # download a model
ollama run model-name      # run a model
ollama rm model-name       # delete a model
ollama ps                  # see running models
ollama serve               # serve model on local URL
```

### Access via LangChain

Import `ChatOllama` from `langchain_ollama`. Pass the local model name (from `ollama list`) as the model parameter. No API key needed. Same 4-step invoke pattern as all other providers.

> The model name must already exist locally. Pull it first with `ollama pull` before using it in LangChain.

---

## Part 2 — The 3-Stage LLM Training Pipeline

Every LLM — Llama, Mistral, Gemini, GPT — goes through these 3 stages. You can pick the model from any stage and fine-tune it on your own data.

```
Stage 1: Pre-training (Raw model)
  └── Trained on internet-scale data
  └── Task: next token prediction
  └── Architecture: decoder-only transformer
  └── Output: knows language, not instructions

Stage 2: Supervised Fine-Tuning / SFT (Instruct model)
  └── Trained on input → output pairs
  └── Teaches model to follow instructions
  └── Output: conversational, instruction-following

Stage 3: Preference Alignment (Preference-tuned model)
  └── Trained on human preference data (chosen vs rejected)
  └── Aligns output with how humans want responses
  └── Not always mandatory
```

### Which Model to Pick for Fine-tuning?

- **Use pre-trained (raw) model** → if your task is text generation, report generation, or data synthesis. Not conversational.
- **Use instruct model** → if your task is a chatbot, QA system, or conversational AI. Already knows how to follow instructions — just add your domain data on top.

---

## Part 3 — What is Fine-tuning?

Taking an already-trained model and training it further on domain-specific data so it performs better for your use case.

> 99% of real-world AI work is RAG or agentic systems — NOT fine-tuning. But fine-tuning knowledge is essential for interviews and for the rare cases where it is needed.

### Full Fine-tuning vs PEFT

| Type | What it does | Cost | Feasibility |
|---|---|---|---|
| Full fine-tuning | Retrains ALL weights | Extremely expensive | Only Meta, Google do this |
| PEFT | Trains a small subset of parameters | Feasible on single GPU | What we implement |

---

## Part 4 — PEFT Techniques

### Hierarchy

```
PEFT (Parameter Efficient Fine-Tuning)
├── Old-school (for CNN, BERT, T5)
│   ├── Freeze all layers, train only last output layer
│   └── Freeze starting layers, retrain last layers
│   └── ❌ Fails on latest LLMs (too large, too expensive)
│
└── Modern (for LLMs)
    ├── LoRa (Low-Rank Adaptation) ← main one
    ├── QLoRa (Quantized LoRa) ← LoRa + quantization
    ├── DoRA (Weight Decomposition LoRa) ← more expressive
    ├── BitFit (train only biases)
    └── IA³ (scales activations directly)
```

---

### 1. LoRa — Low-Rank Adaptation

**Architecture:**
```
W (frozen) + A (r×d) × B (d×r) = W + AB (merged output)
```

**How it works:**

Instead of retraining the huge weight matrix W, LoRa adds two tiny trainable matrices A and B alongside it:

- **W** = original frozen weight matrix (e.g. 4096×4096 = 16M parameters)
- **A** = small matrix, shape r×d
- **B** = small matrix, shape d×r
- **r** = rank (a small number: 8, 16, 64)

A×B gives back a matrix the same shape as W, so they can be added: `W + AB`. Only A and B are trained (~0.1–1% of parameters). After training, AB is merged into W — no extra overhead at inference.

**Key parameters:**
- `r` (rank) — higher = more parameters trained = better performance but more memory
- `alpha` — scaling factor, usually set equal to r
- `target_modules` — which weight matrices to apply LoRa to (Q, K, V in attention)

---

### 2. QLoRa — Quantized LoRa

**Architecture:**
```
W (4-bit quantized, frozen) + A (bf16) × B (bf16) = dequant + merge
```

**How it works:**

LoRa + quantization. Converts model weights from float32 → 4-bit integer before applying LoRa. 8× less memory for the base model. Adapters A and B stay in bf16 (higher precision). Slight accuracy trade-off but negligible in practice with Unsloth.

**When to use:** Low VRAM (less than 8GB). Standard choice for Colab free tier T4 GPU.

---

### 3. DoRA — Weight Decomposition LoRa

**Architecture:**
```
W → split → m (magnitude, trainable) × (V + AB) (direction + LoRa adapters)
```

**How it works:**

Any weight matrix W can be decomposed into:
- **m** (magnitude) — how big the weights are (a small 1D vector per column). Trained directly.
- **V** (direction) — which direction the weights point (normalized unit vectors). Frozen but updated via LoRa adapters AB.

Final formula: `output = m × (V + AB) / norm`

**Why better than LoRa:** Plain LoRa updates magnitude and direction together — they get tangled. DoRA separates them so each can be fine-tuned more precisely. More expressive with the same parameter count.

---

### 4. BitFit — Bias-only Fine-tuning

**Architecture:**
```
Full model (frozen)
├── W₁, W₂, W₃... (frozen)
├── Q, K, V matrices (frozen)
└── b₁, b₂, b₃... bias terms ← only these train
```

**How it works:**

Trains only the bias terms in each layer. Everything else stays completely frozen. Extremely cheap (~0.01% of parameters). Only works for simple domain shifts — not suitable for complex task adaptation.

---

### 5. IA³ — Infused Adapter by Inhibiting and Amplifying Inner Activations

**Architecture:**
```
Attention block
├── Q (unchanged)
├── K × lₖ  ← scaled by learned vector
├── V × lᵥ  ← scaled by learned vector
└── FFN output × lff ← scaled by learned vector

lₖ, lᵥ, lff = tiny learned 1D vectors
```

**How it works:**

Instead of adding new matrices, IA³ puts a tiny learned scale vector on specific activations:
- **lₖ** → scales Key activations
- **lᵥ** → scales Value activations
- **lff** → scales FFN output

Each number in the vector is a multiplier:
- Value > 1 → amplifies the activation
- Value < 1 → inhibits the activation
- Value = 1 → no change

No weight updates — only scales what is already there. Q is not scaled because scaling K already affects the Q×K attention score indirectly.

---

### Comparison Table

| Technique | What trains | Params trained | Best for |
|---|---|---|---|
| LoRa | Low-rank adapters A, B | ~0.1–1% | Most fine-tuning tasks |
| QLoRa | LoRa on quantized base | ~0.1–1% | Low VRAM (≤8GB GPU) |
| DoRA | Magnitude + direction | ~0.1–1% | Higher expressivity needed |
| BitFit | Bias terms only | ~0.01% | Simple domain shifts |
| IA³ | Scaling vectors lₖ lᵥ lff | ~0.01% | Minimal compute, few-shot |

---

### When to Use What

**Decision tree:**

```
Low VRAM (< 8GB)?
└── Yes → QLoRa

High expressivity needed?
└── Yes → DoRA

Standard fine-tuning on decent GPU?
└── Yes → LoRa (default choice)

Simple domain shift, same language?
└── Yes → BitFit

Few-shot or minimal compute?
└── Yes → IA³
```

**Rule of thumb:** Start with QLoRa. If results are not good enough, try LoRa with higher rank. If still not good enough, try DoRA. Don't jump to complex techniques before trying the simple ones first.

---

## Part 5 — Preference Alignment Techniques (Stage 3)

### What is Preference Data?

Each sample has 3 parts: prompt + chosen response + rejected response. The model learns which type of response humans prefer.

| Technique | Used by | How it works |
|---|---|---|
| RLHF + PPO | OpenAI (ChatGPT) | Reinforcement learning from human feedback. Needs a reward model. Complex. |
| DPO | Most practitioners | Direct Preference Optimization. No RL needed. Trains directly on chosen vs rejected. |
| GRPO | DeepSeek | Updated RLHF. Behind DeepSeek R1's reasoning capability. |
| ORPO | Research | Updated DPO. Latest technique. |

> Course covers RLHF and DPO in detail. GRPO and ORPO are self-study after those are understood.

---

## Part 6 — PyTorch

Deep learning framework used to code LLMs from scratch. Written in Python, built by Meta. Works on GPU via NVIDIA's CUDA kernel — a low-level GPU programming layer that processes tensor operations in parallel.

> For this course, you don't need to learn PyTorch deeply. Hugging Face handles the low-level PyTorch for you in fine-tuning. You need it only for understanding transformer code from scratch.

---

## Part 7 — Hugging Face Ecosystem

### 3 Main Components

- **Hub** — repository for models. Like GitHub but for ML models. Upload your fine-tuned model here.
- **Dataset Hub** — repository for datasets. CSV, JSON, Parquet formats supported.
- **Spaces** — host your Streamlit or Gradio app. Good for demos and portfolio.

### Core Libraries

| Library | Purpose |
|---|---|
| `transformers` | Load LLMs |
| `datasets` | Load and process datasets |
| `tokenizers` | Tokenize input text |
| `peft` | LoRa, DoRA, IA³ adapters |
| `accelerate` | Multi-GPU training |
| `trl` | RLHF, DPO, PPO, SFT training |
| `bitsandbytes` | Quantization (QLoRa) |
| `safetensors` | Save model weights |
| `evaluate` | Accuracy, BLEU, ROUGE metrics |
| `unsloth` | Faster fine-tuning (not built by HF) |

> All fine-tuning frameworks (Unsloth, Llama Factory, Axolotl) are built on top of Hugging Face. HF is the foundation layer.

### Fine-tuning Pipeline Steps

1. Load dataset from HF Hub or custom CSV/JSON
2. Preprocess and clean data
3. Tokenize — convert text to tokens
4. Load model from HF Hub via `transformers`
5. Apply PEFT config (LoRa adapter via `peft` library)
6. Apply quantization if needed (`bitsandbytes` → QLoRa)
7. Train — SFT or preference tuning via `trl` library
8. Evaluate — `evaluate` or `lighteval`
9. Save checkpoint (safetensors or GGUF format)
10. Push to Hugging Face Hub
11. Inference and deployment
12. Monitor and iterate

---

## Part 8 — Unsloth

### Why Unsloth is Faster

| | Hugging Face (raw) | Unsloth |
|---|---|---|
| Kernel | Standard CUDA | Custom Triton kernel |
| VRAM usage | High | 2–3× less |
| Training speed | Baseline | 2–3× faster |
| Example | 10 hours, 40GB RAM | 5 hours, 12–16GB RAM |
| Context window | Limited | Up to 78K tokens |

### Technical Reasons

- **Triton kernel** — custom GPU kernel built on top of CUDA. Python-like GPU language with higher optimization than standard CUDA
- **Flash Attention** — open-source optimized attention algorithm (by Dao AI Lab)
- **Fused Attention** — combines operations that were previously separate
- **Smart checkpointing** — saves memory by only keeping needed states during training
- **Automatic sequence packing** — packs sequences efficiently to reduce wasted compute
- **No approximation tricks** — accuracy is not sacrificed. Full precision output maintained

> Real numbers: Llama 3-8B fine-tuning needs 24GB VRAM with standard HF. Unsloth does the same in 7–8GB VRAM at ~6× faster training.

---

## Part 9 — Google Colab GPU Setup

### Why Colab?

LLM fine-tuning needs GPU. Colab provides free T4 GPU (15GB VRAM, 12.7GB RAM). Enough for learning and small models with Unsloth.

### Setup Steps

1. Go to [colab.research.google.com](https://colab.research.google.com) → create new notebook
2. Runtime → Change runtime type → GPU → T4 (free)
3. Connect → verify resources (RAM, VRAM, disk shown top right)
4. Click the key icon → add secrets: `HF_TOKEN`, `GOOGLE_API_KEY` etc.
5. Access in code: `from google.colab import userdata; userdata.get('HF_TOKEN')`

### Alternative GPU Platforms

- Kaggle Notebooks (free GPU, 30hrs/week)
- RunPod (paid, production-grade)
- Vast.ai (paid, cheaper than RunPod)
- PaperSpace Gradient
- Lightning.ai
- Lambda.ai
- AWS / GCP / Azure GPU instances

---

## Part 10 — Practical Fine-tuning Pipeline

### SFT (Supervised Fine-Tuning) with Unsloth

1. Load quantized model via `FastLanguageModel.from_pretrained()` with `load_in_4bit=True`
2. Apply LoRa config via `get_peft_model()` — set r, target_modules, alpha
3. Load dataset from HF Hub using `load_dataset()`
4. Apply chat template / prompt formatting to dataset
5. Train using `SFTTrainer` from `trl` library
6. Set training args: epochs, batch size, learning rate, max steps
7. Save as LoRa adapter (~4MB) or merge and save full model
8. Push to Hugging Face Hub with write token

### DPO (Preference Tuning) with Unsloth

1. Dataset needs 3 columns: `prompt`, `chosen`, `rejected`
2. Load already SFT-trained model as base
3. Train using `DPOTrainer` from `trl` library
4. Model learns to prefer chosen over rejected via a contrastive loss function
5. No reward model needed (unlike RLHF) — simpler and more practical

### Key Parameters to Understand

| Parameter | What it controls |
|---|---|
| `r` (rank) | LoRa adapter size. Higher = more params = better performance but more memory |
| `alpha` | Scaling factor for LoRa. Usually set equal to r |
| `target_modules` | Which weight matrices get LoRa (Q, K, V in attention) |
| `max_seq_length` | Max token length per training sample |
| `load_in_4bit` | Enables QLoRa (4-bit quantization) |
| `per_device_train_batch_size` | Samples per GPU per step |
| `gradient_accumulation_steps` | Simulate larger batch by accumulating gradients |

---

## Assignment

- Set up Google Colab with T4 GPU — verify with a simple print statement
- Create Hugging Face read + write tokens — add to Colab secrets
- Run the SFT fine-tuning notebook from GitHub on Colab
- Run the DPO preference tuning notebook after SFT works
- Push your fine-tuned adapter to your own Hugging Face Hub repository
- Self-study: read about GRPO and ORPO once RLHF and DPO are understood

---

## Resources

- Ollama download: [ollama.com](https://ollama.com)
- Hugging Face Hub: [huggingface.co/models](https://huggingface.co/models)
- Unsloth GitHub: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- TRL docs: [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- PEFT docs: [huggingface.co/docs/peft](https://huggingface.co/docs/peft)
- Google Colab: [colab.research.google.com](https://colab.research.google.com)
- Kaggle (free GPU): [kaggle.com](https://kaggle.com)