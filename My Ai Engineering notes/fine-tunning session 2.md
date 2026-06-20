# Fine-tuning Practical Sessions — Full Pipeline

**Course:** Krish Naik Academy — Generative AI Engineering
**Sessions:** 7 June, 13 June, 14 June
**Instructor:** Sunny Savita

---

## What These Sessions Covered

These 3 sessions are the **hands-on implementation** of the fine-tuning theory. They complete the full 3-stage LLM training pipeline end-to-end on Google Colab.

| Session | Topic |
|---|---|
| 7 June | Non-instruction + Instruction fine-tuning practical (HuggingFace) |
| 13 June | Preference tuning (DPO) practical + LoRa merge strategies |
| 14 June | LoRa theory deep dive + RLHF theory + Unsloth practical |

---

## The 3-Stage Pipeline — Big Picture

Before diving into details, understand this: **every LLM goes through 3 stages of training.**

```
Stage 1: Non-Instruction Fine-tuning
  → Teach the model domain-specific knowledge (pharma, legal, finance...)
  → Data: raw text (PDF, TXT, HTML)
  → Model learns: what the domain looks like

Stage 2: Instruction Fine-tuning (SFT)
  → Teach the model to have conversations
  → Data: instruction + input + output pairs
  → Model learns: how to respond to questions

Stage 3: Preference Alignment (DPO/RLHF)
  → Align model responses with what humans actually prefer
  → Data: prompt + chosen response + rejected response
  → Model learns: which answer is better
```

Each stage builds on the previous. You take the output of Stage 1 as the base for Stage 2, and so on.

---

## Part 1 — How to Get Training Data in Real World

### For Non-Instruction Fine-tuning (Stage 1)

Raw data is available everywhere inside companies:

- **Storage locations:** OneDrive, Google Drive, SharePoint, Confluence, S3, Azure Blob, internal portals
- **File formats:** PDF, Word, HTML, TXT, Markdown
- **How to access:** Python SDK → pull data → save to intermediate JSONL/CSV → run pipeline

### For Instruction Fine-tuning (Stage 2)

This is harder — you need question-answer pairs, not raw text. Three ways to get it:

**Method 1 — Manual creation**
Domain experts (SMEs) manually write instruction → input → output rows. Highest quality but expensive. OpenAI hired thousands of people just for this. Sunny himself labeled 1 lakh rows manually in 2021 for an NLP project.

**Method 2 — Synthetic data generation (most common today)**
Take your raw documents → give to ChatGPT/Claude with a prompt like "generate QA pairs from this document" → get instruction data back automatically. Fast and cheap. Most companies use this approach.

**Method 3 — Existing conversations**
Support tickets, chat logs, email threads, Reddit/Quora/StackOverflow. Already in Q&A format naturally.

### For Preference Alignment (Stage 3)

Same two options: manual curation by SMEs, or generate using AI. For AI generation: give raw document + ask LLM to generate both a good answer and a bad answer for the same question.

---

## Part 2 — Data Formats

### Alpaca Format (Stage 2 — Instruction)

The industry-standard format for SFT. Created by Stanford University.

```json
{
  "instruction": "Summarize the given input",
  "input": "Sunny is an AI mentor who teaches GenAI",
  "output": "Sunny is an AI mentor"
}
```

**Rules:**
- `instruction` = what you want the model to do
- `input` = the content to act on (can be left blank)
- `output` = the expected response
- If no separate instruction, just put the question in `instruction` and leave `input` blank

### OpenAI / ChatGPT Format (alternative)

```json
{
  "system": "You are a helpful pharma assistant",
  "user": "What does metformin do?",
  "assistant": "Metformin activates AMPK..."
}
```

Same concept, different column names. Used when fine-tuning OpenAI-compatible models.

### Preference / DPO Format (Stage 3)

```json
{
  "prompt": "Explain the mechanism of metformin",
  "chosen": "Metformin primarily works by activating AMPK, a central metabolic regulator...",
  "rejected": "Metformin mainly works by increasing insulin secretion from the pancreas..."
}
```

**Why is rejected answer needed?**
The model needs to see BOTH answers to learn the difference. Without seeing the rejected answer, it cannot learn what to avoid.

---

## Part 3 — Tokenization (The Part Everyone Gets Confused About)

### The Problem

Different sentences have different lengths. Models need uniform-length inputs.

```
Sentence 1: 413 tokens
Sentence 2: 512 tokens
Sentence 3: 1000 tokens
Sentence 4: 30 tokens
→ Cannot feed these to model as-is
```

### Solution 1 — Padding + Truncation (Simple)

Add special EOS tokens to shorter sentences. Cut longer sentences at max_length.

```python
tokenizer(text, padding="max_length", truncation=True, max_length=512)
```

- Padded positions get label = -100 → PyTorch ignores them during loss calculation
- ✅ Simple, works for beginners
- ❌ Wastes GPU memory on padding, truncation loses information

### Solution 2 — Custom Block Sizing (Industry Standard)

Concatenate all tokens → split into fixed-size blocks of 512 → drop remainder.

```
All tokens: [10, 20, 30, 40, 50, 60, 70, 80, 90]
Block size = 4
Block 1: [10, 20, 30, 40]  ← one training example
Block 2: [50, 60, 70, 80]  ← one training example
Dropped:  [90]              ← acceptable loss
```

- ✅ No padding waste
- ✅ No truncation loss
- ✅ Better GPU utilization
- ✅ Industry-grade causal LM pre-training

### Why labels = input_ids

For next-token prediction, the model's target at each position is the NEXT token. Setting `labels = input_ids.copy()` tells HuggingFace: "the label for position N is the token at position N+1." HuggingFace handles the shift automatically.

---

## Part 4 — LoRa Adapter Strategies

After training, you have two options for how to use your LoRa adapter.

### Option A — Temporary Append (for inferencing)

Load base model + adapter together in RAM. No permanent file change.

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, adapter_path)
# Use for inference
# Adapter is NOT permanently merged
```

### Option B — Permanent Merge

Bake the adapter weights permanently into the base model. Creates one single model file.

```python
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model_directory")
```

**Both give identical inference results.** The difference is storage and deployment preference.

---

## Part 5 — Two Deployment Strategies

### Strategy 1 — Sequential Merge (Recommended for Production)

```
Stage 1: base_model + LoRa_1 → merge → merged_1
Stage 2: merged_1  + LoRa_2 → merge → merged_2
Stage 3: merged_2  + LoRa_3 → merge → final_model
```

Each stage's adapter gets permanently baked in before the next stage. Result: one single model with all 3 stages of knowledge. Best for production deployment.

### Strategy 2 — Separate Adapters Per Task

```
base_model + LoRa_1 (non-instruction) → for generation tasks
base_model + LoRa_2 (instruction)     → for conversation tasks
base_model + LoRa_3 (preference)      → for aligned responses
```

Train separate adapters for each task on the same base model. Load whichever adapter fits the current task at runtime. More flexible but each adapter only knows its own stage.

---

## Part 6 — LoRa Theory (Why It Works)

### The Core Idea

When you fine-tune a model, you're not changing all the weights dramatically. The change (ΔW) needed is actually low-rank — it can be approximated by two small matrices.

```
Original weight matrix W:  shape 4096 × 4096 = 16M parameters

LoRa approximation:
  A: shape 4096 × 16  = 65K parameters
  B: shape 16 × 4096  = 65K parameters
  Total: 130K parameters (< 1% of original)

W_new = W_frozen + (A × B)
```

**r = rank.** Common values: 8, 16, 64. Higher r = more trainable parameters = better quality but more VRAM.

### Where LoRa is Applied

`target_modules` in your config controls which weight matrices get LoRa adapters. Typically:

- `q_proj` — Query matrix in attention
- `k_proj` — Key matrix in attention
- `v_proj` — Value matrix in attention
- `o_proj` — Output projection
- Optionally: `gate_proj`, `up_proj`, `down_proj` in FFN layers

More target modules = better fine-tuning = more VRAM needed.

### The Real-World Implication

Companies like Sarvam AI didn't train LLMs from scratch. They took a base model (Llama, Mistral, Gemma) and trained LoRa adapters. The adapter file is ~4MB. The base model is 7GB. Ship the adapter, not the whole model. Anyone with the same base model can load and use it. **This is how most LLM fine-tuning companies work.**

---

## Part 7 — RLHF Theory

### The Full RLHF Pipeline (What OpenAI Used)

```
Step 1: Collect preference data
        → prompt + chosen response + rejected response

Step 2: Train a Reward Model (RM)
        → RM learns to score responses (higher = better)

Step 3: Use PPO to fine-tune the LLM
        → LLM generates response
        → RM scores it
        → PPO updates LLM weights to maximize reward
        → Repeat
```

**Why it is complex:** 3 components running simultaneously — LLM being trained, frozen reference LLM (to prevent model from drifting too far), reward model. Very resource and memory intensive.

**Algorithm behind RLHF:** PPO (Proximal Policy Optimization) — a Reinforcement Learning algorithm. You don't need to learn the full RL math; just understand the concept.

### Why DPO Replaced RLHF in Practice

| | RLHF | DPO |
|---|---|---|
| Reward model | Required (separate training) | Not needed |
| Complexity | High (3 components) | Low (1 component) |
| Memory | Very high | Normal |
| Result quality | Good | Equivalent to RLHF |
| Industry adoption | Declining | Dominant |

DPO uses a mathematical proof to show that optimizing directly on chosen/rejected pairs gives the same result as RLHF, without needing the reward model step.

---

## Part 8 — Unsloth vs HuggingFace

### Numbers

| Metric | HuggingFace Native | Unsloth |
|---|---|---|
| Training speed | Baseline | 2–3× faster |
| VRAM usage | Baseline | ~60% less |
| Context window (T4 GPU) | ~2K–4K tokens | Up to 78K tokens |
| Lines of code | More | Fewer |
| Accuracy | Baseline | Same (no approximations) |

### What Changes in Code

Replace these HuggingFace calls with Unsloth equivalents:

```python
# HuggingFace
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name, ...)
model = get_peft_model(model, lora_config)

# Unsloth (replacement)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(model_name, ...)
model = FastLanguageModel.get_peft_model(model, ...)

# Before training (Unsloth only)
FastLanguageModel.for_training(model)

# Before inferencing (Unsloth only)
FastLanguageModel.for_inference(model)
```

Everything else — dataset loading, tokenization, DPOTrainer, SFTTrainer, saving, pushing to Hub — is identical.

### When to Use Which

- **HuggingFace native** → learning, debugging, maximum control, Unsloth doesn't support your model yet
- **Unsloth** → production training, resource-constrained environments, faster iteration. **Default choice.**

---

## Part 9 — RAG vs Fine-tuning

| | RAG | Fine-tuning |
|---|---|---|
| Cost | Low | High (GPU cost in lakhs for 1–2 days) |
| Complexity | Medium | High |
| Data freshness | Easy to update | Needs retraining |
| Model quality | Uses best models (GPT, Claude) | Limited by base model |
| Data privacy | Data leaves your system | Can stay on-premise |
| When to use | 90% of cases | Own model needed, deep domain patterns |

**Rule:** Default to RAG. Only do fine-tuning when the business explicitly needs an owned model, has budget for GPU infrastructure, and has data that cannot go to external APIs.

---

## Part 10 — Guardrails (Handling PII and Bias)

When working with sensitive data (healthcare, banking):

**Data bias in training:**
- Sit with SMEs and business analysts before data collection
- Define rules for what constitutes acceptable vs unacceptable responses
- Review and filter training data before it goes into the pipeline

**PII leaking in model outputs:**
- Use guardrail libraries that intercept model output before showing to user
- Libraries: **NeMo Guardrails** (NVIDIA), **Guardrails AI**, OpenAI moderation API
- Guardrails can detect and mask PII in generated text programmatically

**Data access control:**
- Restricted data lives in enterprise workspaces with admin-controlled access
- Cloud options: Azure OpenAI, AWS Bedrock — both provide enterprise-grade data isolation contracts

---

## Complete Pipeline Code Flow (Reference)

```
1. Load data (PDF → parse with PyMuPDF → clean with regex)
2. Convert to HuggingFace dataset format
3. Split into train/validation
4. Load tokenizer
5. Tokenize data (block sizing or padding)
6. Load base model (4-bit quantized)
7. Apply LoRa config (get_peft_model)
8. Initialize data collator
9. Set training arguments
10. Initialize trainer (SFTTrainer or DPOTrainer)
11. Train (trainer.train())
12. Save LoRa adapter
13. Push to HuggingFace Hub (optional)
14. Load for inference (append or merge)
15. Generate responses
16. Merge model for next stage (merge_and_unload)
17. Repeat stages 6–16 for each training stage
```

---

## Assignment

- Run the full 3-stage notebook end-to-end on Colab: non-instruction → instruction → DPO
- Redo the same pipeline using Unsloth — compare training time and VRAM
- Collect 50+ instruction pairs on your own domain using synthetic data generation
- Self-study: GRPO and ORPO — both available in TRL library
- Interview prep: explain the full pipeline, LoRa math, why DPO replaced RLHF, RAG vs fine-tuning tradeoffs

**Next module: RAG (Retrieval Augmented Generation)**

---

## Resources

- Unsloth GitHub: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- TRL library (DPOTrainer, SFTTrainer): [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- NeMo Guardrails: [github.com/NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- Guardrails AI: [guardrailsai.com](https://guardrailsai.com)
- DPO paper: search "Direct Preference Optimization" on arXiv
- LoRa paper: search "LoRA Low-Rank Adaptation of Large Language Models" on arXiv
- Alpaca dataset: [huggingface.co/datasets/tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
- Google Colab GPU: [colab.research.google.com](https://colab.research.google.com)