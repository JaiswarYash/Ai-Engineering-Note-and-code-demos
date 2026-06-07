# Session 1 — Dev Environment Setup & Introduction to Generative AI

**Course:** Krish Naik Academy — Generative AI Engineering (Modern Path)
**Date:** March 22, 2026
**Instructor:** Monal (substitute for Sunny Savita)

---

## What We Covered

Two topics in this session:

1. **Dev environment setup** — VS Code, UV, Python, VENV
2. **Introduction to Generative AI** — from first principles (AI → ML → DL → Gen AI → Agentic AI)

This is the foundation class. The next session starts with Transformers.

---

## Part 1 — Dev Environment Setup

### The Three Types of Code Editors

| Type | Example | What it does | Size |
|---|---|---|---|
| Editor | Notepad | Write, create, save files only | Tiny |
| Code Editor | VS Code | Editor + syntax highlighting + extensions + project management | ~500MB |
| IDE | IntelliJ, PyCharm | Code Editor + full debugging + language-specific tools | 2–9GB |

> VS Code is a **Code Editor**, not an IDE. Common misconception.
> We use VS Code because it is lightweight, extensible, and production-standard.

---

### Three Ways to Get Python

| Method | Tool | Pros | Cons |
|---|---|---|---|
| Direct | python.org | Simple | One version only, no env management |
| Package manager | Anaconda / Miniconda | Powerful, manages versions + envs | Heavy (2.7–5.7GB) |
| Modern tool | **UV** ✅ | Fast (written in Rust), lightweight, replaces pip + VENV + version management | — |

---

### VENV — What It Is and Why It Exists

**Problem:** Without isolation, all projects share one global Python lib folder. Install NumPy 1.2 for Project A, it breaks Project B needing NumPy 1.6.

**VENV solution:** Creates a separate lib folder per project so dependencies don't conflict.

**VENV limitation:** It copies your existing Python. So if system has Python 3.6, every VENV also has 3.6. It cannot manage different Python versions across projects.

**UV solution:** Manages Python versions themselves, with VENV built in.

---

### UV Command Reference

#### Install UV (one-time, system-wide)

```bash
# Windows PowerShell (Option A)
irm https://astral.sh/uv/install.ps1 | iex

# Windows (Option B — winget)
winget install astral-sh.uv

# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After install: close and reopen terminal. Verify with `uv`.

#### Per-project setup (run once per project)

```bash
# Step 1 — install and pin Python version
uv python install 3.12
uv python pin 3.12

# Step 2 — create project files (pyproject.toml, uv.lock, .python-version)
uv init

# Step 3 — create virtual environment
uv sync

# Step 4 — activate environment
# Windows CMD: right-click .venv/scripts/activate → copy relative path → paste in CMD
# Mac/Linux: source .venv/bin/activate

# Step 5 — run a Python file
uv run python hello_world.py
```

#### Managing libraries

```bash
uv add numpy       # install → auto-updates pyproject.toml
uv remove numpy    # uninstall → auto-updates pyproject.toml
```

---

### pyproject.toml and uv.lock

- **pyproject.toml** — central config file. Tracks project name, Python version, all dependencies. Share this with teammates so they can run `uv sync` and get the identical environment.
- **uv.lock** — locked snapshot of every installed package version. Tracks revision history. Do not delete either file.

---

### Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `uv not recognized` | Path not updated yet | Restart VS Code after installing UV |
| `incompatible Python version` | Created project files before pinning Python | Delete all UV files except .py files, then: pin → init → sync |
| Activate not working in PowerShell | Wrong terminal | Use CMD for activation, not PowerShell |
| File not saving | VS Code doesn't auto-save | Ctrl+S (Windows) / Cmd+S (Mac) before running |
| Created files inside .venv | Wrong location | All code goes in root project folder, never inside .venv |
| Python version conflict after change | Wrong order | Edit pyproject.toml → `uv lock` → `uv sync` |

---

## Part 2 — Introduction to Generative AI

### The Hierarchy: AI → ML → Deep Learning → Generative AI

```
AI
└── Machine Learning
    └── Deep Learning
        └── Generative AI
            └── Agentic AI
```

Each level is a specialization of the one above.

---

### AI

Machines programmed to exhibit **human-like decision making**. Any system that learns and makes intelligent decisions on its own.

---

### Machine Learning

Systems that **learn patterns from data without being explicitly programmed**.

- You don't write the formula — the algorithm derives it from data
- Goal: **fit the line** (find the mathematical relationship in data)
- Example: pass 1 year of weather data → ML derives patterns → predicts tomorrow's rain

---

### Deep Learning

Specialization of ML using **Neural Networks** — algorithms inspired by how the human brain learns.

- Flexible: one architecture handles text, images, audio (unlike classic ML which needs a separate algorithm per data type)
- Neural networks are old but only got famous once the internet created enough data to feed them (they are data-hungry)
- Reduces complexity of classic ML: one powerful algorithm instead of dozens

---

### Why Generative AI Exists — The Data Scarcity Problem

ML and DL need huge amounts of data. For niche domains like MRI scans, hospitals won't share millions of patient records.

**The need:** Can we take 100 MRI samples and generate more synthetic-but-realistic MRI data from those 100?

This required algorithms that don't just fit the line — they had to **learn the distribution of the data** to generate new samples that look like the originals.

---

### Generative AI

A specialized branch of deep learning that moves beyond classification and prediction to **create entirely new content from scratch**.

**Key difference:**

| Traditional ML/DL (Discriminative) | Generative AI |
|---|---|
| Learns boundaries between classes | Learns the distribution of data |
| Fits the line | Generates from the distribution |
| Output: prediction, classification | Output: text, image, audio, code |
| Example: will it rain tomorrow? (Yes/No + probability) | Example: generate 7 days of synthetic weather data |

---

### Timeline — How We Got to Transformers

**2013 — VAE (Variational Autoencoders)**
Compress data → tiny representation → decompress back. Proved we can learn a compressed representation and generate new samples from it.

**2014 — GANs (Generative Adversarial Networks)**
Two networks compete: generator (creates fake data) vs discriminator (judges real vs fake). Led to realistic image generation (e.g. the viral Will Smith deepfake era).

**2017 — Transformers**
Understood data from **context**, not just position. Optimized for parallel processing during training. This is what made BERT, GPT, and all modern LLMs possible. ChatGPT is a product — it uses GPT models, which are Transformer derivatives.

> Next class starts here — Sunny covers Transformers from scratch.

---

### Agentic AI

**Simple definition:** Gen AI understands. Agentic AI understands **and acts**.

- ChatGPT understands "call my brother" — but can't do it without tools
- Give it a calling tool → it understands intent AND executes → that's Agentic AI
- Agentic AI = Gen AI + access to tools (code execution, file system, terminal, APIs)
- Closest thing we have to true AI (Jarvis from Iron Man)

---

## Assignments

- Complete Python up to OOP minimum — check Krish's YouTube Python playlist
- Watch 2–4 ML algorithm videos — understand the flow, not the depth
- Practice the terminal — navigation, creating folders, running files
- Complete VS Code + UV setup independently (pin → init → sync → activate → run)
- Next class: Transformers from scratch with Sunny

---

## Resources

- UV documentation: [docs.astral.sh/uv](https://docs.astral.sh/uv)
- Krish Naik Python playlist: search "Krish Naik Python playlist" on YouTube
- Krish Naik ML playlist: search "Krish Naik machine learning playlist" on YouTube
- Course dashboard: [learn.krishnaikacademy.com](https://learn.krishnaikacademy.com)