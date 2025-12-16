# LLM-Supported Natural Language to Bash Translation  
_Reproducibility Study of NL2SH-ALFA (CS421 Project)_

## Overview

This repo contains a reproducibility study of the paper:

> Westenfelder et al., “LLM-Supported Natural Language to Bash Translation” (NL2SH-ALFA)

The goal is to reproduce and analyze how well small language models can translate **natural language instructions** into **bash commands**, and to study how different **evaluation metrics** change the reported performance.

We work with two small open-source models:

- `meta-llama/Llama-3.2-1B-Instruct`
- `Qwen/Qwen2.5-Coder-0.5B-Instruct`

We:

- Evaluate them **zero-shot** (no fine-tuning)
- **Fine-tune** them on NL2SH-ALFA
- Validate our **evaluation approach** using a Functional Equivalence Heuristics (FEH) comparison

---

## Dataset: NL2SH-ALFA

We use the **NL2SH-ALFA** dataset from Hugging Face:

- Hugging Face: https://huggingface.co/datasets/westenfelder/NL2SH-ALFA

Basic properties:

- ≈ 40,639 training examples
- 600 test examples
- Each example is a pair:
  - Natural language instruction (English)
  - A corresponding bash command

The commands cover tasks like:

- Listing files (`ls`, `ls -la`, `find . -type f`)
- Printing text (`echo "hello world"`)
- Searching files (`grep`, `find ... -exec grep ...`)
- Working with dates and environment variables (`date`, `echo $HOME`, etc.)

The dataset is designed specifically for **natural language to shell (NL2SH)** translation and is the same dataset used in the original NL2SH-ALFA paper.

> We do not redistribute the dataset in this repo. Please download it directly from Hugging Face.

---

## Models

Base models:

- **Llama-3.2-1B-Instruct**  
  - Hugging Face ID: `meta-llama/Llama-3.2-1B-Instruct`

- **Qwen2.5-Coder-0.5B-Instruct**  
  - Hugging Face ID: `Qwen/Qwen2.5-Coder-0.5B-Instruct`

We use standard Hugging Face Transformers code to:

- Load the base models
- Fine-tune them on NL2SH-ALFA
- Save the fine-tuned versions locally

---

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```


This should install:

- `transformers`
- `datasets`
- `accelerate`
- `sentence-transformers`
- `scikit-learn`
- `jupyter`
- and other required libraries.

---

### 2. Download the NL2SH-ALFA dataset

The dataset is hosted on Hugging Face:

- https://huggingface.co/datasets/westenfelder/NL2SH-ALFA

In the notebooks, you can load it directly via `datasets`:

```
from datasets import load_dataset

train_dataset = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
test_dataset = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train")
```


Make sure you are logged in to Hugging Face if needed.

---

### 3. Run the base model demo (zero-shot) – `example.ipynb`

Start by exploring how the **base model** behaves without any fine-tuning.

Open:

```
example.ipynb
```

This notebook:

- Loads the base model:  
  `model_id = "meta-llama/Llama-3.2-1B-Instruct"`
- Uses prompts like:  
  `"Task: list all python files in the current directory"`  
  and asks the model to produce a bash command.
- Optionally evaluates **zero-shot performance** on a subset of NL2SH-ALFA using your chosen metric.

This gives you a feel for:

- What kinds of commands the model generates out-of-the-box
- Baseline performance (before fine-tuning)

---

### 4. Fine-tune the models – `finetune.ipynb`

Next, fine-tune both Llama and Qwen on NL2SH-ALFA.

Open:

```
finetune.ipynb
```


This notebook:

- Loads the NL2SH-ALFA train/test splits
- Fine-tunes:
  - `meta-llama/Llama-3.2-1B-Instruct`
  - `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Typical configuration:
  - Epochs: 10
  - Batch size: 15
  - Learning rate: 1e-5
  - Precision: bfloat16
  - GPU: A100 (recommended)
- Saves the fine-tuned models to directories like:
  - `llama_1b_nl2sh_finetuned/`
  - `qwen_0.5b_nl2sh_finetuned/`

After this step, you will have:

- Zero-shot results from `example.ipynb`
- Fine-tuned models ready for evaluation

---

### 5. Validate evaluation metrics (FEH comparison) – `feh_comparison.ipynb`

This is where we **validate our evaluation approach**.

Open:

```
feh_comparison.ipynb
```


This notebook:

- Loads a subset of the NL2SH-ALFA test set
- Uses 8 different heuristics to decide whether two commands are functionally equivalent:

  - `bleu`
  - `nl2cmd`
  - `tfidf`
  - `exec_tfidf`
  - `mxbai_embed`
  - `exec_mxbai_embed`
  - `llama3`
  - `exec_llama3`

- Produces a table with:
  - TP, FP, TN, FN
  - Precision, Recall, F1
  - Accuracy
- Writes detailed results to `feh_results/*.csv`

**Key validation result:**

- Execution-based methods (which actually run the commands) achieve:
  - `exec_mxbai_embed`: **90%** accuracy
  - `exec_tfidf`: **82%** accuracy
- Our semantic method `mxbai_embed` (no execution) also achieves **82%** accuracy.

This shows that **semantic similarity is as accurate as a strong execution-based method**, but without needing Docker or a sandbox.

This is why we trust it as our main evaluation metric for the fine-tuned models.

---

## Evaluation Methods

We consider two main ways to score generated commands:

### 1. Exact Match

- Simple string equality:
  - `predicted_command == ground_truth_command`
- Pros:
  - Easy to implement
- Cons:
  - **Too strict** for code: many different commands can do the same job
  - Small changes (extra spaces, different flags, different but equivalent commands) all count as wrong

Example:

- Ground truth: `ls`
- Model: `find . -type f`

Both list files, but exact match marks it as incorrect.

### 2. Semantic Similarity (mxbai_embed)

- Convert ground truth and predicted commands into embeddings using a sentence-embedding model.
- Compute cosine similarity.
- If similarity ≥ 0.8 → treat as **functionally equivalent**.

Pros:

- Captures meaning rather than exact syntax
- Matches execution-based TF-IDF accuracy (82%) in our FEH study
- Does not require running the commands

This is the main metric we use for reporting model performance.

---

## Main Results (Summary)

### Zero-shot (no fine-tuning, semantic similarity)

- **Llama-3.2-1B**: ~32%
- **Qwen-0.5B**: ~36%

### Fine-tuned (semantic similarity)

- **Llama-3.2-1B**: ~57%
- **Qwen-0.5B**: ~60.33%

Improvements:

- Llama: **+25 percentage points** (32 → 57)
- Qwen: **+24.33 percentage points** (36 → 60.33)

With strict exact-match, post-finetuning accuracy is only ≈11–14%, but this is due to the metric being too strict, not because the models fail to generate working commands.

---

## Reproducibility

### Easy parts

- **Dataset**: free on Hugging Face (`westenfelder/NL2SH-ALFA`)
- **Models**: open-source on Hugging Face
- **Tools**: standard Python, Transformers, Datasets, Jupyter
- **Code**: notebooks are self-contained

### Harder parts

- **GPU requirements**:
  - Free Colab T4 (15GB) is often not enough.
  - An A100 (40GB) via Colab Pro works well.
- **Evaluation**:
  - Original paper uses Docker-based execution.
  - We use semantic similarity, then validate it using FEH.

Overall difficulty: **Moderate** – a student with Colab Pro and some patience can reproduce the main results.

---

## Key Takeaways

- Evaluation choice is critical:
  - Exact-match: 11–14% (too strict for code)
  - Semantic similarity: 57–60% (reflects functional correctness)
- Fine-tuning small models is very effective:
  - ~25 percentage point gains from fine-tuning on 40k examples
- Small models are practical:
  - 500M–1B parameter models can achieve ≈60% accuracy on NL2SH
  - More accessible than huge models
- Semantic similarity is a valid evaluation metric:
  - Matches execution-based TF-IDF at 82% accuracy in FEH comparison

---

## Safety Note

Generated bash commands can be **unsafe** if executed blindly.

Examples of dangerous patterns:

- `rm -rf /`
- `dd if=/dev/zero of=/dev/sda`
- Fork bombs: `:(){ :|:& }; :`

Always have a human expert review commands before running them on any important system.

---

## Acknowledgements

- Original paper and dataset authors (Westenfelder et al.)  
- Hugging Face for hosting models and datasets  
- CS421 course staff