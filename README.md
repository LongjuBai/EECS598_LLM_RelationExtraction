
## 📌 Overview

Relation Extraction (RE) aims to identify semantic relations between entities in text and is a core component of information extraction and knowledge base construction. Recent large language models (LLMs) enable RE through prompting and reasoning without task-specific fine-tuning.

This work presents a **systematic empirical study** of LLM-based relation extraction, investigating:

- Prompt-based and reasoning-based RE (including Chain-of-Thought and Tree-of-Thought)
- Sentence-level vs. entity-level embedding strategies
- Dataset-dependent behaviors across ADE, and CoNLL04
- Robustness and consistency of LLM predictions

---

## 📂 Repository Structure

The repository structure exactly mirrors the current GitHub layout:

```text
EECS598_LLM_RelationExtraction/
├── datasets598/
├── log/
├── outputs/
├── prompts/
├── .gitignore
├── README.md
├── cluster_embeddings.py
├── cot.py
├── generate_embeddings.py
├── lh.log
├── main.py
├── query_sentence_relation_type.py
├── run_ade.sh
├── run_conll04.sh
├── run_conll04_cluster_entity_embeddings.sh
├── run_conll04_cluster_sentence_embeddings.sh
├── run_conll04_entity_embed.sh
├── run_conll04_sentence_embed.sh
├── stash_pull_stashPop.sh
├── tot.py
├── tot2.py
├── tot2_multi.py
└── utils.py
```

⸻

📁 Directory Description

datasets598/

Contains datasets used in the experiments, including:
	•	ADE
	•	CoNLL04

Datasets are preprocessed into formats suitable for LLM prompting and embedding generation.

⸻

prompts/

Prompt templates used in experiments, including:
	•	Zero-shot prompts
	•	Chain-of-Thought (CoT) prompts
	•	Tree-of-Thought (ToT) prompts

These prompts are used verbatim in the paper to ensure reproducibility.

⸻

outputs/

Stores model predictions, intermediate reasoning traces, and final outputs for each experiment.

⸻

log/

Execution logs for debugging and experiment tracking.

⸻

🧠 Core Python Scripts

main.py

Entry point for most experiments.
Handles dataset loading, prompt construction, LLM inference, and result saving.

⸻

query_sentence_relation_type.py

Queries the LLM to predict the relation type for a given sentence and entity pair.

⸻

cot.py

Implements Chain-of-Thought (CoT) prompting for relation extraction.

⸻

tot.py, tot2.py, tot2_multi.py

Implement Tree-of-Thought (ToT) style reasoning with different search and aggregation strategies.

⸻

generate_embeddings.py

Generates sentence-level or entity-level embeddings used for clustering and analysis.

⸻

cluster_embeddings.py

Clusters embeddings to analyze relation grouping and representation structure.

⸻

utils.py

Shared utility functions for data loading, formatting, evaluation, and logging.

⸻

🧪 Experiment Scripts

The following shell scripts reproduce the experiments reported in the paper:

ADE dataset
```
bash run_ade.sh
```

CoNLL04 baseline
```
bash run_conll04.sh
```

⸻

⚙️ Environment Setup

python >= 3.9

⚠️ Some experiments require access to proprietary LLM APIs.
Please set API keys via environment variables before running.

⸻

📊 Results & Analysis
	•	Quantitative results (precision, recall, F1) are reported in the paper.
	•	This repository provides raw outputs for transparency and reproducibility.
	•	Qualitative analyses (reasoning paths, failure cases) are derived from stored CoT/ToT traces.

⸻

🧪 Reproducibility Notes
	•	Prompts are fixed and versioned.
	•	Dataset splits follow prior RE literature.
	•	Random seeds are controlled where applicable.
	•	All scripts correspond directly to reported experiments.

---


ADE and CoNLL04 datasets can be downloaded from [SpERT](https://github.com/lavis-nlp/spert/blob/master/scripts/fetch_datasets.sh) and should be put under datasets/ade/ and datasets/conll04/, respectively. 

| Model         | Micro Prec    | Micro Rec     | Micro F1      | Macro Prec    | Macro Rec     | Macro F1      |
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| **CoNLL04 Dataset** |
| COT | 0.3396 | 0.5215 | 0.3996 | 0.2976 | 0.4914 | 0.3707 |
| GRE (ours) | **0.4364** | **0.5867** | **0.4941** | **0.3748** | **0.5700** | **0.4522** |
| **CoNLL04 Dataset (After Annotated)** |
| COT | 0.4408 | 0.5075 | 0.4488 | 0.3899 | 0.4662 | 0.4246 |
| GRE (ours) | **0.5862** | **0.6206** | **0.5985** | **0.5525** | **0.6085** | **0.5792** |
| **ADE Dataset** |
| COT | 0.6505 | **0.6264** | 0.6382 | 0.6505 | **0.6264** | 0.6382 |
| GRE (ours) | **0.7565** | 0.6213 | **0.6822** | **0.7565** | 0.6213 | **0.6822** |
