# 📄 Graphical Reasoning: LLM-based Semi-Open Relation Extraction  
**ACL 2026 Student Research Workshop (SRW)** — *Submitted to ACL 2026*  
📌 *Preprint (arXiv):* https://arxiv.org/abs/2405.00216

---

## 🔍 Overview

Relation Extraction (RE) is a core task in natural language processing that identifies structured relationships between entities in unstructured text. This project explores how **Large Language Models (LLMs)** can be harnessed to perform *semi-open* relation extraction, combining pragmatic reasoning with flexible relation discovery.

We introduce a novel **Graphical Reasoning (GRE)** framework that decomposes relation extraction into sequential, interpretable reasoning subtasks. By leveraging advanced prompting techniques — including *Chain of Thought* (CoT) reasoning and example-based in-context learning — we achieve more reliable and adaptable extraction performance.

The proposed approach is evaluated across multiple datasets, demonstrating improvements in accuracy and robustness.

---

## 🚀 Key Contributions

- 📌 **Toolkit for LLM-based Relation Extraction**  
  A framework integrating LLM reasoning with graph-structured decomposition that supports semi-open extraction settings.

- 🧠 **Graphical Reasoning (GRE)**  
  A structured breakdown of relational inference into reasoning steps, enabling better handling of complex and overlapping relational patterns.

- 🧩 **Chain of Thought (CoT) Prompting**  
  Application of step-by-step reasoning prompts to enhance LLM understanding and decision-making during relation identification.

- 📊 **Empirical Evaluation**  
  Demonstrated improvements over baseline extraction methods on diverse datasets, including custom annotated corpora.

---

## 🧪 Experimental Setup

The experiments use **in-context learning** with GPT-3.5 to evaluate the proposed GRE pipeline. Detailed comparisons against existing approaches show consistent gains across standard evaluation metrics such as precision, recall, and F1 score.

> See the paper for complete dataset descriptions, experimental configurations, and quantitative results.

---

## 📂 Repository Structure

```text
📦 .
├── data/
│   └── README.md                   # Dataset descriptions and access instructions
├── models/
│   └── reasoning_prompt_templates/ # Prompt templates for in-context learning
├── experiments/
│   ├── eval_scripts/
│   └── results/                    # Evaluation outputs and logs
├── notebooks/
│   └── demo.ipynb                  # End-to-end extraction demo
├── src/
│   └── graphical_reasoning.py      # Core GRE implementation
├── paper.pdf                       # ACL SRW 2026 submission (local copy)
├── LICENSE
└── README.md
```

⸻

📦 Dependencies
	•	Python 3.8+
	•	transformers
	•	openai (or other LLM API clients)
	•	networkx
	•	scikit-learn

⸻

📖 Citation

If you use this work, please cite:
```
@misc{tao2024graphical,
  title={Graphical Reasoning: LLM-based Semi-Open Relation Extraction},
  author={Tao, Yicheng and Wang, Yiqun and Bai, Longju},
  year={2024},
  eprint={2405.00216},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

⸻

📌 Contact

For questions or collaboration inquiries, please open an issue on GitHub or contact the authors using the information provided in the paper.

⸻


ADE and CoNLL04 datasets can be downloaded from [SpERT](https://github.com/lavis-nlp/spert/blob/master/scripts/fetch_datasets.sh) and should be put under datasets/ade/ and datasets/conll04/, respectively. NYT dataset can be downloaded from [Zeng et al., 2018](https://github.com/xiangrongzeng/copy_re) and should be put under datasets/nyt/.

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
