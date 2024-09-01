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

Few-shot experiments compared with [UIE](https://universal-ie.github.io):
| Model         | 1-Shot    | 5-Shot     | 10-Shot      |
| ------------- |:-------------:|:-------------:|:-------------:|
| **CoNLL04 Dataset** |
| UIE | 0.2186 | 0.4404 | **0.5114** |
| LLM-base | 0.3875 | 0.4887 | 0.4481 |
| GRE (ours) | **0.5069** | **0.5418** | 0.4898 |
To run GRE, use tot2_fs.py. To run base LLM model (GPT-4o), use tot2_fs_base.py.
