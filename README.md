# Natural Language Processing – Vehicle Complaints Classification

## What is NER?
Named Entity Recognition (NER) is an NLP technique used to automatically identify and categorize important pieces of information—such as names, locations, dates, or domain‑specific entities—from unstructured text. NER converts raw sentences into structured elements enabling better information extraction and downstream analysis.

## Project Overview
In this project, raw vehicle complaint narratives are transformed from unstructured text into a structured and machine‑readable format. This includes text cleaning, tokenization, encoding, and assigning entity labels. Structuring the dataset enables deep learning models to learn complaint patterns, identify main issues, and classify vehicle defects effectively.

### Dataset 
<img width="804" height="414" alt="image" src="https://github.com/user-attachments/assets/9e4d2a80-d66a-4806-8370-de791e797337" />


## Repository Structure

- `data/` – Raw and intermediate data files used by the notebooks (CSV and related assets). 
- `Roberta code and model/` – Code and artifacts specific to RoBERTa fine‑tuning and evaluation.
- `DL_DataCleaning_splitting.ipynb` – Notebook for data cleaning, preprocessing, and train/validation/test splits.
- `255_Project_RNN.ipynb` – Recurrent Neural Network baseline implementation on the vehicle dataset.
- `255_Project_LSTM_BiLSTM.ipynb` – LSTM and BiLSTM implementations and experiments.
- `BiLstm__Implementation_vehicle_dataset.ipynb` – Additional BiLSTM experiments on the vehicle dataset. 
- `BERT/` – BERT fine‑tuning on the vehicle complaints dataset.  

## Recommended environment:
- Python 3.8+.
- Jupyter Notebook / JupyterLab.
- Common ML/NLP libraries:
  - `pandas`, `numpy`, `scikit-learn`.
  - `matplotlib` / `seaborn` (for visualizations, if used).
  - `torch`, `torchvision`, `torchaudio`.
  - `transformers` (for BERT/RoBERTa).
  - `tqdm` and `datasets` (optional, depending on notebook).

## Models and Methods

| Model    | File / Folder                                       | Notes |
|---------|------------------------------------------------------|-------|
| RNN     | `255_Project_RNN.ipynb`                              | Simple recurrent baseline for sequence modeling on complaint texts. |
| LSTM    | `255_Project_LSTM_BiLSTM.ipynb`                      | Unidirectional LSTM on vehicle complaint texts. |
| BiLSTM  | `255_Project_LSTM_BiLSTM.ipynb`, `BiLstm__Implementation_vehicle_dataset.ipynb` | Bidirectional LSTM to capture context from both directions. |
| BERT    | `BERT_Implementation_vehicle_dataset.ipynb`          | Transformer‑based model fine‑tuned for text classification. |
| RoBERTa | `Roberta code and model/`                            | Robustly optimized BERT variant fine‑tuned on the same dataset. |

##  Results
## Results

| Model                  | Precision        | Recall           | F1 Score         | Accuracy         |
|------------------------|-------------------|-------------------|-------------------|-------------------|
| RNN                    | 0.4356 ± 0.01     | 0.6438 ± 0.01     | 0.5196 ± 0.01     | 0.6438 ± 0.01     |
| RNN (main entities)    | 0.6957            | 0.8696            | 0.7594            | 0.8228            |
| LSTM                   | 0.4551 ± 0.02     | 0.3997 ± 0.02     | 0.3967 ± 0.02     | 0.7390 ± 0.14     |
| BiLSTM                 | 0.8510 ± 0.01     | 0.8490 ± 0.01     | 0.8528 ± 0.01     | 0.8810 ± 0.01     |
| BERT-base              | 0.5567± 0.01      | 0.6950± 0.01      | 0.6182± 0.01      | 0.6587± 0.01      |
| **RoBERTa**            | **0.95303 ± 0.01**| **0.9603 ± 0.01** | **0.9566 ± 0.01** | **0.9813 ± 0.01** |


RoBERTa achieves the best performance with an F1‑score of **0.9566** and accuracy of **0.9813**, outperforming all other evaluated models due to its strong contextual encoding and ability to capture fine‑grained complaint patterns.

## RoBERTa model's Output
Each complaint sentence is broken into tokens, and the model predicts the correct entity label, converting text into structured data.
<img width="1020" height="64" alt="image" src="https://github.com/user-attachments/assets/0fee34bc-b2d0-4c70-ab46-88663b8b6072" />


### Example:
```
Sentence: "The engine makes a loud rattling noise when accelerating."

Token        Predicted TAG
--------------------------------
The          O
engine       PART
makes        O
a            O
loud         ATTRIBUTE
rattling     ISSUE
noise        ISSUE
when         O
accelerating ACTION
.            O
```

