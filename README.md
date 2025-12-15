# Natural Language Processing – Vehicle Complaints Classification

This repository contains coursework and experiments for a Natural Language Processing project focused on classifying vehicle complaints using deep learning architectures, including RNN, LSTM, BiLSTM, BERT, and RoBERTa.[web:22][web:25] All implementations are provided as Jupyter notebooks.[web:22]

## Project Overview

The goal of this project is to build and compare multiple neural architectures for text classification on a vehicle complaints dataset.[web:22][web:28] The workflow covers data cleaning and splitting, classical recurrent models, and transformer‑based models.[web:22][web:6]

Main objectives:
- Preprocess and explore the vehicle complaints dataset.[web:22]
- Train and evaluate RNN, LSTM, and BiLSTM baselines.[web:22][web:6]
- Fine‑tune BERT and RoBERTa for improved performance.[web:22][web:6]
- Compare models using standard classification metrics such as accuracy, precision, recall, and F1.[web:25][web:28]

A copy of the full project artifacts (data, models, reports) is also available in Google Drive: `https://drive.google.com/drive/folders/1TipN06_9-l3ppV64pVarUOoMNtfkmaUq`.[web:22]

## Repository Structure

- `data/` – Raw and intermediate data files used by the notebooks (CSV and related assets).[web:22]  
- `Roberta code and model/` – Code and artifacts specific to RoBERTa fine‑tuning and evaluation.[web:22][web:14]  
- `DL_DataCleaning_splitting.ipynb` – Notebook for data cleaning, preprocessing, and train/validation/test splits.[web:22]  
- `255_Project_RNN.ipynb` – Recurrent Neural Network baseline implementation on the vehicle dataset.[web:22]  
- `255_Project_LSTM_BiLSTM.ipynb` – LSTM and BiLSTM implementations and experiments.[web:22]  
- `BiLstm__Implementation_vehicle_dataset.ipynb` – Additional BiLSTM experiments on the vehicle dataset.[web:22]  
- `BERT_Implementation_vehicle_dataset.ipynb` – BERT fine‑tuning on the vehicle complaints dataset.[web:22]  

> Note: `.DS_Store` is a macOS metadata file and not required to run the project.[web:22]

## Setup and Requirements

These notebooks can be run locally or in a hosted environment such as Google Colab.[web:6]

Recommended environment:
- Python 3.8+.[web:6]
- Jupyter Notebook / JupyterLab.[web:6]
- Common ML/NLP libraries:
  - `pandas`, `numpy`, `scikit-learn`.[web:6][web:25]
  - `matplotlib` / `seaborn` (for visualizations, if used).[web:6]
  - `torch`, `torchvision`, `torchaudio`.[web:6]
  - `transformers` (for BERT/RoBERTa).[web:6][web:14]
  - `tqdm` and `datasets` (optional, depending on notebook).[web:6]

To create a basic environment with `pip`:

python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
pip install -r requirements.txt # If you add one

text

If no `requirements.txt` exists yet, you can generate one from your current environment using:

pip freeze > requirements.txt

## Models and Methods

| Model    | File / Folder                                       | Notes |
|---------|------------------------------------------------------|-------|
| RNN     | `255_Project_RNN.ipynb`                              | Simple recurrent baseline for sequence modeling on complaint texts.[web:22][web:28] |
| LSTM    | `255_Project_LSTM_BiLSTM.ipynb`                      | Unidirectional LSTM on vehicle complaint texts.[web:22][web:6] |
| BiLSTM  | `255_Project_LSTM_BiLSTM.ipynb`, `BiLstm__Implementation_vehicle_dataset.ipynb` | Bidirectional LSTM to capture context from both directions.[web:22][web:6] |
| BERT    | `BERT_Implementation_vehicle_dataset.ipynb`          | Transformer‑based model fine‑tuned for text classification.[web:22][web:6][web:14] |
| RoBERTa | `Roberta code and model/`                            | Robustly optimized BERT variant fine‑tuned on the same dataset.[web:22][web:14] |


Goodle Drive link to the project : https://drive.google.com/drive/folders/1TipN06_9-l3ppV64pVarUOoMNtfkmaUq
