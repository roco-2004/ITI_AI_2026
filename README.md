# House Price Predictor

An end-to-end machine learning application for predicting house prices in India. The project combines a trained regression model with a FastAPI backend and a React frontend to provide an interactive house price prediction experience.

**Status: Ready for instructor review. All notebooks include verified saved outputs; raw datasets and large training checkpoints are intentionally excluded.**

## Project Structure

```text
ITI_AI_2026/
├── README.md
├── requirements.txt
├── .gitignore
│
├── ML/
│   ├── README.md
│   ├── 01_Bank_Customer_Churn_Classification.ipynb
│   └── 02_Bengaluru_House_Price_Regression.ipynb
│
├── CNN/
│   ├── README.md
│   ├── 01_CIFAR10_Baseline_CNN.ipynb
│   └── 02_CIFAR10_Improved_CNN.ipynb
│
└── Final-Project/
    ├── README.md
    ├── docker-compose.yml
    ├── pyproject.toml
    │
    ├── backend/
    │   ├── app/
    │   │   ├── api/
    │   │   ├── core/
    │   │   ├── schemas/
    │   │   ├── services/
    │   │   └── utils/
    │   └── tests/
    │
    ├── frontend/
    │   ├── src/
    │   ├── e2e/
    │   └── package.json
    │
    ├── models/
    │   ├── house_price.pkl
    │   ├── locations.json
    │   └── model_metadata.json
    │
    ├── notebooks/
    │   ├── house_price_model.ipynb
    │   └── data/
    │
    ├── docs/
    │   └── screenshots/
    │
    └── scripts/
        ├── build_notebook.py
        ├── smoke_test.py
        └── verify_project.py
```

## Machine Learning

- [Bank Customer Churn Classification](ML/01_Bank_Customer_Churn_Classification.ipynb)
- [Bengaluru House Price Regression](ML/02_Bengaluru_House_Price_Regression.ipynb)

## CNN

- [CIFAR-10 Baseline CNN](CNN/01_CIFAR10_Baseline_CNN.ipynb)
- [CIFAR-10 Improved CNN](CNN/02_CIFAR10_Improved_CNN.ipynb)

## Final Project

- [India House Price Predictor](Final-Project/README.md) — an executed regression notebook, exported scikit-learn pipeline, FastAPI backend, React interface, tests, screenshots, and Docker Compose workflow.

## Project summary

| Notebook | Task type | Dataset | Verified metric | Status |
| --- | --- | --- | --- | --- |
| [Bank Customer Churn Classification](ML/01_Bank_Customer_Churn_Classification.ipynb) | Binary classification | Churn Modelling | Accuracy 0.8660; ROC-AUC 0.8677 | Complete |
| [Bengaluru House Price Regression](ML/02_Bengaluru_House_Price_Regression.ipynb) | Regression | Bengaluru House Prices | Test R² 0.6996; RMSE 71.9076 | Complete |
| [CIFAR-10 Baseline CNN](CNN/01_CIFAR10_Baseline_CNN.ipynb) | Image classification | CIFAR-10 | Test accuracy 0.7292; macro F1 0.7304 | Complete; fresh 25-epoch run |
| [CIFAR-10 Improved CNN](CNN/02_CIFAR10_Improved_CNN.ipynb) | Image classification | CIFAR-10 | Test accuracy 0.9008; macro F1 0.9006 | Complete; checkpoint evaluation verified |

## Setup with Python 3.11

Run these commands in PowerShell:

```powershell
Set-Location "D:\ITI-GitHub-Portfolio"
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter notebook
```

Open a notebook and run its cells in order. The notebooks expect datasets under the repository-root `datasets/` directory and provide actionable missing-file messages. The improved CNN can require a long CPU training run of up to 70 epochs.

## Data availability and saved outputs

Raw datasets and model checkpoints are excluded because of privacy, licensing, and GitHub file-size considerations. In particular, CSV files, CIFAR image arrays, and Keras checkpoints are not committed.

The notebooks retain their valid saved outputs, plots, training histories, and verified metrics. GitHub can therefore display the completed work directly without downloading the excluded data or rerunning training.

## Limitations

- The churn classifier's recall is 0.5577, so it misses a meaningful share of positive churn cases.
- The Bengaluru model drops from validation to test performance and is not a production-grade pricing model.
- The baseline CNN is intentionally compact and reaches lower accuracy than the improved model.
- The improved CNN's final evaluation was reproduced from its validation-selected checkpoint; the complete 70-epoch CPU retraining was not repeated during the audit, and the checkpoint is not published.

