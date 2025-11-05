# Credit Approval Prediction Project

This project implements a machine learning model to predict credit approval for applicants based on their financial and demographic data. The model uses logistic regression and provides metrics such as accuracy, ROC-AUC, and feature importance (odds ratios).

---

## Project Structure

```
CST-600-AA-Week-2/
│
├── data/
│   └── clean_dataset.csv          # Cleaned dataset used for training and evaluation
│
├── src/
│   └── main.py                    # Main Python script for data loading, modeling, and evaluation
│
├── screenshots/                   # Folder to save graphs, plots, and other visuals
│   └── example_plot.png
│
├── requirements.txt               # Python dependencies for the project
│
└── README.md                      # Project documentation
```

---

## Installation

1. **Clone the repository**:

```bash
git clone <your-repo-url>
cd CST-600-AA-Week-2
```

2. **Create a virtual environment**:

```bash
python -m venv .venv311
```

3. **Activate the virtual environment**:

* Windows:

```bash
.venv311\Scripts\activate
```

* macOS/Linux:

```bash
source .venv311/bin/activate
```

4. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script:

```bash
python src/main.py
```

Or, open the Jupyter Notebook for interactive exploration:

```bash
jupyter notebook
```

---

## Output

The project outputs the following:

* Dataset summary and class balance
* Confusion matrix and classification report
* ROC-AUC score
* Odds ratios for features to understand their influence on credit approval

> **Note:** Sensitive features such as Gender, Ethnicity, and Marital Status are included for demonstration purposes. In real applications, avoid using these features directly to prevent bias in predictions.

---

## Dependencies

```text
pandas==2.2.3
numpy==2.1.0
scikit-learn==1.6.1
matplotlib==3.10.7
seaborn==0.13.2
```

---

## Future Improvements

* Implement additional classification algorithms (Random Forest, XGBoost) for comparison
* Include feature engineering and normalization steps
* Save generated graphs in `screenshots/` and reference them in the README
