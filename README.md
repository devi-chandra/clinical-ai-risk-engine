# Clinical AI Risk Stratification Engine

> **End-to-End Machine Learning Project for Healthcare AI**

<p align="center">
  <strong>Explainable Machine Learning for Clinical Dengue Risk Prediction</strong><br/>
  An end-to-end AI-powered clinical decision support system that predicts dengue disease risk from patient laboratory data, compares multiple ML models, and delivers interpretable results through an interactive Streamlit dashboard.
</p>

---

## Overview

Dengue fever is one of the fastest-spreading vector-borne diseases globally, with an estimated 400 million infections annually across tropical and subtropical regions. Early, accurate identification of high-risk patients is critical — delayed diagnosis often leads to progression into dengue hemorrhagic fever or dengue shock syndrome, both of which carry significant mortality if untreated.

Traditional clinical triage relies on symptom scoring and physician experience, which is time-intensive and inconsistent across settings. Machine learning offers a complementary approach: models trained on patient laboratory data can flag high-risk individuals earlier and more objectively, allowing clinicians to prioritize care.

This project builds a complete ML pipeline for dengue risk prediction — from raw clinical data through preprocessing, model training, multi-model comparison, and a production Streamlit web application. Every prediction includes a confidence score, while the application provides global feature importance to help interpret the trained model.

> **Medical Disclaimer:** This application is an AI-assisted clinical decision support tool. Predictions are intended to support — not replace — physician judgment. All outputs should be interpreted alongside patient history, clinical examination, and laboratory investigations.

---

## Key Features

### Multi-Model Training and Comparison
Three machine learning algorithms—Logistic Regression, Random Forest, and XGBoost—are trained, evaluated, and compared using Accuracy, Precision, Recall, F1 Score, and ROC-AUC. The best-performing model is automatically selected.

### High Predictive Performance
The Random Forest model achieved a ROC-AUC of **0.9088** and an Accuracy of **92.42%** on the test dataset.

### Interactive Patient Risk Assessment
The Streamlit application allows users to enter patient demographic and laboratory values and instantly predicts whether the patient is at **High** or **Low Dengue Risk**, along with a confidence score.

### Model Performance Visualization
The dashboard includes ROC Curve, Confusion Matrix, and comparative performance metrics for all trained models.

### Feature Importance Analysis
The application visualizes Random Forest feature importance, helping users understand which clinical variables contribute most to the trained model.

### Clinical Report
A downloadable clinical report summarizes patient inputs, prediction results, and confidence score for documentation purposes.

### Data Preprocessing Pipeline
Missing values are imputed, categorical variables are encoded, and preprocessing artifacts are saved to ensure consistent predictions during deployment.

### Explainability Support
A separate SHAP analysis script is included for post-training model interpretation and explainability experiments.
---

## Screenshots
 screen shots are available in seperate folder.
---

## Machine Learning Workflow

```
1. Data Collection
   └── 989 patient records with clinical and laboratory features
       from dengue_dataset.csv

2. Data Cleaning & Preprocessing
   ├── Missing value imputation (saved as imputer.pkl)
   ├── Feature standardization (saved as scaler.pkl)
   └── Feature name serialization (saved as feature_names.pkl)

3. Feature Engineering
   └── Derived binary feature: is_child (age-based)
       Gender encoded as numeric
       All features scaled to zero mean, unit variance

4. Train / Test Split
   └── Stratified split to preserve class balance

5. Model Training
   ├── Logistic Regression
   ├── Random Forest (n_estimators tuned)
   └── XGBoost

6. Model Evaluation
   └── Accuracy, Precision, Recall, F1 Score, ROC-AUC
       across all three models

7. Model Selection
   └── Random Forest selected (highest ROC-AUC: 0.9088)
       Serialized as risk_model.pkl

8. Prediction & Deployment
   └── Streamlit app loads pkl artifacts and runs inference
       Results displayed with confidence score, feature
       importance chart, ROC curve, confusion matrix,
       and downloadable clinical report
```

---

## Dataset

| Property | Details |
|---|---|
| Source | Clinical dengue patient records |
| Total Patients | 989 |
| Features | 9 (clinical + laboratory) |
| Target Variable | Dengue risk label (binary: positive / negative) |
| Format | CSV (`data/dengue_dataset.csv`) |

**Input Features:**

| Feature | Type | Clinical Significance |
|---|---|---|
| Age | Numeric | Risk factor; children have distinct profiles |
| Gender | Categorical | Encoded as binary numeric |
| Hemoglobin (g/dL) | Numeric | Anemia indicator in dengue |
| WBC Count | Numeric | Leukopenia is a classic dengue marker |
| Differential Count | Numeric | Lymphocyte/neutrophil ratio shifts in dengue |
| RBC Count | Numeric | Red cell volume changes with hemoconcentration |
| Platelet Count | Numeric | Thrombocytopenia is the hallmark dengue lab finding |
| Platelet Distribution Width | Numeric | Platelet morphology indicator |
| is_child (engineered) | Binary | Derived from age; pediatric cases treated distinctly |

**Preprocessing steps applied:**
- Missing value imputation (median/mode strategy)
- StandardScaler normalization
- Gender label encoding
- All artifacts serialized via Joblib for reproducible inference

---

## Models Used

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9141 | 0.9058 | 0.9690 | 0.9363 | 0.8975 |
| **Random Forest** ⭐ | **0.9242** | **0.9254** | **0.9612** | **0.9430** | **0.9088** |
| XGBoost | 0.9242 | 0.9254 | 0.9612 | 0.9430 | 0.9027 |

> ⭐ **Best Model: Random Forest** — selected on highest ROC-AUC score. In clinical settings, ROC-AUC is preferred over accuracy as it captures model performance across all decision thresholds, which matters when the cost of false negatives (missed dengue cases) is high.

---

## Evaluation

### ROC Curve
The Receiver Operating Characteristic curve plots the True Positive Rate against the False Positive Rate across all classification thresholds. An AUC of **0.9088** indicates the Random Forest model has a 90.88% probability of correctly ranking a randomly selected positive patient above a randomly selected negative one — a strong result for a clinical screening task.

### Confusion Matrix
The confusion matrix reveals the distribution of true positives, false positives, true negatives, and false negatives on the test set. With recall of 0.9612, the model captures the overwhelming majority of actual dengue-positive patients, which is the clinically critical metric to optimize.

### Feature Importance
The model's built-in feature importance scores reveal that **platelet count** and **WBC count** are the dominant predictors — consistent with established clinical knowledge that thrombocytopenia and leukopenia are the most reliable laboratory markers of dengue infection. This alignment between model behavior and clinical domain knowledge is a positive signal for trustworthiness.

---

## Streamlit Application

The dashboard is organized into four distinct sections:

**Patient Risk Assessment**
The primary interface. Input fields accept patient values for all eight laboratory parameters. Clicking "Predict Disease Risk" runs the full preprocessing pipeline and trained model in real time, returning a risk label (High / Low) and prediction confidence percentage.

**Model Performance Comparison**
A side-by-side table of all three models across five metrics, followed by a Plotly bar chart comparing ROC-AUC scores visually. Automatically highlights the best-performing model.

**Model Evaluation**
Displays the ROC curve and confusion matrix plots generated during training, alongside a Clinical Interpretation section and Medical Disclaimer.

**Feature Importance & Clinical Report**
A color-mapped horizontal bar chart shows each feature's contribution to the model's predictions. Below this, a Patient Summary table lists all entered values, followed by a formatted Clinical Report noting the tools used and appropriate disclaimers.

---

## Project Architecture

```
dengue_dataset.csv
        │
        ▼
┌─────────────────────┐
│   analyze_data.py   │  ← Exploratory data analysis
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   train_model.py    │  ← Preprocessing, training, serialization
│                     │
│  ┌───────────────┐  │
│  │ Imputer       │  │
│  │ Scaler        │  │
│  │ Random Forest │  │  → risk_model.pkl
│  │ Logistic Reg  │  │  → scaler.pkl
│  │ XGBoost       │  │  → imputer.pkl
│  └───────────────┘  │  → feature_names.pkl
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  evaluate_model.py  │  ← Metrics, ROC curve, confusion matrix
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  shap_analysis.py   │  ← SHAP explainability (post-hoc)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  utils/predictor.py │  ← Inference wrapper (loads pkl artifacts)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│      app.py         │  ← Streamlit dashboard
│                     │
│  • Patient Input    │
│  • Risk Prediction  │
│  • Model Comparison │
│  • Feature Charts   │
│  • Clinical Report  │
└─────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/devi-chandra/clinical-ai-risk-engine.git
cd clinical-ai-risk-engine

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
# Run exploratory data analysis (optional)
python analyze_data.py

# Train all models and serialize artifacts
python train_model.py

# Evaluate the best model (generates ROC curve and confusion matrix)
python evaluate_model.py

# Run SHAP explainability analysis (optional)
python shap_analysis.py
```

### Launching the Application

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## Folder Structure

```
clinical-ai-risk-engine/
│
├── data/
│   └── dengue_dataset.csv          # Raw patient dataset (989 records)
│
├── models/
│   ├── risk_model.pkl              # Trained Random Forest model
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── imputer.pkl                 # Fitted SimpleImputer
│   ├── feature_names.pkl           # Ordered feature name list
│   ├── model_results.csv           # Metrics for all three models
│   ├── roc_curve.png               # ROC curve plot (saved)
│   └── confusion_matrix.png        # Confusion matrix plot (saved)
│
├── screenshots/
│   ├── dashboard.png
│   ├── prediction.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── utils/
│   └── predictor.py                # Inference wrapper — loads artifacts, runs prediction
│
├── app.py                          # Main Streamlit dashboard
├── train_model.py                  # Preprocessing + model training pipeline
├── analyze_data.py                 # Exploratory data analysis
├── evaluate_model.py               # Model evaluation and plot generation
├── shap_analysis.py                # SHAP explainability script
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Integrate SHAP visualizations directly into the Streamlit dashboard.
- Generate professional PDF-based clinical reports.
- Deploy the application on Streamlit Community Cloud or AWS.
- Support additional tropical diseases such as Malaria and Typhoid.
- Integrate real hospital datasets for improved generalization.
- Add physician authentication and patient history management.
- Perform hyperparameter optimization using Optuna or GridSearchCV.
---

## Why This Project Stands Out

This project was built to reflect what real production ML looks like in a high-stakes domain — not a notebook with an accuracy score, but a complete system with a preprocessing pipeline, serialized artifacts, multi-model comparison, explainability tooling, and a functional deployment.

**What it demonstrates for technical recruiters:**

| Skill | How It Appears in This Project |
|---|---|
| Machine Learning | Three algorithms trained and rigorously compared across five metrics |
| Healthcare AI | Domain-appropriate feature selection, clinical framing, medical disclaimer |
| Explainability | Feature importance visualization; dedicated SHAP analysis script |
| Data Engineering | Imputation, scaling, feature engineering, artifact serialization pipeline |
| Deployment | Streamlit app with real-time inference, clean multi-section UI |
| Model Evaluation | ROC-AUC prioritized over accuracy; confusion matrix and ROC curve included |
| Production Mindset | Modular codebase, separated concerns (train / evaluate / predict / app) |
| Code Quality | Clean architecture with `utils/predictor.py` decoupling inference from UI |

In healthcare AI specifically, the ability to explain a model's decision is not optional — it is a clinical and regulatory requirement. This project treats explainability as a first-class concern, not an afterthought.

---

## Author

**Devi**

Final Year B.Tech – Computer Science & Engineering (Artificial Intelligence & Data Science)

LinkedIn: https://linkedin.com/in/jdevi23
