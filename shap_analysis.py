import joblib
import shap
import pandas as pd
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("data/Dengue_diseases_dataset_modified (1).csv")

df["is_child"] = (df["gender"] == "Child").astype(int)
df["gender"] = df["gender"].map({"Male":1,"Female":0,"Child":0})

X = df.drop("dengue_label", axis=1)

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

model = joblib.load("models/risk_model.pkl")

explainer = shap.Explainer(model)

exp = explainer(X.iloc[:1])

print("SHAP version:", shap.__version__)
print("Explanation shape:", exp.values.shape)
print("Base values shape:", exp.base_values.shape)