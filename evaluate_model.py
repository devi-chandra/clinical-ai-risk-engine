import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
df = pd.read_csv("data/Dengue_diseases_dataset_modified (1).csv")

df["is_child"] = (df["gender"] == "Child").astype(int)
gender_map = {"Male": 1, "Female": 0, "Child": 0}
df["gender"] = df["gender"].map(gender_map)

X = df.drop("dengue_label", axis=1)
y = df["dengue_label"]

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load("models/risk_model.pkl")

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.savefig("models/confusion_matrix.png", dpi=300)
plt.close()

# ROC Curve
roc = RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.savefig("models/roc_curve.png", dpi=300)
plt.close()

print("Evaluation plots saved successfully!")