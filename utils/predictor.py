import joblib
import pandas as pd

# Load saved objects
model = joblib.load("models/risk_model.pkl")
imputer = joblib.load("models/imputer.pkl")

NUMERIC_COLUMNS = [
    "hemoglobin_g_dl",
    "wbc_count",
    "differential_count",
    "rbc_count",
    "platelet_count",
    "platelet_distribution_width",
]


def preprocess_input(data):

    is_child = 1 if data["gender"] == "Child" else 0

    gender = {
        "Male": 1,
        "Female": 0,
        "Child": 0
    }[data["gender"]]

    df = pd.DataFrame([{
        "age": data["age"],
        "gender": gender,
        "hemoglobin_g_dl": data["hemoglobin"],
        "wbc_count": data["wbc"],
        "differential_count": data["differential"],
        "rbc_count": data["rbc"],
        "platelet_count": data["platelet"],
        "platelet_distribution_width": data["pdw"],
        "is_child": is_child
    }])

    df[NUMERIC_COLUMNS] = imputer.transform(df[NUMERIC_COLUMNS])

    return df


def predict_patient(data):

    patient = preprocess_input(data)

    prediction = model.predict(patient)[0]

    confidence = model.predict_proba(patient)[0][1]

    if confidence < 0.40:
        risk = "LOW"

    elif confidence < 0.70:
        risk = "MODERATE"

    else:
        risk = "HIGH"

    return {

        "prediction": int(prediction),

        "confidence": round(confidence * 100, 2),

        "risk": risk,

        "data": patient

    }