import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from utils.predictor import predict_patient

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Clinical AI Risk Stratification Engine",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

results = pd.read_csv("models/model_results.csv", index_col=0)

# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html,body,[class*="css"]{
font-family:'Inter',sans-serif;
}

.stApp{
background:#020617;
color:white;
}

section[data-testid="stSidebar"]{
background:#0F172A;
}

h1{
color:#00E5FF;
font-weight:700;
}

h2,h3{
color:#A855F7;
}

.metric-card{
background:#111827;
padding:20px;
border-radius:18px;
border:1px solid rgba(255,255,255,.08);
box-shadow:0 0 25px rgba(0,229,255,.08);
margin-bottom:18px;
}

.prediction-card{
background:#111827;
padding:30px;
border-radius:20px;
border:1px solid rgba(0,229,255,.25);
text-align:center;
}

hr{
border:1px solid #1E293B;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🩺 Clinical AI Risk Stratification Engine")

st.caption(
    "Explainable Machine Learning for Clinical Dengue Risk Prediction"
)
c1,c2,c3,c4 = st.columns(4)

c1.metric("Patients",989)
c2.metric("Features",9)
c3.metric("Best ROC-AUC","90.88%")
c4.metric("Best Model","Random Forest")

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Clinical Dashboard")

st.sidebar.success("✔ Random Forest Model")

st.sidebar.metric(
    "ROC-AUC",
    "0.9088"
)

st.sidebar.metric(
    "Accuracy",
    "92.42%"
)

st.sidebar.metric(
    "Dataset",
    "989 Patients"
)

st.sidebar.metric(
    "Features",
    "9"
)

st.sidebar.info(
"""
Models

• Logistic Regression

• Random Forest

• XGBoost
"""
)

# --------------------------------------------------
# PATIENT INPUT
# --------------------------------------------------

st.header("Patient Risk Assessment")

left,right=st.columns(2)

with left:

    age=st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

    gender=st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Child"
        ]
    )

    hemoglobin=st.number_input(
        "Hemoglobin (g/dL)",
        value=13.5
    )

    wbc=st.number_input(
        "WBC Count",
        value=7000.0
    )

with right:

    differential=st.number_input(
        "Differential Count",
        value=60
    )

    rbc=st.number_input(
        "RBC Count",
        value=5
    )

    platelet=st.number_input(
        "Platelet Count",
        value=180000.0
    )

    pdw=st.number_input(
        "Platelet Distribution Width",
        value=15.0
    )

predict_button=st.button(
    "🔍 Predict Disease Risk",
    use_container_width=True
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    patient={

        "age":age,
        "gender":gender,
        "hemoglobin":hemoglobin,
        "wbc":wbc,
        "differential":differential,
        "rbc":rbc,
        "platelet":platelet,
        "pdw":pdw

    }

    result=predict_patient(patient)

    st.divider()

    c1,c2,c3=st.columns(3)

    with c1:

        st.metric(
            "Risk Level",
            result["risk"]
        )

    with c2:

        st.metric(
            "Confidence",
            f'{result["confidence"]:.2f}%'
        )

    with c3:

        st.metric(
            "Prediction",
            "Positive" if result["prediction"]==1 else "Negative"
        )

    if result["prediction"]==1:

        st.error("⚠️ High probability of Dengue detected.")

    else:

        st.success("✅ Low probability of Dengue detected.")

    st.info(
        "Prediction generated successfully. Scroll down for model performance."
    )

st.divider()

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.header("Model Performance Comparison")

st.dataframe(
    results.style.format("{:.4f}"),
    use_container_width=True
)

best_model=results["ROC-AUC"].idxmax()

st.success(f"Best Performing Model : {best_model}")
# --------------------------------------------------
# MODEL VISUALIZATIONS
# --------------------------------------------------

st.divider()

st.header("Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("ROC Curve")

    try:
        st.image(
            "models/roc_curve.png",
            use_container_width=True
        )
    except:
        st.warning("ROC curve not found.")

with col2:
    st.subheader("Confusion Matrix")

    try:
        st.image(
            "models/confusion_matrix.png",
            use_container_width=True
        )
    except:
        st.warning("Confusion Matrix not found.")

# --------------------------------------------------
# PERFORMANCE CHART
# --------------------------------------------------

st.divider()

st.header("Model Comparison")

metrics = results.reset_index().rename(
    columns={"index":"Model"}
)

chart = px.bar(
    metrics,
    x="Model",
    y="ROC-AUC",
    text="ROC-AUC",
    color="Model",
    template="plotly_dark"
)

chart.update_layout(
    plot_bgcolor="#020617",
    paper_bgcolor="#020617",
    font_color="white",
    height=500
)

st.plotly_chart(
    chart,
    use_container_width=True
)

# --------------------------------------------------
# AI CLINICAL INSIGHTS
# --------------------------------------------------

st.divider()

st.header("Clinical Interpretation")

if predict_button:

    confidence = result["confidence"]

    if confidence >= 80:

        st.success(
            """
### AI Assessment

The model has very high confidence in this prediction.

Recommended actions:

• Review platelet count carefully.

• Confirm laboratory findings.

• Perform physician assessment.

• Consider repeat CBC if symptoms persist.
"""
        )

    elif confidence >= 60:

        st.warning(
            """
### AI Assessment

Moderate confidence prediction.

Recommended actions:

• Monitor patient.

• Repeat blood investigation.

• Observe symptoms over 24–48 hours.
"""
        )

    else:

        st.info(
            """
### AI Assessment

Low confidence prediction.

Recommended actions:

• Continue observation.

• Clinical correlation recommended.

• AI prediction alone should not determine diagnosis.
"""
        )

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.divider()

st.warning(
"""
### Medical Disclaimer

This application is an AI-assisted clinical decision support tool.

Predictions should NOT replace physician diagnosis.

Always interpret results together with patient history,
clinical examination and laboratory investigations.
"""
)
# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()
st.header("Model Feature Importance")

try:

    rf_model = joblib.load("models/risk_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        template="plotly_dark"
    )

    fig.update_layout(
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font_color="white",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except Exception as e:
    st.error(f"Unable to load feature importance: {e}")

# --------------------------------------------------
# PATIENT SUMMARY
# --------------------------------------------------

st.divider()
st.header("Patient Summary")

summary = pd.DataFrame({
    "Parameter":[
        "Age",
        "Gender",
        "Hemoglobin",
        "WBC Count",
        "Differential Count",
        "RBC Count",
        "Platelet Count",
        "Platelet Distribution Width"
    ],
    "Value":[
        age,
        gender,
        hemoglobin,
        wbc,
        differential,
        rbc,
        platelet,
        pdw
    ]
})

st.dataframe(
    summary,
    use_container_width=True
)

# --------------------------------------------------
# AI EXPLANATION
# --------------------------------------------------

if predict_button:

    st.divider()

    st.header("AI Clinical Explanation")

    if result["prediction"] == 1:

        st.markdown(f"""
<div style="
background:#7F1D1D;
padding:20px;
border-radius:15px;
border-left:6px solid red;
padding:25px;
">

<h2 style="color:white;">🔴 HIGH RISK</h2>

<h3 style="color:white;">
Confidence: {result['confidence']:.2f}%
</h3>

</div>
""", unsafe_allow_html=True)

    else:

        st.markdown(f"""
<div style="
background:#14532D;
padding:20px;
border-radius:15px;
border-left:6px solid #22C55E;
padding:25px;
">

<h2 style="color:white;">🟢 LOW RISK</h2>

<h3 style="color:white;">
Confidence: {result['confidence']:.2f}%
</h3>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TOP FEATURES
# --------------------------------------------------

if predict_button:

    st.subheader("Top Influential Features")

    try:

        top3 = importance_df.head(3)

        for _, row in top3.iterrows():

            st.write(
                f"✅ {row['Feature']} "
                f"(Importance: {row['Importance']:.3f})"
            )

    except:
        st.info("Feature importance unavailable.")

# --------------------------------------------------
# REPORT DOWNLOADok i on
# --------------------------------------------------

st.divider()

st.header("Clinical Report")

if predict_button:

    report = f"""
Clinical AI Risk Stratification Engine

Prediction : {result['risk']}

Confidence : {result['confidence']:.2f} %

Model : Random Forest

-------------------------------------

Patient Information

Age : {age}

Gender : {gender}

Hemoglobin : {hemoglobin}

WBC : {wbc}

Differential : {differential}

RBC : {rbc}

Platelet : {platelet}

Platelet Distribution Width : {pdw}

-------------------------------------

AI Assisted Clinical Decision Support

This prediction should not replace
professional medical diagnosis.
"""

    st.download_button(
        label="📄 Download Clinical Report",
        data=report,
        file_name="clinical_AI_report.txt",
        mime="text/plain"
    )
    st.divider()

st.markdown("""
<div style='text-align:center;padding:20px;color:#94A3B8'>

### Clinical AI Risk Stratification Engine

Developed using

Python • Scikit-Learn • Random Forest • Streamlit • Plotly

AI-assisted clinical decision support system for dengue risk prediction.

⚠️ This application is intended for educational and research purposes only.

</div>
""", unsafe_allow_html=True)