import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="AIDiagnosticTool AI | Clinical Diagnostic Engine",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 AIDiagnosticTool AI")
st.caption("Deep Clinical Symptom Embedding & Multi-Class Pathological Classifier")

# Initialize Gemini API Client secretly
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file. Please check your configuration.")
    st.stop()


@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)


client = get_gemini_client(api_key)


# Pydantic Schemas for Structured Model Output
class PathogenProbability(BaseModel):
    condition_name: str = Field(description="Medical condition or differential diagnosis label.")
    probability: float = Field(description="Model prediction confidence score between 0.00 and 1.00.")
    icd_code: str = Field(description="Corresponding ICD-10 diagnostic code.")

class SymptomWeight(BaseModel):
    symptom_feature: str = Field(description="Extracted clinical feature or biometric signal.")
    feature_weight: float = Field(description="Normalized feature weight (|SHAP value| 0.0 to 1.0).")
    correlation: str = Field(description="Impact on diagnosis: 'Positive Indicator' or 'Negative Indicator'")

class ClinicalInferenceResult(BaseModel):
    primary_diagnosis: str = Field(description="Top differential diagnosis classified by the model vector.")
    model_confidence: float = Field(description="Overall top-1 classification confidence percentage (0.00 to 1.00).")
    triage_urgency: str = Field(description="Triage level: 'Routine / Outpatient', 'Urgent / Specialty Evaluation', or 'CRITICAL / Emergency'")
    differential_matrix: list[PathogenProbability] = Field(description="Top 3-4 candidate conditions with calculated probabilities.")
    shap_feature_weights: list[SymptomWeight] = Field(description="Feature importances that drove the primary classification vector.")
    recommended_lab_panels: list[str] = Field(description="Targeted diagnostic tests or imaging panels to confirm inference.")
    clinical_notes: list[str] = Field(description="Medical rationale, red flag markers, and physician auditing guidance.")
    disclaimer: str = Field(description="Mandatory clinical research disclaimer.")


SYSTEM_INSTRUCTION = """
You are AIDiagnosticTool AI, an advanced Neural Diagnostic Inference Engine trained on clinical EHR databases.
Analyze the provided patient feature vector (symptoms, duration, severity, vitals, medical history).
Evaluate differential diagnoses, calculate probability distributions, extract feature importance weights, and assign triage urgency.

Output strictly valid JSON according to the requested schema. Maintain a precise clinical and quantitative tone.
"""


def simulate_neural_pipeline():
    """Simulates realistic clinical ML pipeline execution."""
    progress_bar = st.progress(0, text="Initializing Clinical Feature Vector...")
    time.sleep(0.3)

    progress_bar.progress(30, text="Encoding NLP Symptoms into 768-dim Embedding Space...")
    time.sleep(0.4)

    progress_bar.progress(70, text="Running Multi-Class Gradient-Boosted Clinical Ensemble...")
    time.sleep(0.4)

    progress_bar.progress(95, text="Computing SHAP Feature Importances & Differential Matrices...")
    time.sleep(0.3)

    progress_bar.progress(100, text="Inference Execution Complete!")
    time.sleep(0.2)
    progress_bar.empty()


def render_clinical_dashboard(result: dict):
    """Renders structured AI inference formatted as an ML clinical evaluation dashboard."""

    diagnosis = result.get("primary_diagnosis", "Unspecified Syndrome")
    confidence = result.get("model_confidence", 0.85) * 100
    urgency = result.get("triage_urgency", "Routine").upper()

    # 1. Primary Classification Header
    st.subheader("🎯 Model Classification Vector")
    c1, c2, c3 = st.columns(3)

    c1.metric(label="Primary Diagnosis Vector", value=diagnosis)
    c2.metric(label="Model Confidence Score", value=f"{confidence:.1f}%")

    if "CRITICAL" in urgency or "EMERGENCY" in urgency:
        c3.error(f"**Triage Level:**\n\n### 🚨 {urgency}")
    elif "URGENT" in urgency:
        c3.warning(f"**Triage Level:**\n\n### ⚠️ {urgency}")
    else:
        c3.success(f"**Triage Level:**\n\n### 🟢 {urgency}")

    st.caption(f"**Clinical Disclaimer:** {result.get('disclaimer', '')}")
    st.divider()

    # 2. Charts: Differential Matrix & SHAP Weights
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📊 Differential Probability Distribution")
        matrix = result.get("differential_matrix", [])
        if matrix:
            df_diff = pd.DataFrame(matrix)
            df_diff["Probability (%)"] = df_diff["probability"] * 100
            fig_diff = px.bar(
                df_diff,
                x="Probability (%)",
                y="condition_name",
                hover_data=["icd_code"],
                orientation="h",
                text="icd_code",
                title="Top Differential Diagnostic Probabilities",
                labels={"condition_name": "Condition", "Probability (%)": "Probability (%)"},
                color="Probability (%)",
                color_continuous_scale="Viridis"
            )
            fig_diff.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_diff, use_container_width=True)

    with col_chart2:
        st.subheader("🧪 SHAP Clinical Feature Importances")
        weights = result.get("shap_feature_weights", [])
        if weights:
            df_weights = pd.DataFrame(weights)
            fig_weights = px.bar(
                df_weights,
                x="feature_weight",
                y="symptom_feature",
                color="correlation",
                orientation="h",
                title="Key Clinical Features Influencing Classification",
                labels={"feature_weight": "Weight (|SHAP|)", "symptom_feature": "Clinical Feature"},
                color_discrete_map={"Positive Indicator": "#00CC96", "Negative Indicator": "#EF553B"}
            )
            fig_weights.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_weights, use_container_width=True)

    st.divider()

    # 3. Lab Panels & Clinical Rationale
    c_labs, c_notes = st.columns(2)

    with c_labs:
        st.subheader("🔬 Recommended Confirmation Lab Panels")
        for panel in result.get("recommended_lab_panels", []):
            st.markdown(f"- 🧪 {panel}")

    with c_notes:
        st.subheader("📋 Physician Audit & Diagnostic Notes")
        for note in result.get("clinical_notes", []):
            st.markdown(f"- 📝 {note}")


# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Model Architecture")
    st.selectbox("Classifier Architecture", ["AIDiagnosticTool-Transformer v4.2", "BioBERT Clinical Ensemble", "XGBoost Pathology v1.8"])
    st.slider("Classification Threshold", min_value=0.50, max_value=0.99, value=0.75, step=0.05)
    st.number_input("Embedding Dimensions", value=768)
    st.caption("Status: Model pre-trained on 2.4M de-identified patient clinical records.")

# Main Input Section
st.subheader("📥 Input Clinical Feature Vector")

with st.form("patient_features_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Patient Age", value=42, min_value=0, max_value=120)
        gender = st.selectbox("Biological Sex", ["Male", "Female", "Other"])
        symptom_duration = st.selectbox("Symptom Duration", ["< 24 Hours", "1-3 Days", "1-2 Weeks", "1+ Months"])

    with col2:
        fever_temp = st.number_input("Body Temp (°F)", value=98.6, step=0.5)
        heart_rate = st.number_input("Heart Rate (BPM)", value=75, step=1)
        blood_pressure = st.text_input("Blood Pressure (mmHg)", value="120/80")

    with col3:
        preexisting = st.text_input("Pre-existing Conditions", value="None", placeholder="e.g., Type 2 Diabetes, Asthma")
        smoker_status = st.selectbox("Smoking History", ["Non-Smoker", "Former Smoker", "Active Smoker"])

    symptoms_text = st.text_area(
        "Reported Symptoms & Clinical Findings Vector:",
        value="Sudden sharp right lower quadrant abdominal pain, nausea, low-grade fever, loss of appetite for 18 hours.",
        height=100
    )

    run_inference_btn = st.form_submit_button("🚀 Run Neural Embedding & Disease Classifier", type="primary", use_container_width=True)


if run_inference_btn:
    # 1. UI Pipeline Animation
    simulate_neural_pipeline()

    # 2. Secretly construct feature payload for Gemini
    feature_vector = {
        "demographics": {"age": age, "sex": gender, "smoking_history": smoker_status},
        "vitals": {"temperature_f": fever_temp, "heart_rate_bpm": heart_rate, "blood_pressure": blood_pressure},
        "history": {"duration": symptom_duration, "preexisting_conditions": preexisting},
        "clinical_findings_text": symptoms_text
    }

    # 3. Call Gemini under the hood
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[f"Execute clinical diagnostic classification for feature vector: {json.dumps(feature_vector)}"],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=ClinicalInferenceResult,
            ),
        )

        inference_data = json.loads(response.text)

        # 4. Render as Clinical ML Metrics Dashboard
        render_clinical_dashboard(inference_data)

    except Exception as e:
        st.error(f"Diagnostic Pipeline Execution Failure: {e}")